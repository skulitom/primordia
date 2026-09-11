// Particle Life rendering.
//
//   fs_fade      : dims the HDR trail texture (multiplicative blend)
//   vs/fs_trail  : soft glows splatted additively into the trail. Each glow is
//                  stretched back along the path its particle covered since the
//                  last frame, so fast movers leave continuous streaks rather
//                  than a bead per frame
//   vs/fs_head   : crisp anti-aliased discs, additively into a cleared buffer
//   fs_composite : average colour x log-compressed density into the scene
//
// The trail and heads textures ("the canvas") normally match the output
// pixels. Particles are instanced 4-vertex quads: instance `ii` draws particle
// `ii % count` in torus tile `ii / count`, so particles near an edge (and the
// tiling when zoomed out) appear wherever the camera looks. When the view spans
// many periods, the canvas instead holds exactly one domain period (the sprite
// view is then the identity) and the composite repeats it with wrap_i.

struct Draw {
    view: ViewXform,     // sprite passes: canvas uv -> world uv
    screen: ViewXform,   // composite: output uv -> world uv (the camera)
    domain: vec2<f32>,
    canvas: vec2<f32>,   // canvas size in pixels
    tiles: vec2<u32>,
    count: u32,
    wrap: u32,           // 1 = the canvas holds one domain period
    head_radius: f32,    // world units
    trail_radius: f32,   // world units
    head_gain: f32,
    trail_gain: f32,
    speed_ref: f32,      // world units / s at which a particle is ~63% "hot"
    speed_glow: f32,
    fade: f32,
    knee: f32,           // highlight roll-off threshold
    motion: f32,         // simulated seconds since the last frame (streak length)
    relief: f32,         // strength of the density shading (0 = flat)
    zoom1_ppu: f32,      // canvas pixels per world unit at zoom 1 (glow widths)
    _pad0: f32,
    ground: vec4<f32>,
    colors: array<vec4<f32>, 8>, // rgb = linear colour, a = size multiplier
};

@group(0) @binding(0) var<uniform> draw: Draw;
@group(0) @binding(1) var<storage, read> particles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> velocities: array<vec2<f32>>;
@group(1) @binding(0) var trail_tex: texture_2d<f32>;
@group(1) @binding(1) var heads_tex: texture_2d<f32>;

// Quad half-size of each kernel in multiples of its radius.
const HEAD_EXTENT: f32 = 2.0;
const TRAIL_EXTENT: f32 = 3.0;
// Discs are never drawn thinner than this (their energy is scaled down instead).
const MIN_RADIUS_PX: f32 = 0.8;
const SQRT_PI: f32 = 1.77245385;

struct SpriteOut {
    @builtin(position) position: vec4<f32>,
    // Pixels from the particle centre in the streak frame: x along the motion
    // (0 = now, negative = behind), y across it.
    @location(0) local: vec2<f32>,
    @location(1) color: vec3<f32>, // species colour x brightness
    @location(2) radius: f32,      // kernel radius in pixels
    @location(3) weight: f32,      // coverage weight (gain x energy)
    @location(4) streak: f32,      // streak length in pixels (0 for heads)
    @location(5) core: f32,        // flat core of a zoomed-in glow, in pixels
};

fn sprite(vi: u32, ii: u32, trail: bool) -> SpriteOut {
    var out: SpriteOut;
    let n = max(draw.count, 1u);
    let a = particles[ii % n];
    let v = velocities[ii % n];
    let tile = ii / n;
    let tint = draw.colors[min(u32(a.z), 7u)];

    // Kernel size in pixels. Zooming in magnifies the particle but not its
    // glow: beyond zoom 1 a trail glow becomes a flat core as wide as the
    // head's growth, followed by the Gaussian falloff it has at zoom 1, so a
    // close-up shows crisp discs with a thin rim of light instead of a haze.
    // (At zoom <= 1 the core is empty and the kernel is the plain Gaussian.)
    // Below MIN_RADIUS_PX the footprint is kept and the intensity lowered
    // instead, so zoomed-out views keep their overall brightness.
    let px_per_unit = draw.canvas.x / (draw.view.scale.x * draw.domain.x);
    let head_px = draw.head_radius * tint.a * px_per_unit;
    var r_true = head_px;
    var core = 0.0;
    var extent = HEAD_EXTENT;
    var gain = draw.head_gain;
    var streak = vec2<f32>(0.0);
    if (trail) {
        let ppu = min(px_per_unit, draw.zoom1_ppu);
        r_true = draw.trail_radius * tint.a * ppu;
        core = max(px_per_unit - draw.zoom1_ppu, 0.0) * draw.head_radius * tint.a;
        extent = TRAIL_EXTENT;
        gain = draw.trail_gain;
        // Pixels (y down, like world uv) covered since the last frame.
        streak = v * (draw.motion * px_per_unit);
    }
    let r_px = max(r_true, MIN_RADIUS_PX);
    let energy = (r_true * r_true) / (r_px * r_px);
    let half_px = core + r_px * extent + 1.0;
    let len = length(streak);
    let dir = select(vec2<f32>(1.0, 0.0), streak / max(len, 1e-6), len > 1e-3);

    // The copies of this particle that can touch the canvas are uv + k for
    // integer k with uv + k in [offset - margin, offset + scale + margin];
    // instance `tile` takes the tile-th one of them. The margin covers the
    // quad and its streak.
    let uv = a.xy / draw.domain;
    let margin = (half_px + len) / draw.canvas * draw.view.scale;
    let first = ceil(draw.view.offset - margin - uv);
    let world = uv + first + vec2<f32>(f32(tile % draw.tiles.x), f32(tile / draw.tiles.x));
    let visible = all(world <= draw.view.offset + draw.view.scale + margin);
    let centre = (world - draw.view.offset) / draw.view.scale * draw.canvas;

    // The quad runs from half_px behind the streak's tail to half_px ahead of
    // the particle, and half_px to either side.
    let along = mix(-len - half_px, half_px, f32(vi & 1u));
    let side = (f32(vi >> 1u) * 2.0 - 1.0) * half_px;
    let pix = centre + dir * along + vec2<f32>(-dir.y, dir.x) * side;
    let ndc = vec2<f32>(pix.x / draw.canvas.x * 2.0 - 1.0, 1.0 - pix.y / draw.canvas.y * 2.0);
    out.position = select(vec4<f32>(-4.0, -4.0, 0.0, 1.0), vec4<f32>(ndc, 0.0, 1.0), visible);
    out.local = vec2<f32>(along, side);
    out.radius = r_px;
    out.core = core;
    out.streak = len;

    // Colour from the species; fast particles burn brighter in their own hue
    // (never whiter, so every species keeps its identity when it blooms).
    let heat = 1.0 - exp(-length(v) / max(draw.speed_ref, 1e-3));
    let jitter = 0.8 + 0.4 * f32(pcg_hash(u32(a.w)) & 255u) / 255.0;
    let brightness = (0.55 + 0.6 * draw.speed_glow * heat) * jitter;
    out.color = tint.rgb * brightness;
    // The round kernel (flat core c, then a Gaussian of radius r) covers
    // pi (c^2 + sqrt(pi) c r + r^2); swept along a segment of length L it gains
    // L (2c + sqrt(pi) r). Divide by the ratio so a streak carries the same light.
    let round = PI * (core * core + SQRT_PI * core * r_px + r_px * r_px);
    out.weight = gain * energy / (1.0 + len * (2.0 * core + SQRT_PI * r_px) / round);
    return out;
}

@vertex
fn vs_head(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SpriteOut {
    return sprite(vi, ii, false);
}

@vertex
fn vs_trail(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SpriteOut {
    return sprite(vi, ii, true);
}

@fragment
fn fs_head(in: SpriteOut) -> @location(0) vec4<f32> {
    let d = length(in.local);
    let x = d / in.radius;
    // Anti-aliased disc, slightly brighter in the middle, with a tight halo.
    let disc = clamp(in.radius + 0.5 - d, 0.0, 1.0) * (1.25 - 0.5 * min(x * x, 1.0));
    let halo = 0.22 * exp(-2.5 * x * x);
    let w = in.weight * (disc + halo);
    return vec4<f32>(in.color * w, w);
}

@fragment
fn fs_trail(in: SpriteOut) -> @location(0) vec4<f32> {
    // Distance to the streak segment from (-streak, 0) to (0, 0), past the core.
    let behind = in.local.x - clamp(in.local.x, -in.streak, 0.0);
    let x = max(length(vec2<f32>(behind, in.local.y)) - in.core, 0.0) / in.radius;
    let w = in.weight * exp(-x * x);
    return vec4<f32>(in.color * w, w);
}

// Multiplied into the trail texture (blend: dst * src).
@fragment
fn fs_fade(in: FullscreenOut) -> @location(0) vec4<f32> {
    return vec4<f32>(draw.fade);
}

// Both buffers hold (sum of colour x weight, sum of weight). Their ratio is the
// local average colour. Brightness follows the accumulated weight linearly up
// to the knee and only logarithmically beyond it, so a dense cluster of
// hundreds of particles shows gradients rather than a flat highlight (and does
// not flood the bloom).
//
// Where species overlap, the average drifts towards grey (magenta + cyan is
// lavender). A chroma floor pushes such mixtures back out from their peak
// channel, so they keep a hue instead of reading white once tonemapped.
const CHROMA_FLOOR: f32 = 0.55;
// Above HIGHLIGHT_START (peak channel), light is compressed along its own hue
// towards HIGHLIGHT_LIMIT, so the densest, fastest streaks stay coloured
// instead of running into the tonemapper's white shoulder.
const HIGHLIGHT_START: f32 = 1.0;
const HIGHLIGHT_LIMIT: f32 = 1.5;
// Density shading: log-density is treated as a height field and lit from the
// upper left, so dense cores read as domes with a lit and a shaded side.
const RELIEF_STEP: i32 = 2;           // pixels between gradient samples
const RELIEF_SCALE: f32 = 6.0;        // height-field exaggeration
const RELIEF_LIGHT: vec3<f32> = vec3<f32>(-0.48, -0.6, 0.64);

// Summed canvas at texel `p`: wrapped onto the period in wrap mode, clamped to
// the edge otherwise (textureLoad out of bounds is undefined).
fn canvas_sum(p: vec2<i32>) -> vec4<f32> {
    let size = vec2<i32>(textureDimensions(trail_tex));
    var q = clamp(p, vec2<i32>(0), size - vec2<i32>(1));
    if (draw.wrap != 0u) {
        q = wrap_i(p, size);
    }
    return clamp(textureLoad(trail_tex, q, 0) + textureLoad(heads_tex, q, 0), vec4<f32>(0.0), vec4<f32>(60000.0));
}

fn height(p: vec2<i32>, k: f32) -> f32 {
    return log(1.0 + canvas_sum(p).a / k);
}

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    var p = vec2<i32>(in.position.xy);
    if (draw.wrap != 0u) {
        // The canvas is one period: find this pixel's place in it.
        p = vec2<i32>(floor(view_apply(draw.screen, in.uv) * draw.canvas));
    }
    let sum = canvas_sum(p);
    let k = max(draw.knee, 0.05);
    let h = log(1.0 + sum.a / k);

    var color = sum.rgb / max(sum.a, 1e-4);
    let mx = max(max(color.r, color.g), color.b);
    let mn = min(min(color.r, color.g), color.b);
    let chroma = (mx - mn) / max(mx, 1e-4);
    if (chroma < CHROMA_FLOOR) {
        color = vec3<f32>(mx) - (vec3<f32>(mx) - color) * (CHROMA_FLOOR / max(chroma, 0.08));
    }
    var light = color * (k * h);

    if (draw.relief > 0.0) {
        let dx = vec2<i32>(RELIEF_STEP, 0);
        let dy = vec2<i32>(0, RELIEF_STEP);
        let grad = vec2<f32>(height(p + dx, k) - height(p - dx, k), height(p + dy, k) - height(p - dy, k))
            * (RELIEF_SCALE / f32(2 * RELIEF_STEP));
        let normal = normalize(vec3<f32>(-grad, 1.0));
        let lambert = max(dot(normal, normalize(RELIEF_LIGHT)), 0.0) / normalize(RELIEF_LIGHT).z;
        // Only dense structures are shaded; thin lace and dust stay flat.
        let amount = draw.relief * smoothstep(1.0, 3.0, h);
        light = light * mix(1.0, lambert, amount);
    }

    let peak = max(max(light.r, light.g), light.b);
    let over = max(peak - HIGHLIGHT_START, 0.0);
    let span = HIGHLIGHT_LIMIT - HIGHLIGHT_START;
    let rolled = min(peak, HIGHLIGHT_START + over / (1.0 + over / span));
    light = light * (rolled / max(peak, 1e-4));
    return vec4<f32>(draw.ground.rgb + light, 1.0);
}
