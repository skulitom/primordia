// ---------------------------------------------------------------------------
// Lenia display: lights the composed state as a height field and colours each
// channel through its own palette, tinting the medium with the light that
// life casts into it (`cs_light` in lenia_render.wgsl).
//
// A module of its own: DirectX 12 translates a whole module for every
// pipeline, and its samplers must not share binding numbers with the compute
// passes' buffers (see the conventions in src/world/mod.rs).
// ---------------------------------------------------------------------------

struct Draw {
    view: ViewXform,
    size: vec2<f32>,     // domain in cells
    grid: vec2<u32>,     // light field in blocks
    channels: u32,
    relief: f32,         // height-field lighting strength
    gain: f32,           // density -> palette position
    brightness: f32,
    tint: f32,           // palette position of halos and wakes
    mix_power: f32,      // colour mixing: 1 blends channels evenly, higher lets the dominant one win
    ground: f32,         // strength of the medium's own deep hue
    medium: f32,         // strength of the light that life casts into the medium
    level: vec4<f32>,    // per channel: brightness
    rim: vec4<f32>,      // per channel: luminous membrane along creature edges
    core: vec4<f32>,     // per channel: density where nuclei start to glow
    glow: vec4<f32>,     // per channel: HDR emission of those nuclei
    hue: vec4<f32>,      // per channel: palette position where bodies start
    ground_channel: u32, // channel whose palette tints the medium
    sharp: f32,          // colour reconstruction: 0 B-spline (soft), 1 Catmull-Rom, 2/3 Mitchell-Netravali
    _p0: f32,
    _p1: f32,
};

@group(0) @binding(0) var<uniform> draw: Draw;
@group(0) @binding(1) var state_tex: texture_2d<f32>;
@group(0) @binding(2) var glow_tex: texture_2d<f32>;
@group(0) @binding(3) var wrap_samp: sampler;
@group(0) @binding(4) var lut0: texture_2d<f32>;
@group(0) @binding(5) var lut1: texture_2d<f32>;
@group(0) @binding(6) var lut2: texture_2d<f32>;
@group(0) @binding(7) var lut_samp: sampler;
@group(0) @binding(8) var<storage, read> dlight: array<vec4<f32>>;

// Cubic B-spline weights of the four taps around a sample at fraction `f`.
fn bspline(f: f32) -> vec4<f32> {
    let f2 = f * f;
    let f3 = f2 * f;
    return vec4<f32>(1.0 - 3.0 * f + 3.0 * f2 - f3, 4.0 - 6.0 * f2 + 3.0 * f3, 1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3, f3)
        * (1.0 / 6.0);
}

// Cubic B-spline reconstruction from four bilinear taps (Sigg & Hadwiger,
// GPU Gems 2 ch. 20). C2-smooth, so relief lighting shows no bilinear diamonds
// in close-ups; the repeat sampler wraps the torus seamlessly. It blurs a
// little, so it only feeds the lighting and the glow.
fn sample_smooth(t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let p = uv * draw.size - 0.5;
    let i = floor(p);
    let f = p - i;
    let wx = bspline(f.x);
    let wy = bspline(f.y);
    let g0 = vec2<f32>(wx.x + wx.y, wy.x + wy.y);
    let g1 = vec2<f32>(wx.z + wx.w, wy.z + wy.w);
    let h0 = (i - 0.5 + vec2<f32>(wx.y, wy.y) / g0) / draw.size;
    let h1 = (i + 1.5 + vec2<f32>(wx.w, wy.w) / g1) / draw.size;
    let a = textureSampleLevel(t, wrap_samp, vec2<f32>(h0.x, h0.y), 0.0);
    let b = textureSampleLevel(t, wrap_samp, vec2<f32>(h1.x, h0.y), 0.0);
    let c = textureSampleLevel(t, wrap_samp, vec2<f32>(h0.x, h1.y), 0.0);
    let d = textureSampleLevel(t, wrap_samp, vec2<f32>(h1.x, h1.y), 0.0);
    return g0.y * (g0.x * a + g1.x * b) + g1.y * (g0.x * c + g1.x * d);
}

// Catmull-Rom reconstruction from nine bilinear taps (the middle two weights
// of each axis share one tap). It interpolates the cells exactly, so colour
// stays crisp when zoomed in; it can overshoot slightly, so callers clamp.
fn sample_sharp(t: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let p = uv * draw.size;
    let c1 = floor(p - 0.5) + 0.5;
    let f = p - c1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let p0 = (c1 - 1.0) / draw.size;
    let p12 = (c1 + w2 / w12) / draw.size;
    let p3 = (c1 + 2.0) / draw.size;
    var s = vec4<f32>(0.0);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p0.x, p0.y), 0.0) * (w0.x * w0.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p12.x, p0.y), 0.0) * (w12.x * w0.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p3.x, p0.y), 0.0) * (w3.x * w0.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p0.x, p12.y), 0.0) * (w0.x * w12.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p12.x, p12.y), 0.0) * (w12.x * w12.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p3.x, p12.y), 0.0) * (w3.x * w12.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p0.x, p3.y), 0.0) * (w0.x * w3.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p12.x, p3.y), 0.0) * (w12.x * w3.y);
    s += textureSampleLevel(t, wrap_samp, vec2<f32>(p3.x, p3.y), 0.0) * (w3.x * w3.y);
    return s;
}

fn height(uv: vec2<f32>) -> f32 {
    return dot(sample_smooth(state_tex, uv).rgb, vec3<f32>(1.0));
}

// Light of the medium at world uv: a cubic B-spline over the block grid
// (smooth, no blocky facets), which is stretched to span the torus exactly
// so it wraps without a seam.
fn medium_light(uv: vec2<f32>) -> vec3<f32> {
    let p = uv * vec2<f32>(draw.grid) - 0.5;
    let i = vec2<i32>(floor(p));
    let f = p - floor(p);
    let wx = bspline(f.x);
    let wy = bspline(f.y);
    var sum = vec3<f32>(0.0);
    for (var y = 0; y < 4; y++) {
        var row = vec3<f32>(0.0);
        for (var x = 0; x < 4; x++) {
            row += dlight[wrap_index(i + vec2<i32>(x - 1, y - 1), draw.grid)].xyz * wx[x];
        }
        sum += row * wy[y];
    }
    return sum;
}

fn lut(c: u32, t: f32) -> vec3<f32> {
    if (c == 0u) {
        return palette_lookup(lut0, lut_samp, t);
    }
    if (c == 1u) {
        return palette_lookup(lut1, lut_samp, t);
    }
    return palette_lookup(lut2, lut_samp, t);
}

@fragment
fn fs_draw(in: FullscreenOut) -> @location(0) vec4<f32> {
    let w = view_apply(draw.view, in.uv);
    let texel = 1.0 / draw.size;
    let on = vec3<f32>(vec3<u32>(0u, 1u, 2u) < vec3<u32>(draw.channels));
    // Colour: a blend of Catmull-Rom (follows the cells exactly) and the
    // B-spline (soft). Both cubics are affine in Mitchell and Netravali's
    // (B, C), so the default blend of 2/3 is their filter B = C = 1/3: crisp
    // close-ups without the ringing Catmull-Rom leaves on diagonal edges.
    var cells = sample_sharp(state_tex, w).rgb;
    if (draw.sharp < 1.0) {
        cells = mix(sample_smooth(state_tex, w).rgb, cells, draw.sharp);
    }
    let a = clamp(cells, vec3<f32>(0.0), vec3<f32>(1.0)) * on;
    let glow = max(sample_smooth(glow_tex, w).rgb, vec3<f32>(0.0)) * on;
    let total = dot(a, vec3<f32>(1.0));

    // Total density as a height field: its slope outlines every creature
    // (a luminous membrane) and lights it like translucent jelly.
    let slope = 0.5 * vec2<f32>(
        height(w + vec2<f32>(texel.x, 0.0)) - height(w - vec2<f32>(texel.x, 0.0)),
        height(w + vec2<f32>(0.0, texel.y)) - height(w - vec2<f32>(0.0, texel.y)),
    );
    let n = normalize(vec3<f32>(-slope * 5.0, 1.0));
    let l = normalize(vec3<f32>(-0.45, -0.6, 0.65));
    let hv = normalize(l + vec3<f32>(0.0, 0.0, 1.0));
    let diffuse = max(dot(n, l), 0.0);
    let spec = pow(max(dot(n, hv), 0.0), 48.0);
    let edge = smoothstep(0.015, 0.12, length(slope)) * (1.0 - smoothstep(0.4, 0.9, total));

    // Each channel through its own palette. Colours mix by each channel's
    // share of the local density raised to `mix_power` (high: the locally
    // dominant channel's hue wins; low: channels blend into gradients).
    // Bodies span 0.4 of the palette from `hue` (its middle) so they keep
    // their colour; only dense nuclei reach the light end (0.85-1) and go
    // into HDR.
    let sp = pow(max(a, vec3<f32>(1e-4)), vec3<f32>(draw.mix_power)) * on;
    let inv = 1.0 / max(dot(sp, vec3<f32>(1.0)), 1e-20);
    // Halos mix the same way, weighted by their own strength, and shine as
    // bright as the strongest of them (overlapping halos do not add to white).
    let g2 = glow * glow;
    let ginv = 1.0 / max(dot(g2, vec3<f32>(1.0)), 1e-9);
    let gpeak = max(glow.x, max(glow.y, glow.z));
    let lit = medium_light(w) * on;
    // Bodies fade in with the total density, so the faint fringe of one
    // channel at a creature's edge does not show as specks of its hue.
    let cover = smoothstep(0.0, 0.25, total);
    var body = vec3<f32>(0.0);
    var core = vec3<f32>(0.0);
    var rim = vec3<f32>(0.0);
    var aura = vec3<f32>(0.0);
    var life = 0.0;
    for (var c = 0u; c < draw.channels; c++) {
        let v = a[c];
        let lvl = draw.level[c];
        let share = sp[c] * inv * lvl;
        let t = draw.hue[c] + 0.4 * smoothstep(0.0, 1.0, v * draw.gain);
        body += share * cover * lut(c, t);
        rim += share * draw.rim[c] * lut(c, 0.62);
        // Dense cores push into HDR so bloom makes them glow.
        let hot = smoothstep(draw.core[c], 1.0, v);
        core += share * lut(c, 0.85 + 0.15 * hot) * (hot * hot * draw.glow[c]);
        aura += (g2[c] * ginv * gpeak * lvl) * lut(c, draw.tint);
        // Light cast into the medium saturates: one creature already lights
        // its surroundings, a crowd does not flood the ground.
        life += (1.0 - exp(-60.0 * lit[c])) * lvl;
    }
    // The medium is one substance: one palette's deep hue, a little lighter
    // where life is near.
    let gc = draw.ground_channel;
    let ground = lut(gc, 0.14) * draw.ground + lut(gc, 0.36) * (draw.medium * min(life, 1.5));
    let dense = smoothstep(0.02, 0.35, total);
    let shade = mix(1.0, diffuse * 1.3 + 0.25, draw.relief * dense);
    var col = ground + body * shade + core + aura + rim * edge;
    col += vec3<f32>(spec * draw.relief * 0.5 * dense);
    col = col * draw.brightness;
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(64.0)), 1.0);
}
