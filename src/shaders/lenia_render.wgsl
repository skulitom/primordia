// ---------------------------------------------------------------------------
// Lenia: brush, respawning, explosion quench, per-frame composition, the
// light of the medium and the display.
//
// State planes are f32 (channel-major) as written by lenia.wgsl; `growth`
// holds each channel's growth rate from the last step (vec4 per cell).
// `mass` is shared bookkeeping in fixed point (x MASS_SCALE): [0] is the total
// mass summed by the last compose pass and [1..4] each channel's share of it;
// [4..8] hold the mass under each candidate spot of a creature stamp
// (`cs_probe`).
// ---------------------------------------------------------------------------

const MASS_SCALE: f32 = 256.0;

// === brush, respawning and revival ============================================

struct Brush {
    size: vec2<u32>,
    center: vec2<f32>, // cells (may lie outside the domain: wrapped by torus_delta)
    radius: f32,       // cells
    mode: u32,         // 1 paint noise, 2 erase, 3 noise if extinct, 4 stamp a creature
    seed: u32,
    channels: u32,
    amplitude: f32,
    threshold: u32,    // mode 3: act only while the total mass is below this; mode 4: the watched mass
    side: u32,         // mode 4: side of the stamp window in cells
    room: u32,         // mode 4: most mass a candidate spot may already hold
    watch: u32,        // mode 4: bit c set = channel c's mass counts against the threshold
    probe: u32,        // mode 4: side of the window a candidate spot must find (nearly) empty
    base: u32,         // mode 4: first float of the window in `stamp`
    _p1: u32,
    spots: array<vec4<f32>, 2>, // mode 4: four candidate centres (xy, zw); the first free one wins
};

@group(0) @binding(0) var<uniform> brush: Brush;
@group(0) @binding(1) var<storage, read_write> cells: array<f32>;
@group(0) @binding(2) var<storage, read_write> mass: array<atomic<u32>>;
// Every brood window in every quarter turn, three planes each (uploaded once
// per run; `brush.base` selects one).
@group(0) @binding(3) var<storage, read> stamp: array<f32>;

fn spot(i: u32) -> vec2<f32> {
    let s = brush.spots[i / 2u];
    return select(s.xy, s.zw, (i & 1u) == 1u);
}

// Top-left cell of a `side`-square window centred on `centre`.
fn window_origin(centre: vec2<f32>, side: u32) -> vec2<i32> {
    return vec2<i32>(floor(centre)) - vec2<i32>(i32(side / 2u));
}

// One workgroup per candidate spot: sums the mass already under the probe
// window around it (the creature plus a margin, smaller than the stamp,
// whose corners are empty anyway).
@compute @workgroup_size(256)
fn cs_probe(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    let side = brush.probe;
    let origin = window_origin(spot(wg.x), side);
    let n = brush.size.x * brush.size.y;
    let count = side * side;
    var sum = 0.0;
    for (var k = li; k < count; k += 256u) {
        let offset = vec2<u32>(k % side, k / side);
        let idx = wrap_index(origin + vec2<i32>(offset), brush.size);
        for (var c = 0u; c < brush.channels; c++) {
            sum += cells[c * n + idx];
        }
    }
    atomicAdd(&mass[4u + wg.x], u32(sum * MASS_SCALE));
}

// Mode 4: copies the stamp window (one plane per channel) onto the first
// candidate spot with room, blending with max so nothing already there is
// erased.
fn stamp_creature(local: vec2<u32>) {
    // Only while the watched species are below their target mass.
    var watched = 0u;
    for (var c = 0u; c < 3u; c++) {
        if ((brush.watch & (1u << c)) != 0u) {
            watched += atomicLoad(&mass[1u + c]);
        }
    }
    if (local.x >= brush.side || local.y >= brush.side || watched >= brush.threshold) {
        return;
    }
    var pick = 4u;
    for (var i = 0u; i < 4u; i++) {
        if (pick == 4u && atomicLoad(&mass[4u + i]) <= brush.room) {
            pick = i;
        }
    }
    if (pick == 4u) {
        return;
    }
    let n = brush.size.x * brush.size.y;
    let idx = wrap_index(window_origin(spot(pick), brush.side) + vec2<i32>(local), brush.size);
    let plane = brush.side * brush.side;
    let s = brush.base + local.y * brush.side + local.x;
    for (var c = 0u; c < brush.channels; c++) {
        let i = c * n + idx;
        cells[i] = max(cells[i], stamp[s + c * plane]);
    }
}

@compute @workgroup_size(16, 16)
fn cs_brush(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (brush.mode == 4u) {
        stamp_creature(gid.xy);
        return;
    }
    if (gid.x >= brush.size.x || gid.y >= brush.size.y) {
        return;
    }
    if (brush.mode == 3u && atomicLoad(&mass[0]) >= brush.threshold) {
        return;
    }
    let size = vec2<f32>(brush.size);
    let d = torus_delta(brush.center, vec2<f32>(gid.xy) + 0.5, size);
    let r = length(d) / max(brush.radius, 1.0);
    if (r >= 1.0) {
        return;
    }
    let n = brush.size.x * brush.size.y;
    let idx = gid.y * brush.size.x + gid.x;
    let fall = 1.0 - smoothstep(0.65, 1.0, r);
    for (var c = 0u; c < brush.channels; c++) {
        let i = c * n + idx;
        if (brush.mode == 2u) {
            cells[i] = cells[i] * (1.0 - fall);
        } else {
            // Per-cell white noise: Lenia's classic "random soup".
            let v = rand3(gid.x, gid.y, brush.seed * 4u + c) * brush.amplitude;
            cells[i] = mix(cells[i], v, fall);
        }
    }
}

// === explosion quench =========================================================
// Some species boil over into space-filling labyrinths when creatures collide.
// Once per frame, each channel of every 16x16 block whose neighbourhood holds
// far more of that channel's mass than a few of its creatures fades a little,
// so a boiling patch dies out instead of taking over the torus. Isolated
// creatures never come near the limit. Neighbourhoods are sized per channel
// (by that species' own radius), so small species are watched as closely as
// large ones.

struct Quench {
    size: vec2<u32>,
    grid: vec2<u32>,    // blocks of 16x16 cells (one per compose workgroup)
    reach: vec4<u32>,   // per channel: neighbourhood radius in blocks
    limit: vec4<u32>,   // per channel: most mass (fixed point) a neighbourhood may hold
    keep: f32,          // factor applied to the crowded channels of a block
    channels: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<uniform> qn: Quench;
@group(0) @binding(1) var<storage, read_write> qcells: array<f32>;
@group(0) @binding(2) var<storage, read> qblocks: array<vec4<u32>>;

var<workgroup> crowded: u32; // bit c set = channel c is over its limit here

@compute @workgroup_size(16, 16)
fn cs_quench(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li == 0u) {
        let r = i32(max(qn.reach.x, max(qn.reach.y, qn.reach.z)));
        var sum = vec3<u32>(0u);
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                let m = qblocks[wrap_index(vec2<i32>(wg.xy) + vec2<i32>(dx, dy), qn.grid)].xyz;
                // Each channel only counts blocks within its own reach.
                let d = u32(max(abs(dx), abs(dy)));
                sum += select(vec3<u32>(0u), m, vec3<u32>(d) <= qn.reach.xyz);
            }
        }
        let over = sum > qn.limit.xyz;
        crowded = u32(over.x) | (u32(over.y) << 1u) | (u32(over.z) << 2u);
    }
    workgroupBarrier();
    if (crowded == 0u || gid.x >= qn.size.x || gid.y >= qn.size.y) {
        return;
    }
    let n = qn.size.x * qn.size.y;
    let idx = gid.y * qn.size.x + gid.x;
    for (var c = 0u; c < qn.channels; c++) {
        if ((crowded & (1u << c)) != 0u) {
            qcells[c * n + idx] *= qn.keep;
        }
    }
}

// === composition ============================================================
// Once per frame: packs the state and an emissive "glow" (growth-field halo +
// soft wakes) into filterable textures, and sums the mass per 16x16 block
// and in total.

struct Compose {
    size: vec2<u32>,
    channels: u32,
    decay: f32,  // wake persistence this frame (1 while paused)
    halo: f32,
    trail: f32,
    spread: f32, // how far the wake diffuses this frame (1 = a binomial 3x3 step, 0 while paused)
    _p0: f32,
    rest: vec4<f32>, // growth rate of each channel in empty space
};

@group(0) @binding(0) var<uniform> cmp: Compose;
@group(0) @binding(1) var<storage, read> cstate: array<f32>;
@group(0) @binding(2) var<storage, read> cgrowth: array<vec4<f32>>;
// Wakes are ping-ponged: each frame diffuses last frame's (read) into the other.
@group(0) @binding(3) var<storage, read> trail_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> trail_out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> total: array<atomic<u32>>;
@group(0) @binding(6) var out_state: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var out_glow: texture_storage_2d<rgba16float, write>;
@group(0) @binding(8) var<storage, read_write> blocks: array<vec4<u32>>; // per channel (xyz) and total (w)

var<workgroup> wg_mass: array<atomic<u32>, 3>;

fn density_at(p: vec2<i32>) -> vec3<f32> {
    let n = cmp.size.x * cmp.size.y;
    let i = wrap_index(p, cmp.size);
    return vec3<f32>(cstate[i], cstate[n + i], cstate[2u * n + i]);
}

@compute @workgroup_size(16, 16)
fn cs_compose(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    if (li < 3u) {
        atomicStore(&wg_mass[li], 0u);
    }
    workgroupBarrier();
    if (gid.x < cmp.size.x && gid.y < cmp.size.y) {
        let n = cmp.size.x * cmp.size.y;
        let idx = gid.y * cmp.size.x + gid.x;
        let p = vec2<i32>(gid.xy);
        let on = vec3<f32>(vec3<u32>(0u, 1u, 2u) < vec3<u32>(cmp.channels));
        let a = vec3<f32>(cstate[idx], cstate[n + idx], cstate[2u * n + idx]) * on;

        // Wake: every creature body deposits into a decaying maximum that
        // diffuses by about a cell per frame, so a glider leaves one soft
        // comet tail. The deposit comes from the density blurred over 5x5
        // cells (a sparse 3x3 at stride 2), so it is a single smooth blob
        // rather than the creature's thin dense arms.
        var blurred = vec3<f32>(0.0);
        for (var dy = -2; dy <= 2; dy += 2) {
            for (var dx = -2; dx <= 2; dx += 2) {
                blurred += density_at(p + vec2<i32>(dx, dy));
            }
        }
        let deposit = smoothstep(vec3<f32>(0.3), vec3<f32>(0.7), blurred * (1.0 / 9.0)) * on;
        var spread = vec3<f32>(0.0);
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let w = f32((2 - abs(dx)) * (2 - abs(dy))); // binomial 1-2-1
                spread += trail_in[wrap_index(p + vec2<i32>(dx, dy), cmp.size)].xyz * w;
            }
        }
        let before = mix(trail_in[idx].xyz, spread * (1.0 / 16.0), cmp.spread);
        let tr = max(before * cmp.decay, deposit);
        trail_out[idx] = vec4<f32>(tr, 0.0);
        let wake = max(tr - deposit, vec3<f32>(0.0));

        // Halo: where the growth field rises above its empty-space level, i.e.
        // the zone around each creature where its potential sits near mu.
        let rest = cmp.rest.xyz;
        let lift = clamp((cgrowth[idx].xyz - rest) / max(1.0 - rest, vec3<f32>(1e-3)), vec3<f32>(0.0), vec3<f32>(1.0));
        let glow = (lift * lift * cmp.halo + wake * cmp.trail) * on;
        textureStore(out_state, p, vec4<f32>(a, 0.0));
        textureStore(out_glow, p, vec4<f32>(glow, 0.0));
        let m = vec3<u32>(a * MASS_SCALE);
        atomicAdd(&wg_mass[0], m.x);
        atomicAdd(&wg_mass[1], m.y);
        atomicAdd(&wg_mass[2], m.z);
    }
    workgroupBarrier();
    if (li == 0u) {
        let m = vec3<u32>(atomicLoad(&wg_mass[0]), atomicLoad(&wg_mass[1]), atomicLoad(&wg_mass[2]));
        let sum = m.x + m.y + m.z;
        atomicAdd(&total[0], sum);
        atomicAdd(&total[1], m.x);
        atomicAdd(&total[2], m.y);
        atomicAdd(&total[3], m.z);
        blocks[wg.y * groups.x + wg.x] = vec4<u32>(m, sum);
    }
}

// === light of the medium =======================================================
// A coarse field (one value per 16x16 block and channel) of how much life is
// nearby: block masses blurred over about a hundred cells and smoothed over
// time. The display tints the dark ground with it, so the medium glows
// faintly around crowds and along recent paths.

struct Light {
    grid: vec2<u32>,
    keep: f32,  // share of last frame's light kept (1 while paused, 0 on a fresh run)
    scale: f32, // fixed-point block mass -> mean density of the block
};

@group(0) @binding(0) var<uniform> lt: Light;
@group(0) @binding(1) var<storage, read> lblocks: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> light: array<vec4<f32>>;

@compute @workgroup_size(8, 8)
fn cs_light(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= lt.grid.x || gid.y >= lt.grid.y) {
        return;
    }
    // Gaussian over 7x7 blocks (sigma 1.6 blocks), wrapping around the torus.
    var sum = vec3<f32>(0.0);
    var weight = 0.0;
    for (var dy = -3; dy <= 3; dy++) {
        for (var dx = -3; dx <= 3; dx++) {
            let w = exp(-f32(dx * dx + dy * dy) * (1.0 / (2.0 * 1.6 * 1.6)));
            sum += vec3<f32>(lblocks[wrap_index(vec2<i32>(gid.xy) + vec2<i32>(dx, dy), lt.grid)].xyz) * w;
            weight += w;
        }
    }
    let idx = gid.y * lt.grid.x + gid.x;
    let now = sum * (lt.scale / weight);
    light[idx] = vec4<f32>(mix(now, light[idx].xyz, lt.keep), 0.0);
}

// === display ================================================================

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
