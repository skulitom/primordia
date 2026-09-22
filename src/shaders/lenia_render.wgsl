// ---------------------------------------------------------------------------
// Lenia: brush, respawning, explosion quench, per-frame composition and the
// light of the medium. The display is in lenia_draw.wgsl.
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
