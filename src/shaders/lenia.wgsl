// ---------------------------------------------------------------------------
// Lenia simulation step: Bert Chan's continuous cellular automaton in its
// multi-channel, multi-kernel ("expanded") form.
//
//   U_k = K_k * A_src(k)                          (kernels normalised to sum 1)
//   G_k = 2 exp(-(U_k - mu_k)^2 / (2 sigma_k^2)) - 1
//   A_c <- clip(A_c + dt * sum_{k -> c} h_k G_k / sum_{k -> c} h_k, 0, 1)
//
// The state is three f32 planes (channel-major) in ping-ponged storage buffers
// on a torus. Kernels that read the same channel are packed four at a time into
// a *group*, and every group is one dispatch of `cs_step`:
//
// 1. the workgroup copies its 32x32 output tile plus a `halo`-cell apron of the
//    source channel into shared memory (wrapping around the torus);
// 2. each thread convolves four horizontally adjacent cells with the group's
//    four kernels at once. Tap weights are vec4s (one lane per kernel) stored
//    row by row with empty taps trimmed away, and a register sliding window
//    lets one shared-memory read plus one weight fetch feed 16 multiply-adds;
// 3. the four potentials go through their growth functions and are mixed into
//    per-channel growth rates, accumulated in `growth` across the groups of a
//    step. The last group of the step applies the update into the other buffer.
//
// lenia.rs prepends `const TILE_CAP: u32 = ...;` (floats of shared memory),
// chosen from the device's workgroup-storage limit, before compiling this.
// ---------------------------------------------------------------------------

struct Group {
    size: vec2<u32>,
    src: u32,        // channel read by this group's kernels
    flags: u32,      // bit 0: first group of the step, bit 1: last group
    halo: u32,       // apron width = ceil(largest kernel radius in the group)
    tile_w: u32,     // shared-tile row stride (odd, so rows start on different banks)
    row_start: u32,  // first row descriptor of this group in `rows`
    row_count: u32,
    dt: f32,
    _p0: u32,
    // Size of the torus the domain wraps on: the whole domain for the world,
    // or one nursery cell (a multiple of TILE) when hatching creatures in
    // many isolated tori at once.
    wrap: vec2<u32>,
    mu: vec4<f32>,     // growth centre per kernel slot
    inv2s2: vec4<f32>, // 1 / (2 sigma^2) per kernel slot
    // Column k shares kernel slot k's growth out to the channels:
    // h_k / sum(h) into its target channel, zero for empty slots.
    to_channel: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> grp: Group;
@group(0) @binding(1) var<storage, read> state_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> state_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> growth: array<vec4<f32>>;
// Row descriptors: (first weight, weight count including 3 zero pads, tile offset, unused).
@group(0) @binding(4) var<storage, read> rows: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> taps: array<vec4<f32>>;

const TILE: u32 = 32u;     // output cells per workgroup along x and y
const THREADS: u32 = 256u; // 8 x 32 threads, each owning 4 cells along x

var<workgroup> tile: array<f32, TILE_CAP>;

// Turns one cell's four potentials into growth, accumulates it and, in the
// last group of the step, applies the clipped Euler update.
fn finish(x: u32, y: u32, u: vec4<f32>) {
    if (x >= grp.size.x) {
        return;
    }
    let d = u - grp.mu;
    let g = 2.0 * exp(-d * d * grp.inv2s2) - 1.0;
    var rate = grp.to_channel * g;
    let idx = y * grp.size.x + x;
    if ((grp.flags & 1u) == 0u) {
        rate += growth[idx];
    }
    growth[idx] = rate;
    if ((grp.flags & 2u) != 0u) {
        let n = grp.size.x * grp.size.y;
        state_out[idx] = clamp(state_in[idx] + grp.dt * rate.x, 0.0, 1.0);
        state_out[n + idx] = clamp(state_in[n + idx] + grp.dt * rate.y, 0.0, 1.0);
        state_out[2u * n + idx] = clamp(state_in[2u * n + idx] + grp.dt * rate.z, 0.0, 1.0);
    }
}

@compute @workgroup_size(8, 32)
fn cs_step(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let size = grp.size;
    let span = TILE + 2u * grp.halo;
    // Corner of the torus this workgroup lives on (0 unless nursery cells).
    let torus = (wg.xy * TILE / grp.wrap) * grp.wrap;
    let origin = vec2<i32>(wg.xy * TILE - torus) - vec2<i32>(i32(grp.halo));
    let plane = grp.src * size.x * size.y;
    for (var i = li; i < span * span; i += THREADS) {
        let ty = i / span;
        let tx = i - ty * span;
        let q = vec2<u32>(wrap_i(origin + vec2<i32>(i32(tx), i32(ty)), vec2<i32>(grp.wrap))) + torus;
        tile[ty * grp.tile_w + tx] = state_in[plane + q.y * size.x + q.x];
    }
    workgroupBarrier();

    // Sliding window: at tap t of a row, cell j of the thread's four needs the
    // weight t - j, so the last three weights ride along in registers.
    let base = lid.y * grp.tile_w + lid.x * 4u;
    var u0 = vec4<f32>(0.0);
    var u1 = vec4<f32>(0.0);
    var u2 = vec4<f32>(0.0);
    var u3 = vec4<f32>(0.0);
    let row_end = grp.row_start + grp.row_count;
    for (var r = grp.row_start; r < row_end; r++) {
        let row = rows[r];
        var w1 = vec4<f32>(0.0);
        var w2 = vec4<f32>(0.0);
        var w3 = vec4<f32>(0.0);
        var t = base + row.z;
        let end = row.x + row.y;
        for (var k = row.x; k < end; k++) {
            let v = tile[t];
            let w0 = taps[k];
            u0 += v * w0;
            u1 += v * w1;
            u2 += v * w2;
            u3 += v * w3;
            w3 = w2;
            w2 = w1;
            w1 = w0;
            t++;
        }
    }

    let y = wg.y * TILE + lid.y;
    if (y < size.y) {
        let x = wg.x * TILE + lid.x * 4u;
        finish(x, y, u0);
        finish(x + 1u, y, u1);
        finish(x + 2u, y, u2);
        finish(x + 3u, y, u3);
    }
}
