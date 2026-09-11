// Particle Life simulation (Jeffrey Ventrella's "Clusters", Tom Mohr's force model).
//
// N particles of K species live on a torus. Species i feels species j through
// an asymmetric attraction A[i][j] in [-1, 1] and a piecewise-linear kernel of
// the normalised distance r = d / (r_max * R[i][j]):
//
//   r <  beta : r / beta - 1                                   (universal repulsion)
//   r <  1    : A[i][j] * (1 - |2r - 1 - beta| / (1 - beta))    (tent of attraction)
//   otherwise : 0
//
// Neighbour search uses a uniform grid (cell size >= r_max) that is rebuilt
// every sub-step with a counting sort:
//   cs_count     -> cell of each particle + its rank inside the cell (atomics)
//   cs_scan      -> exclusive prefix sum of the counts = cell start offsets
//   cs_scatter   -> particles copied into cell order (pos_b / vel_b)
//   cs_force     -> partial forces of each sorted particle from each cell of
//                   its 3x3 block (nine threads per particle)
//   cs_integrate -> sums the partials, integrates, writes pos_a / vel_a
//
// pos = (x, y, species, id). The id is the particle's fixed index (exact as
// f32); hashed, it picks the species through the current species weights, so
// changing K or the proportions needs no respawn.

const WG: u32 = 256u;
const MAX_KINDS: u32 = 8u;
// Forces are accumulated as fixed-point integers. Integer addition is
// associative, so the sum does not depend on the (atomic, hence arbitrary)
// order particles land in within a cell, and every run is reproducible.
const FIXED: f32 = 16384.0;

struct Sim {
    domain: vec2<f32>,
    grid: vec2<u32>,
    cell: vec2<f32>,
    count: u32,
    kinds: u32,
    r_max: f32,
    beta: f32,
    // Acceleration per unit of summed kernel force (already scaled by r_max and
    // the neighbour-count normalisation).
    force: f32,
    // Velocity multiplier per sub-step (from the friction half-life).
    friction: f32,
    dt: f32,
    max_speed: f32,
    pointer: vec2<f32>,
    pointer_radius: f32,
    pointer_mode: u32, // 0 = none, 1 = attract, 2 = repel
    // Spring constant of the attracting brush (1/s^2); the repelling brush
    // pushes with pointer_strength * pointer_radius (units/s^2).
    pointer_strength: f32,
    cells: u32,
    salt: u32, // per-seed salt of the species hash
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    mat: array<vec4<f32>, 16>, // A[i][j] at flat index i * 8 + j
    rad: array<vec4<f32>, 16>, // R[i][j] (radius multipliers)
    cdf: array<vec4<f32>, 2>,  // cumulative species weights
};

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read_write> pos_a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel_a: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> pos_b: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel_b: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> starts: array<u32>;
// Scratch (9 x count entries): (cell, rank) per particle between cs_count and
// cs_scatter, then fixed-point partial forces between cs_force and cs_integrate.
@group(0) @binding(7) var<storage, read_write> scratch: array<vec2<u32>>;

fn particle_index(gid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return gid.x + gid.y * nwg.x * WG;
}

fn cell_of(p: vec2<f32>) -> vec2<i32> {
    let c = vec2<i32>(floor(p / sim.cell));
    return clamp(c, vec2<i32>(0), vec2<i32>(sim.grid) - vec2<i32>(1));
}

// Species of particle `id` from its hashed gene and the cumulative weights
// (mirrored on the CPU in particle_life.rs). 840 = lcm(1..8), so equal weights
// split exactly evenly.
fn species(id: u32) -> f32 {
    let u = (f32(pcg_hash(id ^ sim.salt) % 840u) + 0.5) / 840.0;
    var k = 0u;
    for (var j = 0u; j + 1u < sim.kinds; j++) {
        k += select(0u, 1u, u >= sim.cdf[j / 4u][j % 4u]);
    }
    return f32(k);
}

// --- 1. bin particles into grid cells ----------------------------------------

@compute @workgroup_size(256)
fn cs_count(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = particle_index(gid, nwg);
    if (i >= sim.count) {
        return;
    }
    let c = cell_of(pos_a[i].xy);
    let cell = u32(c.y) * sim.grid.x + u32(c.x);
    scratch[i] = vec2<u32>(cell, atomicAdd(&counts[cell], 1u));
}

// --- 2. prefix sum of the cell counts (single workgroup) ---------------------

var<workgroup> scan_sums: array<u32, 256>;

// Each thread owns a contiguous chunk of cells: it sums its chunk, the 256
// chunk totals are scanned in shared memory (Hillis-Steele), then each thread
// writes its chunk's start offsets. The counters are zeroed on the way out so
// the next sub-step can count again without a separate clear pass.
@compute @workgroup_size(256)
fn cs_scan(@builtin(local_invocation_index) li: u32) {
    let n = sim.cells;
    let chunk = (n + 255u) / 256u;
    let begin = min(li * chunk, n);
    let end = min(begin + chunk, n);
    var total = 0u;
    for (var c = begin; c < end; c++) {
        total += atomicLoad(&counts[c]);
    }
    scan_sums[li] = total;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset = offset << 1u) {
        var add = 0u;
        if (li >= offset) {
            add = scan_sums[li - offset];
        }
        workgroupBarrier();
        scan_sums[li] = scan_sums[li] + add;
        workgroupBarrier();
    }
    var run = scan_sums[li] - total;
    for (var c = begin; c < end; c++) {
        let k = atomicLoad(&counts[c]);
        starts[c] = run;
        run += k;
        atomicStore(&counts[c], 0u);
    }
    if (li == 255u) {
        starts[n] = scan_sums[255];
    }
}

// --- 3. scatter into cell order ------------------------------------------------

@compute @workgroup_size(256)
fn cs_scatter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = particle_index(gid, nwg);
    if (i >= sim.count) {
        return;
    }
    let slot = scratch[i];
    let dst = starts[slot.x] + slot.y;
    var p = pos_a[i];
    p.z = species(u32(p.w));
    pos_b[dst] = p;
    vel_b[dst] = vel_a[i];
}

// --- 4. pairwise forces: one thread per (particle, neighbouring cell) ----------
//
// Thread t handles sorted particle i = t % count against cell b = t / count of
// its 3x3 block. Consecutive threads are consecutive sorted particles (mostly
// sharing a cell) looking at the same neighbour cell, so a warp walks one
// particle list in lockstep; splitting the block nine ways also keeps the GPU
// busy when a few cells are very crowded. Partial sums go to `scratch` as
// fixed-point integers and are added up in a fixed order by cs_integrate.

var<workgroup> wg_mat: array<f32, 64>;
var<workgroup> wg_inv_radius: array<f32, 64>;

@compute @workgroup_size(256)
fn cs_force(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    // Stage the K x K tables in workgroup memory: lookups with a per-pair
    // (divergent) index are much cheaper there than in the uniform buffer.
    if (li < MAX_KINDS * MAX_KINDS) {
        wg_mat[li] = sim.mat[li / 4u][li % 4u];
        wg_inv_radius[li] = 1.0 / (sim.r_max * sim.rad[li / 4u][li % 4u]);
    }
    workgroupBarrier();

    let t = particle_index(gid, nwg);
    if (t >= sim.count * 9u) {
        return;
    }
    let i = t % sim.count;
    let block = t / sim.count;

    let me = pos_b[i];
    let p = me.xy;
    let row = min(u32(me.z), MAX_KINDS - 1u) * MAX_KINDS;
    let grid = vec2<i32>(sim.grid);

    // Neighbour cell on the torus, plus the whole-period shift that brings its
    // particles next to us: raw - wrapped is exactly 0 or +-grid per axis.
    let raw = cell_of(p) + vec2<i32>(i32(block % 3u) - 1, i32(block / 3u) - 1);
    let n = wrap_i(raw, grid);
    let offset = vec2<f32>((raw - n) / grid) * sim.domain;

    let reach2 = sim.r_max * sim.r_max;
    let beta = sim.beta;
    let inv_beta = 1.0 / beta;
    let inv_band = 1.0 / (1.0 - beta);
    let shift = offset - p;
    let cell = u32(n.y) * sim.grid.x + u32(n.x);
    let end = starts[cell + 1u];
    var acc = vec2<i32>(0);
    for (var j = starts[cell]; j < end; j++) {
        let q = pos_b[j];
        let d = q.xy + shift;
        let d2 = dot(d, d);
        if (d2 < reach2 && d2 > 1e-6) {
            let k = row + u32(q.z);
            let dist = sqrt(d2);
            let r = dist * wg_inv_radius[k];
            if (r < 1.0) {
                let attract = wg_mat[k] * (1.0 - abs(2.0 * r - 1.0 - beta) * inv_band);
                let f = select(attract, r * inv_beta - 1.0, r < beta);
                acc += vec2<i32>(d * (f / dist * FIXED));
            }
        }
    }
    scratch[t] = bitcast<vec2<u32>>(acc);
}

// --- 5. integration ------------------------------------------------------------

@compute @workgroup_size(256)
fn cs_integrate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = particle_index(gid, nwg);
    if (i >= sim.count) {
        return;
    }
    var acc = vec2<i32>(0);
    for (var b = 0u; b < 9u; b++) {
        acc += bitcast<vec2<i32>>(scratch[b * sim.count + i]);
    }

    let me = pos_b[i];
    let p = me.xy;
    let force = vec2<f32>(acc) * (sim.force / FIXED);
    var v = vel_b[i] * sim.friction + force * sim.dt;

    // Brush, with a smooth bump profile w (1 at the cursor, 0 at the rim).
    if (sim.pointer_mode != 0u) {
        let d = torus_delta(p, sim.pointer, sim.domain); // particle -> cursor
        let x = length(d) / sim.pointer_radius;
        if (x < 1.0) {
            let w = (1.0 - x * x) * (1.0 - x * x);
            if (sim.pointer_mode == 1u) {
                // A spring towards the cursor: it vanishes at the centre, so the
                // gathered swarm keeps living instead of jittering on one point.
                v += d * (sim.pointer_strength * w * sim.dt);
            } else {
                let away = select(vec2<f32>(1.0, 0.0), -d / max(length(d), 1e-4), x > 1e-4);
                v += away * (sim.pointer_strength * sim.pointer_radius * 3.0 * w * sim.dt);
            }
        }
    }

    // Speed limit: a safety net against runaway forces (the component clamp
    // also bounds any value the length-based scale could not).
    let speed = length(v);
    v = v * min(1.0, sim.max_speed / max(speed, 1e-6));
    v = clamp(v, vec2<f32>(-sim.max_speed), vec2<f32>(sim.max_speed));

    // One step moves less than r_max (< domain), so a single conditional
    // period shift wraps exactly; the final clamp only guards the far edge
    // against rounding (x + domain can round up to exactly domain).
    var q = p + v * sim.dt;
    q = select(q, q + sim.domain, q < vec2<f32>(0.0));
    q = select(q, q - sim.domain, q >= sim.domain);
    q = clamp(q, vec2<f32>(0.0), sim.domain - vec2<f32>(0.001));
    // Write back in canonical (id) order rather than sorted order: particles
    // are then always drawn in the same order, so even the additive blending
    // is bit-for-bit reproducible.
    let id = min(u32(me.w), sim.count - 1u);
    pos_a[id] = vec4<f32>(q, me.z, me.w);
    vel_a[id] = v;
}
