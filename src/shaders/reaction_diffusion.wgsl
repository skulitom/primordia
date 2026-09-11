// Gray-Scott reaction-diffusion.
//
//   U + 2V -> 3V        (autocatalysis)
//   dU/dt = Du * lap(U) - U V^2 + F (1 - U)
//   dV/dt = Dv * lap(V) + U V^2 - (F + k) V
//
// State lives in two ping-ponged storage buffers of vec2(U, V) on a torus.
// Everything that varies across space (atlas sweeps, k weather, lagoons,
// pattern-size modulation, ridge flow, film swirls, paper fibre) is built
// from sines with integer wave vectors or from wrapped lattices, so it is
// periodic on the unit torus and never shows a seam.
//
// Per displayed frame:
//   cs_inject        brush strokes, rain drops and the revive safety net edit
//                    the latest state in place
//   cs_step    x N   explicit Euler sub-steps
//   cs_prepare       bakes V, U, a blurred V and |dV/dt| into a filterable
//                    texture and histograms V and U
//   cs_resolve       turns the histograms into a smoothed auto-contrast window
//   fs_display       lights the field as a height map

// --- simulation --------------------------------------------------------------

struct Sim {
    size: vec2<u32>,
    feed: f32,
    kill: f32,
    du: f32,
    dv: f32,
    dt: f32,
    atlas: u32,
    drift: f32,               // amplitude of the slow travelling k modulation
    drift_phase: f32,         // wrapped every 200 pi (see DRIFT_PERIOD in the .rs)
    scale_var: f32,           // log-amplitude of the spatial pattern-size modulation
    frame: u32,
    pointer: vec2<f32>,       // brush centre in world uv (not wrapped)
    pointer_radius: f32,      // cells
    // Bits 0-1: 0 = none, 1 = create, 2 = erase. Bit 2: the stroke continues
    // from `prev_pointer` (the same button was held on the previous frame).
    pointer_mode: u32,
    create_state: vec2<f32>,  // chemistry "create" lays down: seeds, or the filled state for foam
    erase_state: vec2<f32>,   // what "erase" restores: bare U
    spray: vec2<f32>,         // brush spray: x = grid spacing, y = dot radius (cells)
    create_jitter: f32,       // random spread added to the V of created chemistry
    // Seeds: per-frame chance of a dot per grid cell at the brush centre.
    // Foam: chance that a grid cell holds a hole.
    spray_density: f32,
    drop_count: u32,
    filled_ground: u32,       // 1 = the filled state is the ground (seeds die on bare U)
    prev_pointer: vec2<f32>,  // brush centre on the previous frame of the stroke (world uv)
    ground: f32,              // kill added inside the drifting lagoon mask
    flow_phase: f32,          // shape of the ridge-flow potential (from the seed)
    aniso: f32,               // ridge-flow anisotropy of the diffusion of V
    _pad: f32,
    // Automatic droplets for this frame: xy = centre in cells, z = radius
    // (w is padding).
    drops: array<vec4<f32>, 4>,
};

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> src: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<f32>>;
// Contrast window + revive state written by cs_resolve (see there for the layout).
@group(0) @binding(3) var<storage, read> stats: array<f32>;

// Critical (saddle-node) kill rate: non-trivial states exist for k below it,
// and every interesting morphology lives in a thin band around it.
fn critical_kill(f: f32) -> f32 {
    return sqrt(f) * 0.5 - f;
}

// Travelling plane waves with integer wave vectors, so they are periodic on
// the unit torus. Returns roughly -1..1.
fn drift_field(uv: vec2<f32>, phase: f32) -> f32 {
    let a = sin(TAU * (uv.x + uv.y) + phase);
    let b = sin(TAU * (2.0 * uv.x - uv.y) - 0.8 * phase + 2.1);
    let c = sin(TAU * (uv.x - 2.0 * uv.y) + 0.6 * phase + 4.2);
    return clamp((a + b + c) * 0.5, -1.0, 1.0);
}

fn scale_field(uv: vec2<f32>, phase: f32) -> f32 {
    let a = sin(TAU * (2.0 * uv.x + uv.y) + 0.35 * phase + 0.7);
    let b = sin(TAU * (uv.x - uv.y) - 0.25 * phase + 3.3);
    return (a + b) * 0.5;
}

// Lagoons: 0..1 over a few large regions (about a third of the torus) that
// drift slowly. `ground` lifts k there, above the survival band for presets
// that use it, so structure dies back into open water that the pattern
// re-colonises as the lagoon moves on.
fn lagoon(uv: vec2<f32>, phase: f32) -> f32 {
    return smoothstep(0.15, 0.6, drift_field(uv, 0.3 * phase + 1.7));
}

// Radius (in units of the torus' short side, times 2 pi) of the isotropic
// core around each whorl and delta of the ridge flow.
const RIDGE_CORE: f32 = 0.18;

// Ridge flow: (cos 2t, sin 2t) of the direction t along the level sets of the
// periodic potential psi = cos 2 pi x + cos 2 pi y + 0.7 cos(2 pi (x - y) + phase).
// Stripes that follow it close into whorls around the extrema of psi and meet
// in deltas at its saddles. The phase of the diagonal term drifts with the
// weather, so whorls and deltas migrate slowly and the ridges keep
// re-aligning instead of freezing. The direction is undefined where psi is
// flat, so the result fades to zero there.
fn ridge_flow(uv: vec2<f32>) -> vec2<f32> {
    let s = sin(TAU * uv);
    let diag = 0.7 * sin(TAU * (uv.x - uv.y) + sim.flow_phase + 0.2 * sim.drift_phase);
    // -grad psi / 2 pi per cell, in units of the short side (the torus is not square).
    let world = vec2<f32>(sim.size);
    let g = vec2<f32>(s.x + diag, s.y - diag) * (min(world.x, world.y) / world);
    // The ridge runs along (-g.y, g.x) / |g|.
    return vec2<f32>(g.y * g.y - g.x * g.x, -2.0 * g.x * g.y) / (dot(g, g) + RIDGE_CORE * RIDGE_CORE);
}

// Kill offsets from the critical curve that the atlas sweeps, at feed 0.01,
// 0.02, ..., 0.08. Trimmed from the survival band (`KILL_BAND` in
// reaction_diffusion.rs), with room for the weather drift: the low-feed end
// starts above the homogeneous oscillations, the high-feed end neither
// floods into the filled state nor dies back to bare ground.
fn atlas_band(f: f32) -> vec2<f32> {
    var lo = array<f32, 8>(0.0040, 0.0030, 0.0006, -0.0012, -0.0014, -0.0008, -0.0006, -0.0006);
    var hi = array<f32, 8>(0.0072, 0.0068, 0.0064, 0.0060, 0.0040, 0.0020, -0.0004, -0.0005);
    let x = clamp(f * 100.0 - 1.0, 0.0, 6.999);
    let i = u32(x);
    let t = x - f32(i);
    return vec2<f32>(mix(lo[i], lo[i + 1u], t), mix(hi[i], hi[i + 1u], t));
}

// Atlas feed at the top and bottom rows and along the middle row.
const ATLAS_FEED_EDGE: f32 = 0.020;
const ATLAS_FEED_MID: f32 = 0.066;

// Atlas: feed peaks along the middle row and kill along the middle column.
// Both follow a full cosine period across the domain, so the parameter map
// (and with it the pattern) is periodic on the torus. Each axis is sheared by
// a sine of the other, so the two halves of a sweep differ and the map reads
// as one continuous chart rather than a mirrored wallpaper.
fn atlas_feed_kill(uv: vec2<f32>) -> vec2<f32> {
    let warped = uv + 0.12 * sin(TAU * uv.yx + vec2<f32>(0.9, 2.3));
    let across = 0.5 - 0.5 * cos(TAU * warped);
    let f = mix(ATLAS_FEED_EDGE, ATLAS_FEED_MID, across.y);
    let band = atlas_band(f);
    return vec2<f32>(f, critical_kill(f) + mix(band.x, band.y, across.x));
}

@compute @workgroup_size(16, 16)
fn cs_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sim.size.x || gid.y >= sim.size.y) {
        return;
    }
    let size = vec2<i32>(sim.size);
    let p = vec2<i32>(gid.xy);
    // Neighbouring columns and rows, wrapped on the torus.
    let lo = wrap_i(p - vec2<i32>(1), size);
    let hi = wrap_i(p + vec2<i32>(1), size);
    let row = vec3<u32>(u32(lo.y), gid.y, u32(hi.y)) * sim.size.x;
    let col = vec3<u32>(u32(lo.x), gid.x, u32(hi.x));
    let c = src[row.y + col.y];
    let east = src[row.y + col.z];
    let west = src[row.y + col.x];
    let north = src[row.x + col.y]; // y - 1
    let south = src[row.z + col.y]; // y + 1
    let se = src[row.z + col.z];    // (+1, +1)
    let nw = src[row.x + col.x];    // (-1, -1)
    let ne = src[row.x + col.z];    // (+1, -1)
    let sw = src[row.z + col.x];    // (-1, +1)
    // Karl Sims' 3x3 Laplacian: 0.2 edges, 0.05 corners, -1 centre. It is
    // 0.3 times the continuum Laplacian for smooth fields.
    var lap = (east + west + north + south) * 0.2 + (se + nw + ne + sw) * 0.05 - c;

    let uv = (vec2<f32>(p) + 0.5) / vec2<f32>(sim.size);
    if (sim.aniso != 0.0) {
        // V diffuses faster along the ridge flow than across it (U stays
        // isotropic), so stripes grow along the flow. A diffusion tensor
        // with mean D and deviation a D along direction t adds
        // a D (cos 2t (v_xx - v_yy) + 2 sin 2t v_xy) to D lap(v), scaled by
        // 0.3 like the stencil above.
        let rf = ridge_flow(uv) * (0.3 * sim.aniso);
        let straight = (east.y + west.y) - (north.y + south.y);
        let diagonal = (se.y + nw.y) - (ne.y + sw.y);
        lap.y += rf.x * straight + rf.y * 0.5 * diagonal;
    }

    var fk = vec2<f32>(sim.feed, sim.kill);
    if (sim.atlas != 0u) {
        fk = atlas_feed_kill(uv);
    }
    // Slow "weather": k drifts across morphology boundaries so settled
    // patterns keep reorganising (stripes pinch into spots and regrow).
    fk.y += sim.drift * drift_field(uv, sim.drift_phase);
    if (sim.ground != 0.0) {
        fk.y += sim.ground * lagoon(uv, sim.drift_phase);
    }
    // Scaling both diffusions only rescales space, so this varies the
    // pattern size across the domain without changing the morphology.
    let diffusion = vec2<f32>(sim.du, sim.dv) * exp(sim.scale_var * scale_field(uv, sim.drift_phase));

    let uvv = c.x * c.y * c.y;
    let u = c.x + (diffusion.x * lap.x - uvv + fk.x * (1.0 - c.x)) * sim.dt;
    let v = c.y + (diffusion.y * lap.y + uvv - (fk.x + fk.y) * c.y) * sim.dt;
    dst[row.y + col.y] = clamp(vec2<f32>(u, v), vec2<f32>(0.0), vec2<f32>(1.0));
}

// Pearson's seed chemistry, perturbed (0.5, 0.25).
fn seed_chemistry(p: vec2<u32>) -> vec2<f32> {
    return vec2<f32>(0.5, 0.25 + 0.1 * (rand3(p.x, p.y, sim.frame) - 0.5));
}

// What "create" paints at cell `p`, with a little per-cell jitter in V so the
// new structure breaks symmetry.
fn created(p: vec2<u32>) -> vec2<f32> {
    let n = rand3(p.x ^ 0x9e37u, p.y, sim.frame) - 0.5;
    return vec2<f32>(sim.create_state.x, clamp(sim.create_state.y + n * sim.create_jitter, 0.0, 1.0));
}

// Grid cells per axis for the spray and foam grids: a whole number that tiles
// the torus exactly, so a dot or hole on the seam is the same from both sides.
fn spray_cells() -> vec2<i32> {
    return max(vec2<i32>(round(vec2<f32>(sim.size) / sim.spray.x)), vec2<i32>(1));
}

// Spray-paint nucleation for large brushes: every cell of a grid may receive
// one small dot per frame, at a random spot and with a probability that
// fades towards the rim. A held brush keeps sprouting fresh structure instead
// of pinning a disc of seed chemistry. Dots may straddle grid lines, so the
// 3x3 neighbouring grid cells are tested.
fn sprayed(pos: vec2<f32>, centre: vec2<f32>, radius: f32) -> bool {
    let world = vec2<f32>(sim.size);
    let cells = spray_cells();
    let cell = world / vec2<f32>(cells);
    let home = vec2<i32>(floor(pos / cell));
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let g = home + vec2<i32>(i, j);
            // Hash the wrapped cell, but place its dot next to `pos`.
            let key = bitcast<vec2<u32>>(wrap_i(g, cells));
            let h = hash3u(key.x, key.y, sim.frame);
            let spot = (vec2<f32>(g) + vec2<f32>(rand2(h, 1u), rand2(h, 2u))) * cell;
            let fade = 1.0 - smoothstep(0.45 * radius, radius, length(torus_delta(spot, centre, world)));
            if (u32_to_unit(h) < sim.spray_density * fade && length(torus_delta(pos, spot, world)) < sim.spray.y) {
                return true;
            }
        }
    }
    return false;
}

// Fresh foam for regimes whose ground is the filled state: a jittered grid of
// holes that ignores the frame, so the pattern a stroke lays down is stable.
fn foam_hole(pos: vec2<f32>) -> bool {
    let world = vec2<f32>(sim.size);
    let cells = spray_cells();
    let cell = world / vec2<f32>(cells);
    let home = vec2<i32>(floor(pos / cell));
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let g = home + vec2<i32>(i, j);
            let key = bitcast<vec2<u32>>(wrap_i(g, cells));
            let h = hash2u(key.x ^ 0x51f3u, key.y);
            let spot = (vec2<f32>(g) + vec2<f32>(rand2(h, 1u), rand2(h, 2u))) * cell;
            if (u32_to_unit(h) < sim.spray_density && length(torus_delta(pos, spot, world)) < sim.spray.y) {
                return true;
            }
        }
    }
    return false;
}

// Runs once per displayed frame before the sub-steps and edits the latest
// state in place (it is bound as `dst`).
@compute @workgroup_size(16, 16)
fn cs_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sim.size.x || gid.y >= sim.size.y) {
        return;
    }
    let idx = gid.y * sim.size.x + gid.x;
    let world = vec2<f32>(sim.size);
    var s = dst[idx];
    let pos = vec2<f32>(gid.xy) + 0.5;
    let mode = sim.pointer_mode & 3u;

    if (mode != 0u) {
        let r = sim.pointer_radius;
        let centre = sim.pointer * world;
        // Distance to the segment the brush swept since the previous frame,
        // so fast strokes stay continuous.
        let stroke = torus_delta(sim.prev_pointer * world, centre, world);
        let rel = torus_delta(sim.prev_pointer * world, pos, world);
        let t = clamp(dot(rel, stroke) / max(dot(stroke, stroke), 1e-6), 0.0, 1.0);
        let d = length(rel - stroke * t);
        // Solid prints are stamped only on the leading edge of a stroke:
        // cells the brush already covered on the previous frame are left to
        // evolve, so a resting brush does not pin a frozen disc.
        let fresh = (sim.pointer_mode & 4u) == 0u || length(rel) >= r;
        if (d < r) {
            if (mode == 2u) {
                s = mix(s, sim.erase_state, 1.0 - smoothstep(0.7 * r, r, d));
            } else if (sim.filled_ground != 0u) {
                if (fresh) {
                    let foam = select(created(gid.xy), sim.erase_state, foam_hole(pos));
                    s = mix(s, foam, 1.0 - smoothstep(0.8 * r, r, d));
                }
            } else if (r < sim.spray.x) {
                // Brushes smaller than the spray grid paint solid dots.
                if (fresh) {
                    s = created(gid.xy);
                }
            } else if (sprayed(pos, centre, r)) {
                s = created(gid.xy);
            }
        }
    }

    // Rain seeds new structure; on filled ground a drop pops a hole instead.
    for (var i = 0u; i < min(sim.drop_count, 4u); i++) {
        let drop = sim.drops[i];
        let d = torus_delta(pos, drop.xy, world);
        if (dot(d, d) < drop.z * drop.z) {
            s = select(created(gid.xy), sim.erase_state, sim.filled_ground != 0u);
        }
    }

    // Safety net: once the field has stayed uniform for a moment (erased, or
    // completely filled) and nobody is painting, restart it once so the
    // screen can never stay blank. Filled regimes, where seeds die on bare U,
    // get a fresh foam (the filled state riddled with holes); the others get
    // coarse seeds, or holes when the filled state has swallowed everything.
    if (stats[5] > 0.5 && mode == 0u) {
        let block = gid.xy / 8u;
        let r = rand3(block.x, block.y, sim.frame);
        if (sim.filled_ground != 0u) {
            s = select(sim.create_state, vec2<f32>(1.0, 0.0), r < 0.04);
        } else if (r < 0.015) {
            s = select(seed_chemistry(gid.xy), vec2<f32>(1.0, 0.0), s.y > 0.1);
        }
    }
    dst[idx] = s;
}

// --- analysis: field texture + auto contrast ----------------------------------

// V histogram covers 0..V_RANGE (V rarely exceeds ~0.6); U covers 0..1.
const V_RANGE: f32 = 0.8;
const BINS: u32 = 128u;
// Consecutive uniform frames (half a second at 60 fps) before the revive
// fires, so it never answers a brush stroke frame by frame, and even a
// regime that keeps dying is re-seeded less than twice a second.
const REVIVE_FRAMES: f32 = 30.0;

struct Prep {
    size: vec2<u32>,
    snap: u32,        // 1 = jump straight to the new contrast window
    _p0: u32,
    act_scale: f32,   // turns |dV| per sub-step into display units
    rate: f32,        // contrast window smoothing per frame
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<uniform> prep: Prep;
@group(0) @binding(1) var<storage, read> latest: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> previous: array<vec2<f32>>;
@group(0) @binding(3) var field_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<storage, read_write> hist: array<atomic<u32>, 256>;
// [0] V low, [1] V high, [2] U low, [3] U high, [4] consecutive uniform
// frames, [5] revive flag. The CPU clears [4..8) on every reset.
@group(0) @binding(5) var<storage, read_write> win_out: array<f32, 8>;

var<workgroup> local_hist: array<atomic<u32>, 256>;

fn latest_v(p: vec2<i32>) -> f32 {
    return latest[wrap_index(p, prep.size)].y;
}

@compute @workgroup_size(16, 16)
fn cs_prepare(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    atomicStore(&local_hist[li], 0u);
    workgroupBarrier();

    if (gid.x < prep.size.x && gid.y < prep.size.y) {
        let p = vec2<i32>(gid.xy);
        let idx = gid.y * prep.size.x + gid.x;
        let c = latest[idx];
        let before = previous[idx];
        // Soft neighbourhood average (radius ~3 cells) for occlusion and glow.
        var blur = c.y * 2.0;
        blur += latest_v(p + vec2<i32>(3, 0)) + latest_v(p + vec2<i32>(-3, 0));
        blur += latest_v(p + vec2<i32>(0, 3)) + latest_v(p + vec2<i32>(0, -3));
        blur += latest_v(p + vec2<i32>(2, 2)) + latest_v(p + vec2<i32>(-2, 2));
        blur += latest_v(p + vec2<i32>(2, -2)) + latest_v(p + vec2<i32>(-2, -2));
        blur *= 0.1;
        let activity = min(abs(c.y - before.y) * prep.act_scale, 60000.0);
        textureStore(field_out, p, vec4<f32>(c.y, c.x, blur, activity));

        let bv = min(u32(c.y * (f32(BINS) / V_RANGE)), BINS - 1u);
        let bu = min(u32(c.x * f32(BINS)), BINS - 1u);
        atomicAdd(&local_hist[bv], 1u);
        atomicAdd(&local_hist[BINS + bu], 1u);
    }

    workgroupBarrier();
    let n = atomicLoad(&local_hist[li]);
    if (n > 0u) {
        atomicAdd(&hist[li], n);
    }
}

// Value (as a 0..1 fraction of the histogram's range) below which a fraction
// `q` of the samples in histogram `base` lie.
fn hist_percentile(base: u32, total: f32, q: f32) -> f32 {
    let goal = q * total;
    var cum = 0.0;
    for (var i = 0u; i < BINS; i++) {
        let c = f32(atomicLoad(&hist[base + i]));
        if (c > 0.0 && cum + c >= goal) {
            return (f32(i) + clamp((goal - cum) / c, 0.0, 1.0)) / f32(BINS);
        }
        cum += c;
    }
    return 1.0;
}

@compute @workgroup_size(1)
fn cs_resolve() {
    var total = 0.0;
    var live = 0.0;
    // Cells with V above ~0.03 count as "alive".
    let live_bin = u32(0.03 / V_RANGE * f32(BINS)) + 1u;
    for (var i = 0u; i < BINS; i++) {
        let c = f32(atomicLoad(&hist[i]));
        total += c;
        if (i >= live_bin) {
            live += c;
        }
    }
    if (total > 0.0) {
        let v_lo = hist_percentile(0u, total, 0.03) * V_RANGE;
        let v_hi = max(hist_percentile(0u, total, 0.997) * V_RANGE, v_lo + 0.1);
        let u_lo = hist_percentile(BINS, total, 0.003);
        let u_hi = max(hist_percentile(BINS, total, 0.97), u_lo + 0.1);
        let rate = select(clamp(prep.rate, 0.0, 1.0), 1.0, prep.snap != 0u);
        win_out[0] = mix(win_out[0], v_lo, rate);
        win_out[1] = mix(win_out[1], v_hi, rate);
        win_out[2] = mix(win_out[2], u_lo, rate);
        win_out[3] = mix(win_out[3], u_hi, rate);
        // Uniform: (almost) nothing alive, or a mostly filled field whose V
        // barely varies (e.g. the filled state swallowed every hole).
        let fraction = live / total;
        let spread = (hist_percentile(0u, total, 0.999) - hist_percentile(0u, total, 0.001)) * V_RANGE;
        let uniform = fraction < 0.0002 || (fraction > 0.5 && spread < 0.02);
        win_out[4] = select(0.0, min(win_out[4] + 1.0, 1000.0), uniform);
        win_out[5] = select(0.0, 1.0, win_out[4] > REVIVE_FRAMES);
    }
    for (var i = 0u; i < 2u * BINS; i++) {
        atomicStore(&hist[i], 0u);
    }
}

// --- display -----------------------------------------------------------------

struct Draw {
    view: ViewXform,
    size: vec2<u32>,
    time: f32,
    material: u32,       // 0 lacquer, 1 nacre, 2 luminous, 3 ink, 4 dark-field
    contrast: vec2<f32>, // body mask edges on the normalised level
    pal_range: vec2<f32>,
    relief: f32,
    gloss: f32,
    glow: f32,
    halo: f32,
    iridescence: f32,
    activity: f32,
    shadow: f32,
    brightness: f32,
    invert: u32,         // 1 = the filled state is the ground, holes are raised
    aura_tint: f32,      // palette position of the U-depletion halo
    reflect: f32,        // nacre: how much the ground reflects the film (0..1)
    clarity: f32,        // local contrast of the level (0 = off)
};

@group(0) @binding(0) var<uniform> draw: Draw;
@group(0) @binding(1) var field_tex: texture_2d<f32>;
@group(0) @binding(2) var field_samp: sampler;
@group(0) @binding(3) var lut: texture_2d<f32>;
@group(0) @binding(4) var lut_samp: sampler;
@group(0) @binding(5) var<storage, read> win: array<f32>;

// Hardware-filtered sample at world uv. The sampler repeats, which is exactly
// the torus wrap (and seamless for any zoom or pan).
fn field_at(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(field_tex, field_samp, uv, 0.0);
}

// V normalised by the auto window: 0 = ground, 1 = crest. `invert` swaps them
// for regimes whose ground is the filled (high V) state.
fn level(v: f32) -> f32 {
    let n = clamp((v - win[0]) / max(win[1] - win[0], 1e-3), 0.0, 1.0);
    return select(n, 1.0 - n, draw.invert != 0u);
}

// Level of a field texel with optional local contrast: `clarity` pushes it
// away from the blurred level, lifting shallow ripples (such as honeycomb
// holes closing in a flooded plateau) out of flat mid-tones.
fn lift(s: vec4<f32>) -> f32 {
    let h = level(s.x);
    return clamp(h + draw.clarity * (h - level(s.z)), 0.0, 1.0);
}

fn pal(t: f32) -> vec3<f32> {
    return palette_lookup(lut, lut_samp, mix(draw.pal_range.x, draw.pal_range.y, clamp(t, 0.0, 1.0)));
}

// Thin-film interference colour for a film `thickness` micrometres thick
// (red, green and blue wavelengths), like oil on water or nacre.
fn thin_film(thickness: f32) -> vec3<f32> {
    let phase = thickness / vec3<f32>(0.65, 0.54, 0.45);
    return 0.5 + 0.5 * cos(TAU * phase);
}

// Keeps 70% of a colour's saturation.
fn soften(c: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(luminance(c)), c, 0.7);
}

// Slowly flowing thickness variations of a film (the swirls on a soap
// bubble). Integer wave vectors keep it periodic on the unit torus.
fn film_swirl(uv: vec2<f32>, t: f32) -> f32 {
    let a = sin(TAU * (2.0 * uv.x + uv.y) + 0.13 * t);
    let b = sin(TAU * (uv.x - 3.0 * uv.y) - 0.09 * t + 1.7);
    let c = sin(TAU * (3.0 * uv.x + 2.0 * uv.y) + 0.07 * t + 4.1);
    return (a + b + 0.5 * c) * 0.4;
}

// Smooth value noise on an n-cell lattice wrapped over the unit torus.
fn torus_noise(uv: vec2<f32>, n: vec2<i32>, salt: u32) -> f32 {
    let p = uv * vec2<f32>(n);
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(i);
    let e = f * f * (3.0 - 2.0 * f);
    let a = bitcast<vec2<u32>>(wrap_i(i, n));
    let b = bitcast<vec2<u32>>(wrap_i(i + vec2<i32>(1, 0), n));
    let c = bitcast<vec2<u32>>(wrap_i(i + vec2<i32>(0, 1), n));
    let d = bitcast<vec2<u32>>(wrap_i(i + vec2<i32>(1, 1), n));
    let top = mix(rand3(a.x, a.y, salt), rand3(b.x, b.y, salt), e.x);
    let bottom = mix(rand3(c.x, c.y, salt), rand3(d.x, d.y, salt), e.x);
    return mix(top, bottom, e.y);
}

// Paper fibre: two streaky octaves at right angles over a fine tooth, 0..1.
fn paper_fibre(uv: vec2<f32>) -> f32 {
    let cells = vec2<f32>(draw.size);
    let across = torus_noise(uv, max(vec2<i32>(cells / vec2<f32>(9.0, 1.5)), vec2<i32>(1)), 11u);
    let down = torus_noise(uv, max(vec2<i32>(cells / vec2<f32>(1.5, 9.0)), vec2<i32>(1)), 23u);
    let tooth = torus_noise(uv, max(vec2<i32>(cells / 1.2), vec2<i32>(1)), 37u);
    return 0.35 * across + 0.35 * down + 0.3 * tooth;
}

@fragment
fn fs_display(in: FullscreenOut) -> @location(0) vec4<f32> {
    let w = view_apply(draw.view, in.uv);
    let texel = 1.0 / vec2<f32>(draw.size);
    let f = field_at(w);
    // The level is both the height and the palette coordinate, so bodies are
    // rounded and shade from their rim colour to their crest colour; the
    // contrast window only decides where ground ends and body begins.
    let h = lift(f);
    let m = smoothstep(draw.contrast.x, draw.contrast.y, h);
    // Blurred level: how much body surrounds this point.
    let density = level(f.z);

    // Height-field normal (central differences one cell apart).
    let h_px = lift(field_at(w + vec2<f32>(texel.x, 0.0)));
    let h_mx = lift(field_at(w - vec2<f32>(texel.x, 0.0)));
    let h_py = lift(field_at(w + vec2<f32>(0.0, texel.y)));
    let h_my = lift(field_at(w - vec2<f32>(0.0, texel.y)));
    let hx = h_px - h_mx;
    let hy = h_py - h_my;
    let n = normalize(vec3<f32>(-vec2<f32>(hx, hy) * (draw.relief * 4.0), 1.0));
    // Macro normal from the blurred field two cells either side: the
    // orientation of the neighbourhood rather than of each small feature.
    let b_x = level(field_at(w + vec2<f32>(2.0 * texel.x, 0.0)).z) - level(field_at(w - vec2<f32>(2.0 * texel.x, 0.0)).z);
    let b_y = level(field_at(w + vec2<f32>(0.0, 2.0 * texel.y)).z) - level(field_at(w - vec2<f32>(0.0, 2.0 * texel.y)).z);
    let n_macro = normalize(vec3<f32>(-vec2<f32>(b_x, b_y) * (draw.relief * 2.0), 1.0));
    let l = normalize(vec3<f32>(-0.55, -0.65, 0.55));
    let half_vec = normalize(l + vec3<f32>(0.0, 0.0, 1.0));
    let ndl = dot(n, l);
    let diffuse = max(ndl, 0.0);
    let wrapped = max((ndl + 0.6) / 1.6, 0.0);
    let ndh = max(dot(n, half_vec), 0.0);
    // Sharp glints only on convex surface (crests of bodies): the rims of
    // holes and the creases where stripes meet are concave, and a pinpoint
    // highlight there reads as a stray white tick rather than as gloss. The
    // glint also follows a blend with the macro normal, so small holes and
    // necks do not each catch an identical tick on their lit shoulder.
    let curvature = h_px + h_mx + h_py + h_my - 4.0 * h;
    let convex = 1.0 - smoothstep(-0.03, 0.05, curvature);
    let spec = pow(max(dot(normalize(mix(n, n_macro, 0.6)), half_vec), 0.0), 60.0) * convex;
    let sheen = pow(ndh, 10.0);
    let fres = pow(clamp(1.0 - n.z, 0.0, 1.0), 1.2);

    // Contact occlusion (ground hugging raised structure) and a cast shadow
    // from whatever stands between this point and the light.
    let cavity = clamp((density - h) * 1.8, 0.0, 1.0);
    let occluder = lift(field_at(w + normalize(l.xy) * texel * 2.5));
    let shadow = clamp((occluder - h) * 1.3, 0.0, 1.0) * draw.shadow;
    let ao = (1.0 - cavity * 0.6) * (1.0 - shadow * 0.75);
    // U is depleted around structures: a soft aura reaching further than V.
    // It fades where the neighbourhood is dense, so the gaps of a packed
    // labyrinth keep a deep ground instead of a grey haze.
    let u_level = clamp((win[3] - f.y) / max(win[3] - win[2], 1e-3), 0.0, 1.0);
    let aura = u_level * u_level * (1.0 - m) * (1.0 - density);
    // Growth glow, compressed so fast waves never blow out.
    let fronts = f.w / (1.0 + f.w) * draw.activity;
    // Slowly flowing film thickness shared by the iridescent materials.
    let swirl = film_swirl(w, draw.time);

    let ground = pal(0.0);
    let body = pal(h);
    var col: vec3<f32>;
    switch draw.material {
        case 1u: {
            // Nacre: thin-film interference whose thickness is dominated by
            // the slow swirls, so hue flows across the image in broad bands
            // like a soap film; height and density only bend the bands. The
            // film fades in with body size (decaying specks would otherwise
            // read as saturated dots), and the ground faintly reflects it.
            let thickness = 0.45 + 0.35 * h + 0.2 * density + 0.1 * n.z + 1.6 * swirl;
            let film = thin_film(thickness);
            // Squaring pushes the pastel interference colours towards soap-film vividness.
            let vivid = soften(film * film * 1.6);
            let sheet = smoothstep(0.2, 0.5, density);
            let base = body * (0.25 + 0.85 * diffuse);
            let reflection = mix(ground, pal(0.06) + 0.03 * thin_film(0.6 + swirl), draw.reflect);
            col = mix(reflection, base, m) * ao;
            col += vivid * (0.3 + fres * 1.4 + sheen * 0.6) * draw.iridescence * m * sheet * mix(vec3<f32>(1.0), body, 0.25);
            col += spec * draw.gloss * 2.5 * mix(vec3<f32>(1.0), film, 0.5) * m;
        }
        case 2u: {
            // Luminous: translucent bodies lit from within; bright cores bloom.
            let core = pow(h, 1.6);
            let base = body * (0.15 + 0.5 * wrapped);
            col = mix(ground, base, m) * ao;
            col += pal(0.45 + 0.55 * h) * core * m * draw.glow * 2.2;
            col += soften(thin_film(0.4 + 0.6 * h + 1.2 * swirl)) * fres * m * draw.iridescence * 0.7;
            col += spec * draw.gloss * 1.5 * m * mix(vec3<f32>(1.0), body, 0.4);
        }
        case 3u: {
            // Ink on paper: pigment pools along the flanks of each ridge
            // (the centre stays a shade lighter), the inking is slightly
            // uneven, and the paper has a faint fibre and is embossed by the
            // relief.
            let edge = clamp(length(vec2<f32>(hx, hy)) * 1.4, 0.0, 1.0);
            let pooled = m * (0.8 + 0.2 * edge) + edge * 0.15 * (1.0 - m);
            let ink = clamp(pooled * (0.9 + 0.1 * film_swirl(w, 0.0)), 0.0, 1.0);
            let emboss = 1.0 + (ndl - l.z) * 0.35 * draw.relief;
            let paper = 0.95 + 0.05 * paper_fibre(w);
            col = pal(ink) * paper * emboss * (1.0 - cavity * 0.08 - shadow * 0.12);
            col += vec3<f32>(spec * draw.gloss * 0.3 * m);
        }
        case 4u: {
            // Dark-field microscopy: light scattered off the membranes (the
            // level band where ground turns into body, plus any steep slope)
            // outlines every body; the interior stays dim and translucent
            // around a small bright nucleus.
            let edge_mid = mix(draw.contrast.x, draw.contrast.y, 0.5);
            let band = smoothstep(draw.contrast.x, edge_mid, h) * (1.0 - smoothstep(edge_mid, draw.contrast.y + 0.3, h));
            let slope = clamp(length(vec2<f32>(hx, hy)) * 2.0, 0.0, 1.0);
            let membrane = max(band, slope * slope);
            let nucleus = smoothstep(0.9, 1.0, h);
            col = mix(ground, pal(0.3 + 0.3 * h) * (0.08 + 0.2 * wrapped), m) * ao;
            col += pal(0.6 + 0.4 * h) * membrane * draw.glow * 1.5;
            col += pal(1.0) * nucleus * draw.glow * 0.5;
            col += soften(thin_film(0.5 + 0.5 * h + 1.2 * swirl)) * membrane * draw.iridescence * 0.6;
            col += spec * draw.gloss * m * mix(vec3<f32>(1.0), body, 0.3);
        }
        default: {
            // Lacquer: glossy enamel with sharp glints.
            let base = body * (0.28 + 0.95 * diffuse);
            col = mix(ground, base, m) * ao;
            col += body * fres * 0.35 * m;
            col += spec * draw.gloss * 3.0 * mix(vec3<f32>(1.0), body, 0.3) * m;
            col += soften(thin_film(0.5 + 0.5 * h + swirl)) * fres * draw.iridescence * m * 0.6;
        }
    }

    if (draw.material != 3u) {
        col += pal(draw.aura_tint) * aura * draw.halo;
        col += pal(0.85) * fronts * 0.8;
    }
    col *= draw.brightness;
    // Every term above is bounded (the state is clamped to 0..1 and every
    // division is guarded), so this only caps extreme highlights.
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(64.0)), 1.0);
}
