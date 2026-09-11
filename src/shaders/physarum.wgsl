// Physarum polycephalum transport networks: Jeff Jones' (2010) agent model,
// extended to four species coupled through an interaction matrix.
//
// Per sub-step:
//   cs_agents   every agent senses the trail at three points ahead of it (left,
//               centre, right), steers towards the strongest signal, moves, and
//               counts its deposit into its species' channel of `counts`. The
//               counters are integer atomics: race-free, overflow-proof (at most
//               DEPOSIT_UNITS per agent) and bit-exactly deterministic.
//   cs_diffuse  every cell folds its counts into the trail (with a soft crowding
//               limit), blends towards the 3x3 mean, decays, and clears the
//               counts. It also updates `traffic`, a display-only long exposure
//               of where agents actually travel (linear in the agent count).
//
// `TRAIL_FORMAT` and `TRAFFIC_FORMAT` are substituted by physarum.rs (the trail
// is rgba32float when the adapter can filter it, rgba16float otherwise).

struct Agent {
    pos: vec2<f32>,   // cells, in [0, size)
    heading: f32,     // radians, in [0, TAU)
    state: u32,       // bits 0-1: species; bits 2-31: per-agent Weyl counter
};

struct Sim {
    size: vec2<u32>,
    agent_count: u32,
    species_count: u32,
    decay: f32,
    diffusion: f32,
    food: f32,
    pointer_mode: u32,             // 0 = none, 1 = drop food, 2 = repel + erase
    pointer: vec2<f32>,            // cells (not wrapped)
    pointer_radius: f32,           // cells, in [2, 0.45 * min(size)]
    trail_cap: f32,                // upper bound of the trail and traffic fields
    motion: array<vec4<f32>, 4>,   // per species: sensor angle, sensor distance, turn angle, speed
    extra: array<vec4<f32>, 4>,    // per species: wander, curl, size variety (log2), -
    interact: array<vec4<f32>, 4>, // row s: weight species s gives each trail channel
    deposit: vec4<f32>,            // per species, already normalised by the host
    // x: crowding: agents per cell at which deposits saturate (0 = linear)
    // y: sensing saturation in trail units (0 = off)
    // z: renewal: probability per sub-step that an agent is reborn at the layout
    // w: swirl: angular speed (rad / sub-step) of the rotation at the swirl's core
    tune: vec4<f32>,
    // x: traffic persistence, y: traffic blur, z: traffic deposit per agent
    traffic: vec4<f32>,
    // xy: swirl centre (cells), z: core radius r0, w: outer radius (cells)
    swirl: vec4<f32>,
    // x: terrain strength (log2 of the largest change of the decay rate),
    // yz: terrain drift offset (lattice units), w: log2 of the largest change
    // of the agents' size (reach and stride)
    terrain: vec4<f32>,
    // xy: terrain lattice cells across the domain (coarsest octave), z: seed, w: -
    terrain_cells: vec4<u32>,
    // x: core pull, an attractant at the swirl centre (trail units),
    // y: 1 / its radius^2 (cells), zw: -
    core: vec4<f32>,
};

struct Seeding {
    size: vec2<u32>,
    count: u32,
    pattern: u32,       // Layout in physarum.rs
    seed: u32,
    species_count: u32,
    radius: f32,        // spawn radius in cells
    clusters: u32,      // rings / clusters for the layouts that use them
    centre: vec2<f32>,  // centre of the centred layouts (cells)
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var trail_src: texture_2d<f32>;
@group(0) @binding(3) var trail_samp: sampler; // linear + repeat: wraps the torus
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var trail_dst: texture_storage_2d<TRAIL_FORMAT, write>;
@group(0) @binding(6) var<uniform> seeding: Seeding;
@group(0) @binding(7) var traffic_src: texture_2d<f32>;
@group(0) @binding(8) var traffic_dst: texture_storage_2d<TRAFFIC_FORMAT, write>;

const AGENT_WG: u32 = 256u;
// Fixed-point units one agent deposits per sub-step. Divisible by 1-4, so the
// deposit splits exactly over the 1-4 splats of a long stride.
const DEPOSIT_UNITS: u32 = 12u;
const MAX_SPLATS: f32 = 4.0;

fn unit_dir(a: f32) -> vec2<f32> {
    return vec2<f32>(cos(a), sin(a));
}

// Linear agent index for a dispatch split by `gpu::dispatch_linear`.
fn agent_index(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * AGENT_WG;
}

// Mask of the channels that belong to live species.
fn live_channels(count: u32) -> vec4<f32> {
    return select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(count));
}

// Brings a position that is less than one period outside [0, size) back onto
// the torus. Exact: no float division, so no seams (see `wrap_i`). The result
// can equal `size` when `p` is a hair below 0; `deposit_cell` handles that.
fn wrap_near(p: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    let q = select(p, p - size, p >= size);
    return select(q, q + size, q < vec2<f32>(0.0));
}

// Wraps an arbitrary position onto the torus. As in `wrap_i`, the float
// quotient only has to land within one period; `wrap_near` makes it exact.
fn wrap_pos(p: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    return wrap_near(p - size * floor(p / size), size);
}

// Linear index of the cell containing position `p`, wrapped exactly.
fn deposit_cell(p: vec2<f32>) -> u32 {
    return wrap_index(vec2<i32>(floor(p)), sim.size);
}

// --- terrain ---------------------------------------------------------------------

// Random value at an integer lattice point of a lattice with `n` cells across
// the torus. The lattice wraps, so the noise is exactly periodic.
fn lattice_value(c: vec2<i32>, n: vec2<i32>, seed: u32) -> f32 {
    let w = wrap_i(c, n);
    return u32_to_unit(pcg_hash(u32(w.y * n.x + w.x) ^ seed));
}

// Smooth periodic value noise in [0, 1] at lattice coordinates `u`.
fn value_noise(u: vec2<f32>, n: vec2<i32>, seed: u32) -> f32 {
    let i = floor(u);
    let f = u - i;
    let s = f * f * (3.0 - 2.0 * f);
    let c = vec2<i32>(i);
    let a = lattice_value(c, n, seed);
    let b = lattice_value(c + vec2<i32>(1, 0), n, seed);
    let d = lattice_value(c + vec2<i32>(0, 1), n, seed);
    let e = lattice_value(c + vec2<i32>(1, 1), n, seed);
    return mix(mix(a, b, s.x), mix(d, e, s.x), s.y);
}

// A slow landscape over the torus, in [-1, 1]: three octaves of periodic value
// noise, stretched so that broad regions reach the extremes. It sets where the
// trail lasts (fertile ground: dense networks) and where it fades fast
// (barren ground: quiet voids), giving every layout a macro composition.
fn terrain(p: vec2<f32>) -> f32 {
    let n = vec2<i32>(sim.terrain_cells.xy);
    let seed = sim.terrain_cells.z;
    let u = p / vec2<f32>(sim.size) * vec2<f32>(n) + sim.terrain.yz;
    let v = 0.57 * value_noise(u, n, seed)
        + 0.29 * value_noise(u * 2.0 + vec2<f32>(17.3, 5.9), n * 2, seed ^ 0x68e31da4u)
        + 0.14 * value_noise(u * 4.0 + vec2<f32>(3.1, 11.7), n * 4, seed ^ 0xb5297a4du);
    return clamp((v - 0.5) * 2.6, -1.0, 1.0);
}

// --- seeding -------------------------------------------------------------------

struct Spawn {
    pos: vec2<f32>,
    heading: f32,
};

// Birth place and heading of an agent of species `s` under the current layout,
// from the random word `h`. Used both for seeding and for renewal, so reborn
// agents keep feeding the layout's structure (a radial burst, a vortex, a
// species' home territory...).
fn spawn(s: u32, h: u32) -> Spawn {
    let size = vec2<f32>(seeding.size);
    let centre = seeding.centre;
    let k = max(seeding.species_count, 1u);
    let h1 = pcg_hash(h);
    let h2 = pcg_hash(h1);
    let h3 = pcg_hash(h2);
    let u0 = u32_to_unit(h);
    let u1 = u32_to_unit(h1);
    let u2 = u32_to_unit(h2);

    // Default: uniform in a disk (sqrt for equal area), facing outwards.
    let theta = u1 * TAU;
    var pos = centre + seeding.radius * sqrt(u0) * unit_dir(theta);
    var heading = theta;
    switch seeding.pattern {
        case 0u: { // uniform scatter
            pos = vec2<f32>(u0, u1) * size;
            heading = u2 * TAU;
        }
        case 2u: { // disk facing inwards
            heading = theta + PI;
        }
        case 3u: { // concentric rings (species s owns rings s, s + k, ...), tangential
            let per = max(seeding.clusters / k, 1u);
            let ring = s + k * (h3 % per);
            let r = seeding.radius * (f32(ring) + 1.0) / f32(k * per) + (u0 - 0.5) * 4.0;
            pos = centre + r * unit_dir(theta);
            heading = theta + select(-0.5, 0.5, (ring & 1u) == 0u) * PI;
        }
        case 4u: { // one angular sector per species, facing outwards
            let a = (f32(s) + u1) / f32(k) * TAU;
            pos = centre + seeding.radius * sqrt(u0) * unit_dir(a);
            heading = a;
        }
        case 5u: { // one vertical band per species
            pos = vec2<f32>((f32(s) + u0) / f32(k), u1) * size;
            heading = u2 * TAU;
        }
        case 6u: { // clusters scattered over the torus (species s owns blobs s, s + k, ...)
            let per = max(seeding.clusters / k, 1u);
            let blob = s + k * (h3 % per);
            let bh = hash2u(blob, seeding.seed ^ 0x51ed270bu);
            let bc = vec2<f32>(u32_to_unit(bh), u32_to_unit(pcg_hash(bh))) * size;
            let br = seeding.radius / sqrt(f32(k * per));
            pos = bc + br * sqrt(u0) * unit_dir(theta);
            heading = u2 * TAU;
        }
        case 7u: { // vortex: born on the rim, heading inwards with a swirl
            pos = centre + seeding.radius * (0.9 + 0.1 * u0) * unit_dir(theta);
            heading = theta + 1.3 * PI;
        }
        case 8u: { // galaxy: born along `clusters` spiral arms, travelling along them
            let arms = max(seeding.clusters, 1u);
            let arm = h3 % arms;
            let t = sqrt(u0);
            let twist = 1.2 * PI;
            let a = f32(arm) / f32(arms) * TAU + twist * t + (u2 - 0.5) * 0.7;
            pos = centre + seeding.radius * t * unit_dir(a);
            heading = a + atan(twist * t) + select(0.0, PI, ((h3 >> 16u) & 1u) == 1u);
        }
        default: {} // 1: disk facing outwards
    }
    heading = heading - TAU * floor(heading / TAU);
    return Spawn(wrap_pos(pos, size), heading);
}

@compute @workgroup_size(256)
fn cs_init(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = agent_index(gid, groups);
    if (i >= seeding.count) {
        return;
    }
    let species = i % max(seeding.species_count, 1u);
    let h = hash2u(i, seeding.seed);
    let born = spawn(species, h);
    agents[i] = Agent(born.pos, born.heading, (hash2u(h, 0x2c1b3c6du) & ~3u) | species);
}

// Zeroes the destination trail and traffic textures (no CLEAR_TEXTURE feature).
@compute @workgroup_size(16, 16)
fn cs_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= seeding.size.x || gid.y >= seeding.size.y) {
        return;
    }
    textureStore(trail_dst, vec2<i32>(gid.xy), vec4<f32>(0.0));
    textureStore(traffic_dst, vec2<i32>(gid.xy), vec4<f32>(0.0));
}

// --- agents ----------------------------------------------------------------------

// Signal perceived at `p` (cells) by an agent whose species weighs the trail
// channels with `row`.
fn sense(p: vec2<f32>, row: vec4<f32>) -> f32 {
    let size = vec2<f32>(sim.size);
    var t = textureSampleLevel(trail_src, trail_samp, p / size, 0.0);
    if (sim.tune.y > 0.0) {
        // Michaelis-Menten saturation: dense veins stop out-shouting their neighbours.
        t = t / (vec4<f32>(1.0) + t / sim.tune.y);
    }
    var v = dot(t, row);
    if (sim.core.x != 0.0) {
        // Core pull: a long-tailed attractant at the swirl centre that every
        // species feels. Agents stream inwards and the swirl shears the
        // streams into spiral arms around a bright bulge.
        let d = torus_delta(sim.swirl.xy, p, size);
        v += sim.core.x / (1.0 + dot(d, d) * sim.core.y);
    }
    if (sim.pointer_mode == 1u) {
        // Food: an attractant centred on the cursor that every species likes,
        // whatever its interaction matrix says. The falloff is steep enough
        // that a held brush gathers the network into one radial hub.
        let d = torus_delta(sim.pointer, p, size);
        let q = 1.0 + dot(d, d) / (sim.pointer_radius * sim.pointer_radius);
        v += sim.food / (q * q);
    }
    return v;
}

@compute @workgroup_size(256)
fn cs_agents(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = agent_index(gid, groups);
    if (i >= sim.agent_count) {
        return;
    }
    var a = agents[i];
    let s = a.state & 3u;
    let m = sim.motion[s];
    let e = sim.extra[s];
    let row = sim.interact[s];
    let size = vec2<f32>(sim.size);

    // Fresh randomness per agent and sub-step. The state's upper 30 bits are a
    // Weyl counter (+4 per sub-step, which leaves the species bits alone),
    // hashed together with the agent index. For every agent that is a
    // bijection of its counter, and no two agents share an index, so random
    // streams never merge. (Re-hashing and masking the previous word instead
    // is many-to-one: agents' streams collide, and with renewal they are then
    // reborn together as exact clones, collapsing the population in minutes.)
    let h = hash2u(i, a.state);
    let r_side = u32_to_unit(h);
    let r_jitter = u32_to_unit(pcg_hash(h ^ 0x9e3779b9u)) * 2.0 - 1.0;

    // Renewal (Jones' population turnover): now and then an agent is reborn
    // at the layout. Newcomers wander, draw faint filigree and seed new
    // branches, so the network keeps reorganising instead of only coarsening.
    if (u32_to_unit(pcg_hash(h ^ 0x7f4a7c15u)) < sim.tune.z) {
        let born = spawn(s, pcg_hash(h ^ 0x165667b1u));
        a.pos = born.pos;
        a.heading = born.heading;
    }

    // Fixed per-agent size (from the agent index): sensor reach and stride
    // scale together, so one species weaves several network scales at once.
    var agent_scale = exp2(e.z * (u32_to_unit(pcg_hash(i ^ 0xa511e9b3u)) * 2.0 - 1.0));
    if (sim.terrain.w != 0.0) {
        // The terrain scales every agent too: fertile ground grows a fine,
        // dense mesh (slower agents linger there), barren ground a coarse,
        // sparse one.
        agent_scale *= exp2(-sim.terrain.w * terrain(a.pos));
    }
    let reach = m.y * agent_scale;
    let stride = m.w * agent_scale;

    let fl = sense(a.pos + unit_dir(a.heading + m.x) * reach, row);
    let fc = sense(a.pos + unit_dir(a.heading) * reach, row);
    let fr = sense(a.pos + unit_dir(a.heading - m.x) * reach, row);

    // Jones' steering rules.
    var turn = 0.0;
    if (fc >= fl && fc >= fr) {
        turn = 0.0; // strongest ahead: keep going
    } else if (fc < fl && fc < fr) {
        turn = select(-m.z, m.z, r_side < 0.5); // both sides stronger: pick one
    } else if (fl > fr) {
        turn = m.z;
    } else {
        turn = -m.z;
    }
    var heading = a.heading + turn + e.x * r_jitter + e.y;
    var p = a.pos + unit_dir(heading) * stride;

    if (sim.pointer_mode == 2u) {
        // Repel: ease agents out of the brush (a fraction of the way to the
        // rim per sub-step) and steer them along the rim, slightly outwards,
        // on whichever side they were already heading. They skirt the brush
        // like water round a stone and draw a bright rim vein.
        let d = torus_delta(sim.pointer, p, size);
        let dist = length(d);
        if (dist < sim.pointer_radius) {
            let n = select(unit_dir(heading), d / max(dist, 1e-3), dist > 1e-3);
            p += n * ((sim.pointer_radius - dist) * 0.3);
            let along = vec2<f32>(-n.y, n.x) * select(-1.0, 1.0, dot(unit_dir(heading), vec2<f32>(-n.y, n.x)) >= 0.0);
            let want = along + 0.5 * n;
            var turn_out = atan2(want.y, want.x) - heading;
            turn_out = turn_out - TAU * floor(turn_out / TAU + 0.5); // into [-PI, PI)
            heading += 0.5 * turn_out;
        }
    }

    // Usually far less than a period, but large agents on barren terrain in
    // a small domain can stride further, so wrap exactly from any distance.
    p = wrap_pos(p, size);

    if (sim.tune.w != 0.0) {
        // Swirl: differential rotation about the swirl centre with a flat
        // rotation curve (angular speed ~ r0 / (r + r0)), which winds the
        // network into logarithmic arms, fading out before the outer radius.
        // The outer radius is below half the shorter side and distances are
        // taken on the torus, so the swirl is seamless wherever its centre is.
        let d = torus_delta(sim.swirl.xy, p, size);
        let r = length(d);
        let w = sim.tune.w * sim.swirl.z / (r + sim.swirl.z) * (1.0 - smoothstep(0.7, 1.0, r / sim.swirl.w));
        if (w != 0.0) {
            let cw = cos(w);
            let sw = sin(w);
            p = wrap_near(sim.swirl.xy + vec2<f32>(d.x * cw - d.y * sw, d.x * sw + d.y * cw), size);
            heading += w;
        }
    }
    heading = heading - TAU * floor(heading / TAU);

    // Deposit one agent's worth, split over enough splats along the step that
    // every cell it crosses gets some: fast or large agents draw continuous
    // lines instead of dotted ones.
    let moved = torus_delta(a.pos, p, size);
    let splats = u32(clamp(ceil(max(abs(moved.x), abs(moved.y))), 1.0, MAX_SPLATS));
    let units = DEPOSIT_UNITS / splats;
    for (var k = 1u; k <= splats; k++) {
        let q = a.pos + moved * (f32(k) / f32(splats));
        atomicAdd(&counts[deposit_cell(q) * 4u + s], units);
    }

    agents[i] = Agent(p, heading, a.state + 4u);
}

// --- trail ----------------------------------------------------------------------

@compute @workgroup_size(16, 16)
fn cs_diffuse(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= sim.size.x || gid.y >= sim.size.y) {
        return;
    }
    let dims = vec2<i32>(sim.size);
    let p = vec2<i32>(gid.xy);
    // Wrapped neighbour rows / columns, shared by all taps.
    let lo = wrap_i(p - vec2<i32>(1), dims);
    let hi = wrap_i(p + vec2<i32>(1), dims);
    let c = textureLoad(trail_src, p, 0);
    let sum = c
        + textureLoad(trail_src, vec2<i32>(lo.x, lo.y), 0)
        + textureLoad(trail_src, vec2<i32>(p.x, lo.y), 0)
        + textureLoad(trail_src, vec2<i32>(hi.x, lo.y), 0)
        + textureLoad(trail_src, vec2<i32>(lo.x, p.y), 0)
        + textureLoad(trail_src, vec2<i32>(hi.x, p.y), 0)
        + textureLoad(trail_src, vec2<i32>(lo.x, hi.y), 0)
        + textureLoad(trail_src, vec2<i32>(p.x, hi.y), 0)
        + textureLoad(trail_src, vec2<i32>(hi.x, hi.y), 0);
    var v = mix(c, sum * (1.0 / 9.0), sim.diffusion);

    // Fold in this sub-step's deposits (in agents) and reset the counters.
    // During this pass only this invocation touches these four counters.
    let base = (gid.y * sim.size.x + gid.x) * 4u;
    let n = vec4<f32>(
        f32(atomicExchange(&counts[base], 0u)),
        f32(atomicExchange(&counts[base + 1u], 0u)),
        f32(atomicExchange(&counts[base + 2u], 0u)),
        f32(atomicExchange(&counts[base + 3u], 0u)),
    ) * (1.0 / f32(DEPOSIT_UNITS));
    var crowded = n;
    if (sim.tune.x > 0.0) {
        // Soft exclusion: a crowded cell deposits like ~`crowding` agents at most,
        // which stops the population collapsing into a few dense worms.
        crowded = n / (vec4<f32>(1.0) + n / sim.tune.x);
    }
    var decay = sim.decay;
    var fertility = 1.0;
    if (sim.terrain.x != 0.0) {
        // Fertile ground is marked more strongly and keeps its marks longer:
        // deposits and the trail's lifetime both scale by 2^(+-strength).
        fertility = exp2(sim.terrain.x * terrain(vec2<f32>(p) + 0.5));
        decay = clamp(1.0 - (1.0 - sim.decay) / fertility, 0.0, 0.999);
    }
    v = (v + crowded * (sim.deposit * fertility)) * decay;

    // Traffic: long exposure of the raw agent counts, lightly blurred.
    let g0 = textureLoad(traffic_src, p, 0);
    let cross = textureLoad(traffic_src, vec2<i32>(lo.x, p.y), 0)
        + textureLoad(traffic_src, vec2<i32>(hi.x, p.y), 0)
        + textureLoad(traffic_src, vec2<i32>(p.x, lo.y), 0)
        + textureLoad(traffic_src, vec2<i32>(p.x, hi.y), 0);
    var g = mix(g0, (g0 + cross) * 0.2, sim.traffic.y) * sim.traffic.x + n * (sim.traffic.z * fertility);

    if (sim.pointer_mode != 0u) {
        let size = vec2<f32>(sim.size);
        let d = torus_delta(vec2<f32>(p) + 0.5, sim.pointer, size);
        let q = dot(d, d) / (sim.pointer_radius * sim.pointer_radius);
        if (q < 1.0) {
            let w = (1.0 - q) * (1.0 - q);
            if (sim.pointer_mode == 1u) {
                // A little food in the trail too, so the hub shows at once.
                v += live_channels(sim.species_count) * (sim.food * 0.01 * w * w);
            } else {
                v *= 1.0 - w;
                g *= 1.0 - w;
            }
        }
    }
    textureStore(trail_dst, p, clamp(v, vec4<f32>(0.0), vec4<f32>(sim.trail_cap)));
    textureStore(traffic_dst, p, clamp(g, vec4<f32>(0.0), vec4<f32>(sim.trail_cap)));
}
