// Live measurements of Physarum: a 16x16 workgroup per block of trail cells
// and a 256-thread workgroup per block of agents, both writing into one
// partials buffer (the agent records follow the cell records; see metrics.wgsl
// for the contract). The lanes match `METRICS` in physarum.rs.
//
// Trail values are compared as relative densities: each species' trail is
// divided by its mean steady-state level (`mean_levels` in physarum.rs), so
// 1 means "as much trail as an even spread of the agents would leave".
struct Measure {
    size: vec2<u32>,
    agent_count: u32,
    species_count: u32,
    inv_cells: f32,
    inv_agents: f32,        // 0 when there are no agents
    cell_workgroups: u32,   // partial records written by the cell kernel
    _p0: u32,
    inv_level: vec4<f32>,   // 1 / mean trail level per species, 0 for dead species
    thresholds: vec4<f32>,  // marked ground x, vein x, travelled traffic, reserved
};
struct Agent {
    pos: vec2<f32>,
    heading: f32,
    state: u32,             // bits 0-1: species
};
@group(0) @binding(0) var<uniform> mu: Measure;
@group(0) @binding(1) var trail: texture_2d<f32>;
@group(0) @binding(2) var trail_prev: texture_2d<f32>;  // one sub-step older
@group(0) @binding(3) var traffic: texture_2d<f32>;
@group(0) @binding(4) var<storage, read> agents: array<Agent>;
@group(0) @binding(5) var<storage, read_write> partials: array<vec4<f32>>;

const LOG_FLOOR: f32 = 0.000000059604645; // 2^-24: an empty cell's log2 density
const LOG_CEIL: f32 = 16.0;

fn store_partials(record: u32, r: array<vec4<f32>, 4>) {
    let base = record * 4u;
    partials[base] = r[0];
    partials[base + 1u] = r[1];
    partials[base + 2u] = r[2];
    partials[base + 3u] = r[3];
}

// Total trail of a cell relative to the species' mean levels.
fn relative_density(t: vec4<f32>) -> f32 {
    return dot(max(finite_or_zero4(t), vec4<f32>(0.0)), mu.inv_level);
}

@compute @workgroup_size(16, 16)
fn cs_measure_cells(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var m: array<vec4<f32>, 4>;
    if (all(gid.xy < mu.size)) {
        let p = vec2<i32>(gid.xy);
        let x = relative_density(textureLoad(trail, p, 0));
        let x0 = relative_density(textureLoad(trail_prev, p, 0));
        let live = select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(mu.species_count));
        let g = dot(max(finite_or_zero4(textureLoad(traffic, p, 0)), vec4<f32>(0.0)), live);
        let w = mu.inv_cells;
        // ground, veins, concentration, travelled
        m[0] = vec4<f32>(
            f32(x > mu.thresholds.x),
            f32(x > mu.thresholds.y),
            log2(clamp(x, LOG_FLOOR, LOG_CEIL)),
            f32(g > mu.thresholds.z),
        ) * w;
        // trail_mass, reinforced, (agent lanes below), reserved
        m[1] = vec4<f32>(min(x, 10000.0), f32(x > x0), 0.0, 0.0) * w;
    }
    let r = metric_reduce(li, m);
    if (li == 0u) {
        store_partials(wg.y * nwg.x + wg.x, r);
    }
}

@compute @workgroup_size(256)
fn cs_measure_agents(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.x + gid.y * nwg.x * 256u;
    var m: array<vec4<f32>, 4>;
    if (i < mu.agent_count) {
        let a = agents[i];
        let s = a.state & 3u;
        // Positions can equal `size` after a wrap; `wrap_i` brings them back.
        let cell = wrap_i(vec2<i32>(floor(finite_or_zero2(a.pos))), vec2<i32>(mu.size));
        var t = max(finite_or_zero4(textureLoad(trail, cell, 0)), vec4<f32>(0.0));
        var inv_level = mu.inv_level;
        let xs = clamp(t[s] * inv_level[s], 0.0, 10000.0);
        let w = mu.inv_agents;
        // on_vein, agent_trail
        m[1].z = f32(xs > mu.thresholds.y) * w;
        m[1].w = xs * w;
    }
    let r = metric_reduce(li, m);
    if (li == 0u) {
        store_partials(mu.cell_workgroups + wg.y * nwg.x + wg.x, r);
    }
}
