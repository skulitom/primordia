// Live measurements of Particle Life: one thread per particle (and, for the
// empty-cell lane, per grid cell) reduces the state `step` just produced (see
// metrics.wgsl for the contract). The lanes match `METRICS` in
// particle_life.rs. Cell occupancies come from the last sub-step's counting
// sort: `sorted[i]` is the i-th particle in cell order and `starts` holds the
// cell start offsets, so both describe the same snapshot.
struct Measure {
    count: u32,
    cells: u32,           // live grid cells (the buffer is allocated for more)
    grid: vec2<u32>,
    cell: vec2<f32>,      // cell size in world units
    inv_count: f32,       // 0 when there are no particles
    inv_cells: f32,
    inv_r_max: f32,
    hot: f32,             // speed from which a particle counts as hot
    slow: f32,            // speed below which a particle counts as settled
    inv_crowd: f32,       // 1 / (1 + expected particles per cell)
    dense: f32,           // cell-mates from which a particle counts as clustered
    mix: f32,             // expected same-species share among cell-mates
    inv_unmix: f32,       // 1 / (1 - mix), 0 for a single species
    _pad: f32,
};
@group(0) @binding(0) var<uniform> mu: Measure;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;     // canonical order (unused lanes read it for symmetry)
@group(0) @binding(2) var<storage, read> vel: array<vec2<f32>>;     // canonical order
@group(0) @binding(3) var<storage, read> sorted: array<vec4<f32>>;  // cell order of the last sub-step
@group(0) @binding(4) var<storage, read> starts: array<u32>;        // cell start offsets (cells + 1 entries)
@group(0) @binding(5) var<storage, read_write> partials: array<vec4<f32>>;

fn cell_index(p: vec2<f32>) -> u32 {
    let c = clamp(vec2<i32>(floor(p / mu.cell)), vec2<i32>(0), vec2<i32>(mu.grid) - vec2<i32>(1));
    return u32(c.y) * mu.grid.x + u32(c.x);
}

@compute @workgroup_size(256)
fn cs_measure(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let i = gid.x + gid.y * nwg.x * 256u;
    var m: array<vec4<f32>, 4>;
    if (i < mu.count) {
        let v = finite_or_zero2(vel[i]);
        let speed = min(length(v), 1e6);
        let q = finite_or_zero4(sorted[i]);
        let c = cell_index(q.xy);
        let first = starts[c];
        let last = max(starts[c + 1u], first);
        let n = f32(max(last - first, 1u));
        var same = 0.0;
        for (var j = first; j < last; j++) {
            same += f32(sorted[j].z == q.z);
        }
        var segregation = 0.0;
        if (n > 1.0) {
            segregation = clamp(((same - 1.0) / (n - 1.0) - mu.mix) * mu.inv_unmix, -1.0, 1.0);
        }
        let w = mu.inv_count;
        // speed, hot, slow, crowding
        m[0] = vec4<f32>(speed * mu.inv_r_max, f32(speed > mu.hot), f32(speed < mu.slow), n * mu.inv_crowd) * w;
        // dense, segregation, (void: per cell below), reserved
        m[1] = vec4<f32>(f32(n > mu.dense), segregation, 0.0, 0.0) * w;
    }
    if (i < mu.cells) {
        m[1].z = f32(starts[i + 1u] == starts[i]) * mu.inv_cells;
    }
    let r = metric_reduce(li, m);
    if (li == 0u) {
        let base = (wg.y * nwg.x + wg.x) * 4u;
        partials[base] = r[0];
        partials[base + 1u] = r[1];
        partials[base + 2u] = r[2];
        partials[base + 3u] = r[3];
    }
}
