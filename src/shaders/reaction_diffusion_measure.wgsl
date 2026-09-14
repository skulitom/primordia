// Live measurements of the Gray-Scott field: a 16x16 workgroup per block of
// cells reduces the state `step` just produced (see metrics.wgsl for the
// contract). The lanes match `METRICS` in reaction_diffusion.rs.
struct Measure {
    size: vec2<u32>,
    inv_cells: f32,
    _p0: f32,
    thresholds: vec4<f32>, // alive V, body V, filled V, active |dV|
};
@group(0) @binding(0) var<uniform> mu: Measure;
@group(0) @binding(1) var<storage, read> latest: array<vec2<f32>>;    // (U, V)
@group(0) @binding(2) var<storage, read> previous: array<vec2<f32>>;  // one step older
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<f32>>;

fn alive_at(p: vec2<i32>) -> bool {
    return finite_or_zero(latest[wrap_index(p, mu.size)].y) > mu.thresholds.x;
}

@compute @workgroup_size(16, 16)
fn cs_measure(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var m: array<vec4<f32>, 4>;
    if (all(gid.xy < mu.size)) {
        let i = gid.y * mu.size.x + gid.x;
        let c = clamp(finite_or_zero2(latest[i]), vec2<f32>(0.0), vec2<f32>(1.0));
        let b = clamp(finite_or_zero2(previous[i]), vec2<f32>(0.0), vec2<f32>(1.0));
        let dv = c.y - b.y;
        let alive = c.y > mu.thresholds.x;
        // A living cell with a dead 4-neighbour lies on the pattern's boundary.
        var edge = 0.0;
        if (alive) {
            let p = vec2<i32>(gid.xy);
            let surrounded = alive_at(p + vec2<i32>(1, 0)) && alive_at(p - vec2<i32>(1, 0))
                && alive_at(p + vec2<i32>(0, 1)) && alive_at(p - vec2<i32>(0, 1));
            edge = f32(!surrounded);
        }
        let w = mu.inv_cells;
        // alive, body, v_mean, u_mean
        m[0] = vec4<f32>(f32(alive), f32(c.y > mu.thresholds.y), c.y, c.x) * w;
        // active, v_drift, edge, filled
        m[1] = vec4<f32>(f32(abs(dv) > mu.thresholds.w), dv, edge, f32(c.y > mu.thresholds.z)) * w;
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
