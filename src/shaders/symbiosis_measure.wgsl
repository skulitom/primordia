// Live measurements of one Symbiosis habitat: a 16x16 workgroup per block of
// cells reduces the state `step` just produced (see metrics.wgsl for the
// contract). The lanes match `METRICS` in symbiosis.rs.
struct Measure {
    size: vec2<u32>,
    count: u32,             // agents; 0 = none
    _p0: u32,
    inv_cells: f32,
    inv_count: f32,         // 0 when there are no agents
    _p1: vec2<f32>,
    thresholds: vec4<f32>,  // growth V, active |dV|, exhausted fertility, busy-route trail
};
@group(0) @binding(0) var<uniform> mu: Measure;
@group(0) @binding(1) var<storage, read> latest: array<vec4<f32>>;    // (U, V, trail, geography)
@group(0) @binding(2) var<storage, read> previous: array<vec4<f32>>;  // one chemistry step older
@group(0) @binding(3) var<storage, read> fertility: array<f32>;
@group(0) @binding(4) var<storage, read> deposits: array<u32>;        // agents per cell this frame
@group(0) @binding(5) var<storage, read_write> partials: array<vec4<f32>>;

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
        let c = finite_or_zero4(latest[i]);
        let p = finite_or_zero4(previous[i]);
        let v = clamp(c.y, 0.0, 1.0);
        let dv = v - clamp(p.y, 0.0, 1.0);
        let f = clamp(finite_or_zero(fertility[i]), 0.0, 1.0);
        let d = f32(min(deposits[i], mu.count));
        let growing = f32(v > mu.thresholds.x);
        let w = mu.inv_cells;
        // growth_cover, growth_mean, growth_active, growth_drift
        m[0] = vec4<f32>(growing, v, f32(abs(dv) > mu.thresholds.y), dv) * w;
        // routes, fertility_mean, exhausted, agents_on_growth
        m[1] = vec4<f32>(f32(c.z > mu.thresholds.w) * w, f * w, f32(f < mu.thresholds.z) * w, d * growing * mu.inv_count);
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
