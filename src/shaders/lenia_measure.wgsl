// Live measurements of a Lenia domain: a 16x16 workgroup per block of cells
// reduces the state `step` just produced (see metrics.wgsl for the contract).
// The lanes match `METRICS` in lenia.rs. The state is planar and
// channel-major: plane c of cell i is `state[c * n + i]`.
struct Measure {
    size: vec2<u32>,
    channels: u32,      // active channels; stale planes above it are ignored
    _p0: u32,
    inv_cells: f32,
    occupied: f32,      // summed value from which a cell counts as occupied
    changing: f32,      // summed change from which a cell counts as changing
    dense: f32,         // summed value from which a cell counts as a dense core
};
@group(0) @binding(0) var<uniform> mu: Measure;
@group(0) @binding(1) var<storage, read> latest: array<f32>;
@group(0) @binding(2) var<storage, read> previous: array<f32>;      // one full step older
@group(0) @binding(3) var<storage, read> growth: array<vec4<f32>>;  // last step's growth per channel
@group(0) @binding(4) var<storage, read_write> partials: array<vec4<f32>>;

fn planes(buffer_index: u32, n: u32, i: u32) -> vec3<f32> {
    if (buffer_index == 0u) {
        return vec3<f32>(finite_or_zero(latest[i]), finite_or_zero(latest[n + i]), finite_or_zero(latest[2u * n + i]));
    }
    return vec3<f32>(finite_or_zero(previous[i]), finite_or_zero(previous[n + i]), finite_or_zero(previous[2u * n + i]));
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
        let n = mu.size.x * mu.size.y;
        let i = gid.y * mu.size.x + gid.x;
        let on = vec3<f32>(vec3<u32>(0u, 1u, 2u) < vec3<u32>(mu.channels));
        let a = clamp(planes(0u, n, i), vec3<f32>(0.0), vec3<f32>(1.0)) * on;
        let b = clamp(planes(1u, n, i), vec3<f32>(0.0), vec3<f32>(1.0)) * on;
        let g = clamp(finite_or_zero4(growth[i]).xyz, vec3<f32>(-1.0), vec3<f32>(1.0)) * on;
        let total = a.x + a.y + a.z;
        let change = abs(a - b);
        let w = mu.inv_cells;
        // mass, mass_1, mass_2, mass_3
        m[0] = vec4<f32>(total, a) * w;
        // occupied, active, growth, dense
        m[1] = vec4<f32>(
            f32(total > mu.occupied),
            f32(change.x + change.y + change.z > mu.changing),
            g.x + g.y + g.z,
            f32(total > mu.dense),
        ) * w;
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
