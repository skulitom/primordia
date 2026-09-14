// Sums the per-workgroup partials written by a world's `cs_measure` (four
// vec4<f32> per workgroup) into one 64-byte totals record. Compiled after
// metrics.wgsl, whose `metric_reduce` does the in-workgroup tree.

struct ReduceParams {
    count: u32, // partial records to sum (workgroups the measure kernel ran)
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var<uniform> reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read> reduce_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> reduce_out: array<vec4<f32>, 4>;

@compute @workgroup_size(256)
fn cs_reduce(@builtin(local_invocation_index) li: u32) {
    var acc: array<vec4<f32>, 4>;
    for (var i = li; i < reduce_params.count; i += 256u) {
        let base = i * 4u;
        for (var k = 0u; k < 4u; k++) {
            acc[k] += reduce_in[base + k];
        }
    }
    let r = metric_reduce(li, acc);
    if (li == 0u) {
        for (var k = 0u; k < 4u; k++) {
            reduce_out[k] = r[k];
        }
    }
}
