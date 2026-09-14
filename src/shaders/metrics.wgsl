// ---------------------------------------------------------------------------
// Live measurements: the workgroup reduction shared by every world.
//
// A world's `cs_measure` kernel turns each cell (or agent) into up to sixteen
// contributions, already multiplied by their normalisation (1 / cells for a
// mean or a fraction, 1 / agents for an agent share), folds them across its
// 256-thread workgroup with `metric_reduce` and stores the four resulting
// vec4s at `partials[workgroup * 4 ..]`. The engine's `cs_reduce` (see
// metrics_reduce.wgsl) then sums every workgroup's partials into one 64-byte
// totals record. Everything stays in f32: the contributions are small and a
// tree of additions keeps the rounding error far below anything a sparkline
// could show.
//
// `metric_reduce` contains barriers, so every thread of the workgroup must
// call it exactly once per dispatch: kernels pass zeros for threads outside
// the domain instead of returning early.
// ---------------------------------------------------------------------------

// Half the workgroup parks its values here (8 KiB); the other half folds them
// in from registers before the barrier tree finishes the sum.
var<workgroup> metric_sums: array<array<vec4<f32>, 4>, 128>;

// Sums `m` over the calling 256-thread workgroup. Valid in thread 0 only.
fn metric_reduce(li: u32, m: array<vec4<f32>, 4>) -> array<vec4<f32>, 4> {
    var acc = m;
    if (li >= 128u) {
        metric_sums[li - 128u] = acc;
    }
    workgroupBarrier();
    if (li < 128u) {
        for (var k = 0u; k < 4u; k++) {
            acc[k] += metric_sums[li][k];
        }
        metric_sums[li] = acc;
    }
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride >>= 1u) {
        if (li < stride) {
            for (var k = 0u; k < 4u; k++) {
                acc[k] += metric_sums[li + stride][k];
            }
            metric_sums[li] = acc;
        }
        workgroupBarrier();
    }
    return acc;
}
