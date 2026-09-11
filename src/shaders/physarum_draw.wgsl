// Physarum display. Two fields are combined per species: the diffused trail
// (soft haze around the veins) and the traffic long exposure (crisp paths,
// brighter where more agents travel). Densities span three to five decades,
// from a lone agent's streak to a vein carrying thousands, so they go through
// a logarithmic tone curve: faint trails keep a delicate filigree while dense
// veins cross the knee and glow into HDR. The result is painted either through
// the palette (total density) or with each species' own colour.
//
// `cs_ground` runs first every frame and adapts the curve's black point to
// the image (see below); `fs_draw` then paints with it.

struct Draw {
    view: ViewXform,
    size: vec2<u32>,
    species_count: u32,
    mode: u32,                // 0 = palette by total density, 1 = species colours
    trail_gain: vec4<f32>,    // per species; display density 1 = the knee (a dense vein)
    traffic_gain: vec4<f32>,  // per species
    brightness: f32,
    black_point: f32,         // configured black point, 10^-filigree (never undercut)
    glow: f32,                // extra HDR emission of veins at and above the knee
    smoothing: f32,           // 0 = bilinear, 1 = cubic B-spline reconstruction
    ground: f32,              // fraction of the frame kept dark by the adaptive black point (0 = off)
    ground_rate: f32,         // per-frame glide of the adaptive black point (1 = instant)
    frame: u32,               // frame counter: re-draws the black point's sample jitter
    palette_span: f32,        // tone that reaches the top of the palette (>= 1)
    colors: array<vec4<f32>, 4>,
};

// Adaptive black point in display density units.
struct Ground {
    black_point: f32,         // 0 after a reset: the next measurement is taken as is
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> draw: Draw;
@group(0) @binding(1) var trail: texture_2d<f32>;
@group(0) @binding(2) var field_samp: sampler; // linear + repeat: wraps the torus
@group(0) @binding(3) var lut: texture_2d<f32>;
@group(0) @binding(4) var lut_samp: sampler;
@group(0) @binding(5) var traffic: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> ground_in: Ground;         // fs_draw
@group(0) @binding(7) var<storage, read_write> ground_out: Ground;  // cs_ground

// The black point never rises above this: at least one decade of tone remains.
const MAX_BLACK_POINT: f32 = 0.1;

fn live_mask() -> vec4<f32> {
    return select(vec4<f32>(0.0), vec4<f32>(1.0), vec4<u32>(0u, 1u, 2u, 3u) < vec4<u32>(draw.species_count));
}

// --- adaptive black point ------------------------------------------------------

const GROUND_GRID: u32 = 64u;                           // 64 x 64 stratified samples
const GROUND_SAMPLES: u32 = GROUND_GRID * GROUND_GRID;
const GROUND_BINS: u32 = 64u;
const GROUND_LOG_LO: f32 = -24.0;                       // log2 display density range of the histogram
const GROUND_LOG_HI: f32 = 4.0;

var<workgroup> histogram: array<atomic<u32>, GROUND_BINS>;

// How densities are spread depends entirely on the parameters: a network
// leaves most of the frame empty, while lanes or a fine lace can pack it. This
// single workgroup histograms the display density at 4096 jittered cells, finds
// the density below which the `ground` fraction of the frame lies and raises the
// black point to it, so that fraction stays dark whatever the parameters. It
// never goes below the configured black point, so sparse networks (whose
// ground is empty anyway) keep their full filigree.
@compute @workgroup_size(256)
fn cs_ground(@builtin(local_invocation_index) li: u32) {
    if (li < GROUND_BINS) {
        atomicStore(&histogram[li], 0u);
    }
    workgroupBarrier();

    let dims = vec2<f32>(draw.size);
    let live = live_mask();
    for (var k = 0u; k < GROUND_SAMPLES / 256u; k++) {
        let n = li + k * 256u;
        // One sample at a random spot inside each cell of the grid, re-drawn
        // every frame so the smoothed estimate cannot alias with periodic
        // patterns such as a honeycomb.
        let h = hash2u(n, draw.frame ^ 0x6a09e667u);
        let jitter = vec2<f32>(u32_to_unit(h), u32_to_unit(pcg_hash(h)));
        let uv = (vec2<f32>(f32(n % GROUND_GRID), f32(n / GROUND_GRID)) + jitter) / f32(GROUND_GRID);
        let p = min(vec2<i32>(uv * dims), vec2<i32>(draw.size) - vec2<i32>(1));
        let x = (textureLoad(trail, p, 0) * draw.trail_gain + textureLoad(traffic, p, 0) * draw.traffic_gain) * live;
        let l = (log2(max(x.x + x.y + x.z + x.w, 1e-12)) - GROUND_LOG_LO) / (GROUND_LOG_HI - GROUND_LOG_LO);
        atomicAdd(&histogram[u32(clamp(l, 0.0, 1.0) * f32(GROUND_BINS - 1u))], 1u);
    }
    workgroupBarrier();

    if (li == 0u) {
        var goal = draw.black_point;
        if (draw.ground > 0.0) {
            let wanted = u32(draw.ground * f32(GROUND_SAMPLES));
            var seen = 0u;
            var bin = 0u;
            loop {
                seen += atomicLoad(&histogram[bin]);
                if (seen >= wanted || bin + 1u >= GROUND_BINS) {
                    break;
                }
                bin += 1u;
            }
            // Upper edge of the bin holding the percentile.
            let level = exp2(GROUND_LOG_LO + f32(bin + 1u) / f32(GROUND_BINS - 1u) * (GROUND_LOG_HI - GROUND_LOG_LO));
            goal = max(min(level, MAX_BLACK_POINT), draw.black_point);
        }
        // Glide in log space so the image breathes slowly instead of
        // flickering; after a reset (stored 0) take the measurement as is.
        let prev = ground_out.black_point;
        let glided = exp2(mix(log2(max(prev, 1e-12)), log2(goal), draw.ground_rate));
        ground_out.black_point = select(goal, glided, prev > 0.0);
    }
}

// --- display ---------------------------------------------------------------------

// Cubic B-spline sample of `tex` at world uv (bilinear at smoothing 0, a blend
// in between). The B-spline is built from four bilinear taps (Sigg &
// Hadwiger); its weights are positive, so close-ups are smooth without
// ringing. The repeat sampler wraps every tap onto the torus, so zoomed-out
// tiling is seamless. `draw.smoothing` is uniform, so the branches are free.
fn field(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    if (draw.smoothing <= 0.0) {
        return max(textureSampleLevel(tex, field_samp, uv, 0.0), vec4<f32>(0.0));
    }
    let size = vec2<f32>(draw.size);
    let p = uv * size - 0.5;
    let i = floor(p);
    let f = p - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1; // >= 1/6
    let g1 = w2 + w3; // >= 1/6
    let c0 = (i - 0.5 + w1 / g0) / size;
    let c1 = (i + 1.5 + w3 / g1) / size;
    let cubic = g0.y * (g0.x * textureSampleLevel(tex, field_samp, vec2<f32>(c0.x, c0.y), 0.0)
            + g1.x * textureSampleLevel(tex, field_samp, vec2<f32>(c1.x, c0.y), 0.0))
        + g1.y * (g0.x * textureSampleLevel(tex, field_samp, vec2<f32>(c0.x, c1.y), 0.0)
            + g1.x * textureSampleLevel(tex, field_samp, vec2<f32>(c1.x, c1.y), 0.0));
    if (draw.smoothing >= 1.0) {
        return max(cubic, vec4<f32>(0.0));
    }
    let linear = textureSampleLevel(tex, field_samp, uv, 0.0);
    return max(mix(linear, cubic, draw.smoothing), vec4<f32>(0.0));
}

// Logarithmic tone curve: equal steps of brightness for equal density ratios.
// 0 at the black point `bp`, 1 at the knee (x = 1), above 1 for the densest veins.
fn tone(x: vec4<f32>, bp: f32) -> vec4<f32> {
    return log2(vec4<f32>(1.0) + x / bp) / log2(1.0 + 1.0 / bp);
}

// Emission gain: 1 in the filigree and the body of the network, rising
// smoothly from tone 0.8 so veins at the knee and beyond glow into HDR.
fn glow_gain(s: f32) -> f32 {
    let hot = clamp((s - 0.8) * 5.0, 0.0, 3.0);
    return 1.0 + draw.glow * hot * hot;
}

@fragment
fn fs_draw(in: FullscreenOut) -> @location(0) vec4<f32> {
    let w = view_apply(draw.view, in.uv);
    let x = (field(trail, w) * draw.trail_gain + field(traffic, w) * draw.traffic_gain) * live_mask();
    let bp = clamp(ground_in.black_point, 1e-6, MAX_BLACK_POINT);
    // Brightness follows the tone of the total density in both modes, so
    // where species overlap the image gets brighter, never bleached.
    let total = tone(vec4<f32>(x.x + x.y + x.z + x.w), bp).x;

    var col: vec3<f32>;
    if (draw.mode == 0u) {
        // Palette: position from the tone, reaching the top stop only at
        // `palette_span`, so the densest cores alone turn the lightest colour.
        col = palette_lookup(lut, lut_samp, total / draw.palette_span) * (draw.brightness * glow_gain(total));
    } else {
        // Species colours: the hue is the mean of the species' colours
        // weighted by (tone / strongest tone)^4, so the locally dominant
        // species keeps its own saturated colour and only true overlaps blend.
        let s = tone(x, bp);
        let r = s / max(max(max(s.x, s.y), max(s.z, s.w)), 1e-6);
        let wt = r * r * r * r * live_mask();
        let hue = (draw.colors[0].rgb * wt.x + draw.colors[1].rgb * wt.y + draw.colors[2].rgb * wt.z
            + draw.colors[3].rgb * wt.w) / max(wt.x + wt.y + wt.z + wt.w, 1e-6);
        // Squared: tone is perceptual, emission is linear light.
        col = hue * (total * total * glow_gain(total) * draw.brightness);
    }
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(64.0)), 1.0);
}
