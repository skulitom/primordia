// Post-processing: physically-based bloom (13-tap downsample / tent upsample
// chain, Jimenez 2014) followed by exposure, tonemapping, vignette, grain and
// dithering into the output target.

struct PostParams {
    exposure: f32,
    bloom_strength: f32,
    bloom_threshold: f32,
    vignette: f32,
    tonemap: u32,
    // 1 / (sum of the per-level weights), so strength does not depend on how many
    // levels the current resolution has.
    bloom_norm: f32,
    time: f32,
    saturation: f32,
    encode_srgb: u32,
    grain: f32,
    // Weight applied each time a coarser level is folded into a finer one: level
    // j reaches the composite with weight falloff^j, so wide halos stay subtle and
    // the image does not wash out with the global average brightness.
    bloom_falloff: f32,
    _pad0: f32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_samp: sampler;
@group(0) @binding(2) var<uniform> params: PostParams;
// Only used by the composite pass.
@group(0) @binding(3) var bloom_tex: texture_2d<f32>;

fn src(uv: vec2<f32>) -> vec3<f32> {
    // Clamp away NaN/Inf so a single bad texel cannot poison the whole chain.
    return clamp(textureSampleLevel(src_tex, src_samp, uv, 0.0).rgb, vec3<f32>(0.0), vec3<f32>(60000.0));
}

fn karis_weight(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + luminance(c));
}

@fragment
fn fs_downsample_first(in: FullscreenOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let uv = in.uv;
    let a = src(uv + t * vec2<f32>(-2.0, -2.0));
    let b = src(uv + t * vec2<f32>(0.0, -2.0));
    let c = src(uv + t * vec2<f32>(2.0, -2.0));
    let d = src(uv + t * vec2<f32>(-2.0, 0.0));
    let e = src(uv);
    let f = src(uv + t * vec2<f32>(2.0, 0.0));
    let g = src(uv + t * vec2<f32>(-2.0, 2.0));
    let h = src(uv + t * vec2<f32>(0.0, 2.0));
    let i = src(uv + t * vec2<f32>(2.0, 2.0));
    let j = src(uv + t * vec2<f32>(-1.0, -1.0));
    let k = src(uv + t * vec2<f32>(1.0, -1.0));
    let l = src(uv + t * vec2<f32>(-1.0, 1.0));
    let m = src(uv + t * vec2<f32>(1.0, 1.0));

    // Karis average of the five 2x2 groups suppresses single-pixel fireflies.
    let g0 = (j + k + l + m) * 0.25;
    let g1 = (a + b + d + e) * 0.25;
    let g2 = (b + c + e + f) * 0.25;
    let g3 = (d + e + g + h) * 0.25;
    let g4 = (e + f + h + i) * 0.25;
    let w0 = 0.5 * karis_weight(g0);
    let w1 = 0.125 * karis_weight(g1);
    let w2 = 0.125 * karis_weight(g2);
    let w3 = 0.125 * karis_weight(g3);
    let w4 = 0.125 * karis_weight(g4);
    var o = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);

    // Soft-knee threshold (identity when the threshold is 0).
    let threshold = params.bloom_threshold;
    let knee = max(threshold * 0.5, 1e-4);
    let brightness = max(o.r, max(o.g, o.b));
    var soft = clamp(brightness - threshold + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-5);
    let contribution = max(soft, brightness - threshold) / max(brightness, 1e-5);
    o = o * clamp(contribution, 0.0, 1.0);
    return vec4<f32>(o, 1.0);
}

@fragment
fn fs_downsample(in: FullscreenOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let uv = in.uv;
    let a = src(uv + t * vec2<f32>(-2.0, -2.0));
    let b = src(uv + t * vec2<f32>(0.0, -2.0));
    let c = src(uv + t * vec2<f32>(2.0, -2.0));
    let d = src(uv + t * vec2<f32>(-2.0, 0.0));
    let e = src(uv);
    let f = src(uv + t * vec2<f32>(2.0, 0.0));
    let g = src(uv + t * vec2<f32>(-2.0, 2.0));
    let h = src(uv + t * vec2<f32>(0.0, 2.0));
    let i = src(uv + t * vec2<f32>(2.0, 2.0));
    let j = src(uv + t * vec2<f32>(-1.0, -1.0));
    let k = src(uv + t * vec2<f32>(1.0, -1.0));
    let l = src(uv + t * vec2<f32>(-1.0, 1.0));
    let m = src(uv + t * vec2<f32>(1.0, 1.0));
    var o = e * 0.125;
    o += (a + c + g + i) * 0.03125;
    o += (b + d + f + h) * 0.0625;
    o += (j + k + l + m) * 0.125;
    return vec4<f32>(o, 1.0);
}

// 3x3 tent filter, weighted by the falloff; additively blended into the next
// larger level.
@fragment
fn fs_upsample(in: FullscreenOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src_tex));
    let uv = in.uv;
    var o = src(uv) * 4.0;
    o += (src(uv + vec2<f32>(-t.x, 0.0)) + src(uv + vec2<f32>(t.x, 0.0)) + src(uv + vec2<f32>(0.0, -t.y)) + src(uv + vec2<f32>(0.0, t.y))) * 2.0;
    o += src(uv + vec2<f32>(-t.x, -t.y)) + src(uv + vec2<f32>(t.x, -t.y)) + src(uv + vec2<f32>(-t.x, t.y)) + src(uv + vec2<f32>(t.x, t.y));
    return vec4<f32>(o * (params.bloom_falloff / 16.0), 1.0);
}

fn bloom_at(uv: vec2<f32>) -> vec3<f32> {
    return clamp(textureSampleLevel(bloom_tex, src_samp, uv, 0.0).rgb, vec3<f32>(0.0), vec3<f32>(60000.0));
}

// The finest bloom level is half resolution; tent-filter it up to avoid blocky glows.
fn bloom_tent(uv: vec2<f32>) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(bloom_tex));
    var o = bloom_at(uv) * 4.0;
    o += (bloom_at(uv + vec2<f32>(-t.x, 0.0)) + bloom_at(uv + vec2<f32>(t.x, 0.0)) + bloom_at(uv + vec2<f32>(0.0, -t.y)) + bloom_at(uv + vec2<f32>(0.0, t.y))) * 2.0;
    o += bloom_at(uv + vec2<f32>(-t.x, -t.y)) + bloom_at(uv + vec2<f32>(t.x, -t.y)) + bloom_at(uv + vec2<f32>(-t.x, t.y)) + bloom_at(uv + vec2<f32>(t.x, t.y));
    return o / 16.0;
}

// --- tonemapping -------------------------------------------------------------

fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

// Minimal AgX (Troy Sobotka's AgX, polynomial fit by Benjamin Wrensch).
fn tonemap_agx(color: vec3<f32>) -> vec3<f32> {
    let agx_mat = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let agx_mat_inv = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = agx_mat * max(color, vec3<f32>(1e-10));
    v = clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    v = agx_contrast(v);
    v = agx_mat_inv * v;
    // AgX output is display-encoded. Undo exactly the sRGB curve the composite
    // re-applies, so display values come out as AgX intended (a plain 2.2 power
    // would crush the shadows).
    return srgb_to_linear(clamp(v, vec3<f32>(0.0), vec3<f32>(1.0)));
}

// Narkowicz's ACES filmic fit.
fn tonemap_aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Reinhard on luminance; channels are rescaled (not clipped) above 1 so bright
// saturated colours keep their hue.
fn tonemap_reinhard(x: vec3<f32>) -> vec3<f32> {
    let y = x / (1.0 + luminance(x));
    return y / max(max(max(y.r, y.g), y.b), 1.0);
}

@fragment
fn fs_composite(in: FullscreenOut) -> @location(0) vec4<f32> {
    var hdr = src(in.uv);
    if (params.bloom_strength > 0.0) {
        hdr += bloom_tent(in.uv) * (params.bloom_norm * params.bloom_strength);
    }
    hdr = hdr * params.exposure;
    let l = luminance(hdr);
    hdr = max(mix(vec3<f32>(l), hdr, params.saturation), vec3<f32>(0.0));

    var c: vec3<f32>;
    switch params.tonemap {
        case 0u: { c = tonemap_agx(hdr); }
        case 1u: { c = tonemap_aces(hdr); }
        case 2u: { c = tonemap_reinhard(hdr); }
        default: { c = clamp(hdr, vec3<f32>(0.0), vec3<f32>(1.0)); }
    }

    // Vignette.
    let d = (in.uv - 0.5) * 1.41421356;
    c = c * (1.0 - params.vignette * smoothstep(0.25, 1.0, dot(d, d)));

    // Work in sRGB for grain + dither so the noise is perceptually even.
    var s = linear_to_srgb(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
    let px = vec2<u32>(in.position.xy);
    let frame = u32(params.time * 60.0);
    let n1 = rand3(px.x, px.y, frame);
    let n2 = rand3(px.x + 7919u, px.y + 104729u, frame);
    let tri = n1 + n2 - 1.0; // triangular noise in (-1, 1)
    s = s + vec3<f32>(tri * (1.0 / 255.0));
    if (params.grain > 0.0) {
        s = s + vec3<f32>((n1 - 0.5) * params.grain * 0.12);
    }
    s = clamp(s, vec3<f32>(0.0), vec3<f32>(1.0));

    if (params.encode_srgb != 0u) {
        return vec4<f32>(s, 1.0);
    }
    return vec4<f32>(srgb_to_linear(s), 1.0);
}
