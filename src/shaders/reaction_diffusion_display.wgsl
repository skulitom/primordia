// Gray-Scott reaction-diffusion display: lights the field baked by
// `cs_prepare` (reaction_diffusion.wgsl) as a height map in one of five
// materials, normalised by the auto-contrast window of `cs_resolve`.
//
// A module of its own: DirectX 12 translates a whole module for every
// pipeline, and its samplers must not share binding numbers with the compute
// passes' buffers (see the conventions in src/world/mod.rs).

struct Draw {
    view: ViewXform,
    size: vec2<u32>,
    time: f32,
    material: u32,       // 0 lacquer, 1 nacre, 2 luminous, 3 ink, 4 dark-field
    contrast: vec2<f32>, // body mask edges on the normalised level
    pal_range: vec2<f32>,
    relief: f32,
    gloss: f32,
    glow: f32,
    halo: f32,
    iridescence: f32,
    activity: f32,
    shadow: f32,
    brightness: f32,
    invert: u32,         // 1 = the filled state is the ground, holes are raised
    aura_tint: f32,      // palette position of the U-depletion halo
    reflect: f32,        // nacre: how much the ground reflects the film (0..1)
    clarity: f32,        // local contrast of the level (0 = off)
};

@group(0) @binding(0) var<uniform> draw: Draw;
@group(0) @binding(1) var field_tex: texture_2d<f32>;
@group(0) @binding(2) var field_samp: sampler;
@group(0) @binding(3) var lut: texture_2d<f32>;
@group(0) @binding(4) var lut_samp: sampler;
@group(0) @binding(5) var<storage, read> win: array<f32>;

// Hardware-filtered sample at world uv. The sampler repeats, which is exactly
// the torus wrap (and seamless for any zoom or pan).
fn field_at(uv: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(field_tex, field_samp, uv, 0.0);
}

// V normalised by the auto window: 0 = ground, 1 = crest. `invert` swaps them
// for regimes whose ground is the filled (high V) state.
fn level(v: f32) -> f32 {
    let n = clamp((v - win[0]) / max(win[1] - win[0], 1e-3), 0.0, 1.0);
    return select(n, 1.0 - n, draw.invert != 0u);
}

// Level of a field texel with optional local contrast: `clarity` pushes it
// away from the blurred level, lifting shallow ripples (such as honeycomb
// holes closing in a flooded plateau) out of flat mid-tones.
fn lift(s: vec4<f32>) -> f32 {
    let h = level(s.x);
    return clamp(h + draw.clarity * (h - level(s.z)), 0.0, 1.0);
}

fn pal(t: f32) -> vec3<f32> {
    return palette_lookup(lut, lut_samp, mix(draw.pal_range.x, draw.pal_range.y, clamp(t, 0.0, 1.0)));
}

// Thin-film interference colour for a film `thickness` micrometres thick
// (red, green and blue wavelengths), like oil on water or nacre.
fn thin_film(thickness: f32) -> vec3<f32> {
    let phase = thickness / vec3<f32>(0.65, 0.54, 0.45);
    return 0.5 + 0.5 * cos(TAU * phase);
}

// Keeps 70% of a colour's saturation.
fn soften(c: vec3<f32>) -> vec3<f32> {
    return mix(vec3<f32>(luminance(c)), c, 0.7);
}

// Slowly flowing thickness variations of a film (the swirls on a soap
// bubble). Integer wave vectors keep it periodic on the unit torus.
fn film_swirl(uv: vec2<f32>, t: f32) -> f32 {
    let a = sin(TAU * (2.0 * uv.x + uv.y) + 0.13 * t);
    let b = sin(TAU * (uv.x - 3.0 * uv.y) - 0.09 * t + 1.7);
    let c = sin(TAU * (3.0 * uv.x + 2.0 * uv.y) + 0.07 * t + 4.1);
    return (a + b + 0.5 * c) * 0.4;
}

// Smooth value noise on an n-cell lattice wrapped over the unit torus.
fn torus_noise(uv: vec2<f32>, n: vec2<i32>, salt: u32) -> f32 {
    let p = uv * vec2<f32>(n);
    let i = vec2<i32>(floor(p));
    let f = p - vec2<f32>(i);
    let e = f * f * (3.0 - 2.0 * f);
    let a = bitcast<vec2<u32>>(wrap_i(i, n));
    let b = bitcast<vec2<u32>>(wrap_i(i + vec2<i32>(1, 0), n));
    let c = bitcast<vec2<u32>>(wrap_i(i + vec2<i32>(0, 1), n));
    let d = bitcast<vec2<u32>>(wrap_i(i + vec2<i32>(1, 1), n));
    let top = mix(rand3(a.x, a.y, salt), rand3(b.x, b.y, salt), e.x);
    let bottom = mix(rand3(c.x, c.y, salt), rand3(d.x, d.y, salt), e.x);
    return mix(top, bottom, e.y);
}

// Paper fibre: two streaky octaves at right angles over a fine tooth, 0..1.
fn paper_fibre(uv: vec2<f32>) -> f32 {
    let cells = vec2<f32>(draw.size);
    let across = torus_noise(uv, max(vec2<i32>(cells / vec2<f32>(9.0, 1.5)), vec2<i32>(1)), 11u);
    let down = torus_noise(uv, max(vec2<i32>(cells / vec2<f32>(1.5, 9.0)), vec2<i32>(1)), 23u);
    let tooth = torus_noise(uv, max(vec2<i32>(cells / 1.2), vec2<i32>(1)), 37u);
    return 0.35 * across + 0.35 * down + 0.3 * tooth;
}

@fragment
fn fs_display(in: FullscreenOut) -> @location(0) vec4<f32> {
    let w = view_apply(draw.view, in.uv);
    let texel = 1.0 / vec2<f32>(draw.size);
    let f = field_at(w);
    // The level is both the height and the palette coordinate, so bodies are
    // rounded and shade from their rim colour to their crest colour; the
    // contrast window only decides where ground ends and body begins.
    let h = lift(f);
    let m = smoothstep(draw.contrast.x, draw.contrast.y, h);
    // Blurred level: how much body surrounds this point.
    let density = level(f.z);

    // Height-field normal (central differences one cell apart).
    let h_px = lift(field_at(w + vec2<f32>(texel.x, 0.0)));
    let h_mx = lift(field_at(w - vec2<f32>(texel.x, 0.0)));
    let h_py = lift(field_at(w + vec2<f32>(0.0, texel.y)));
    let h_my = lift(field_at(w - vec2<f32>(0.0, texel.y)));
    let hx = h_px - h_mx;
    let hy = h_py - h_my;
    let n = normalize(vec3<f32>(-vec2<f32>(hx, hy) * (draw.relief * 4.0), 1.0));
    // Macro normal from the blurred field two cells either side: the
    // orientation of the neighbourhood rather than of each small feature.
    let b_x = level(field_at(w + vec2<f32>(2.0 * texel.x, 0.0)).z) - level(field_at(w - vec2<f32>(2.0 * texel.x, 0.0)).z);
    let b_y = level(field_at(w + vec2<f32>(0.0, 2.0 * texel.y)).z) - level(field_at(w - vec2<f32>(0.0, 2.0 * texel.y)).z);
    let n_macro = normalize(vec3<f32>(-vec2<f32>(b_x, b_y) * (draw.relief * 2.0), 1.0));
    let l = normalize(vec3<f32>(-0.55, -0.65, 0.55));
    let half_vec = normalize(l + vec3<f32>(0.0, 0.0, 1.0));
    let ndl = dot(n, l);
    let diffuse = max(ndl, 0.0);
    let wrapped = max((ndl + 0.6) / 1.6, 0.0);
    let ndh = max(dot(n, half_vec), 0.0);
    // Sharp glints only on convex surface (crests of bodies): the rims of
    // holes and the creases where stripes meet are concave, and a pinpoint
    // highlight there reads as a stray white tick rather than as gloss. The
    // glint also follows a blend with the macro normal, so small holes and
    // necks do not each catch an identical tick on their lit shoulder.
    let curvature = h_px + h_mx + h_py + h_my - 4.0 * h;
    let convex = 1.0 - smoothstep(-0.03, 0.05, curvature);
    let spec = pow(max(dot(normalize(mix(n, n_macro, 0.6)), half_vec), 0.0), 60.0) * convex;
    let sheen = pow(ndh, 10.0);
    let fres = pow(clamp(1.0 - n.z, 0.0, 1.0), 1.2);

    // Contact occlusion (ground hugging raised structure) and a cast shadow
    // from whatever stands between this point and the light.
    let cavity = clamp((density - h) * 1.8, 0.0, 1.0);
    let occluder = lift(field_at(w + normalize(l.xy) * texel * 2.5));
    let shadow = clamp((occluder - h) * 1.3, 0.0, 1.0) * draw.shadow;
    let ao = (1.0 - cavity * 0.6) * (1.0 - shadow * 0.75);
    // U is depleted around structures: a soft aura reaching further than V.
    // It fades where the neighbourhood is dense, so the gaps of a packed
    // labyrinth keep a deep ground instead of a grey haze.
    let u_level = clamp((win[3] - f.y) / max(win[3] - win[2], 1e-3), 0.0, 1.0);
    let aura = u_level * u_level * (1.0 - m) * (1.0 - density);
    // Growth glow, compressed so fast waves never blow out.
    let fronts = f.w / (1.0 + f.w) * draw.activity;
    // Slowly flowing film thickness shared by the iridescent materials.
    let swirl = film_swirl(w, draw.time);

    let ground = pal(0.0);
    let body = pal(h);
    var col: vec3<f32>;
    switch draw.material {
        case 1u: {
            // Nacre: thin-film interference whose thickness is dominated by
            // the slow swirls, so hue flows across the image in broad bands
            // like a soap film; height and density only bend the bands. The
            // film fades in with body size (decaying specks would otherwise
            // read as saturated dots), and the ground faintly reflects it.
            let thickness = 0.45 + 0.35 * h + 0.2 * density + 0.1 * n.z + 1.6 * swirl;
            let film = thin_film(thickness);
            // Squaring pushes the pastel interference colours towards soap-film vividness.
            let vivid = soften(film * film * 1.6);
            let sheet = smoothstep(0.2, 0.5, density);
            let base = body * (0.25 + 0.85 * diffuse);
            let reflection = mix(ground, pal(0.06) + 0.03 * thin_film(0.6 + swirl), draw.reflect);
            col = mix(reflection, base, m) * ao;
            col += vivid * (0.3 + fres * 1.4 + sheen * 0.6) * draw.iridescence * m * sheet * mix(vec3<f32>(1.0), body, 0.25);
            col += spec * draw.gloss * 2.5 * mix(vec3<f32>(1.0), film, 0.5) * m;
        }
        case 2u: {
            // Luminous: translucent bodies lit from within; bright cores bloom.
            let core = pow(h, 1.6);
            let base = body * (0.15 + 0.5 * wrapped);
            col = mix(ground, base, m) * ao;
            col += pal(0.45 + 0.55 * h) * core * m * draw.glow * 2.2;
            col += soften(thin_film(0.4 + 0.6 * h + 1.2 * swirl)) * fres * m * draw.iridescence * 0.7;
            col += spec * draw.gloss * 1.5 * m * mix(vec3<f32>(1.0), body, 0.4);
        }
        case 3u: {
            // Ink on paper: pigment pools along the flanks of each ridge
            // (the centre stays a shade lighter), the inking is slightly
            // uneven, and the paper has a faint fibre and is embossed by the
            // relief.
            let edge = clamp(length(vec2<f32>(hx, hy)) * 1.4, 0.0, 1.0);
            let pooled = m * (0.8 + 0.2 * edge) + edge * 0.15 * (1.0 - m);
            let ink = clamp(pooled * (0.9 + 0.1 * film_swirl(w, 0.0)), 0.0, 1.0);
            let emboss = 1.0 + (ndl - l.z) * 0.35 * draw.relief;
            let paper = 0.95 + 0.05 * paper_fibre(w);
            col = pal(ink) * paper * emboss * (1.0 - cavity * 0.08 - shadow * 0.12);
            col += vec3<f32>(spec * draw.gloss * 0.3 * m);
        }
        case 4u: {
            // Dark-field microscopy: light scattered off the membranes (the
            // level band where ground turns into body, plus any steep slope)
            // outlines every body; the interior stays dim and translucent
            // around a small bright nucleus.
            let edge_mid = mix(draw.contrast.x, draw.contrast.y, 0.5);
            let band = smoothstep(draw.contrast.x, edge_mid, h) * (1.0 - smoothstep(edge_mid, draw.contrast.y + 0.3, h));
            let slope = clamp(length(vec2<f32>(hx, hy)) * 2.0, 0.0, 1.0);
            let membrane = max(band, slope * slope);
            let nucleus = smoothstep(0.9, 1.0, h);
            col = mix(ground, pal(0.3 + 0.3 * h) * (0.08 + 0.2 * wrapped), m) * ao;
            col += pal(0.6 + 0.4 * h) * membrane * draw.glow * 1.5;
            col += pal(1.0) * nucleus * draw.glow * 0.5;
            col += soften(thin_film(0.5 + 0.5 * h + 1.2 * swirl)) * membrane * draw.iridescence * 0.6;
            col += spec * draw.gloss * m * mix(vec3<f32>(1.0), body, 0.3);
        }
        default: {
            // Lacquer: glossy enamel with sharp glints.
            let base = body * (0.28 + 0.95 * diffuse);
            col = mix(ground, base, m) * ao;
            col += body * fres * 0.35 * m;
            col += spec * draw.gloss * 3.0 * mix(vec3<f32>(1.0), body, 0.3) * m;
            col += soften(thin_film(0.5 + 0.5 * h + swirl)) * fres * draw.iridescence * m * 0.6;
        }
    }

    if (draw.material != 3u) {
        col += pal(draw.aura_tint) * aura * draw.halo;
        col += pal(0.85) * fronts * 0.8;
    }
    col *= draw.brightness;
    // Every term above is bounded (the state is clamped to 0..1 and every
    // division is guarded), so this only caps extreme highlights.
    return vec4<f32>(clamp(col, vec3<f32>(0.0), vec3<f32>(64.0)), 1.0);
}
