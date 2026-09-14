// ---------------------------------------------------------------------------
// Primordia common WGSL prelude.
//
// `Gpu::shader` prepends this file to every shader module, so everything here
// is available to all worlds: a fullscreen-triangle vertex shader, the view
// transform, hashing / random numbers, colour helpers and palette lookup.
// ---------------------------------------------------------------------------

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

struct FullscreenOut {
    @builtin(position) position: vec4<f32>,
    // Screen uv: (0,0) is the top-left corner, (1,1) the bottom-right corner.
    @location(0) uv: vec2<f32>,
};

// Oversized triangle that covers the whole viewport. Draw with `draw(0..3, 0..1)`.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> FullscreenOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: FullscreenOut;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// Maps screen uv -> world uv (`world = screen * scale + offset`). World uv may
// fall outside 0..1 when zoomed out or panned: every Primordia world is a torus,
// so wrap with `fract()` (or integer modulo) when sampling.
struct ViewXform {
    scale: vec2<f32>,
    offset: vec2<f32>,
};

fn view_apply(v: ViewXform, screen_uv: vec2<f32>) -> vec2<f32> {
    return screen_uv * v.scale + v.offset;
}

// Positive modulo for wrapping integer cell coordinates onto the torus.
// The float quotient only has to land within one period of the true floor
// (GPU division is not correctly rounded); the two selects then make the
// result exact for |p| < 2^24. Verified on-device by `primordia selftest`.
fn wrap_i(p: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    var q = p - size * vec2<i32>(floor(vec2<f32>(p) / vec2<f32>(size)));
    q = select(q, q - size, q >= size);
    q = select(q, q + size, q < vec2<i32>(0));
    return q;
}

// Unsigned flavour of `wrap_i` for linear indexing: returns y * width + x.
fn wrap_index(p: vec2<i32>, size: vec2<u32>) -> u32 {
    let q = wrap_i(p, vec2<i32>(size));
    return u32(q.y) * size.x + u32(q.x);
}

// Shortest displacement from `a` to `b` on a torus of the given size.
fn torus_delta(a: vec2<f32>, b: vec2<f32>, size: vec2<f32>) -> vec2<f32> {
    var d = b - a;
    d = d - size * round(d / size);
    return d;
}

// --- finite guards ----------------------------------------------------------
// Measurements must not let a NaN in a state buffer poison a whole sum. The
// exponent-bit test is used because WGSL lets implementations assume floats
// are never NaN, so `x != x` may be folded away. Apply these before any clamp,
// mask multiply or comparison (NaN * 0 is still NaN).

fn is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn finite_or_zero(x: f32) -> f32 {
    return select(0.0, x, is_finite(x));
}

fn finite_or_zero2(v: vec2<f32>) -> vec2<f32> {
    let finite = (bitcast<vec2<u32>>(v) & vec2<u32>(0x7f800000u)) != vec2<u32>(0x7f800000u);
    return select(vec2<f32>(0.0), v, finite);
}

fn finite_or_zero4(v: vec4<f32>) -> vec4<f32> {
    let finite = (bitcast<vec4<u32>>(v) & vec4<u32>(0x7f800000u)) != vec4<u32>(0x7f800000u);
    return select(vec4<f32>(0.0), v, finite);
}

// --- hashing / random numbers ----------------------------------------------

fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash2u(a: u32, b: u32) -> u32 {
    return pcg_hash(a ^ pcg_hash(b));
}

fn hash3u(a: u32, b: u32, c: u32) -> u32 {
    return pcg_hash(a ^ pcg_hash(b ^ pcg_hash(c)));
}

// Uniform float in [0, 1).
fn u32_to_unit(h: u32) -> f32 {
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

fn rand1(a: u32) -> f32 {
    return u32_to_unit(pcg_hash(a));
}

fn rand2(a: u32, b: u32) -> f32 {
    return u32_to_unit(hash2u(a, b));
}

fn rand3(a: u32, b: u32, c: u32) -> f32 {
    return u32_to_unit(hash3u(a, b, c));
}

// --- colour helpers --------------------------------------------------------

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((max(c, vec3<f32>(0.0)) + 0.055) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

// Sample a palette LUT (see palette.rs: a 256x1 sRGB texture, returned in
// linear space). `t` is clamped to [0, 1].
fn palette_lookup(lut: texture_2d<f32>, samp: sampler, t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0) * (255.0 / 256.0) + (0.5 / 256.0);
    return textureSampleLevel(lut, samp, vec2<f32>(x, 0.5), 0.0).rgb;
}

// Cheap cosine-gradient palette (Inigo Quilez) for procedural colour.
fn cosine_palette(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(TAU * (c * t + d));
}

// ---------------------------------------------------------------------------
// End of prelude. World shader source follows.
// ---------------------------------------------------------------------------
