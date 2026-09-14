struct Draw {
    view: ViewXform,
    size: vec2<u32>, layer: u32, brightness: f32,
};
@group(0) @binding(0) var<uniform> draw: Draw;
@group(0) @binding(1) var<storage, read> field: array<vec4<f32>>;
@group(0) @binding(2) var lut: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

fn cell(p: vec2<i32>) -> vec4<f32> { return field[wrap_index(p, draw.size)]; }
fn sample_field(p: vec2<f32>) -> vec4<f32> {
    let q = vec2<i32>(floor(p));
    let t = fract(p);
    return mix(mix(cell(q), cell(q + vec2<i32>(1, 0)), t.x),
               mix(cell(q + vec2<i32>(0, 1)), cell(q + vec2<i32>(1, 1)), t.x), t.y);
}
@fragment
fn fs_display(in: FullscreenOut) -> @location(0) vec4<f32> {
    let p = fract(view_apply(draw.view, in.uv)) * vec2<f32>(draw.size);
    let f = sample_field(p);
    let v = clamp(f.y * 2.9, 0.0, 1.0);
    // Suppress diffuse background trails so only the concentrated routes light
    // up. A dense colony should not turn the entire image into a white veil.
    let trail = pow(1.0 - exp(-max(f.z - 0.2, 0.0) * 0.45), 3.0);
    let dx = sample_field(p + vec2<f32>(1.0, 0.0)).y - sample_field(p - vec2<f32>(1.0, 0.0)).y;
    let dy = sample_field(p + vec2<f32>(0.0, 1.0)).y - sample_field(p - vec2<f32>(0.0, 1.0)).y;
    let normal = normalize(vec3<f32>(-dx * 14.0, -dy * 14.0, 1.0));
    let light = 0.4 + 0.6 * max(dot(normal, normalize(vec3<f32>(-0.5, -0.6, 1.0))), 0.0);
    let rim = clamp(length(vec2<f32>(dx, dy)) * 5.0, 0.0, 1.0);
    let chemistry = palette_lookup(lut, samp, 0.08 + 0.64 * v) * (0.12 + 1.2 * v) * light;
    let network_colour = mix(palette_lookup(lut, samp, 0.92), vec3<f32>(0.95, 0.55, 0.12), 0.45);
    var colour = chemistry + network_colour * trail * 0.65
               + palette_lookup(lut, samp, 0.72) * rim * 0.3;
    if draw.layer == 1u { colour = chemistry + palette_lookup(lut, samp, 0.72) * rim * 0.3; }
    if draw.layer == 2u { colour = vec3<f32>(0.002, 0.004, 0.009) + network_colour * trail; }
    return vec4<f32>(colour * draw.brightness, 1.0);
}
