//! On-device checks of the WGSL prelude maths every world relies on
//! (`primordia selftest`). GPUs are allowed to round float division loosely and
//! drivers differ, so torus wrapping is verified against the CPU.

use anyhow::{anyhow, bail, Context as _, Result};
use wgpu::ShaderStages;

use crate::gpu::{layout, Gpu};
use crate::rng::Rng;

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> cases: array<vec2<i32>>;
@group(0) @binding(1) var<storage, read_write> results: array<vec4<i32>>;

// The formula wrap_i originally used, kept to document backend behaviour.
fn legacy_wrap(p: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    return ((p % size) + size) % size;
}

@compute @workgroup_size(256)
fn cs_wrap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&cases)) {
        return;
    }
    let c = cases[i];
    // Exercise both vector lanes, with opposite signs.
    let p = vec2<i32>(c.x, -c.x);
    let s = vec2<i32>(c.y, c.y);
    results[i] = vec4<i32>(wrap_i(p, s), legacy_wrap(p, s));
}
"#;

pub fn run() -> Result<()> {
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None))?;
    println!("GPU: {} ({:?})", gpu.adapter_name(), gpu.adapter.get_info().backend);

    // Every size up to 8192 at the edges that matter, plus random far-away points.
    let mut cases: Vec<[i32; 2]> = Vec::new();
    for s in 1..=8192i32 {
        for p in [-2 * s, -s - 1, -s, -s + 1, -2, -1, 0, 1, s - 2, s - 1, s, s + 1, 2 * s - 1, 2 * s] {
            cases.push([p, s]);
        }
    }
    let mut rng = Rng::new(7);
    for _ in 0..20_000 {
        let s = 1 + rng.below(16_384) as i32;
        let p = rng.below(1 << 22) as i32 - (1 << 21);
        cases.push([p, s]);
    }
    let n = cases.len() as u64;

    let input = gpu.storage_buffer_init("selftest cases", bytemuck::cast_slice(&cases), wgpu::BufferUsages::empty());
    let output = gpu.storage_buffer("selftest results", n * 16, wgpu::BufferUsages::empty());
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("selftest staging"),
        size: n * 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let module = gpu.shader("selftest", SHADER);
    let cs = ShaderStages::COMPUTE;
    let bgl = gpu.bind_group_layout("selftest", &[layout::storage(0, cs, true), layout::storage(1, cs, false)]);
    let pipeline = gpu.compute_pipeline("selftest wrap", &gpu.pipeline_layout("selftest", &[&bgl]), &module, "cs_wrap");
    let group = gpu.bind_group("selftest", &bgl, &[input.as_entire_binding(), output.as_entire_binding()]);

    let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("selftest") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("selftest"), timestamp_writes: None });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, n * 16);
    gpu.queue.submit([encoder.finish()]);

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device.poll(wgpu::PollType::Wait).map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
    rx.recv().context("map callback dropped")?.context("mapping failed")?;
    let results: Vec<[i32; 4]> = bytemuck::pod_collect_to_vec(&slice.get_mapped_range());
    staging.unmap();

    let (mut wrap_bad, mut legacy_bad) = (Vec::new(), Vec::new());
    for (c, r) in cases.iter().zip(&results) {
        let (p, s) = (c[0], c[1]);
        let expected = [p.rem_euclid(s), (-p).rem_euclid(s)];
        if [r[0], r[1]] != expected {
            wrap_bad.push(format!("wrap_i(({p}, {}), {s}) = ({}, {}), expected ({}, {})", -p, r[0], r[1], expected[0], expected[1]));
        }
        if [r[2], r[3]] != expected {
            legacy_bad.push(format!("(({p} % {s}) + {s}) % {s} = {}, expected {}", r[2], expected[0]));
        }
    }

    println!("wrap_i:       {}/{n} cases correct", n as usize - wrap_bad.len());
    for line in wrap_bad.iter().take(8) {
        println!("  {line}");
    }
    println!("signed-% wrap: {}/{n} cases correct (informational)", n as usize - legacy_bad.len());
    for line in legacy_bad.iter().take(4) {
        println!("  {line}");
    }
    if !wrap_bad.is_empty() {
        bail!("wrap_i is wrong on this GPU/driver in {} cases", wrap_bad.len());
    }
    println!("all checks passed");
    Ok(())
}
