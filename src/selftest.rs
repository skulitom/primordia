//! On-device checks of the WGSL prelude maths every world relies on
//! (`primordia selftest`). GPUs are allowed to round float division loosely and
//! drivers differ, so torus wrapping is verified against the CPU.

use anyhow::{anyhow, bail, Context as _, Result};
use wgpu::ShaderStages;

use crate::gpu::{describe_adapter, layout, Gpu, ADAPTER_NAME_ENV};
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

    let (mut wrap_bad, mut legacy_wrong) = (Vec::new(), 0);
    for (c, r) in cases.iter().zip(&results) {
        let (p, s) = (c[0], c[1]);
        let expected = [p.rem_euclid(s), (-p).rem_euclid(s)];
        if [r[0], r[1]] != expected {
            wrap_bad.push(format!("wrap_i(({p}, {}), {s}) = ({}, {}), expected ({}, {})", -p, r[0], r[1], expected[0], expected[1]));
        }
        if [r[2], r[3]] != expected {
            legacy_wrong += 1;
        }
    }

    // The verdict comes first; everything after it is detail.
    let adapter = describe_adapter(&gpu.adapter.get_info());
    if wrap_bad.is_empty() {
        println!("PASS: torus wrapping (wrap_i) is correct in all {n} cases on {adapter}");
    } else {
        println!("FAIL: torus wrapping (wrap_i) is wrong in {} of {n} cases on {adapter}", wrap_bad.len());
        for line in wrap_bad.iter().take(8) {
            println!("  {line}");
        }
    }
    println!();
    println!("{}", reference_line(legacy_wrong, n));
    println!();
    print_adapters(&gpu);
    if !wrap_bad.is_empty() {
        bail!("wrap_i is wrong on this GPU/driver in {} cases", wrap_bad.len());
    }
    Ok(())
}

/// How the naive `((p % s) + s) % s` wrap fared: informational, since the
/// shaders use `wrap_i` precisely because this is often wrong.
fn reference_line(wrong: usize, cases: u64) -> String {
    if wrong == 0 {
        format!(
            "For reference, a naive signed-% wrap is also right in all {cases} cases here.\n\
             Other GPUs and backends get it wrong, so the shaders use wrap_i everywhere."
        )
    } else {
        format!(
            "For reference, a naive signed-% wrap is wrong in {wrong} of {cases} cases here.\n\
             That is expected (% of negative numbers is unreliable on many GPUs and backends) and is why the \
             shaders use wrap_i: nothing to fix."
        )
    }
}

/// Every adapter wgpu can use, marking the one this check ran on.
fn print_adapters(gpu: &Gpu) {
    let in_use = gpu.adapter.get_info();
    let mut lines: Vec<String> = Vec::new();
    for info in gpu.instance.enumerate_adapters(wgpu::Backends::all()).iter().map(wgpu::Adapter::get_info) {
        let line = format!("  {} {}", if info == in_use { "*" } else { " " }, describe_adapter(&info));
        if !lines.contains(&line) {
            lines.push(line);
        }
    }
    println!("Adapters (* = in use; choose another with {ADAPTER_NAME_ENV}=<part of its name>):");
    for line in &lines {
        println!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_signed_modulo_reference_reads_as_expected_behaviour() {
        let wrong = reference_line(126_303, 134_688);
        assert!(wrong.contains("wrong in 126303 of 134688 cases here.\nThat is expected"), "{wrong}");
        assert!(wrong.contains("nothing to fix"), "{wrong}");
        let right = reference_line(0, 134_688);
        assert!(right.contains("also right in all 134688 cases"), "{right}");
    }
}
