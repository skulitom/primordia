//! Development aid for testing frame pacing on a busy GPU.
//!
//! `PRIMORDIA_DEBUG_GPU_STALL_MS=1200 primordia` adds roughly that much pure GPU
//! work to every interactive frame, as if another program were saturating the
//! GPU. The work is split into many short dispatches, so no single one comes
//! near the operating system's GPU-hang timeout (about 2 s on Windows).

use std::time::Instant;

use wgpu::ShaderStages;

use crate::gpu::{layout, Gpu};

const SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;

@compute @workgroup_size(64)
fn cs_spin(@builtin(global_invocation_id) gid: vec3<u32>) {
    var x = f32(gid.x) * 0.001;
    for (var i = 0u; i < 16384u; i++) {
        x = fract(sin(x * 12.9898 + 78.233) * 43758.5453);
    }
    sink[gid.x] = x;
}
"#;

/// Workgroups per dispatch (64 invocations each).
const GROUPS: u32 = 4096;

pub struct Stall {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    dispatches: u32,
}

impl Stall {
    /// Builds and calibrates the stall if `PRIMORDIA_DEBUG_GPU_STALL_MS` is set.
    pub fn from_env(gpu: &Gpu) -> Option<Self> {
        let ms: f32 = std::env::var("PRIMORDIA_DEBUG_GPU_STALL_MS").ok()?.trim().parse().ok()?;
        if ms <= 0.0 {
            return None;
        }
        let module = gpu.shader("debug stall", SHADER);
        let bgl = gpu.bind_group_layout("debug stall", &[layout::storage(0, ShaderStages::COMPUTE, false)]);
        let pipeline =
            gpu.compute_pipeline("debug stall", &gpu.pipeline_layout("debug stall", &[&bgl]), &module, "cs_spin");
        let sink = gpu.storage_buffer("debug stall sink", u64::from(GROUPS) * 64 * 4, wgpu::BufferUsages::empty());
        let group = gpu.bind_group("debug stall", &bgl, &[sink.as_entire_binding()]);
        let mut stall = Self { pipeline, group, dispatches: 1 };

        // Calibrate: warm up, then time a small batch.
        stall.run_blocking(gpu, 2);
        let start = Instant::now();
        stall.run_blocking(gpu, 8);
        let per_dispatch_ms = (start.elapsed().as_secs_f32() * 1000.0 / 8.0).max(0.01);
        stall.dispatches = ((ms / per_dispatch_ms).ceil() as u32).clamp(1, 100_000);
        log::warn!(
            "debug: adding ~{ms:.0} ms of GPU work per frame ({} dispatches of {per_dispatch_ms:.2} ms)",
            stall.dispatches
        );
        Some(stall)
    }

    fn run_blocking(&self, gpu: &Gpu, dispatches: u32) {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("debug stall") });
        self.record_n(&mut encoder, dispatches);
        gpu.queue.submit([encoder.finish()]);
        gpu.wait_idle();
    }

    fn record_n(&self, encoder: &mut wgpu::CommandEncoder, dispatches: u32) {
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("debug stall"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        for _ in 0..dispatches {
            pass.dispatch_workgroups(GROUPS, 1, 1);
        }
    }

    /// Records this frame's extra GPU work.
    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        self.record_n(encoder, self.dispatches);
    }
}
