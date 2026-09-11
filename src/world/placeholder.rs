//! The smallest possible world: one uniform buffer and one fullscreen pass.
//!
//! It is not registered in `WORLDS`. Copy this file to start a new world and
//! grow it from there (see docs/WRITING_A_WORLD.md).
#![allow(dead_code)]

use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use super::{Frame, ViewXform, World};
use crate::gpu::{self, layout, Gpu, SCENE_FORMAT};

const SHADER: &str = r#"
struct U { view: ViewXform, time: f32, hue: f32, _p0: f32, _p1: f32 };
@group(0) @binding(0) var<uniform> u: U;

@fragment
fn fs_main(in: FullscreenOut) -> @location(0) vec4<f32> {
    let w = fract(view_apply(u.view, in.uv));
    let d = length(w - 0.5);
    let rings = 0.5 + 0.5 * sin(d * 40.0 - u.time * 2.0);
    let col = cosine_palette(u.hue + d, vec3<f32>(0.5), vec3<f32>(0.5), vec3<f32>(1.0), vec3<f32>(0.0, 0.33, 0.67));
    return vec4<f32>(col * rings * 0.3, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    view: ViewXform,
    time: f32,
    hue: f32,
    _pad: [f32; 2],
}

pub struct Placeholder {
    id: &'static str,
    name: &'static str,
    size: [u32; 2],
    hue: f32,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl Placeholder {
    pub fn new(gpu: &Gpu, id: &'static str, name: &'static str, size: [u32; 2], hue: f32) -> Self {
        let module = gpu.shader("placeholder", SHADER);
        let bgl = gpu.bind_group_layout("placeholder", &[layout::uniform(0, ShaderStages::FRAGMENT)]);
        let pl = gpu.pipeline_layout("placeholder", &[&bgl]);
        let pipeline = gpu.fullscreen_pipeline("placeholder", &pl, &module, "fs_main", SCENE_FORMAT, None);
        let uniform = gpu.uniform_buffer("placeholder", &Uniforms::zeroed());
        let group = gpu.bind_group("placeholder", &bgl, &[uniform.as_entire_binding()]);
        Self { id, name, size, hue, uniform, group, pipeline }
    }
}

impl World for Placeholder {
    fn id(&self) -> &'static str {
        self.id
    }
    fn name(&self) -> &'static str {
        self.name
    }
    fn size(&self) -> [u32; 2] {
        self.size
    }
    fn presets(&self) -> &'static [&'static str] {
        &["Placeholder"]
    }
    fn preset(&self) -> usize {
        0
    }
    fn load_preset(&mut self, _gpu: &Gpu, _index: usize, _seed: u64) {}
    fn reset(&mut self, _gpu: &Gpu, _seed: u64) {}
    fn mutate(&mut self, _gpu: &Gpu, seed: u64) {
        self.hue = (seed % 1000) as f32 / 1000.0;
    }
    fn step(&mut self, _frame: &Frame, _encoder: &mut wgpu::CommandEncoder) {}
    fn render(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        frame.gpu.write(&self.uniform, &Uniforms { view: frame.view, time: frame.time, hue: self.hue, _pad: [0.0; 2] });
        gpu::fullscreen_pass(encoder, "placeholder", target, Some(wgpu::Color::BLACK), &self.pipeline, &[&self.group]);
    }
    fn ui(&mut self, _gpu: &Gpu, ui: &mut egui::Ui) {
        ui.label("This world is still being grown.");
    }
}
