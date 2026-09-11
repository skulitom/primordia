//! HDR post-processing shared by every world: bloom, exposure, tonemapping,
//! vignette, grain and dithering.

use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use crate::gpu::{self, layout, Gpu, SCENE_FORMAT};

/// Weight of each coarser bloom level relative to the next finer one.
const BLOOM_FALLOFF: f32 = 0.65;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tonemap {
    Agx,
    Aces,
    Reinhard,
    Linear,
}

impl Tonemap {
    pub const ALL: [Tonemap; 4] = [Tonemap::Agx, Tonemap::Aces, Tonemap::Reinhard, Tonemap::Linear];

    pub fn name(self) -> &'static str {
        match self {
            Tonemap::Agx => "AgX",
            Tonemap::Aces => "ACES",
            Tonemap::Reinhard => "Reinhard",
            Tonemap::Linear => "Linear (clamp)",
        }
    }

    fn code(self) -> u32 {
        match self {
            Tonemap::Agx => 0,
            Tonemap::Aces => 1,
            Tonemap::Reinhard => 2,
            Tonemap::Linear => 3,
        }
    }
}

/// User-facing "look" controls. Worlds suggest a starting point per preset via
/// [`crate::world::World::post_settings`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostSettings {
    pub exposure: f32,
    /// Additive bloom strength (0 disables the bloom passes).
    pub bloom: f32,
    /// Brightness where blooming starts (soft knee of half this value below it).
    pub bloom_threshold: f32,
    pub vignette: f32,
    pub saturation: f32,
    pub grain: f32,
    pub tonemap: Tonemap,
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            bloom: 0.6,
            bloom_threshold: 0.6,
            vignette: 0.35,
            saturation: 1.0,
            grain: 0.0,
            tonemap: Tonemap::Aces,
        }
    }
}

impl PostSettings {
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.add(egui::Slider::new(&mut self.exposure, 0.05..=8.0).logarithmic(true).text("Exposure"));
        ui.add(egui::Slider::new(&mut self.bloom, 0.0..=3.0).text("Bloom"));
        ui.add(egui::Slider::new(&mut self.bloom_threshold, 0.0..=4.0).text("Bloom threshold"));
        ui.add(egui::Slider::new(&mut self.vignette, 0.0..=1.0).text("Vignette"));
        ui.add(egui::Slider::new(&mut self.saturation, 0.0..=2.0).text("Saturation"));
        ui.add(egui::Slider::new(&mut self.grain, 0.0..=1.0).text("Film grain"));
        egui::ComboBox::from_label("Tonemap").selected_text(self.tonemap.name()).show_ui(ui, |ui| {
            for t in Tonemap::ALL {
                ui.selectable_value(&mut self.tonemap, t, t.name());
            }
        });
    }
}

/// Mirrors `PostParams` in post.wgsl (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostUniform {
    exposure: f32,
    bloom_strength: f32,
    bloom_threshold: f32,
    vignette: f32,
    tonemap: u32,
    bloom_norm: f32,
    time: f32,
    saturation: f32,
    encode_srgb: u32,
    grain: f32,
    bloom_falloff: f32,
    _pad: f32,
}

struct BloomLevel {
    view: wgpu::TextureView,
    /// Samples the previous (larger) level, or the scene for level 0.
    down_group: wgpu::BindGroup,
    /// Samples this level; drawn additively into the previous level.
    up_group: wgpu::BindGroup,
}

struct Targets {
    // The view keeps its texture alive, so the texture handle itself isn't stored.
    scene_view: wgpu::TextureView,
    levels: Vec<BloomLevel>,
    composite_group: wgpu::BindGroup,
}

pub struct Post {
    size: [u32; 2],
    output_format: wgpu::TextureFormat,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    sample_layout: wgpu::BindGroupLayout,
    composite_layout: wgpu::BindGroupLayout,
    down_first: wgpu::RenderPipeline,
    down: wgpu::RenderPipeline,
    up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    targets: Targets,
}

impl Post {
    pub fn new(gpu: &Gpu, size: [u32; 2], output_format: wgpu::TextureFormat) -> Self {
        let module = gpu.shader("post", include_str!("shaders/post.wgsl"));
        let fs = ShaderStages::FRAGMENT;
        let sample_layout = gpu.bind_group_layout(
            "post sample",
            &[layout::texture(0, fs, true), layout::sampler(1, fs, true), layout::uniform(2, fs)],
        );
        let composite_layout = gpu.bind_group_layout(
            "post composite",
            &[
                layout::texture(0, fs, true),
                layout::sampler(1, fs, true),
                layout::uniform(2, fs),
                layout::texture(3, fs, true),
            ],
        );
        let sample_pl = gpu.pipeline_layout("post sample", &[&sample_layout]);
        let composite_pl = gpu.pipeline_layout("post composite", &[&composite_layout]);

        let down_first =
            gpu.fullscreen_pipeline("bloom downsample (first)", &sample_pl, &module, "fs_downsample_first", SCENE_FORMAT, None);
        let down = gpu.fullscreen_pipeline("bloom downsample", &sample_pl, &module, "fs_downsample", SCENE_FORMAT, None);
        let up = gpu.fullscreen_pipeline(
            "bloom upsample",
            &sample_pl,
            &module,
            "fs_upsample",
            SCENE_FORMAT,
            Some(gpu::BLEND_ADDITIVE),
        );
        let composite = gpu.fullscreen_pipeline("post composite", &composite_pl, &module, "fs_composite", output_format, None);

        let sampler = gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge);
        let uniform = gpu.uniform_buffer("post uniform", &PostUniform::zeroed());
        let size = [size[0].max(1), size[1].max(1)];
        let targets = build_targets(gpu, size, &sample_layout, &composite_layout, &sampler, &uniform);

        Self {
            size,
            output_format,
            sampler,
            uniform,
            sample_layout,
            composite_layout,
            down_first,
            down,
            up,
            composite,
            targets,
        }
    }

    pub fn resize(&mut self, gpu: &Gpu, size: [u32; 2]) {
        let size = [size[0].max(1), size[1].max(1)];
        if size != self.size {
            self.size = size;
            self.targets =
                build_targets(gpu, size, &self.sample_layout, &self.composite_layout, &self.sampler, &self.uniform);
        }
    }

    /// The HDR (`SCENE_FORMAT`) texture worlds render into.
    pub fn scene_view(&self) -> &wgpu::TextureView {
        &self.targets.scene_view
    }

    /// Records bloom + composite: turns the scene into `output`, a view with
    /// `output_format` (any size; it is sampled by uv).
    pub fn run(
        &self,
        gpu: &Gpu,
        encoder: &mut wgpu::CommandEncoder,
        settings: &PostSettings,
        time: f32,
        output: &wgpu::TextureView,
    ) {
        self.bloom(gpu, encoder, settings, time);
        self.composite(encoder, output);
    }

    /// Uploads `settings` and records the bloom chain for the current scene.
    pub fn bloom(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, settings: &PostSettings, time: f32) {
        let levels = &self.targets.levels;
        let weight_sum: f32 = (0..levels.len()).map(|j| BLOOM_FALLOFF.powi(j as i32)).sum();
        let uniform = PostUniform {
            exposure: settings.exposure,
            bloom_strength: settings.bloom,
            bloom_threshold: settings.bloom_threshold,
            vignette: settings.vignette,
            tonemap: settings.tonemap.code(),
            bloom_norm: 1.0 / weight_sum.max(1e-3),
            time,
            saturation: settings.saturation,
            encode_srgb: u32::from(!self.output_format.is_srgb()),
            grain: settings.grain,
            bloom_falloff: BLOOM_FALLOFF,
            _pad: 0.0,
        };
        gpu.write(&self.uniform, &uniform);

        if settings.bloom > 0.0 {
            for (i, level) in levels.iter().enumerate() {
                let pipeline = if i == 0 { &self.down_first } else { &self.down };
                gpu::fullscreen_pass(encoder, "bloom down", &level.view, Some(wgpu::Color::BLACK), pipeline, &[&level.down_group]);
            }
            for i in (1..levels.len()).rev() {
                gpu::fullscreen_pass(encoder, "bloom up", &levels[i - 1].view, None, &self.up, &[&levels[i].up_group]);
            }
        }
    }

    /// Records only the final composite into `output`, reusing the bloom and the
    /// settings of the last [`Post::bloom`] call (e.g. for screenshots and video).
    pub fn composite(&self, encoder: &mut wgpu::CommandEncoder, output: &wgpu::TextureView) {
        gpu::fullscreen_pass(
            encoder,
            "post composite",
            output,
            Some(wgpu::Color::BLACK),
            &self.composite,
            &[&self.targets.composite_group],
        );
    }
}

fn build_targets(
    gpu: &Gpu,
    size: [u32; 2],
    sample_layout: &wgpu::BindGroupLayout,
    composite_layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    uniform: &wgpu::Buffer,
) -> Targets {
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let (_scene, scene_view) = gpu.texture_2d("scene", size, SCENE_FORMAT, usage | wgpu::TextureUsages::COPY_SRC);

    // Level 0 is half resolution; the coarsest level spans ~8 px of the short
    // side, so the halo covers the same fraction of the frame at any resolution.
    let min_side = size[0].min(size[1]).max(1) as f32;
    let count = ((min_side / 8.0).log2().round() as i32).clamp(1, 10) as usize;

    let mut views: Vec<wgpu::TextureView> = Vec::with_capacity(count);
    for i in 0..count {
        let w = (size[0] >> (i + 1)).max(1);
        let h = (size[1] >> (i + 1)).max(1);
        let (_texture, view) = gpu.texture_2d("bloom level", [w, h], SCENE_FORMAT, usage);
        views.push(view);
    }

    let mut levels = Vec::with_capacity(count);
    for i in 0..count {
        let source = if i == 0 { &scene_view } else { &views[i - 1] };
        let down_group = gpu.bind_group(
            "bloom down",
            sample_layout,
            &[
                wgpu::BindingResource::TextureView(source),
                wgpu::BindingResource::Sampler(sampler),
                uniform.as_entire_binding(),
            ],
        );
        let up_group = gpu.bind_group(
            "bloom up",
            sample_layout,
            &[
                wgpu::BindingResource::TextureView(&views[i]),
                wgpu::BindingResource::Sampler(sampler),
                uniform.as_entire_binding(),
            ],
        );
        levels.push(BloomLevel { view: views[i].clone(), down_group, up_group });
    }

    let composite_group = gpu.bind_group(
        "post composite",
        composite_layout,
        &[
            wgpu::BindingResource::TextureView(&scene_view),
            wgpu::BindingResource::Sampler(sampler),
            uniform.as_entire_binding(),
            wgpu::BindingResource::TextureView(&levels[0].view),
        ],
    );

    Targets { scene_view, levels, composite_group }
}
