//! Experimental two-way coupling of Gray-Scott chemistry and Physarum-style
//! trail-following agents. Both inhabit the same periodic grid. Setting the
//! coupling to zero removes both cross-system terms, leaving each running.

use std::f32::consts::TAU;

use anyhow::{Result, ensure};
use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use super::{Frame, ViewXform, World};
use crate::gpu::{Gpu, SCENE_FORMAT, layout};
use crate::library::WorldSettings;
use crate::palette::{self, PaletteLut};
use crate::post::PostSettings;
use crate::rng::Rng;

#[cfg(test)]
#[path = "symbiosis_tests.rs"]
mod tests;

const PRESETS: &[&str] = &["Living Reef", "Wandering Veins", "Coral Maze"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Layer {
    Together,
    Chemistry,
    Trails,
}

impl Layer {
    fn name(self) -> &'static str {
        match self {
            Self::Together => "Together",
            Self::Chemistry => "Chemistry",
            Self::Trails => "Agent trails",
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Params {
    pub coupling: f32,
    pub feed: f32,
    pub kill: f32,
    pub steps: u32,
    pub sensor_distance: f32,
    pub sensor_angle: f32,
    pub turn_angle: f32,
    pub speed: f32,
    /// Trail retained over one displayed frame, independent of chemical steps.
    pub retention: f32,
    pub wander: f32,
    pub brightness: f32,
    pub layer: Layer,
    /// Older saved recipes open in the original single-world view.
    #[serde(default)]
    pub compare: bool,
}

impl Params {
    fn preset(index: usize) -> (Self, usize) {
        let mut p = Self {
            coupling: 0.7,
            feed: 0.0367,
            kill: 0.0649,
            steps: 8,
            sensor_distance: 12.0,
            sensor_angle: 0.65,
            turn_angle: 0.4,
            speed: 1.4,
            retention: 0.94,
            wander: 0.08,
            brightness: 1.25,
            layer: Layer::Together,
            compare: false,
        };
        let palette = match index {
            1 => {
                p.feed = 0.026;
                p.kill = 0.059;
                p.steps = 10;
                p.sensor_distance = 20.0;
                p.speed = 1.8;
                p.retention = 0.97;
                p.coupling = 0.55;
                2 // Aurora
            }
            2 => {
                p.feed = 0.034;
                p.kill = 0.062;
                p.sensor_distance = 9.0;
                p.speed = 1.0;
                p.retention = 0.92;
                p.coupling = 0.85;
                8 // Coral
            }
            _ => 0, // Bioluminescence
        };
        (p, palette)
    }

    fn validate(&self) -> Result<()> {
        // Recipes may be edited outside the app. Reject unsafe values before
        // uploading uniforms; NaN also fails RangeInclusive::contains.
        for (value, lo, hi) in [
            (self.coupling, 0.0, 1.0),
            (self.feed, 0.01, 0.08),
            (self.kill, 0.03, 0.08),
            (self.sensor_distance, 2.0, 40.0),
            (self.sensor_angle, 0.1, 1.6),
            (self.turn_angle, 0.05, 1.2),
            (self.speed, 0.2, 3.0),
            (self.retention, 0.8, 0.99),
            (self.wander, 0.0, 0.5),
            (self.brightness, 0.2, 4.0),
        ] {
            ensure!((lo..=hi).contains(&value), "Invalid Symbiosis setting");
        }
        ensure!((1..=20).contains(&self.steps), "Invalid chemistry speed");
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Agent {
    pos: [f32; 2],
    heading: f32,
    state: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimUniform {
    size: [u32; 2],
    count: u32,
    steps: u32,
    chemistry: [f32; 4],
    motion: [f32; 4],
    trail: [f32; 4],
    pointer: [f32; 2],
    radius: f32,
    pointer_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    view: ViewXform,
    size: [u32; 2],
    layer: u32,
    brightness: f32,
}

pub struct Symbiosis {
    size: [u32; 2],
    params: Params,
    preset: usize,
    post: PostSettings,
    lut: PaletteLut,
    fields: [wgpu::Buffer; 2],
    current: usize,
    agents: wgpu::Buffer,
    count: u32,
    deposits: wgpu::Buffer,
    uniform: wgpu::Buffer,
    groups: [wgpu::BindGroup; 2],
    agent_pipeline: wgpu::ComputePipeline,
    field_pipeline: wgpu::ComputePipeline,
    draw_uniform: wgpu::Buffer,
    draw_groups: [wgpu::BindGroup; 2],
    draw_pipeline: wgpu::RenderPipeline,
    seed: u64,
    /// Created only while comparing. The reference never owns another reference.
    reference: Option<Box<Symbiosis>>,
}

pub fn create(gpu: &Gpu, output: [u32; 2], seed: u64) -> Box<dyn World> {
    // A bounded simulation grid keeps the experiment responsive at 4K too.
    let scale = 0.5_f32.min(1280.0 / output[0].max(output[1]).max(1) as f32);
    let size = output.map(|n| ((n as f32 * scale).round() as u32).max(64));
    Box::new(Symbiosis::new(gpu, size, seed))
}

impl Symbiosis {
    fn new(gpu: &Gpu, size: [u32; 2], seed: u64) -> Self {
        let (params, palette) = Params::preset(0);
        let count = (size[0] * size[1] * 3 / 5).clamp(2048, 500_000);
        let fields = std::array::from_fn(|_| {
            gpu.storage_buffer("symbiosis habitat", size[0] as u64 * size[1] as u64 * 16, wgpu::BufferUsages::empty())
        });
        let agents = gpu.storage_buffer("symbiosis agents", count as u64 * 16, wgpu::BufferUsages::empty());
        let deposits =
            gpu.storage_buffer("symbiosis deposits", size[0] as u64 * size[1] as u64 * 4, wgpu::BufferUsages::empty());
        let uniform = gpu.uniform_buffer("symbiosis simulation", &SimUniform::zeroed());
        let bgl = gpu.bind_group_layout(
            "symbiosis simulation",
            &[
                layout::uniform(0, ShaderStages::COMPUTE),
                layout::storage(1, ShaderStages::COMPUTE, true),
                layout::storage(2, ShaderStages::COMPUTE, false),
                layout::storage(3, ShaderStages::COMPUTE, false),
                layout::storage(4, ShaderStages::COMPUTE, false),
            ],
        );
        let groups = std::array::from_fn(|i| {
            gpu.bind_group(
                "symbiosis simulation",
                &bgl,
                &[
                    uniform.as_entire_binding(),
                    fields[i].as_entire_binding(),
                    fields[1 - i].as_entire_binding(),
                    agents.as_entire_binding(),
                    deposits.as_entire_binding(),
                ],
            )
        });
        let module = gpu.shader("symbiosis simulation", include_str!("../shaders/symbiosis.wgsl"));
        let pl = gpu.pipeline_layout("symbiosis simulation", &[&bgl]);
        let agent_pipeline = gpu.compute_pipeline("symbiosis agents", &pl, &module, "cs_agents");
        let field_pipeline = gpu.compute_pipeline("symbiosis habitat", &pl, &module, "cs_field");
        let lut = PaletteLut::new(gpu, palette);
        let draw_uniform = gpu.uniform_buffer("symbiosis draw", &DrawUniform::zeroed());
        let bgl = gpu.bind_group_layout(
            "symbiosis draw",
            &[
                layout::uniform(0, ShaderStages::FRAGMENT),
                layout::storage(1, ShaderStages::FRAGMENT, true),
                layout::texture(2, ShaderStages::FRAGMENT, true),
                layout::sampler(3, ShaderStages::FRAGMENT, true),
            ],
        );
        let draw_groups = std::array::from_fn(|i| {
            gpu.bind_group(
                "symbiosis draw",
                &bgl,
                &[
                    draw_uniform.as_entire_binding(),
                    fields[i].as_entire_binding(),
                    wgpu::BindingResource::TextureView(&lut.view),
                    wgpu::BindingResource::Sampler(&lut.sampler),
                ],
            )
        });
        let module = gpu.shader("symbiosis draw", include_str!("../shaders/symbiosis_display.wgsl"));
        let pl = gpu.pipeline_layout("symbiosis draw", &[&bgl]);
        let draw_pipeline = gpu.fullscreen_pipeline("symbiosis draw", &pl, &module, "fs_display", SCENE_FORMAT, None);
        let mut world = Self {
            size,
            params,
            preset: 0,
            post: Self::look(),
            lut,
            fields,
            current: 0,
            agents,
            count,
            deposits,
            uniform,
            groups,
            agent_pipeline,
            field_pipeline,
            draw_uniform,
            draw_groups,
            draw_pipeline,
            seed,
            reference: None,
        };
        world.reset(gpu, seed);
        world
    }

    fn look() -> PostSettings {
        PostSettings { bloom: 0.15, bloom_threshold: 1.0, vignette: 0.18, ..PostSettings::default() }
    }

    fn set_comparison(&mut self, gpu: &Gpu, enabled: bool) {
        if enabled == self.params.compare {
            return;
        }
        self.params.compare = enabled;
        if enabled {
            self.reset(gpu, self.seed);
        } else {
            // Keep the coupled experiment running at its current state.
            self.reference = None;
        }
    }

    fn reference_params(&self) -> Params {
        Params { coupling: 0.0, compare: false, ..self.params }
    }

    fn draw_region(
        &self,
        frame: &Frame,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        region: [f32; 2],
        clear: bool,
    ) {
        // Each pane shows the same camera centre at the same pixel scale as
        // the single view. Narrowing the viewport crops, never stretches it.
        let mut view = frame.view;
        view.offset[0] += view.scale[0] * (1.0 - region[1]) * 0.5;
        view.scale[0] *= region[1];
        frame.gpu.write(
            &self.draw_uniform,
            &DrawUniform { view, size: self.size, layer: self.params.layer as u32, brightness: self.params.brightness },
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("symbiosis view"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if clear { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        let [w, h] = frame.target_size.map(|n| n as f32);
        pass.set_viewport(w * region[0], 0.0, w * region[1], h, 0.0, 1.0);
        pass.set_pipeline(&self.draw_pipeline);
        pass.set_bind_group(0, &self.draw_groups[self.current], &[]);
        pass.draw(0..3, 0..1);
    }
}

impl World for Symbiosis {
    fn id(&self) -> &'static str {
        "symbiosis"
    }
    fn name(&self) -> &'static str {
        "Symbiosis"
    }
    fn size(&self) -> [u32; 2] {
        self.size
    }
    fn presets(&self) -> &'static [&'static str] {
        PRESETS
    }
    fn preset(&self) -> usize {
        self.preset
    }

    fn load_preset(&mut self, gpu: &Gpu, index: usize, seed: u64) {
        self.preset = index.min(PRESETS.len() - 1);
        let (params, palette) = Params::preset(self.preset);
        self.params = Params { compare: self.params.compare, ..params };
        self.lut.set(gpu, palette);
        self.post = Self::look();
        self.reset(gpu, seed);
    }

    fn reset(&mut self, gpu: &Gpu, seed: u64) {
        self.seed = seed;
        let mut rng = Rng::new(seed);
        let [w, h] = self.size;
        let mut field = vec![[1.0_f32, 0.0, 0.0, 0.0]; (w * h) as usize];
        // Sparse, irregular islands. Write wrapped disks rather than measuring
        // every cell against every island; reset stays cheap on large outputs.
        for _ in 0..(w * h / 2400).max(5) {
            let cx = rng.below(w) as i32;
            let cy = rng.below(h) as i32;
            let radius = rng.range(4.0, 10.0);
            let r = radius.ceil() as i32;
            for dy in -r..=r {
                for dx in -r..=r {
                    if (dx * dx + dy * dy) as f32 > radius * radius {
                        continue;
                    }
                    let x = (cx + dx).rem_euclid(w as i32) as u32;
                    let y = (cy + dy).rem_euclid(h as i32) as u32;
                    field[(y * w + x) as usize] = [0.5, rng.range(0.22, 0.3), 0.0, 0.0];
                }
            }
        }
        for buffer in &self.fields {
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&field));
        }
        // A separate stream keeps agent positions independent of the chemistry
        // seed layout, useful when comparing changes to either half.
        let mut rng = Rng::new(seed ^ 0x736c_696d_655f_7264);
        let agents: Vec<Agent> = (0..self.count)
            .map(|_| Agent {
                pos: [rng.range(0.0, w as f32), rng.range(0.0, h as f32)],
                heading: rng.range(0.0, TAU),
                state: rng.next_u32(),
            })
            .collect();
        gpu.queue.write_buffer(&self.agents, 0, bytemuck::cast_slice(&agents));
        self.current = 0;
        if self.params.compare {
            let params = self.reference_params();
            let reference = self.reference.get_or_insert_with(|| Box::new(Self::new(gpu, self.size, seed)));
            reference.params = params;
            reference.reset(gpu, seed);
        } else {
            self.reference = None;
        }
    }

    fn mutate(&mut self, gpu: &Gpu, seed: u64) {
        let mut rng = Rng::new(seed);
        let (mut p, _) = Params::preset(rng.below(PRESETS.len() as u32) as usize);
        p.coupling = rng.range(0.25, 1.0);
        p.sensor_distance *= rng.range(0.7, 1.4);
        p.speed *= rng.range(0.8, 1.3);
        p.sensor_angle = rng.range(0.4, 1.0);
        p.retention = rng.range(0.9, 0.98);
        p.kill += rng.range(-0.0006, 0.0006);
        p.compare = self.params.compare;
        self.params = p;
        self.reset(gpu, seed);
    }

    fn settings(&self) -> Result<WorldSettings> {
        Ok(WorldSettings::Symbiosis {
            params: self.params,
            palette: palette::PALETTES[self.lut.index()].name.to_owned(),
            post: self.post,
        })
    }

    fn restore_settings(&mut self, gpu: &Gpu, settings: &WorldSettings, seed: u64) -> Result<()> {
        let WorldSettings::Symbiosis { params, palette, post } = settings else {
            anyhow::bail!("These settings belong to another world");
        };
        params.validate()?;
        let index = palette::find(palette).ok_or_else(|| anyhow::anyhow!("Unknown palette: {palette}"))?;
        self.params = *params;
        self.post = *post;
        self.lut.set(gpu, index);
        self.reset(gpu, seed);
        Ok(())
    }

    fn step(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder) {
        let reference_params = self.reference_params();
        if let Some(reference) = &mut self.reference {
            reference.params = reference_params;
            reference.step(frame, encoder);
        }
        let p = self.params;
        let pointer =
            frame.pointer.unwrap_or(super::Pointer { pos: [0.0; 2], primary: false, secondary: false, radius: 1.0 });
        frame.gpu.write(
            &self.uniform,
            &SimUniform {
                size: self.size,
                count: self.count,
                steps: p.steps,
                chemistry: [p.feed, p.kill, p.coupling, 0.8],
                motion: [p.sensor_distance, p.sensor_angle, p.turn_angle, p.speed],
                trail: [p.retention.powf(1.0 / p.steps as f32), 0.08 / p.steps as f32, p.wander, 0.28 / p.steps as f32],
                pointer: [
                    pointer.pos[0].rem_euclid(1.0) * self.size[0] as f32,
                    pointer.pos[1].rem_euclid(1.0) * self.size[1] as f32,
                ],
                radius: pointer.radius.max(1.0),
                pointer_mode: if pointer.secondary {
                    2
                } else if pointer.primary {
                    1
                } else {
                    0
                },
            },
        );
        encoder.clear_buffer(&self.deposits, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("symbiosis agents"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.agent_pipeline);
            pass.set_bind_group(0, &self.groups[self.current], &[]);
            pass.dispatch_workgroups(self.count.div_ceil(256), 1, 1);
        }
        for _ in 0..p.steps {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("symbiosis habitat"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.field_pipeline);
            pass.set_bind_group(0, &self.groups[self.current], &[]);
            pass.dispatch_workgroups(self.size[0].div_ceil(16), self.size[1].div_ceil(16), 1);
            self.current = 1 - self.current;
        }
    }

    fn render(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        self.draw_region(frame, encoder, target, [0.0, if self.reference.is_some() { 0.5 } else { 1.0 }], true);
        let params = self.reference_params();
        if let Some(reference) = &mut self.reference {
            // Look changes apply even when paused; each pane owns its uniform,
            // so queue writes cannot overwrite the other pane's draw settings.
            reference.params = params;
            reference.lut.set(frame.gpu, self.lut.index());
            reference.draw_region(frame, encoder, target, [0.5, 0.5], false);
        }
    }

    fn map_position(&self, view: ViewXform, mut screen_uv: [f32; 2]) -> [f32; 2] {
        if self.reference.is_some() {
            screen_uv[0] += if screen_uv[0] < 0.5 { 0.25 } else { -0.25 };
        }
        view.apply(screen_uv)
    }

    fn comparison_labels(&self) -> Option<[String; 2]> {
        self.reference.as_ref().map(|_| [format!("COUPLING {:.2}", self.params.coupling), "COUPLING OFF".into()])
    }

    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) {
        use crate::ui::Slider;
        ui.add(Slider::new(&mut self.params.coupling, 0.0..=1.0).text("Coupling strength"))
            .on_hover_text("Chemistry guides agents; their trails encourage chemical growth. Zero lets both systems run independently.");
        ui.label(
            egui::RichText::new(if self.params.coupling == 0.0 {
                "Uncoupled · both systems keep running"
            } else {
                "Chemistry guides agents · trails feed growth"
            })
            .small()
            .color(crate::ui::ACCENT),
        );
        crate::ui::dropdown(ui, "View", self.params.layer.name(), |ui| {
            for layer in [Layer::Together, Layer::Chemistry, Layer::Trails] {
                ui.selectable_value(&mut self.params.layer, layer, layer.name());
            }
        });
        let mut compare = self.params.compare;
        if ui
            .checkbox(&mut compare, "Compare with coupling off")
            .on_hover_text(
                "Starts both sides from this seed. Left uses your coupling setting; right keeps coupling at zero.",
            )
            .changed()
        {
            self.set_comparison(gpu, compare);
        }
        if compare {
            ui.label(
                egui::RichText::new(
                    "Left: your coupling · Right: coupling off\nBrush strokes and all other settings affect both.",
                )
                .small()
                .color(crate::ui::MUTED),
            );
            if ui
                .button("Restart comparison")
                .on_hover_text("Restart both sides with the current settings and the same seed.")
                .clicked()
            {
                self.reset(gpu, self.seed);
            }
            ui.label(egui::RichText::new("H / Tab hides the panel for a wider view.").small().weak());
        } else {
            ui.label(egui::RichText::new("Starts both habitats again from this seed.").small().weak());
        }
        ui.add_space(4.0);
        ui.add(Slider::new(&mut self.params.sensor_distance, 2.0..=40.0).text("Sensing distance"));
        ui.add(Slider::new(&mut self.params.speed, 0.2..=3.0).text("Agent speed"));
        ui.add(Slider::new(&mut self.params.retention, 0.8..=0.99).text("Trail memory"));
        ui.add(Slider::new(&mut self.params.steps, 1..=20).text("Chemistry steps / frame"));
        ui.collapsing("Chemistry & steering", |ui| {
            ui.add(Slider::new(&mut self.params.feed, 0.01..=0.08).text("Feed"));
            ui.add(Slider::new(&mut self.params.kill, 0.03..=0.08).text("Kill"));
            ui.add(Slider::new(&mut self.params.sensor_angle, 0.1..=1.6).text("Sensor angle"));
            ui.add(Slider::new(&mut self.params.turn_angle, 0.05..=1.2).text("Turn angle"));
            ui.add(Slider::new(&mut self.params.wander, 0.0..=0.5).text("Wander"));
        });
        ui.collapsing("Look", |ui| {
            self.lut.ui(gpu, ui);
            ui.add(Slider::new(&mut self.params.brightness, 0.2..=4.0).text("Brightness"));
        });
    }

    fn post_settings(&self) -> PostSettings {
        self.post
    }
    fn stats(&self) -> String {
        if self.reference.is_some() {
            format!("Two {}×{} habitats · {} agents each", self.size[0], self.size[1], self.count)
        } else {
            format!("{}×{} cells · {} agents · two-way feedback", self.size[0], self.size[1], self.count)
        }
    }
    fn controls_hint(&self) -> &'static str {
        if self.reference.is_some() {
            "Brush affects both sides · Left: seed · Right: clear"
        } else {
            "Left: seed chemistry · Right: clear habitat"
        }
    }
}
