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

const PRESETS: &[&str] = &["Living Reef", "Wandering Veins", "Coral Maze", "Spore Tide", "Root Atlas"];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Relationship {
    #[default]
    Cultivate,
    Graze,
    Weave,
}

impl Relationship {
    const ALL: [Self; 3] = [Self::Cultivate, Self::Graze, Self::Weave];

    fn name(self) -> &'static str {
        match self {
            Self::Cultivate => "Cultivate",
            Self::Graze => "Graze",
            Self::Weave => "Weave",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Self::Cultivate => "Agents tend growth margins; their trails nourish colonies.",
            Self::Graze => "Agents seek growth and consume it, leaving space to recover.",
            Self::Weave => "Agents follow growth; busy trails germinate new living threads.",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Seeding {
    #[default]
    Islands,
    Archipelago,
    Threads,
    Fronts,
}

impl Seeding {
    const ALL: [Self; 4] = [Self::Islands, Self::Archipelago, Self::Threads, Self::Fronts];

    fn name(self) -> &'static str {
        match self {
            Self::Islands => "Scattered islands",
            Self::Archipelago => "Clustered colonies",
            Self::Threads => "Living threads",
            Self::Fronts => "Broken wave fronts",
        }
    }
}

fn one() -> f32 {
    1.0
}

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
    #[serde(default)]
    pub relationship: Relationship,
    #[serde(default)]
    pub seeding: Seeding,
    /// Fixed, periodic fertility differences generated from the recipe's seed.
    #[serde(default)]
    pub terrain: f32,
    #[serde(default = "one")]
    pub scale: f32,
    #[serde(default = "one")]
    pub trail_light: f32,
}

impl Params {
    fn preset(index: usize) -> (Self, usize) {
        let mut p = Self {
            coupling: 0.7,
            feed: 0.0545,
            kill: 0.063,
            steps: 12,
            sensor_distance: 12.0,
            sensor_angle: 0.65,
            turn_angle: 0.4,
            speed: 1.4,
            retention: 0.92,
            wander: 0.08,
            brightness: 1.25,
            layer: Layer::Together,
            compare: false,
            relationship: Relationship::Cultivate,
            seeding: Seeding::Archipelago,
            terrain: 0.8,
            scale: 1.35,
            trail_light: 0.55,
        };
        let palette = match index {
            1 => {
                p.relationship = Relationship::Graze;
                p.feed = 0.014;
                p.kill = 0.05;
                p.steps = 16;
                p.sensor_distance = 8.0;
                p.speed = 1.7;
                p.retention = 0.95;
                p.coupling = 0.65;
                p.scale = 1.0;
                p.terrain = 0.3;
                p.seeding = Seeding::Islands;
                p.trail_light = 0.35;
                2 // Aurora
            }
            2 => {
                p.relationship = Relationship::Weave;
                p.feed = 0.042;
                p.kill = 0.062;
                p.sensor_distance = 14.0;
                p.speed = 1.25;
                p.retention = 0.95;
                p.coupling = 0.85;
                p.scale = 0.85;
                p.terrain = 0.5;
                p.seeding = Seeding::Threads;
                p.trail_light = 0.3;
                8 // Coral
            }
            3 => {
                p.relationship = Relationship::Graze;
                p.feed = 0.01;
                p.kill = 0.045;
                p.steps = 20;
                p.sensor_distance = 5.0;
                p.sensor_angle = 1.2;
                p.turn_angle = 0.8;
                p.speed = 2.3;
                p.retention = 0.88;
                p.wander = 0.18;
                p.coupling = 0.5;
                p.scale = 0.8;
                p.terrain = 0.12;
                p.seeding = Seeding::Fronts;
                p.trail_light = 0.25;
                1 // Ember
            }
            4 => {
                p.relationship = Relationship::Weave;
                p.feed = 0.03;
                p.kill = 0.064;
                p.steps = 10;
                p.sensor_distance = 28.0;
                p.sensor_angle = 0.4;
                p.turn_angle = 0.2;
                p.speed = 0.8;
                p.retention = 0.98;
                p.wander = 0.025;
                p.coupling = 0.95;
                p.scale = 1.5;
                p.terrain = 1.0;
                p.seeding = Seeding::Archipelago;
                p.trail_light = 0.7;
                5 // Moss
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
            (self.terrain, 0.0, 1.0),
            (self.scale, 0.6, 2.0),
            (self.trail_light, 0.0, 2.0),
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
    ecology: [f32; 4],
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
    ecology: [f32; 4],
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
            &DrawUniform {
                view,
                size: self.size,
                layer: self.params.layer as u32,
                brightness: self.params.brightness,
                ecology: [self.params.relationship as u32 as f32, self.params.trail_light, 0.0, 0.0],
            },
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
        // A separate stream makes geography reproducible without perturbing
        // the legacy island/agent streams. Integer harmonics tile exactly.
        let mut land = Rng::new(seed ^ 0x6861_6269_7461_7421);
        let phases: [f32; 4] = std::array::from_fn(|_| land.range(0.0, TAU));
        let frequency = [land.below(3) + 1, land.below(3) + 1];
        for y in 0..h {
            for x in 0..w {
                let u = TAU * x as f32 / w as f32;
                let v = TAU * y as f32 / h as f32;
                field[(y * w + x) as usize][3] = 0.5
                    * (u * frequency[0] as f32 + phases[0] + (v + phases[1]).sin()).sin()
                    + 0.3 * (v * frequency[1] as f32 + phases[2]).cos()
                    + 0.2 * (u * 2.0 - v * 3.0 + phases[3]).sin();
            }
        }
        let colonies: Vec<[f32; 2]> =
            (0..land.below(5) + 3).map(|_| [land.range(0.0, w as f32), land.range(0.0, h as f32)]).collect();
        // Sparse, irregular islands. Write wrapped disks rather than measuring
        // every cell against every island; reset stays cheap on large outputs.
        let islands =
            if self.params.seeding == Seeding::Fronts { (w * h / 12000).max(3) } else { (w * h / 2400).max(5) };
        for _ in 0..islands {
            let (cx, cy) = if self.params.seeding == Seeding::Archipelago {
                let c = rng.pick(&colonies);
                let spread = w.min(h) as f32 * 0.09;
                ((c[0] + rng.normal() * spread) as i32, (c[1] + rng.normal() * spread) as i32)
            } else {
                (rng.below(w) as i32, rng.below(h) as i32)
            };
            let radius = rng.range(4.0, 10.0)
                * self.params.scale
                * if self.params.seeding == Seeding::Fronts { 2.5 } else { 1.0 };
            let angle = if self.params.seeding == Seeding::Fronts { rng.range(0.0, TAU) } else { 0.0 };
            let r = radius.ceil() as i32;
            for dy in -r..=r {
                for dx in -r..=r {
                    if (dx * dx + dy * dy) as f32 > radius * radius {
                        continue;
                    }
                    let x = (cx + dx).rem_euclid(w as i32) as u32;
                    let y = (cy + dy).rem_euclid(h as i32) as u32;
                    let cell = &mut field[(y * w + x) as usize];
                    if self.params.seeding == Seeding::Fronts {
                        let along = dx as f32 * angle.cos() + dy as f32 * angle.sin();
                        // A broken arc with a depleted interior launches a
                        // front instead of collapsing into a circular spot.
                        if along < -radius * 0.3 {
                            continue;
                        }
                        cell[0] = 0.15;
                        cell[1] = if (dx * dx + dy * dy) as f32 > (radius - 3.0).powi(2) { 0.32 } else { 0.0 };
                    } else {
                        cell[0] = 0.5;
                        cell[1] = rng.range(0.22, 0.3);
                    }
                }
            }
        }
        if self.params.seeding == Seeding::Threads {
            for y in 0..h {
                for x in 0..w {
                    let u = TAU * x as f32 / w as f32;
                    let v = TAU * y as f32 / h as f32;
                    let thread = (u * 3.0 + phases[0] + (v * 2.0 + phases[1]).sin() * 1.8).sin();
                    if thread.abs() < 0.16 {
                        let cell = &mut field[(y * w + x) as usize];
                        cell[0] = 0.45;
                        cell[1] = rng.range(0.24, 0.32);
                    }
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
        self.preset = rng.below(PRESETS.len() as u32) as usize;
        let (mut p, palette) = Params::preset(self.preset);
        p.coupling = rng.range(0.25, 1.0);
        p.sensor_distance = (p.sensor_distance * rng.range(0.55, 1.5)).clamp(2.0, 40.0);
        p.speed = (p.speed * rng.range(0.7, 1.35)).clamp(0.2, 3.0);
        p.sensor_angle = (p.sensor_angle * rng.range(0.7, 1.3)).clamp(0.1, 1.6);
        p.retention = (p.retention + rng.range(-0.025, 0.015)).clamp(0.8, 0.99);
        p.scale = rng.range(0.65, 1.9);
        p.terrain = rng.range(0.1, 1.0);
        p.wander = rng.range(0.015, 0.25);
        p.seeding = *rng.pick(&Seeding::ALL);
        // Keep each chemistry family near a viable regime while exploring
        // scale, geography, seeding and agents much more broadly.
        p.kill = (p.kill + rng.range(-0.001, 0.001)).clamp(0.03, 0.08);
        p.compare = self.params.compare;
        self.params = p;
        self.lut.set(gpu, palette);
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
                chemistry: [p.feed, p.kill, p.coupling, 0.8 / (p.scale * p.scale).max(1.0)],
                motion: [p.sensor_distance, p.sensor_angle, p.turn_angle, p.speed],
                trail: [p.retention.powf(1.0 / p.steps as f32), 0.08 / p.steps as f32, p.wander, 0.28 / p.steps as f32],
                ecology: [p.relationship as u32 as f32, p.terrain, p.scale * p.scale, 0.0],
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
        crate::ui::dropdown(ui, "Relationship", self.params.relationship.name(), |ui| {
            for relationship in Relationship::ALL {
                ui.selectable_value(&mut self.params.relationship, relationship, relationship.name());
            }
        });
        ui.add(Slider::new(&mut self.params.coupling, 0.0..=1.0).text("Coupling strength")).on_hover_text(
            "Controls both directions of the selected relationship. Zero lets chemistry and agents run independently.",
        );
        ui.label(
            egui::RichText::new(if self.params.coupling == 0.0 {
                "Uncoupled · both systems keep running"
            } else {
                self.params.relationship.hint()
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
        ui.collapsing("Habitat & growth", |ui| {
            ui.add(Slider::new(&mut self.params.scale, 0.6..=2.0).text("Growth scale"));
            ui.add(Slider::new(&mut self.params.terrain, 0.0..=1.0).text("Habitat variation"))
                .on_hover_text("Fertile and sparse regions follow a seamless landscape unique to this seed.");
            crate::ui::dropdown(ui, "Seeding (on restart)", self.params.seeding.name(), |ui| {
                for seeding in Seeding::ALL {
                    ui.selectable_value(&mut self.params.seeding, seeding, seeding.name());
                }
            });
            if ui
                .button("Restart this seed")
                .on_hover_text("Apply seeding changes while keeping this seed and settings.")
                .clicked()
            {
                self.reset(gpu, self.seed);
            }
        });
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
            ui.add(Slider::new(&mut self.params.trail_light, 0.0..=2.0).text("Trail light"));
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
