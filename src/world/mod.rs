//! Worlds are self-contained GPU artificial-life simulations.
//!
//! A world owns all of its GPU state, advances it in [`World::step`] and paints
//! it into the shared HDR scene texture in [`World::render`]; the app then runs
//! bloom + tonemapping ([`crate::post`]) and draws the UI on top.
//!
//! `placeholder.rs` is the smallest possible world (a good file to copy when
//! starting a new one); `reaction_diffusion.rs` is a complete, full-featured
//! reference that uses every convention below.
//!
//! Conventions every world follows:
//! * The domain is a torus of `size()` cells. Screen -> world mapping is given by
//!   `Frame::view` (a [`ViewXform`]); world uv can fall outside 0..1 when the
//!   camera zooms out or pans, so wrap when sampling.
//! * `render` must overwrite the whole target (`SCENE_FORMAT`, linear HDR; values
//!   above ~0.6 start to bloom, above 1.0 clearly glow). Clear it, then draw a
//!   fullscreen pass.
//! * `render` is called for every displayed frame (interactive and headless), so
//!   display-only state such as motion trails may accumulate there. `step` is
//!   skipped while paused.
//! * Shader modules come from `Gpu::shader`, which prepends `shaders/common.wgsl`
//!   (fullscreen vertex shader `vs_fullscreen`, `ViewXform`, hashing, palettes).
//! * Parameters are plain Rust structs mirrored into uniform buffers; `ui` edits
//!   them with egui and the next `step` uploads them.

pub mod lenia;
pub mod particle_life;
pub mod physarum;
pub mod placeholder;
pub mod reaction_diffusion;
pub mod symbiosis;

use anyhow::{anyhow, Result};

use crate::gpu::Gpu;
use crate::post::PostSettings;

/// Maps screen uv (0..1, y down) to world uv: `world = screen * scale + offset`.
/// Layout matches `ViewXform` in `common.wgsl` (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewXform {
    pub scale: [f32; 2],
    pub offset: [f32; 2],
}

/// Camera over a world: `center` in world uv, `zoom` 1 = the world covers the screen.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Camera {
    pub center: [f32; 2],
    pub zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { center: [0.5, 0.5], zoom: 1.0 }
    }
}

impl ViewXform {
    /// "Cover" fit of a `world`-sized domain onto a `target`-sized screen, then
    /// zoomed/panned by `camera`. Aspect ratio is always preserved.
    pub fn fit(world: [u32; 2], target: [u32; 2], camera: &Camera) -> Self {
        let world_aspect = world[0].max(1) as f32 / world[1].max(1) as f32;
        let target_aspect = target[0].max(1) as f32 / target[1].max(1) as f32;
        let (mut ex, mut ey) = if target_aspect >= world_aspect {
            (1.0, world_aspect / target_aspect)
        } else {
            (target_aspect / world_aspect, 1.0)
        };
        let zoom = camera.zoom.max(1e-3);
        ex /= zoom;
        ey /= zoom;
        Self {
            scale: [ex, ey],
            offset: [camera.center[0] - 0.5 * ex, camera.center[1] - 0.5 * ey],
        }
    }

    pub fn apply(&self, screen_uv: [f32; 2]) -> [f32; 2] {
        [screen_uv[0] * self.scale[0] + self.offset[0], screen_uv[1] * self.scale[1] + self.offset[1]]
    }
}

/// Mouse interaction, already mapped into world space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pointer {
    /// World uv (not wrapped; may be outside 0..1).
    pub pos: [f32; 2],
    /// Left button held: the world's "create / attract / paint" action.
    pub primary: bool,
    /// Right button held: the world's "destroy / repel / erase" action.
    pub secondary: bool,
    /// Brush radius in world cells.
    pub radius: f32,
}

/// Everything a world needs for one frame.
pub struct Frame<'a> {
    pub gpu: &'a Gpu,
    /// Seconds of simulated time since the app started (fixed-rate in headless renders).
    pub time: f32,
    /// Seconds since the previous frame (clamped to <= 0.1).
    pub dt: f32,
    /// Frames stepped since the app started.
    pub frame: u64,
    /// Screen uv -> world uv for this frame's render target.
    pub view: ViewXform,
    /// Render target size in pixels.
    pub target_size: [u32; 2],
    pub pointer: Option<Pointer>,
}

pub trait World {
    /// Stable identifier, e.g. `"physarum"`.
    fn id(&self) -> &'static str;
    /// Display name, e.g. `"Physarum"`.
    fn name(&self) -> &'static str;
    /// Simulation domain size in cells (used for aspect ratio and brush size).
    fn size(&self) -> [u32; 2];

    /// Names of the built-in presets.
    fn presets(&self) -> &'static [&'static str];
    /// Index of the active preset (or of the preset the current parameters came from).
    fn preset(&self) -> usize;
    /// Loads preset parameters and restarts the simulation from `seed`.
    fn load_preset(&mut self, gpu: &Gpu, index: usize, seed: u64);
    /// Restarts the simulation from `seed`, keeping the current parameters.
    fn reset(&mut self, gpu: &Gpu, seed: u64);
    /// Jumps to a new randomly generated parameter set (biased towards
    /// interesting behaviour) and restarts.
    fn mutate(&mut self, gpu: &Gpu, seed: u64);

    /// Complete editable recipe, independent of the evolving GPU buffers.
    fn settings(&self) -> Result<crate::library::WorldSettings> {
        anyhow::bail!("This world does not support saved settings")
    }
    fn restore_settings(&mut self, _gpu: &Gpu, _settings: &crate::library::WorldSettings, _seed: u64) -> Result<()> {
        anyhow::bail!("This world does not support saved settings")
    }

    /// Advances the simulation by one displayed frame (usually several sub-steps).
    fn step(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder);
    /// Paints the current state into `target` (`SCENE_FORMAT`, `frame.target_size`).
    fn render(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView);

    /// Map an input position through the world's presentation (e.g. split panes).
    /// The app uses the same mapping for brushes and cursor-anchored zoom.
    fn map_position(&self, view: ViewXform, screen_uv: [f32; 2]) -> [f32; 2] {
        view.apply(screen_uv)
    }

    /// Optional labels for a left/right comparison, drawn by the app's HUD.
    fn comparison_labels(&self) -> Option<[String; 2]> { None }

    /// Parameter controls, drawn inside the side panel.
    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui);

    /// Suggested post-processing for the current preset.
    fn post_settings(&self) -> PostSettings {
        PostSettings::default()
    }

    /// One-line status for the HUD, e.g. "4,194,304 agents".
    fn stats(&self) -> String {
        String::new()
    }

    /// What the mouse buttons do, e.g. "Left: attract · Right: repel".
    fn controls_hint(&self) -> &'static str {
        "Left: paint · Right: erase"
    }
}

pub struct WorldEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub tagline: &'static str,
    /// Creates the world sized for an output of `output_size` pixels.
    pub create: fn(&Gpu, [u32; 2], u64) -> Box<dyn World>,
}

pub const WORLDS: &[WorldEntry] = &[
    WorldEntry {
        id: "physarum",
        name: "Physarum",
        aliases: &["slime", "slime-mold", "slime-mould", "mold", "mould"],
        tagline: "Millions of slime-mould agents weaving living transport networks",
        create: physarum::create,
    },
    WorldEntry {
        id: "particle-life",
        name: "Particle Life",
        aliases: &["particles", "particlelife", "pl", "life"],
        tagline: "Species with asymmetric attractions self-assemble into cells and creatures",
        create: particle_life::create,
    },
    WorldEntry {
        id: "lenia",
        name: "Lenia",
        aliases: &["smoothlife", "continuous-ca"],
        tagline: "Continuous cellular automata that grow soft, gliding organisms",
        create: lenia::create,
    },
    WorldEntry {
        id: "reaction-diffusion",
        name: "Reaction-Diffusion",
        aliases: &["rd", "gray-scott", "grayscott", "turing", "coral"],
        tagline: "Gray-Scott chemistry painting coral, mitosis and fingerprints",
        create: reaction_diffusion::create,
    },
    WorldEntry {
        id: "symbiosis",
        name: "Symbiosis",
        aliases: &["coupled", "hybrid", "ecosystem"],
        tagline: "Trail-following agents and living chemistry shape each other",
        create: symbiosis::create,
    },
];

fn normalize(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// Finds a world by id, name, alias, 1-based index or unambiguous prefix.
pub fn find(query: &str) -> Option<usize> {
    let q = normalize(query);
    if let Ok(n) = q.parse::<usize>() {
        return (1..=WORLDS.len()).contains(&n).then(|| n - 1);
    }
    let exact = WORLDS.iter().position(|w| {
        normalize(w.id) == q || normalize(w.name) == q || w.aliases.iter().any(|a| normalize(a) == q)
    });
    if exact.is_some() {
        return exact;
    }
    let prefixed: Vec<usize> =
        (0..WORLDS.len()).filter(|&i| !q.is_empty() && normalize(WORLDS[i].id).starts_with(&q)).collect();
    (prefixed.len() == 1).then(|| prefixed[0])
}

/// Finds a preset by 1-based index, name, or unambiguous prefix (case/punctuation-insensitive).
pub fn find_preset(presets: &[&str], query: &str) -> Option<usize> {
    let q = normalize(query);
    if let Ok(n) = q.parse::<usize>() {
        return (1..=presets.len()).contains(&n).then(|| n - 1);
    }
    if let Some(i) = presets.iter().position(|p| normalize(p) == q) {
        return Some(i);
    }
    let prefixed: Vec<usize> =
        (0..presets.len()).filter(|&i| !q.is_empty() && normalize(presets[i]).starts_with(&q)).collect();
    (prefixed.len() == 1).then(|| prefixed[0])
}

/// Creates world `query` for an `output_size` target, optionally loading a preset.
pub fn create(
    gpu: &Gpu,
    query: &str,
    output_size: [u32; 2],
    preset: Option<&str>,
    seed: u64,
) -> Result<(usize, Box<dyn World>)> {
    let index = find(query).ok_or_else(|| {
        let ids: Vec<&str> = WORLDS.iter().map(|w| w.id).collect();
        anyhow!("unknown world '{query}' (available: {})", ids.join(", "))
    })?;
    if WORLDS[index].id == "reaction-diffusion" {
        reaction_diffusion::validate_output_size(gpu, output_size)?;
    }
    let mut world = (WORLDS[index].create)(gpu, output_size, seed);
    if let Some(p) = preset {
        let presets = world.presets();
        let i = find_preset(presets, p).ok_or_else(|| {
            anyhow!("unknown preset '{p}' for {} (available: {})", WORLDS[index].name, presets.join(", "))
        })?;
        world.load_preset(gpu, i, seed);
    }
    Ok((index, world))
}
