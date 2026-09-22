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
//! * A sampler must not share its `@binding` with any other resource declared in
//!   the same module. DirectX 12 translates the whole module for every pipeline,
//!   and naga panics when a compute pipeline's layout puts a buffer where the
//!   module declares a sampler (gfx-rs/wgpu#7638). Give a display pass that
//!   samples textures its own module, as `lenia_draw.wgsl` does; a test checks
//!   every file in `src/shaders`.
//! * Parameters are plain Rust structs mirrored into uniform buffers; `ui` edits
//!   them with egui and the next `step` uploads them.
//! * Measurements (`metrics` / `measure`) are optional: a world reduces a few
//!   scalars of its state on the GPU right after `step`, and the engine reads
//!   them back asynchronously for the panel's sparklines and CSV logs
//!   ([`crate::metrics`]).
//! * The static facts about a world (presets, measurements, palettes) are also
//!   registered in [`WORLDS`], so names resolve and `primordia list` works
//!   before (or without) a GPU.

pub mod lenia;
pub mod particle_life;
pub mod physarum;
pub mod placeholder;
pub mod reaction_diffusion;
pub mod symbiosis;

use anyhow::Result;

use crate::failure::Failure;
use crate::gpu::Gpu;
use crate::metrics::MetricDesc;
use crate::palette;
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

/// Zoom factors a camera may have in a render or a recipe (`render --zoom`).
/// The app's wheel and slider stay within 0.5-64.
pub const ZOOM_RANGE: std::ops::RangeInclusive<f32> = 0.05..=256.0;

/// Runs `f` inside out-of-memory and validation error scopes, so a bad
/// parameter set is reported instead of poisoning the device. Running out of
/// memory is a [`Failure::Gpu`], a validation error a [`Failure::Usage`] (the
/// parameters asked for something this GPU cannot do).
pub fn guarded<T>(gpu: &Gpu, f: impl FnOnce() -> T) -> Result<T> {
    gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let result = f();
    let validation = pollster::block_on(gpu.device.pop_error_scope());
    let oom = pollster::block_on(gpu.device.pop_error_scope());
    if let Some(e) = oom {
        return Err(Failure::Gpu.error(format!("out of GPU memory: {e}")));
    }
    if let Some(e) = validation {
        return Err(Failure::Usage.error(format!("GPU validation failed: {e}")));
    }
    Ok(result)
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

    /// Names of the built-in presets (the same table as [`WorldEntry::presets`]).
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

    /// Scalar measurements this world computes on the GPU every frame
    /// (`&[]` = none). The order is the lane order of the totals record the
    /// world pushes in [`World::measure`] and the column order of CSV logs.
    /// It is the table registered as [`WorldEntry::metrics`].
    fn metrics(&self) -> &'static [crate::metrics::MetricDesc] {
        &[]
    }

    /// Records the measurement passes for the state `step` just produced and
    /// pushes one totals record per series into `sink`: once for a single
    /// habitat, twice for a comparison (self first, then the reference, matching
    /// [`World::comparison_labels`]). Called right after `step`, never while
    /// paused. Skip the passes when `!sink.is_live()`.
    fn measure(&mut self, _frame: &Frame, _encoder: &mut wgpu::CommandEncoder, _sink: &mut crate::metrics::Sink<'_>) {}

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

/// A registered world: everything the app, the CLI and `primordia list` know
/// about it without a GPU, plus its constructor.
pub struct WorldEntry {
    pub id: &'static str,
    pub name: &'static str,
    /// Other names `--world` accepts (whole, ignoring case and punctuation).
    pub aliases: &'static [&'static str],
    pub tagline: &'static str,
    /// Names of the built-in presets: the table [`World::presets`] returns.
    pub presets: fn() -> &'static [&'static str],
    /// The measurement table [`World::metrics`] returns.
    pub metrics: &'static [MetricDesc],
    /// Measurements that must stay above zero for an explore candidate to count
    /// as alive: a dead, empty or frozen world is novel in behaviour space but
    /// never worth an archive slot (`--keep-inert` lifts the rule).
    pub vital: &'static [&'static str],
    /// Colour palettes the world's recipes can name, in menu order.
    pub palettes: fn() -> Vec<&'static str>,
    /// Where a recipe (`library::WorldSettings`) picks from `palettes`: `palette`
    /// holds one name, Lenia's `palettes` one name per channel, and Particle
    /// Life's `params.colors` a colour scheme by 0-based index.
    pub palette_setting: &'static str,
    /// Recipe settings that change how the world looks, never what it does:
    /// the palette, colours, tone and display-only trails, and `post`. They are
    /// dotted paths as `--set` takes them, with `*` for every item of a list.
    /// Explore's refinement never perturbs them, so a child keeps its parent's look.
    pub appearance: &'static [&'static str],
    /// Creates the world sized for an output of `output_size` pixels.
    pub create: fn(&Gpu, [u32; 2], u64) -> Box<dyn World>,
}

pub const WORLDS: &[WorldEntry] = &[
    WorldEntry {
        id: "physarum",
        name: "Physarum",
        aliases: &["slime", "slime-mold", "slime-mould", "mold", "mould"],
        tagline: "Millions of slime-mould agents weaving living transport networks",
        presets: physarum::preset_names,
        metrics: physarum::METRICS,
        vital: &["ground"],
        palettes: physarum::palette_names,
        palette_setting: "palette",
        // The traffic long exposure is display-only, even though `travelled` measures it.
        appearance: &[
            "palette", "post", "params.species.*.color", "params.color_mode", "params.exposure",
            "params.trail_weight", "params.traffic_weight", "params.traffic_persistence", "params.traffic_blur",
            "params.brightness", "params.filigree", "params.glow", "params.palette_span", "params.smoothing",
            "params.ground",
        ],
        create: physarum::create,
    },
    WorldEntry {
        id: "particle-life",
        name: "Particle Life",
        aliases: &["particles", "particlelife", "pl", "life"],
        tagline: "Species with asymmetric attractions self-assemble into cells and creatures",
        presets: particle_life::preset_names,
        metrics: particle_life::METRICS,
        vital: &["speed"],
        palettes: particle_life::scheme_names,
        palette_setting: "params.colors",
        // `ground` is the background colour (packed sRGB).
        appearance: &[
            "ground", "post", "params.colors", "params.color_shift", "params.sizes", "params.size", "params.glow",
            "params.speed_glow", "params.trail", "params.trail_gain", "params.trail_scale", "params.knee",
            "params.relief",
        ],
        create: particle_life::create,
    },
    WorldEntry {
        id: "lenia",
        name: "Lenia",
        aliases: &["smoothlife", "continuous-ca"],
        tagline: "Continuous cellular automata that grow soft, gliding organisms",
        presets: lenia::preset_names,
        metrics: lenia::METRICS,
        vital: &["mass", "active"],
        palettes: palette::names,
        palette_setting: "palettes",
        appearance: &[
            "palettes", "post", "params.gain", "params.glow", "params.halo", "params.trail", "params.persistence",
            "params.relief", "params.brightness", "params.tint", "params.level", "params.rim", "params.core",
            "params.hue", "params.mix_power", "params.ground", "params.medium", "params.ground_channel",
            "params.sharp",
        ],
        create: lenia::create,
    },
    WorldEntry {
        id: "reaction-diffusion",
        name: "Reaction-Diffusion",
        aliases: &["rd", "gray-scott", "grayscott", "turing", "coral"],
        tagline: "Gray-Scott chemistry painting coral, mitosis and fingerprints",
        presets: reaction_diffusion::preset_names,
        metrics: reaction_diffusion::METRICS,
        vital: &["alive", "active"],
        palettes: palette::names,
        palette_setting: "palette",
        // `params.ground` is chemistry here (kill added in drifting lagoons), so it is perturbed.
        appearance: &[
            "palette", "post", "params.material", "params.invert", "params.pal_lo", "params.pal_hi",
            "params.contrast_lo", "params.contrast_hi", "params.relief", "params.gloss", "params.glow", "params.halo",
            "params.aura_tint", "params.iridescence", "params.reflect", "params.clarity", "params.activity",
            "params.shadow", "params.brightness",
        ],
        create: reaction_diffusion::create,
    },
    WorldEntry {
        id: "symbiosis",
        name: "Symbiosis",
        aliases: &["coupled", "hybrid", "ecosystem"],
        tagline: "Trail-following agents and living chemistry shape each other",
        presets: symbiosis::preset_names,
        metrics: symbiosis::METRICS,
        vital: &["growth_cover", "growth_active"],
        palettes: palette::names,
        palette_setting: "palette",
        appearance: &["palette", "post", "params.layer", "params.brightness", "params.trail_light"],
        create: symbiosis::create,
    },
];

// --- name lookup -------------------------------------------------------------------

fn normalize(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// Why a name did not pick exactly one entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Miss {
    /// Nothing matches; `suggestion` is a close spelling, when there is one.
    Unknown { suggestion: Option<&'static str> },
    /// The name is a prefix of several entries.
    Ambiguous(Vec<&'static str>),
    /// A number outside `1..=count`.
    OutOfRange,
}

/// A world, preset or measurement name that did not resolve. It is invalid
/// input (exit status 2, see [`crate::failure`]); the message names the choices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameError {
    /// `"world"`, `"preset"` or `"measurement"`.
    pub what: &'static str,
    /// The world a preset or measurement was looked up in.
    pub world: Option<usize>,
    pub query: String,
    pub miss: Miss,
    /// Every valid name in order; worlds and presets also take their 1-based number.
    pub available: Vec<&'static str>,
}

/// "a, b or c".
fn either(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [one] => one.to_string(),
        [rest @ .., last] => format!("{} or {last}", rest.join(", ")),
    }
}

impl std::fmt::Display for NameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (what, query, count) = (self.what, &self.query, self.available.len());
        let owner = self.world.map(|w| format!(" for {}", WORLDS[w].name)).unwrap_or_default();
        let list = match self.world {
            Some(w) => format!("`primordia list --world {}`", WORLDS[w].id),
            None => "`primordia list`".to_string(),
        };
        match &self.miss {
            Miss::Unknown { suggestion } => {
                write!(f, "unknown {what} '{query}'{owner}")?;
                if let Some(s) = suggestion {
                    write!(f, "; did you mean '{s}'?")?;
                }
                let numbers = if what == "measurement" { String::new() } else { format!(", or 1-{count}") };
                write!(f, " (available: {}{numbers}; see {list})", self.available.join(", "))
            }
            Miss::Ambiguous(candidates) => write!(f, "ambiguous {what} '{query}'{owner}: it could be {}", either(candidates)),
            Miss::OutOfRange => {
                let range = match self.world {
                    Some(w) => format!("{} has {what}s 1-{count}", WORLDS[w].name),
                    None => format!("there are {what}s 1-{count}"),
                };
                write!(f, "{what} {query} is out of range: {range} ({})", self.available.join(", "))
            }
        }
    }
}

impl std::error::Error for NameError {}

/// The spelling in `names` closest to `query`, if `query` looks like a typo of it.
fn suggest(query: &str, names: impl IntoIterator<Item = &'static str>) -> Option<&'static str> {
    let q = normalize(query);
    if q.is_empty() {
        return None;
    }
    names
        .into_iter()
        .map(|name| (name, strsim::jaro_winkler(&q, &normalize(name))))
        .filter(|(_, score)| *score >= 0.8)
        // The first of equally close spellings: an id before its display name or aliases.
        .min_by(|a, b| b.1.total_cmp(&a.1))
        .map(|(name, _)| name)
}

/// Resolves `query` among `names`: a 1-based number, a name or one of the
/// `aliases(i)` of entry `i` (whole, ignoring case and punctuation), or a
/// prefix of exactly one name.
fn lookup(names: &[&'static str], aliases: impl Fn(usize) -> Vec<&'static str>, query: &str) -> Result<usize, Miss> {
    let q = normalize(query);
    if !q.is_empty() && q.bytes().all(|b| b.is_ascii_digit()) {
        return match q.parse::<usize>() {
            Ok(n) if (1..=names.len()).contains(&n) => Ok(n - 1),
            _ => Err(Miss::OutOfRange),
        };
    }
    let exact = |i: usize| normalize(names[i]) == q || aliases(i).iter().any(|a| normalize(a) == q);
    if let Some(i) = (0..names.len()).find(|&i| exact(i)) {
        return Ok(i);
    }
    let prefixed: Vec<usize> =
        (0..names.len()).filter(|&i| !q.is_empty() && normalize(names[i]).starts_with(&q)).collect();
    match prefixed[..] {
        [i] => Ok(i),
        [] => {
            let spellings = (0..names.len()).flat_map(|i| std::iter::once(names[i]).chain(aliases(i)));
            Err(Miss::Unknown { suggestion: suggest(query, spellings) })
        }
        _ => Err(Miss::Ambiguous(prefixed.iter().map(|&i| names[i]).collect())),
    }
}

/// Resolves a world by id, name, alias, 1-based number or a prefix of one id.
pub fn resolve(query: &str) -> Result<usize, NameError> {
    let ids: Vec<&'static str> = WORLDS.iter().map(|w| w.id).collect();
    let aliases = |i: usize| std::iter::once(WORLDS[i].name).chain(WORLDS[i].aliases.iter().copied()).collect();
    lookup(&ids, aliases, query)
        .map_err(|miss| NameError { what: "world", world: None, query: query.to_string(), miss, available: ids.clone() })
}

/// Resolves a preset of world `world` by name, 1-based number or a prefix of one name.
pub fn resolve_preset(world: usize, query: &str) -> Result<usize, NameError> {
    let names = (WORLDS[world].presets)();
    lookup(names, |_| Vec::new(), query).map_err(|miss| NameError {
        what: "preset",
        world: Some(world),
        query: query.to_string(),
        miss,
        available: names.to_vec(),
    })
}

/// Resolves a measurement id of world `world` (exactly: ids are CSV column names).
pub fn resolve_metric(world: usize, id: &str) -> Result<usize, NameError> {
    let metrics = WORLDS[world].metrics;
    let ids: Vec<&'static str> = metrics.iter().map(|m| m.id).collect();
    ids.iter().position(|m| *m == id).ok_or_else(|| NameError {
        what: "measurement",
        world: Some(world),
        query: id.to_string(),
        miss: Miss::Unknown { suggestion: suggest(id, ids.iter().copied()) },
        available: ids.clone(),
    })
}

/// Finds a world by id, name, alias, 1-based number or unambiguous prefix.
pub fn find(query: &str) -> Option<usize> {
    resolve(query).ok()
}

/// Creates world `query` for an `output_size` target, optionally loading a
/// preset. Names are resolved before anything touches the GPU.
pub fn create(
    gpu: &Gpu,
    query: &str,
    output_size: [u32; 2],
    preset: Option<&str>,
    seed: u64,
) -> Result<(usize, Box<dyn World>)> {
    let index = resolve(query)?;
    let preset = preset.map(|p| resolve_preset(index, p)).transpose()?;
    Ok((index, create_at(gpu, index, output_size, preset, seed)?))
}

/// Creates `WORLDS[index]` for an `output_size` target, optionally loading preset `preset` (0-based).
pub fn create_at(gpu: &Gpu, index: usize, output_size: [u32; 2], preset: Option<usize>, seed: u64) -> Result<Box<dyn World>> {
    if WORLDS[index].id == "reaction-diffusion" {
        reaction_diffusion::validate_output_size(gpu, output_size).map_err(|e| Failure::Usage.tag(e))?;
    }
    let mut world = (WORLDS[index].create)(gpu, output_size, seed);
    if let Some(i) = preset {
        let count = world.presets().len();
        if i >= count {
            return Err(Failure::Usage.error(format!("{} has presets 1-{count}, not {}", WORLDS[index].name, i + 1)));
        }
        world.load_preset(gpu, i, seed);
    }
    Ok(world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MAX_METRICS;

    #[test]
    fn registry_tables_are_complete_and_every_name_resolves_to_its_world() {
        let mut spellings = std::collections::HashMap::new();
        for (index, entry) in WORLDS.iter().enumerate() {
            for name in [entry.id, entry.name].into_iter().chain(entry.aliases.iter().copied()) {
                assert_eq!(resolve(name), Ok(index), "{name}");
                let other = spellings.insert(normalize(name), index);
                assert!(other.is_none_or(|o| o == index), "'{name}' names two worlds");
            }

            let presets = (entry.presets)();
            assert!(!presets.is_empty(), "{}: no presets", entry.id);
            for (i, name) in presets.iter().enumerate() {
                assert_eq!(resolve_preset(index, name), Ok(i), "{} / {name}", entry.id);
                // Names go into candidates.csv unquoted and slugs into file names.
                assert!(!name.contains([',', '"']), "{}: preset '{name}' needs CSV quoting", entry.id);
                let slug = crate::headless::slug(name);
                assert!(!slug.is_empty(), "{} / {name}: empty slug", entry.id);
                assert!(presets[..i].iter().all(|p| crate::headless::slug(p) != slug), "{}: duplicate slug {slug}", entry.id);
            }

            let metrics = entry.metrics;
            assert!(!metrics.is_empty() && metrics.len() <= MAX_METRICS, "{}: {} metrics", entry.id, metrics.len());
            for (i, metric) in metrics.iter().enumerate() {
                assert!(
                    !metric.id.is_empty()
                        && metric.id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                    "{}: metric id '{}' is not a CSV-friendly identifier",
                    entry.id,
                    metric.id
                );
                assert!(metrics[..i].iter().all(|m| m.id != metric.id), "{}: duplicate id '{}'", entry.id, metric.id);
                assert!(!metric.label.is_empty() && !metric.hint.is_empty(), "{}: {} lacks a label or hint", entry.id, metric.id);
                assert!(!metric.hint.contains('\n'), "{}: {}'s hint must be one line", entry.id, metric.id);
                assert_eq!(resolve_metric(index, metric.id), Ok(i));
            }
            assert!(!entry.vital.is_empty(), "{}: no vital measurement", entry.id);
            for id in entry.vital {
                assert!(metrics.iter().any(|m| m.id == *id), "{}: unknown vital measurement {id}", entry.id);
            }

            let palettes = (entry.palettes)();
            assert!(!palettes.is_empty(), "{}: no palettes", entry.id);
            for (i, name) in palettes.iter().enumerate() {
                assert!(!palettes[..i].contains(name), "{}: palette {name} is listed twice", entry.id);
            }

            // The look and the palette are appearance; every path is a --set key (list items as `*`).
            let appearance = entry.appearance;
            assert!(appearance.contains(&"post") && appearance.contains(&entry.palette_setting), "{}", entry.id);
            for (i, path) in appearance.iter().enumerate() {
                assert!(path.split('.').all(|part| !part.is_empty() && part.trim() == part), "{}: '{path}'", entry.id);
                assert!(!appearance[..i].contains(path), "{}: {path} is listed twice", entry.id);
            }
        }
    }

    #[test]
    fn names_resolve_by_number_alias_and_prefix_and_misses_explain_themselves() {
        assert_eq!(resolve("5"), Ok(4));
        assert_eq!(resolve("Gray Scott"), Ok(3));
        assert_eq!(resolve("REAC"), Ok(3));
        assert_eq!(resolve("l"), Ok(2), "prefixes match ids only, never aliases such as 'life'");
        assert_eq!(find("slime-mould"), Some(0));
        assert_eq!(find("p"), None);

        let typo = resolve("physarm").unwrap_err();
        assert_eq!(typo.miss, Miss::Unknown { suggestion: Some("physarum") });
        assert_eq!(
            typo.to_string(),
            "unknown world 'physarm'; did you mean 'physarum'? (available: physarum, particle-life, lenia, \
             reaction-diffusion, symbiosis, or 1-5; see `primordia list`)"
        );
        assert_eq!(resolve("greyscott").unwrap_err().miss, Miss::Unknown { suggestion: Some("gray-scott") });
        let ambiguous = resolve("p").unwrap_err();
        assert_eq!(ambiguous.miss, Miss::Ambiguous(vec!["physarum", "particle-life"]));
        assert_eq!(ambiguous.to_string(), "ambiguous world 'p': it could be physarum or particle-life");
        for number in ["0", "6", "123456789012345678901234567890"] {
            assert_eq!(resolve(number).unwrap_err().miss, Miss::OutOfRange, "{number}");
        }
        assert!(resolve("9").unwrap_err().to_string().starts_with("world 9 is out of range: there are worlds 1-5"));
        for nothing in ["zzz", "", "--"] {
            assert_eq!(resolve(nothing).unwrap_err().miss, Miss::Unknown { suggestion: None }, "{nothing:?}");
        }

        let lenia = resolve("lenia").unwrap();
        assert_eq!(resolve_preset(lenia, "neck"), Ok(4));
        assert_eq!(resolve_preset(lenia, "pearl-reef"), Ok(3));
        assert_eq!(resolve_preset(lenia, "7"), Ok(6));
        assert_eq!(
            resolve_preset(lenia, "99").unwrap_err().to_string(),
            "preset 99 is out of range: Lenia has presets 1-7 (Orbium, Leviathans, Menagerie, Pearl Reef, Necklaces, \
             Hydrogeminium, Tessellatium)"
        );
        let typo = resolve_preset(lenia, "orbum").unwrap_err();
        assert_eq!(typo.miss, Miss::Unknown { suggestion: Some("Orbium") });
        assert!(typo.to_string().contains("for Lenia; did you mean 'Orbium'?"), "{typo}");
        assert!(typo.to_string().contains("or 1-7; see `primordia list --world lenia`"), "{typo}");
        let physarum = resolve("physarum").unwrap();
        assert_eq!(
            resolve_preset(physarum, "s").unwrap_err().to_string(),
            "ambiguous preset 's' for Physarum: it could be Symbiosis or Synapses"
        );

        let symbiosis = resolve("symbiosis").unwrap();
        assert_eq!(resolve_metric(symbiosis, "growth_cover"), Ok(0));
        let typo = resolve_metric(symbiosis, "growth_cove").unwrap_err();
        assert_eq!(typo.miss, Miss::Unknown { suggestion: Some("growth_cover") });
        assert!(typo.to_string().starts_with("unknown measurement 'growth_cove' for Symbiosis; did you mean"), "{typo}");
        assert!(!typo.to_string().contains("1-"), "measurements are not numbered: {typo}");
        assert!(resolve_metric(symbiosis, "growth").is_err(), "measurement ids never match by prefix");
    }

    /// `(group, binding)`, whether it is a sampler, and the declaration, for
    /// every resource a WGSL source declares (`@group(g) @binding(b) var ...;`).
    fn resource_bindings(source: &str) -> Vec<([u32; 2], bool, String)> {
        let code: Vec<&str> = source.lines().map(|line| line.split("//").next().unwrap_or("")).collect();
        code.join("\n")
            .split(';')
            .filter_map(|statement| {
                let decl = &statement[statement.find("@group(")?..];
                let number = |key: &str| -> Option<u32> {
                    let rest = &decl[decl.find(key)? + key.len()..];
                    rest[..rest.find(')')?].trim().parse().ok()
                };
                let ty = decl.rsplit(':').next()?.trim();
                let text = decl.split_whitespace().collect::<Vec<_>>().join(" ");
                Some(([number("@group(")?, number("@binding(")?], ty.starts_with("sampler"), text))
            })
            .collect()
    }

    #[test]
    fn resource_bindings_reads_declarations() {
        let source = "struct D { a: f32, };\n@group(0) @binding(3) var wrap: sampler; // repeat\n\
                      @group(1) @binding(2)\n    var<storage, read> cells: array<vec4<f32>>;\nfn f() { let x = 1; }";
        let found = resource_bindings(source);
        assert_eq!(found.len(), 2, "{found:?}");
        assert_eq!((found[0].0, found[0].1), ([0, 3], true));
        assert_eq!((found[1].0, found[1].1), ([1, 2], false));
        assert_eq!(found[1].2, "@group(1) @binding(2) var<storage, read> cells: array<vec4<f32>>");
    }

    /// DirectX 12 runs naga's HLSL writer over the whole module for every
    /// pipeline, and it panics ("Sampler buffer of group ... not bound to a
    /// register") when the pipeline's layout has another resource at a sampler's
    /// binding and no sampler in that group. A sampler that shares its binding
    /// with nothing else in its module cannot meet such a layout.
    #[test]
    fn shader_samplers_keep_their_bindings_to_themselves() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
        let (mut modules, mut clashes) = (0, Vec::new());
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "wgsl") {
                continue;
            }
            modules += 1;
            let bindings = resource_bindings(&std::fs::read_to_string(&path).unwrap());
            for (slot, _, sampler) in bindings.iter().filter(|b| b.1) {
                for (_, _, other) in bindings.iter().filter(|b| b.0 == *slot && !b.1) {
                    let file = path.file_name().unwrap().to_string_lossy();
                    clashes.push(format!("{file}: `{sampler}` shares its binding with `{other}`"));
                }
            }
        }
        assert!(modules >= 15, "found only {modules} shader files in {}", dir.display());
        assert!(clashes.is_empty(), "move these samplers to a module of their own:\n{}", clashes.join("\n"));
    }

    /// The template is not registered in `WORLDS`, so nothing else compiles its
    /// inline WGSL: build it, then step and render one small frame through post.
    #[test]
    fn gpu_placeholder_template_renders_a_frame() {
        use crate::capture::Readback;
        use crate::post::Post;
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let size = [64, 64];
        let mut world = placeholder::Placeholder::new(&gpu, "placeholder", "Placeholder", size, 0.3);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let post = Post::new(&gpu, size, format);
        let (texture, view) = gpu.texture_2d(
            "placeholder test",
            size,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let frame = Frame {
            gpu: &gpu,
            time: 0.5,
            dt: 1.0 / 60.0,
            frame: 0,
            view: ViewXform::fit(world.size(), size, &Camera::default()),
            target_size: size,
            pointer: None,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.step(&frame, &mut encoder);
        world.render(&frame, &mut encoder, post.scene_view());
        post.run(&gpu, &mut encoder, &world.post_settings(), frame.time, &view);
        let readback = Readback::new(&gpu, size, format);
        readback.copy_from(&mut encoder, &texture);
        gpu.queue.submit([encoder.finish()]);
        let pixels = readback.read(&gpu).unwrap();
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
        assert!(pixels.chunks_exact(4).any(|px| px[..3].iter().any(|&c| c > 16)), "the rings should be visible");
    }
}
