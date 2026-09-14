//! Physarum polycephalum: millions of slime-mould agents weave living transport
//! networks. Every agent senses the trail ahead of it at three points, steers
//! towards the strongest signal, moves and deposits (Jeff Jones, 2010). Up to
//! four species share the world; an interaction matrix decides how strongly
//! each is drawn to, or repelled by, every species' trail (after Sage Jenson).
//!
//! GPU state:
//! * `agents`: 16 bytes each: position (cells), heading, and a `u32` holding
//!   the species (low two bits) plus a per-agent Weyl counter, hashed with the
//!   agent index for fresh randomness in every sub-step of a submission.
//! * `trail[2]`: ping-ponged RGBA float textures, one channel per species.
//!   This is what agents sense.
//! * `traffic[2]`: ping-ponged display-only long exposure of the raw agent
//!   counts: crisp paths whose brightness follows the traffic they carry.
//! * `counts`: four `atomic<u32>` per cell. Agents *count* their deposits with
//!   integer atomics (race-free and bit-exactly deterministic); the diffuse
//!   pass turns counts into trail and traffic, blurs, decays and clears them.
//! * `ground`: the display's adaptive black point (below).
//!
//! A slow periodic noise "terrain" scales how fast the trail decays across the
//! torus, so networks thicken on fertile ground and thin out into quiet voids:
//! every preset gets a macro composition, which drifts over the minutes.
//!
//! Display (physarum_draw.wgsl): per species, trail haze plus traffic go
//! through a logarithmic tone curve spanning `filigree` decades below a knee
//! at dense-vein density, so a lone agent's streak stays visible as filigree
//! while veins cross the knee and glow into HDR. Every frame one workgroup
//! histograms the display density and raises the black point until the
//! `ground` fraction of the frame is dark, so parameter sets that pack the
//! frame with lanes still sit on a dark ground.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use super::{Frame, ViewXform, World};
use crate::gpu::{self, layout, Gpu, SCENE_FORMAT};
use crate::palette::{self, Palette, PALETTES};
use crate::post::{PostSettings, Tonemap};
use crate::rng::Rng;

const AGENT_WG: u32 = 256;
const CELL_WG: u32 = 16;
const MAX_SPECIES: usize = 4;
const MAX_STEPS: u32 = 12;
/// Trail cells per output pixel along each axis.
const DOMAIN_SCALE: f32 = 1.0;
const MAX_SIDE: u32 = 8192;
const MIN_AGENTS: u32 = 1024;
const MAX_AGENTS: u32 = 1 << 24;
/// Upper bound for trail and traffic values (the mean level is ~1; also safe for f16).
const TRAIL_CAP: f32 = 60_000.0;
/// Strength of the cursor's food relative to the mean trail level.
const FOOD_GAIN: f32 = 30.0;
const TRAFFIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Time constant of the adaptive black point's glide, in seconds.
const GROUND_SECONDS: f32 = 0.33;
/// How strongly the terrain scales agents (reach and stride), relative to how
/// strongly it scales the trail's decay rate (both in log2 units).
const TERRAIN_SIZE: f32 = 0.5;
/// Radius of the core pull, as a fraction of the shorter side.
const CORE_RADIUS: f32 = 0.12;

/// WGSL spelling of the field texture formats used here.
fn wgsl_format(format: wgpu::TextureFormat) -> &'static str {
    match format {
        wgpu::TextureFormat::Rgba32Float => "rgba32float",
        wgpu::TextureFormat::Rgba16Float => "rgba16float",
        other => panic!("physarum fields cannot use {other:?}"),
    }
}

/// How agents are laid out when the simulation (re)starts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Layout {
    Scatter,
    DiskOut,
    DiskIn,
    Rings,
    Sectors,
    Bands,
    Clusters,
    Vortex,
    Galaxy,
}

impl Layout {
    const ALL: [Layout; 9] = [
        Layout::Scatter,
        Layout::DiskOut,
        Layout::DiskIn,
        Layout::Rings,
        Layout::Sectors,
        Layout::Bands,
        Layout::Clusters,
        Layout::Vortex,
        Layout::Galaxy,
    ];

    fn name(self) -> &'static str {
        match self {
            Layout::Scatter => "Uniform scatter",
            Layout::DiskOut => "Disk, facing out",
            Layout::DiskIn => "Disk, facing in",
            Layout::Rings => "Concentric rings",
            Layout::Sectors => "Species sectors",
            Layout::Bands => "Species bands",
            Layout::Clusters => "Clusters",
            Layout::Vortex => "Rim, swirling in",
            Layout::Galaxy => "Spiral arms",
        }
    }

    /// `Seeding::pattern` in physarum.wgsl.
    fn code(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColorMode {
    /// Total density mapped through the palette.
    Palette,
    /// Every species glows in its own colour; overlaps blend their hues.
    Species,
}

impl ColorMode {
    fn name(self) -> &'static str {
        match self {
            ColorMode::Palette => "Palette (density)",
            ColorMode::Species => "Species colours",
        }
    }
}

/// Behaviour of one species. Angles are in degrees, distances in cells.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Species {
    /// Angle between the centre sensor and each side sensor.
    pub sensor_angle: f32,
    /// How far ahead the sensors reach.
    pub sensor_distance: f32,
    /// Rotation applied when steering.
    pub turn_angle: f32,
    /// Cells travelled per sub-step.
    pub speed: f32,
    /// Trail deposited per agent and sub-step (relative between species).
    pub deposit: f32,
    /// Random heading jitter (+-).
    pub wander: f32,
    /// Constant turn per sub-step; curls paths into vortices.
    pub curl: f32,
    /// Per-agent size variety (log2 units): each agent's sensor distance and
    /// speed are scaled by a fixed factor in `2^[-spread, spread]`.
    pub spread: f32,
    /// sRGB display colour (species colour mode).
    pub color: [u8; 3],
}

impl Species {
    /// Clamped into ranges that can never destabilise the simulation.
    fn sanitized(&self) -> Self {
        Self {
            sensor_angle: self.sensor_angle.clamp(1.0, 170.0),
            sensor_distance: self.sensor_distance.clamp(0.5, 96.0),
            turn_angle: self.turn_angle.clamp(0.5, 170.0),
            speed: self.speed.clamp(0.05, 8.0),
            deposit: self.deposit.clamp(0.01, 10.0),
            wander: self.wander.clamp(0.0, 90.0),
            curl: self.curl.clamp(-20.0, 20.0),
            spread: self.spread.clamp(0.0, 2.0),
            color: self.color,
        }
    }

    const fn with_spread(mut self, spread: f32) -> Self {
        self.spread = spread;
        self
    }

    fn linear_color(&self) -> [f32; 4] {
        let [r, g, b] = self.color.map(|c| palette::srgb_to_linear(c as f32 / 255.0));
        [r, g, b, 1.0]
    }
}

const fn rgb(hex: u32) -> [u8; 3] {
    [(hex >> 16) as u8, (hex >> 8) as u8, hex as u8]
}

/// Compact species constructor for the preset table.
#[allow(clippy::too_many_arguments)]
const fn sp(
    sensor_angle: f32,
    sensor_distance: f32,
    turn_angle: f32,
    speed: f32,
    deposit: f32,
    wander: f32,
    curl: f32,
    color: u32,
) -> Species {
    Species { sensor_angle, sensor_distance, turn_angle, speed, deposit, wander, curl, spread: 0.0, color: rgb(color) }
}

/// Symmetric interaction matrix: `diag` for a species' own trail, `off` for the others.
const fn matrix(diag: f32, off: f32) -> [[f32; 4]; 4] {
    [[diag, off, off, off], [off, diag, off, off], [off, off, diag, off], [off, off, off, diag]]
}

/// Cyclic dominance: each species follows the next one's trail and flees the previous one's.
const fn cyclic(follow: f32, flee: f32) -> [[f32; 4]; 4] {
    [[1.0, follow, flee, 0.0], [flee, 1.0, follow, 0.0], [follow, flee, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

const fn look(exposure: f32, bloom: f32, bloom_threshold: f32, vignette: f32, saturation: f32) -> PostSettings {
    PostSettings { exposure, bloom, bloom_threshold, vignette, saturation, grain: 0.0, tonemap: Tonemap::Agx }
}

// --- palettes ------------------------------------------------------------------

/// Palettes made for this world's presets, offered before the shared ones.
/// Their top stops are kept for the densest cores (see `Params::palette_span`),
/// with a saturated colour just below, so veins read as colour, not white.
static OWN_PALETTES: [Palette; 6] = [
    Palette {
        name: "Ember Gold",
        stops: &[
            (0.0, 0x000004),
            (0.15, 0x240a4c),
            (0.3, 0x5c1468),
            (0.45, 0x9a2862),
            (0.6, 0xd24a3e),
            (0.75, 0xf2821a),
            (0.9, 0xffc46b),
            (1.0, 0xfff3d6),
        ],
    },
    Palette {
        name: "Frost",
        stops: &[(0.0, 0x01030a), (0.25, 0x0e1f45), (0.5, 0x2e5fa8), (0.72, 0x7fb8e6), (0.9, 0xbfe6ff), (1.0, 0xfff0d8)],
    },
    Palette {
        name: "Lichen",
        stops: &[(0.0, 0x030302), (0.25, 0x1a2414), (0.5, 0x4f6b35), (0.75, 0xc3cf8e), (1.0, 0xf4eedb)],
    },
    Palette {
        name: "Pearl",
        stops: &[(0.0, 0x07050c), (0.3, 0x2a2350), (0.55, 0x5f8fae), (0.75, 0xcdb8d6), (1.0, 0xfff6ec)],
    },
    Palette {
        name: "Axon",
        stops: &[(0.0, 0x000308), (0.2, 0x04213a), (0.42, 0x0a5c6e), (0.62, 0x1fb5a8), (0.82, 0x7fe3c9), (1.0, 0xffe2a8)],
    },
    Palette {
        name: "Nebula",
        stops: &[
            (0.0, 0x02010a),
            (0.2, 0x1a0b3d),
            (0.38, 0x3b1f8f),
            (0.55, 0x1f7bbf),
            (0.72, 0x21d4a7),
            (0.86, 0xc8f08a),
            (1.0, 0xffe9c4),
        ],
    },
];

fn palette_count() -> usize {
    OWN_PALETTES.len() + PALETTES.len()
}

/// Palette `i` of this world's list: its own palettes, then the shared ones.
fn palette_at(i: usize) -> &'static Palette {
    match OWN_PALETTES.get(i) {
        Some(p) => p,
        None => &PALETTES[(i - OWN_PALETTES.len()).min(PALETTES.len() - 1)],
    }
}

fn find_palette(name: &str) -> usize {
    (0..palette_count()).find(|&i| palette_at(i).name.eq_ignore_ascii_case(name)).unwrap_or(0)
}

/// The palette lookup texture read by `palette_lookup`: `palette::PaletteLut`
/// for this world's palette list (its own palettes plus the shared ones).
struct Lut {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    index: usize,
}

impl Lut {
    fn new(gpu: &Gpu, index: usize) -> Self {
        let (texture, view) = gpu.texture_2d(
            "physarum palette",
            [256, 1],
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let sampler = gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge);
        let mut lut = Self { texture, view, sampler, index: usize::MAX };
        lut.set(gpu, index);
        lut
    }

    fn palette(&self) -> &'static Palette {
        palette_at(self.index)
    }

    fn set(&mut self, gpu: &Gpu, index: usize) {
        let index = index.min(palette_count() - 1);
        if index == self.index {
            return;
        }
        self.index = index;
        gpu.queue.write_texture(
            self.texture.as_image_copy(),
            &palette_at(index).lut_rgba8(),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );
    }

    /// Palette picker with gradient previews.
    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) {
        let mut index = self.index;
        ui.push_id("physarum palette", |ui| {
            ui.spacing_mut().item_spacing.y = 4.0;
            crate::ui::dropdown(ui, "Palette", self.palette().name, |ui| {
                    for i in 0..palette_count() {
                        ui.horizontal(|ui| {
                            palette::swatch(ui, palette_at(i), egui::vec2(44.0, 12.0));
                            ui.selectable_value(&mut index, i, palette_at(i).name);
                        });
                    }
                });
            palette::swatch(ui, palette_at(index), egui::vec2(ui.available_width(), 5.0));
        });
        self.set(gpu, index);
    }
}

// --- parameters -------------------------------------------------------------------

/// Parameters that can change while the simulation runs.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Params {
    pub species: [Species; MAX_SPECIES],
    /// Row `s`: weight species `s` gives each species' trail (negative repels).
    pub interact: [[f32; MAX_SPECIES]; MAX_SPECIES],
    /// Fraction of the trail kept per sub-step.
    pub decay: f32,
    /// Blend between the trail (0) and its 3x3 mean (1) every sub-step.
    pub diffusion: f32,
    pub steps_per_frame: u32,
    /// Agents per cell beyond which extra agents add little trail (0 = linear).
    pub crowding: f32,
    /// Sensed trail saturates around this multiple of the mean level (0 = off).
    pub saturation: f32,
    /// Probability per sub-step that an agent is reborn at the layout.
    pub renewal: f32,
    /// Differential rotation about `centre` (degrees per sub-step at the core).
    pub swirl: f32,
    /// Centre of the swirl and of the centred layouts, in world uv.
    pub centre: [f32; 2],
    /// Attraction of every species to `centre`, relative to the mean trail level.
    pub gravity: f32,
    /// Terrain strength: the trail's decay rate varies by up to 2^(+-terrain)
    /// across the torus (0 = uniform).
    pub terrain: f32,
    /// Size of the terrain's features, as a fraction of the shorter side.
    pub terrain_scale: f32,
    /// Terrain drift in cells per sub-step.
    pub terrain_drift: f32,
    /// Fraction of the traffic image kept per sub-step (longer = longer trails).
    pub traffic_persistence: f32,
    /// Blur of the traffic image per sub-step.
    pub traffic_blur: f32,
    pub color_mode: ColorMode,
    /// Display gain: where density (relative to the mean) times `exposure`
    /// reaches 1, the tone curve's knee, veins start to glow.
    pub exposure: f32,
    /// Weight of the diffused trail (soft haze) in the image.
    pub trail_weight: f32,
    /// Weight of the traffic long exposure (crisp paths) in the image.
    pub traffic_weight: f32,
    pub brightness: f32,
    /// Decades of density below the knee that stay visible: more reveals
    /// fainter filigree (the log tone curve's black point is 10^-filigree).
    pub filigree: f32,
    /// Extra HDR emission of saturated veins.
    pub glow: f32,
    /// Tone at which the palette reaches its top stop (1 = at the knee; more
    /// keeps the lightest colour for the densest cores).
    pub palette_span: f32,
    /// 0 = bilinear, 1 = cubic B-spline reconstruction.
    pub smoothing: f32,
    /// Fraction of the frame the adaptive black point keeps dark (0 = off).
    pub ground: f32,
}

/// Settings that need a reseed (and possibly a reallocation). The UI edits a
/// pending copy that applies on Apply or reset, never mid-frame.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Population {
    pub agents: u32,
    pub species: usize,
    pub layout: Layout,
    /// Spawn radius as a fraction of half the shorter domain side.
    pub spawn_radius: f32,
    /// Rings or clusters for the layouts that use them.
    pub clusters: u32,
}

impl Population {
    fn sanitized(self, max_agents: u32) -> Self {
        Self {
            agents: self.agents.clamp(MIN_AGENTS, max_agents),
            species: self.species.clamp(1, MAX_SPECIES),
            layout: self.layout,
            spawn_radius: self.spawn_radius.clamp(0.02, 1.0),
            clusters: self.clusters.clamp(1, 64),
        }
    }
}

/// Where a preset puts `Params::centre`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Centre {
    Middle,
    /// On the left or right third line, picked by the seed.
    Thirds,
}

struct Preset {
    name: &'static str,
    /// One entry per live species (1..=4).
    species: &'static [Species],
    interact: [[f32; 4]; 4],
    decay: f32,
    diffusion: f32,
    crowding: f32,
    renewal: f32,
    swirl: f32,
    centre: Centre,
    gravity: f32,
    terrain: f32,
    terrain_scale: f32,
    steps: u32,
    /// Agents per trail cell.
    density: f32,
    layout: Layout,
    spawn_radius: f32,
    clusters: u32,
    palette: &'static str,
    palette_span: f32,
    color_mode: ColorMode,
    exposure: f32,
    filigree: f32,
    glow: f32,
    /// Weight of the diffused trail haze next to the crisp paths.
    haze: f32,
    /// Persistence of the display long exposure per sub-step (streak length).
    streaks: f32,
    /// Blur of the display long exposure per sub-step.
    softness: f32,
    post: PostSettings,
}

const PRESETS: &[Preset] = &[
    Preset {
        name: "Dendrites",
        // A soft crowding cap only on the densest veins lets arteries taper
        // and vary in brightness, while renewal keeps newcomers weaving
        // capillaries through the voids between them.
        species: &[sp(22.5, 12.0, 45.0, 1.2, 1.0, 1.0, 0.0, 0xffb347).with_spread(0.6)],
        interact: matrix(1.0, 0.0),
        decay: 0.97,
        diffusion: 1.0,
        crowding: 8.0,
        renewal: 0.002,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 2.2,
        terrain_scale: 0.6,
        steps: 3,
        density: 2.0,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Ember Gold",
        palette_span: 1.2,
        color_mode: ColorMode::Palette,
        exposure: 0.03,
        filigree: 3.0,
        glow: 1.0,
        haze: 0.1,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.45, 1.0, 0.55, 1.05),
    },
    Preset {
        name: "Neural Lace",
        // A mix of agent sizes weaves a fine and a medium lace at once; a
        // moderate crowding cap lets a sparse set of trunk strands form.
        species: &[sp(22.5, 9.0, 45.0, 1.0, 1.0, 0.0, 0.0, 0x9fe8ff).with_spread(1.0)],
        interact: matrix(1.0, 0.0),
        decay: 0.965,
        diffusion: 1.0,
        crowding: 5.0,
        renewal: 0.01,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 2.2,
        terrain_scale: 0.6,
        steps: 3,
        density: 1.5,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Frost",
        palette_span: 1.15,
        color_mode: ColorMode::Palette,
        exposure: 0.045,
        filigree: 3.0,
        glow: 1.8,
        haze: 0.1,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.45, 1.0, 0.6, 1.1),
    },
    Preset {
        name: "Mycelium",
        // Long sensors, strong wander and a very wide size mix: wandering
        // hyphae of every thickness, sheathed in a soft haze (little
        // diffusion keeps the sheath close to each strand).
        species: &[sp(30.0, 24.0, 25.0, 1.4, 1.0, 10.0, 0.0, 0xd8f5a2).with_spread(1.4)],
        interact: matrix(1.0, 0.0),
        decay: 0.93,
        diffusion: 0.5,
        crowding: 3.0,
        renewal: 0.003,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 1.6,
        terrain_scale: 0.6,
        steps: 3,
        density: 2.5,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Lichen",
        palette_span: 1.3,
        color_mode: ColorMode::Palette,
        exposure: 0.04,
        filigree: 3.0,
        glow: 0.8,
        haze: 0.3,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.4, 1.1, 0.55, 0.9),
    },
    Preset {
        name: "Rival Colonies",
        // Three species that dislike each other's trails, each born in its
        // own scattered colonies: irregular territories with contested borders.
        species: &[
            sp(22.5, 10.0, 40.0, 1.1, 1.0, 0.0, 0.0, 0xff7a5c).with_spread(0.4),
            sp(22.5, 10.0, 40.0, 1.1, 1.0, 0.0, 0.0, 0x2bb5a3).with_spread(0.4),
            sp(22.5, 10.0, 40.0, 1.1, 1.0, 0.0, 0.0, 0x8a7dff).with_spread(0.4),
        ],
        interact: matrix(1.0, -0.35),
        decay: 0.95,
        diffusion: 1.0,
        crowding: 0.0,
        renewal: 0.002,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 0.6,
        terrain_scale: 0.5,
        steps: 3,
        density: 2.0,
        layout: Layout::Clusters,
        spawn_radius: 1.0,
        clusters: 9,
        palette: "Coral",
        palette_span: 1.0,
        color_mode: ColorMode::Species,
        exposure: 0.03,
        filigree: 3.0,
        glow: 0.7,
        haze: 0.1,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.45, 1.0, 0.55, 0.95),
    },
    Preset {
        name: "Symbiosis",
        // Two mutually attracted species at very different scales: a fine
        // teal lace threaded along coarse rose arteries.
        species: &[
            sp(25.0, 5.0, 40.0, 1.0, 1.0, 0.5, 0.0, 0x2ee6c4).with_spread(0.5),
            sp(40.0, 24.0, 30.0, 1.0, 1.0, 0.0, 0.0, 0xff5fa2),
        ],
        interact: [[1.0, 0.5, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        decay: 0.95,
        diffusion: 1.0,
        crowding: 6.0,
        renewal: 0.003,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 1.0,
        terrain_scale: 0.5,
        steps: 4,
        density: 2.0,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Coral",
        palette_span: 1.0,
        color_mode: ColorMode::Species,
        exposure: 0.03,
        filigree: 3.0,
        glow: 0.8,
        haze: 0.1,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.45, 1.0, 0.55, 1.0),
    },
    Preset {
        name: "Honeycomb",
        // Wide sensor angle, small turns: the network settles into polygons,
        // with a crowding cap that lets some walls thicken into ribs.
        species: &[sp(70.0, 10.0, 30.0, 1.0, 1.0, 0.0, 0.0, 0xffc9a8)],
        interact: matrix(1.0, 0.0),
        decay: 0.92,
        diffusion: 1.0,
        crowding: 2.5,
        renewal: 0.0,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 1.2,
        terrain_scale: 0.5,
        steps: 3,
        density: 1.5,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Pearl",
        palette_span: 1.4,
        color_mode: ColorMode::Palette,
        exposure: 0.05,
        filigree: 2.0,
        glow: 1.0,
        haze: 0.0,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.45, 1.0, 0.55, 1.0),
    },
    Preset {
        name: "Synapses",
        // Very wide sensors, small turns and a gentle curl: meandering
        // neurites that knot into glowing synapses.
        species: &[sp(75.0, 8.0, 15.0, 1.0, 1.0, 0.0, 1.5, 0x8ff2d0)],
        interact: matrix(1.0, 0.0),
        decay: 0.9,
        diffusion: 1.0,
        crowding: 1.6,
        renewal: 0.002,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 1.0,
        terrain_scale: 0.5,
        steps: 3,
        density: 1.5,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Axon",
        palette_span: 1.2,
        color_mode: ColorMode::Palette,
        exposure: 0.04,
        filigree: 3.0,
        glow: 2.5,
        haze: 0.1,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.5, 1.3, 0.55, 1.0),
    },
    Preset {
        name: "Currents",
        // Four mutually repelling species of different reach squeeze each
        // other into lanes of different widths that meander like streamlines.
        species: &[
            sp(30.0, 8.0, 40.0, 1.2, 1.0, 0.0, 0.0, 0xfff0d6),
            sp(30.0, 11.0, 40.0, 1.2, 1.0, 0.0, 0.0, 0xff7a59),
            sp(30.0, 15.0, 40.0, 1.2, 1.0, 0.0, 0.0, 0x2aa7b8),
            sp(30.0, 20.0, 40.0, 1.2, 1.0, 0.0, 0.0, 0x3b2f86),
        ],
        interact: matrix(1.0, -1.0),
        decay: 0.95,
        diffusion: 1.0,
        crowding: 0.0,
        renewal: 0.002,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 0.8,
        terrain_scale: 0.5,
        steps: 3,
        density: 2.0,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Neon",
        palette_span: 1.0,
        color_mode: ColorMode::Species,
        // Lanes pack the whole frame, so a higher black point (fewer visible
        // decades) keeps the faint cross-traffic between them dark.
        exposure: 0.025,
        filigree: 1.7,
        glow: 1.2,
        haze: 0.0,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.45, 1.0, 0.55, 0.95),
    },
    Preset {
        name: "Chasing Waves",
        // Cyclic dominance (rock-paper-scissors): every species follows the
        // next one's trail and flees the previous one's. Far-sighted, narrow
        // sensors make each species stream after its prey in long braided
        // ribbons that sweep across the torus and never settle.
        species: &[
            sp(20.0, 32.0, 20.0, 2.0, 1.0, 0.0, 0.0, 0xff5a7a),
            sp(20.0, 32.0, 20.0, 2.0, 1.0, 0.0, 0.0, 0xffb84d),
            sp(20.0, 32.0, 20.0, 2.0, 1.0, 0.0, 0.0, 0x4aa8ff),
        ],
        interact: cyclic(0.6, -1.0),
        decay: 0.9,
        diffusion: 1.0,
        crowding: 2.0,
        renewal: 0.0,
        swirl: 0.0,
        centre: Centre::Middle,
        gravity: 0.0,
        terrain: 0.0,
        terrain_scale: 0.6,
        steps: 3,
        density: 2.0,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 6,
        palette: "Neon",
        palette_span: 1.0,
        color_mode: ColorMode::Species,
        exposure: 0.03,
        filigree: 2.4,
        glow: 1.0,
        haze: 0.0,
        streaks: 0.95,
        softness: 0.2,
        post: look(1.0, 0.45, 1.0, 0.55, 1.0),
    },
    Preset {
        name: "Galaxy",
        // A pull towards a centre on a third of the frame plus differential
        // rotation on a flat rotation curve: agents stream inwards and the
        // shear winds the living network into logarithmic arms around a
        // glowing core.
        species: &[sp(22.5, 9.0, 45.0, 1.2, 1.0, 0.0, 0.0, 0xb8f5ff).with_spread(0.4)],
        interact: matrix(1.0, 0.0),
        decay: 0.95,
        diffusion: 1.0,
        crowding: 6.0,
        renewal: 0.004,
        swirl: 0.6,
        centre: Centre::Thirds,
        gravity: 20.0,
        terrain: 0.0,
        terrain_scale: 0.5,
        steps: 3,
        density: 1.5,
        layout: Layout::Scatter,
        spawn_radius: 1.0,
        clusters: 2,
        palette: "Nebula",
        palette_span: 1.1,
        color_mode: ColorMode::Palette,
        exposure: 0.03,
        filigree: 3.0,
        glow: 2.2,
        haze: 0.1,
        streaks: 0.9,
        softness: 0.2,
        post: look(1.0, 0.5, 1.0, 0.5, 1.05),
    },
];

fn preset_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| PRESETS.iter().map(|p| p.name).collect())
}

/// Default species colours (used to pad presets with fewer species).
const SPECIES_COLORS: [u32; 4] = [0xfff0d6, 0xff7a59, 0x2aa7b8, 0x6a5acd];

/// Curated species colour sets for mutation, each laddered in value as well
/// as hue so that lanes and colonies separate even where hues are close.
const COLOR_SETS: &[[u32; 4]] = &[
    [0xfff0d6, 0xff7a59, 0x2aa7b8, 0x5b4fc4],
    [0xffb84d, 0xff5a7a, 0x4aa8ff, 0x8a7dff],
    [0xff7a5c, 0x2bb5a3, 0x8a7dff, 0xffe3b3],
    [0xf4eedb, 0xc3cf8e, 0x5f8fae, 0xb86b8f],
    [0xffd166, 0xef476f, 0x3fb8c9, 0x7058d8],
    [0xfff6ec, 0xe0799b, 0x5f8fae, 0x3e9b7a],
];

impl Params {
    fn from_preset(p: &Preset) -> Self {
        let mut species = [p.species[0]; MAX_SPECIES];
        for (s, slot) in species.iter_mut().enumerate() {
            match p.species.get(s) {
                Some(src) => *slot = *src,
                // Extra species (if the user raises the count) start as copies
                // of the last one in a fresh colour.
                None => {
                    *slot = p.species[p.species.len() - 1];
                    slot.color = rgb(SPECIES_COLORS[s]);
                }
            }
        }
        Self {
            species,
            interact: p.interact,
            decay: p.decay,
            diffusion: p.diffusion,
            steps_per_frame: p.steps,
            crowding: p.crowding,
            saturation: 0.0,
            renewal: p.renewal,
            swirl: p.swirl,
            centre: [0.5, 0.5],
            gravity: p.gravity,
            terrain: p.terrain,
            terrain_scale: p.terrain_scale,
            terrain_drift: 0.02,
            traffic_persistence: p.streaks,
            traffic_blur: p.softness,
            color_mode: p.color_mode,
            exposure: p.exposure,
            trail_weight: p.haze,
            traffic_weight: 1.0,
            brightness: 1.0,
            filigree: p.filigree,
            glow: p.glow,
            palette_span: p.palette_span,
            smoothing: 1.0,
            ground: 0.5,
        }
    }
}

/// A point on the left or right third line, picked by `seed`.
fn thirds_centre(seed: u64) -> [f32; 2] {
    let mut rng = Rng::new(seed ^ 0xCE47_5EED);
    let x = if rng.chance(0.5) { 0.36 } else { 0.64 };
    [x + rng.range(-0.03, 0.03), 0.5 + rng.range(-0.03, 0.03)]
}

// --- GPU mirrors --------------------------------------------------------------------

/// Mirrors `Sim` in physarum.wgsl (352 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimUniform {
    size: [u32; 2],
    agent_count: u32,
    species_count: u32,
    decay: f32,
    diffusion: f32,
    food: f32,
    pointer_mode: u32,
    pointer: [f32; 2],
    pointer_radius: f32,
    trail_cap: f32,
    motion: [[f32; 4]; 4],
    extra: [[f32; 4]; 4],
    interact: [[f32; 4]; 4],
    deposit: [f32; 4],
    /// Crowding (agents per cell), sensing saturation (trail units), renewal, swirl.
    tune: [f32; 4],
    /// Traffic persistence, blur, deposit per agent, -.
    traffic: [f32; 4],
    /// Swirl centre (cells), core radius, outer radius.
    swirl: [f32; 4],
    /// Terrain strength, drift offset (lattice units), -.
    terrain: [f32; 4],
    /// Terrain lattice cells across the domain, seed, -.
    terrain_cells: [u32; 4],
    /// Core pull (trail units), 1 / its radius^2, -, -.
    core: [f32; 4],
}

/// Mirrors `Seeding` in physarum.wgsl (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SeedUniform {
    size: [u32; 2],
    count: u32,
    pattern: u32,
    seed: u32,
    species_count: u32,
    radius: f32,
    clusters: u32,
    centre: [f32; 2],
    _pad: [f32; 2],
}

/// Mirrors `Draw` in physarum_draw.wgsl (160 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    view: ViewXform,
    size: [u32; 2],
    species_count: u32,
    mode: u32,
    trail_gain: [f32; 4],
    traffic_gain: [f32; 4],
    brightness: f32,
    black_point: f32,
    glow: f32,
    smoothing: f32,
    ground: f32,
    ground_rate: f32,
    frame: u32,
    palette_span: f32,
    colors: [[f32; 4]; 4],
}

/// Simulation resources shared by both ping-pong bind groups: everything but
/// the agent buffer, which is reallocated when the population changes.
struct SimResources {
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    seed_uniform: wgpu::Buffer,
    trail: [wgpu::TextureView; 2],
    traffic: [wgpu::TextureView; 2],
    counts: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl SimResources {
    /// `groups[i]` reads `trail[i]` / `traffic[i]` and writes the other pair.
    fn groups(&self, gpu: &Gpu, agents: &wgpu::Buffer) -> [wgpu::BindGroup; 2] {
        [0, 1].map(|i| {
            gpu.bind_group(
                "physarum sim",
                &self.layout,
                &[
                    self.uniform.as_entire_binding(),
                    agents.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&self.trail[i]),
                    wgpu::BindingResource::Sampler(&self.sampler),
                    self.counts.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&self.trail[1 - i]),
                    self.seed_uniform.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&self.traffic[i]),
                    wgpu::BindingResource::TextureView(&self.traffic[1 - i]),
                ],
            )
        })
    }
}

/// Where the terrain is: a seeded pattern that drifts in a seeded direction.
#[derive(Clone, Copy, Debug)]
struct TerrainState {
    seed: u32,
    /// Drift so far, in cells, kept within one period of the torus.
    offset: [f64; 2],
    /// Unit drift direction.
    dir: [f64; 2],
}

impl TerrainState {
    fn new(seed: u64, size: [u32; 2]) -> Self {
        let mut rng = Rng::new(seed ^ 0x7E44_A1D5);
        let seed = rng.next_u32();
        let offset = [f64::from(rng.f32()) * f64::from(size[0]), f64::from(rng.f32()) * f64::from(size[1])];
        let a = f64::from(rng.range(0.0, std::f32::consts::TAU));
        Self { seed, offset, dir: [a.cos(), a.sin()] }
    }
}

pub struct Physarum {
    size: [u32; 2],
    params: Params,
    /// Population the agents were last seeded with.
    population: Population,
    /// Population edited in the UI; applied by Apply or reset.
    pending: Population,
    preset: usize,
    post: PostSettings,
    seed: u64,
    max_agents: u32,
    lut: Lut,
    terrain: TerrainState,
    sim: SimResources,
    agents: wgpu::Buffer,
    /// Agents the current `agents` buffer can hold.
    agent_capacity: u32,
    /// `sim_groups[i]` reads the fields at index `i` and writes `1 - i`.
    sim_groups: [wgpu::BindGroup; 2],
    init_pipeline: wgpu::ComputePipeline,
    clear_pipeline: wgpu::ComputePipeline,
    agent_pipeline: wgpu::ComputePipeline,
    diffuse_pipeline: wgpu::ComputePipeline,
    draw_uniform: wgpu::Buffer,
    draw_pipeline: wgpu::RenderPipeline,
    /// `draw_groups[i]` displays the fields at index `i`.
    draw_groups: [wgpu::BindGroup; 2],
    /// Adaptive black point (`Ground` in physarum_draw.wgsl), measured every frame.
    ground: wgpu::Buffer,
    ground_pipeline: wgpu::ComputePipeline,
    /// `ground_groups[i]` measures the fields at index `i`.
    ground_groups: [wgpu::BindGroup; 2],
    /// Index of the fields holding the latest state.
    current: usize,
}

pub fn create(gpu: &Gpu, output_size: [u32; 2], seed: u64) -> Box<dyn World> {
    Box::new(Physarum::new(gpu, domain_size(gpu, output_size), seed))
}

/// Domain for an output size, within the device's texture and buffer limits.
fn domain_size(gpu: &Gpu, output_size: [u32; 2]) -> [u32; 2] {
    let limits = gpu.device.limits();
    let max_side = MAX_SIDE.min(limits.max_texture_dimension_2d).max(64);
    // One factor for both axes, so the domain keeps the output's aspect ratio.
    let longest = output_size[0].max(output_size[1]).max(1) as f32 * DOMAIN_SCALE;
    let scale = DOMAIN_SCALE * (max_side as f32 / longest).min(1.0);
    let mut size = output_size.map(|s| ((s as f32 * scale) as u32).clamp(64, max_side));
    // The deposit counters (16 bytes per cell) must fit in one storage binding.
    let max_cells = (u64::from(limits.max_storage_buffer_binding_size) / 16).min(limits.max_buffer_size / 16);
    while u64::from(size[0]) * u64::from(size[1]) > max_cells && size[0].min(size[1]) > 64 {
        size = size.map(|s| (s * 3 / 4).max(64));
    }
    size
}

fn agents_for_density(density: f32, size: [u32; 2], max_agents: u32) -> u32 {
    let n = f64::from(density.max(0.0)) * f64::from(size[0]) * f64::from(size[1]);
    (n.min(f64::from(max_agents)) as u32).max(MIN_AGENTS)
}

fn agent_buffer(gpu: &Gpu, count: u32) -> wgpu::Buffer {
    gpu.storage_buffer("physarum agents", u64::from(count) * 16, wgpu::BufferUsages::empty())
}

impl Physarum {
    pub fn new(gpu: &Gpu, size: [u32; 2], seed: u64) -> Self {
        // f32 trails keep single deposits resolvable on top of dense veins;
        // sampling them with filtering needs FLOAT32_FILTERABLE.
        let trail_format = if gpu.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
            wgpu::TextureFormat::Rgba32Float
        } else {
            wgpu::TextureFormat::Rgba16Float
        };
        let sim_source = include_str!("../shaders/physarum.wgsl")
            .replace("TRAIL_FORMAT", wgsl_format(trail_format))
            .replace("TRAFFIC_FORMAT", wgsl_format(TRAFFIC_FORMAT));
        let sim_module = gpu.shader("physarum sim", &sim_source);
        let draw_module = gpu.shader("physarum draw", include_str!("../shaders/physarum_draw.wgsl"));

        let limits = gpu.device.limits();
        let max_agents = MAX_AGENTS
            .min(limits.max_storage_buffer_binding_size / 16)
            .min((limits.max_buffer_size / 16).min(u64::from(u32::MAX)) as u32)
            .max(MIN_AGENTS);

        let field_usage = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING;
        let cs = ShaderStages::COMPUTE;
        let sim = SimResources {
            layout: gpu.bind_group_layout(
                "physarum sim",
                &[
                    layout::uniform(0, cs),
                    layout::storage(1, cs, false),
                    layout::texture(2, cs, true),
                    layout::sampler(3, cs, true),
                    layout::storage(4, cs, false),
                    layout::storage_texture(5, cs, trail_format, wgpu::StorageTextureAccess::WriteOnly),
                    layout::uniform(6, cs),
                    layout::texture(7, cs, true),
                    layout::storage_texture(8, cs, TRAFFIC_FORMAT, wgpu::StorageTextureAccess::WriteOnly),
                ],
            ),
            uniform: gpu.uniform_buffer("physarum sim", &SimUniform::zeroed()),
            seed_uniform: gpu.uniform_buffer("physarum seeding", &SeedUniform::zeroed()),
            trail: [0, 1].map(|_| gpu.texture_2d("physarum trail", size, trail_format, field_usage).1),
            traffic: [0, 1].map(|_| gpu.texture_2d("physarum traffic", size, TRAFFIC_FORMAT, field_usage).1),
            counts: gpu.storage_buffer(
                "physarum deposit counts",
                u64::from(size[0]) * u64::from(size[1]) * 16,
                wgpu::BufferUsages::empty(),
            ),
            sampler: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat),
        };
        let sim_pl = gpu.pipeline_layout("physarum sim", &[&sim.layout]);
        let init_pipeline = gpu.compute_pipeline("physarum init", &sim_pl, &sim_module, "cs_init");
        let clear_pipeline = gpu.compute_pipeline("physarum clear", &sim_pl, &sim_module, "cs_clear");
        let agent_pipeline = gpu.compute_pipeline("physarum agents", &sim_pl, &sim_module, "cs_agents");
        let diffuse_pipeline = gpu.compute_pipeline("physarum diffuse", &sim_pl, &sim_module, "cs_diffuse");

        let population = Population {
            agents: agents_for_density(PRESETS[0].density, size, max_agents),
            species: PRESETS[0].species.len(),
            layout: PRESETS[0].layout,
            spawn_radius: PRESETS[0].spawn_radius,
            clusters: PRESETS[0].clusters,
        };
        let agent_capacity = population.agents;
        let agents = agent_buffer(gpu, agent_capacity);
        let sim_groups = sim.groups(gpu, &agents);

        let fs = ShaderStages::FRAGMENT;
        let draw_layout = gpu.bind_group_layout(
            "physarum draw",
            &[
                layout::uniform(0, fs),
                layout::texture(1, fs, true),
                layout::sampler(2, fs, true),
                layout::texture(3, fs, true),
                layout::sampler(4, fs, true),
                layout::texture(5, fs, true),
                layout::storage(6, fs, true),
            ],
        );
        let draw_pipeline = gpu.fullscreen_pipeline(
            "physarum draw",
            &gpu.pipeline_layout("physarum draw", &[&draw_layout]),
            &draw_module,
            "fs_draw",
            SCENE_FORMAT,
            None,
        );
        let draw_uniform = gpu.uniform_buffer("physarum draw", &DrawUniform::zeroed());
        let ground = gpu.storage_buffer("physarum ground", 16, wgpu::BufferUsages::empty());
        let lut = Lut::new(gpu, find_palette(PRESETS[0].palette));
        let draw_groups = [0, 1].map(|i| {
            gpu.bind_group(
                "physarum draw",
                &draw_layout,
                &[
                    draw_uniform.as_entire_binding(),
                    wgpu::BindingResource::TextureView(&sim.trail[i]),
                    wgpu::BindingResource::Sampler(&sim.sampler),
                    wgpu::BindingResource::TextureView(&lut.view),
                    wgpu::BindingResource::Sampler(&lut.sampler),
                    wgpu::BindingResource::TextureView(&sim.traffic[i]),
                    ground.as_entire_binding(),
                ],
            )
        });

        // The adaptive black point pass shares the draw module and uniform but
        // only touches bindings 0, 1, 5 and 7, so its groups list them explicitly.
        let ground_layout = gpu.bind_group_layout(
            "physarum ground",
            &[
                layout::uniform(0, cs),
                layout::texture(1, cs, false),
                layout::texture(5, cs, false),
                layout::storage(7, cs, false),
            ],
        );
        let ground_pipeline = gpu.compute_pipeline(
            "physarum ground",
            &gpu.pipeline_layout("physarum ground", &[&ground_layout]),
            &draw_module,
            "cs_ground",
        );
        let ground_groups = [0, 1].map(|i| {
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("physarum ground"),
                layout: &ground_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: draw_uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&sim.trail[i]) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&sim.traffic[i]) },
                    wgpu::BindGroupEntry { binding: 7, resource: ground.as_entire_binding() },
                ],
            })
        });

        let mut world = Self {
            size,
            params: Params::from_preset(&PRESETS[0]),
            population,
            pending: population,
            preset: 0,
            post: PRESETS[0].post,
            seed,
            max_agents,
            lut,
            terrain: TerrainState::new(seed, size),
            sim,
            agents,
            agent_capacity,
            sim_groups,
            init_pipeline,
            clear_pipeline,
            agent_pipeline,
            diffuse_pipeline,
            draw_uniform,
            draw_pipeline,
            draw_groups,
            ground,
            ground_pipeline,
            ground_groups,
            current: 0,
        };
        world.load_preset(gpu, 0, seed);
        world
    }

    fn cells(&self) -> f32 {
        self.size[0] as f32 * self.size[1] as f32
    }

    fn min_side(&self) -> f32 {
        self.size[0].min(self.size[1]) as f32
    }

    /// Reallocates the agent buffer when the population outgrows it (or has
    /// shrunk a lot), rebuilding the bind groups that reference it.
    fn ensure_agent_capacity(&mut self, gpu: &Gpu) {
        let needed = self.population.agents;
        if needed > self.agent_capacity || needed < self.agent_capacity / 2 {
            self.agents = agent_buffer(gpu, needed);
            self.agent_capacity = needed;
            self.sim_groups = self.sim.groups(gpu, &self.agents);
        }
    }

    /// Mean steady-state trail level of every live species. Deposits are
    /// normalised so this is `deposit / species` whatever the density or decay.
    fn mean_levels(&self) -> [f32; 4] {
        let k = self.population.species;
        let mut levels = [0.0; 4];
        for (s, level) in levels.iter_mut().enumerate().take(k) {
            *level = self.params.species[s].sanitized().deposit / k as f32;
        }
        levels
    }

    /// Terrain lattice cells across each axis for the current feature size.
    fn terrain_cells(&self) -> [u32; 2] {
        let feature = self.params.terrain_scale.clamp(0.1, 2.0) * self.min_side();
        self.size.map(|s| ((s as f32 / feature).round() as u32).clamp(1, 64))
    }

    fn sim_uniform(&self, frame: &Frame) -> SimUniform {
        let p = &self.params;
        let k = self.population.species;
        let decay = p.decay.clamp(0.5, 0.999);
        // With `n` agents per cell depositing `d` each and a fraction `decay`
        // kept per sub-step, the mean trail settles at n*d*decay/(1-decay).
        // Scale deposits so it settles at d/k instead: display gains, food and
        // the cap then mean the same thing for every density and decay.
        let per_cell = self.population.agents as f32 / self.cells();
        let crowding = p.crowding.max(0.0);
        // Crowding acts on each species' own count, per_cell / k for an even
        // spread, whose deposit it cuts by 1 + count / crowding: compensate.
        let crowd_norm = if crowding > 0.0 { 1.0 + per_cell / (k as f32 * crowding) } else { 1.0 };
        let scale = (1.0 - decay) / (decay * per_cell.max(1e-6)) * crowd_norm;

        let mut motion = [[0.0; 4]; 4];
        let mut extra = [[0.0; 4]; 4];
        let mut interact = [[0.0; 4]; 4];
        let mut deposit = [0.0; 4];
        for s in 0..MAX_SPECIES {
            let sp = p.species[s].sanitized();
            motion[s] = [sp.sensor_angle.to_radians(), sp.sensor_distance, sp.turn_angle.to_radians(), sp.speed];
            extra[s] = [sp.wander.to_radians(), sp.curl.to_radians(), sp.spread, 0.0];
            for j in 0..k {
                interact[s][j] = p.interact[s][j].clamp(-2.0, 2.0);
            }
            if s < k {
                deposit[s] = sp.deposit * scale;
            }
        }
        let levels = self.mean_levels();
        let mean_level = levels.iter().sum::<f32>() / k as f32;
        // Traffic settles at a mean of 1/k per species for an even spread.
        let persistence = p.traffic_persistence.clamp(0.0, 0.995);

        let size = self.size.map(|s| s as f32);
        let (pointer, pointer_radius, pointer_mode) = match frame.pointer {
            Some(ptr) if ptr.primary || ptr.secondary => (
                [ptr.pos[0] * size[0], ptr.pos[1] * size[1]],
                // Below half the shorter side, so a push never jumps a period.
                ptr.radius.clamp(2.0, (0.45 * self.min_side()).max(2.0)),
                if ptr.primary { 1 } else { 2 },
            ),
            _ => ([0.0; 2], 2.0, 0),
        };
        let centre = p.centre.map(|c| c.clamp(0.0, 1.0));
        let terrain = p.terrain.clamp(0.0, 3.0);
        let cells = self.terrain_cells();
        let offset = [0, 1].map(|a| (self.terrain.offset[a] / f64::from(size[a]) * f64::from(cells[a])) as f32);
        SimUniform {
            size: self.size,
            agent_count: self.population.agents,
            species_count: k as u32,
            decay,
            diffusion: p.diffusion.clamp(0.0, 1.0),
            food: FOOD_GAIN * mean_level,
            pointer_mode,
            pointer,
            pointer_radius,
            trail_cap: TRAIL_CAP,
            motion,
            extra,
            interact,
            deposit,
            tune: [
                crowding,
                p.saturation.max(0.0) * mean_level,
                p.renewal.clamp(0.0, 0.05),
                p.swirl.clamp(-10.0, 10.0).to_radians(),
            ],
            traffic: [persistence, p.traffic_blur.clamp(0.0, 1.0), (1.0 - persistence) / per_cell.max(1e-6), 0.0],
            swirl: [
                (centre[0] * size[0]).min(size[0] - 0.5),
                (centre[1] * size[1]).min(size[1] - 0.5),
                0.06 * self.min_side(),
                0.49 * self.min_side(),
            ],
            terrain: [terrain, offset[0], offset[1], TERRAIN_SIZE * terrain],
            terrain_cells: [cells[0], cells[1], self.terrain.seed, 0],
            core: [p.gravity.clamp(0.0, 40.0) * mean_level, 1.0 / (CORE_RADIUS * self.min_side()).powi(2), 0.0, 0.0],
        }
    }

    fn draw_uniform(&self, frame: &Frame) -> DrawUniform {
        let p = &self.params;
        let k = self.population.species;
        let levels = self.mean_levels();
        let exposure = p.exposure.clamp(0.002, 2.0);
        // Log tone curve: the knee sits at display density 1, black at 10^-filigree.
        let black_point = 10f32.powf(-p.filigree.clamp(1.0, 6.0));
        let trail_weight = p.trail_weight.clamp(0.0, 4.0);
        let traffic_weight = p.traffic_weight.clamp(0.0, 4.0);
        let mut trail_gain = [0.0; 4];
        let mut traffic_gain = [0.0; 4];
        for s in 0..k {
            trail_gain[s] = exposure * trail_weight / levels[s].max(1e-6);
            traffic_gain[s] = exposure * traffic_weight * k as f32;
        }
        DrawUniform {
            view: frame.view,
            size: self.size,
            species_count: k as u32,
            mode: match p.color_mode {
                ColorMode::Palette => 0,
                ColorMode::Species => 1,
            },
            trail_gain,
            traffic_gain,
            brightness: p.brightness.clamp(0.0, 8.0),
            black_point,
            glow: p.glow.clamp(0.0, 8.0),
            smoothing: p.smoothing.clamp(0.0, 1.0),
            ground: p.ground.clamp(0.0, 0.9),
            // The adaptive black point glides with a fixed time constant,
            // whatever the display rate.
            ground_rate: 1.0 - (-frame.dt.max(1e-4) / GROUND_SECONDS).exp(),
            // Wrapping is fine: only the sample jitter depends on it.
            frame: frame.frame as u32,
            palette_span: p.palette_span.clamp(1.0, 3.0),
            colors: p.species.map(|s| s.linear_color()),
        }
    }

    /// Spreads the colours of `k` species over the bright half of the current palette.
    fn colors_from_palette(&mut self, k: usize) {
        let palette = self.lut.palette();
        for s in 0..MAX_SPECIES {
            let t = if k <= 1 { 0.75 } else { 0.5 + 0.42 * (s.min(k - 1) as f32 / (k - 1) as f32) };
            self.params.species[s].color = palette.sample_srgb8(t);
        }
    }
}

impl World for Physarum {
    fn settings(&self) -> anyhow::Result<crate::library::WorldSettings> {
        Ok(crate::library::WorldSettings::Physarum {
            params: self.params, population: self.pending, palette: self.lut.palette().name.to_owned(), post: self.post,
        })
    }

    fn restore_settings(&mut self, gpu: &Gpu, settings: &crate::library::WorldSettings, seed: u64) -> anyhow::Result<()> {
        let crate::library::WorldSettings::Physarum { params, population, palette: name, post } = settings else { anyhow::bail!("Wrong world settings"); };
        let index = (0..palette_count()).find(|&i| palette_at(i).name == name).ok_or_else(|| anyhow::anyhow!("Unknown palette: {name}"))?;
        anyhow::ensure!((1..=128).contains(&params.steps_per_frame), "Invalid step count");
        self.params = *params;
        self.pending = *population;
        self.post = *post;
        self.lut.set(gpu, index);
        self.reset(gpu, seed);
        Ok(())
    }

    fn id(&self) -> &'static str {
        "physarum"
    }

    fn name(&self) -> &'static str {
        "Physarum"
    }

    fn size(&self) -> [u32; 2] {
        self.size
    }

    fn presets(&self) -> &'static [&'static str] {
        preset_names()
    }

    fn preset(&self) -> usize {
        self.preset
    }

    fn load_preset(&mut self, gpu: &Gpu, index: usize, seed: u64) {
        let index = index.min(PRESETS.len() - 1);
        let preset = &PRESETS[index];
        self.preset = index;
        self.params = Params::from_preset(preset);
        self.params.centre = match preset.centre {
            Centre::Middle => [0.5, 0.5],
            Centre::Thirds => thirds_centre(seed),
        };
        self.pending = Population {
            agents: agents_for_density(preset.density, self.size, self.max_agents),
            species: preset.species.len(),
            layout: preset.layout,
            spawn_radius: preset.spawn_radius,
            clusters: preset.clusters,
        };
        self.post = preset.post;
        self.lut.set(gpu, find_palette(preset.palette));
        self.reset(gpu, seed);
    }

    fn reset(&mut self, gpu: &Gpu, seed: u64) {
        self.seed = seed;
        self.pending = self.pending.sanitized(self.max_agents);
        self.population = self.pending;
        self.ensure_agent_capacity(gpu);
        self.terrain = TerrainState::new(seed, self.size);

        let pop = self.population;
        let centre = self.params.centre.map(|c| c.clamp(0.0, 1.0));
        gpu.write(
            &self.sim.seed_uniform,
            &SeedUniform {
                size: self.size,
                count: pop.agents,
                pattern: pop.layout.code(),
                seed: Rng::new(seed).next_u32(),
                species_count: pop.species as u32,
                radius: pop.spawn_radius * 0.5 * self.min_side(),
                clusters: pop.clusters,
                centre: [centre[0] * self.size[0] as f32, centre[1] * self.size[1] as f32],
                _pad: [0.0; 2],
            },
        );
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("physarum reset") });
        encoder.clear_buffer(&self.sim.counts, 0, None);
        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("physarum seed"), timestamp_writes: None });
            pass.set_pipeline(&self.clear_pipeline);
            for group in &self.sim_groups {
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(self.size[0].div_ceil(CELL_WG), self.size[1].div_ceil(CELL_WG), 1);
            }
            let (gx, gy) = gpu::dispatch_linear(pop.agents, AGENT_WG);
            pass.set_pipeline(&self.init_pipeline);
            pass.set_bind_group(0, &self.sim_groups[0], &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        // Forget the adaptive black point: the next frame measures afresh.
        gpu.queue.write_buffer(&self.ground, 0, &[0; 16]);
        gpu.queue.submit([encoder.finish()]);
        self.current = 0;
    }

    fn mutate(&mut self, gpu: &Gpu, seed: u64) {
        let mut rng = Rng::new(seed ^ 0x5117_AB1E_D00D);
        let k = *rng.pick(&[1usize, 2, 2, 3, 3, 4, 4]);

        // A shared base behaviour, varied per species.
        let sensor_angle = rng.range(15.0, 60.0);
        let sensor_distance = rng.range(5.0, 26.0);
        let turn_angle = (sensor_angle * rng.range(0.7, 1.6)).clamp(10.0, 75.0);
        let speed = rng.range(0.8, 1.8);
        let curl = if rng.chance(0.2) { rng.range(0.5, 2.0) * if rng.chance(0.5) { 1.0 } else { -1.0 } } else { 0.0 };
        let colors = *rng.pick(COLOR_SETS);
        for (s, color) in colors.iter().enumerate().take(MAX_SPECIES) {
            let v = |rng: &mut Rng| rng.range(0.75, 1.3);
            self.params.species[s] = Species {
                sensor_angle: (sensor_angle * v(&mut rng)).clamp(10.0, 90.0),
                sensor_distance: (sensor_distance * v(&mut rng)).clamp(3.0, 36.0),
                turn_angle: (turn_angle * v(&mut rng)).clamp(10.0, 80.0),
                speed: (speed * v(&mut rng)).clamp(0.6, 2.4),
                deposit: rng.range(0.7, 1.4),
                wander: if rng.chance(0.3) { rng.range(1.0, 6.0) } else { 0.0 },
                curl,
                spread: if rng.chance(0.6) { rng.range(0.3, 1.2) } else { 0.0 },
                color: rgb(*color),
            };
        }

        // Interaction archetype: independent, rivals, symbiosis or a cyclic chase.
        let archetype = rng.below(4);
        for s in 0..MAX_SPECIES {
            for j in 0..MAX_SPECIES {
                self.params.interact[s][j] = if s == j {
                    rng.range(0.85, 1.2)
                } else {
                    match archetype {
                        0 => rng.range(-0.3, 0.3),
                        1 => rng.range(-1.0, -0.4),
                        2 => rng.range(0.3, 0.8),
                        _ => {
                            if (s + 1) % k == j {
                                rng.range(0.6, 1.1)
                            } else {
                                rng.range(-1.1, -0.5)
                            }
                        }
                    }
                };
            }
        }

        self.params.diffusion = rng.range(0.6, 1.0);
        // Two regimes, both living networks: a loose crowding cap with a long
        // memory (tapering arteries fed by capillaries), or a firm cap (an even
        // lace). Renewal keeps both reorganising instead of coarsening into
        // emptiness.
        if rng.chance(0.5) {
            self.params.crowding = rng.range(5.0, 8.0);
            self.params.decay = rng.range(0.93, 0.97);
        } else {
            self.params.crowding = rng.range(0.7, 2.0);
            self.params.decay = rng.range(0.88, 0.95);
        }
        self.params.renewal = rng.range(0.0015, 0.006);
        // Now and then a galaxy, on a third of the frame.
        let galaxy = rng.chance(0.15);
        self.params.swirl = if galaxy { rng.range(0.25, 0.6) * if rng.chance(0.5) { 1.0 } else { -1.0 } } else { 0.0 };
        self.params.centre = if galaxy { thirds_centre(seed) } else { [0.5, 0.5] };
        self.params.gravity = if galaxy { rng.range(4.0, 10.0) } else { 0.0 };
        // Usually a terrain, so the network has dense and quiet regions.
        self.params.terrain = if rng.chance(0.75) { rng.range(0.7, 1.4) } else { 0.0 };
        self.params.terrain_scale = rng.range(0.35, 0.7);
        self.params.terrain_drift = rng.range(0.01, 0.04);
        self.params.saturation = 0.0;
        self.params.steps_per_frame = 3;
        self.params.exposure = rng.range(0.022, 0.036);
        self.params.trail_weight = rng.range(0.0, 0.1);
        self.params.traffic_weight = 1.0;
        self.params.traffic_persistence = rng.range(0.88, 0.94);
        self.params.traffic_blur = 0.2;
        // Fewer visible decades than the hand-tuned presets: some mutations
        // pack lanes over the whole frame, and this keeps their ground dark.
        self.params.filigree = rng.range(2.3, 2.9);
        self.params.glow = rng.range(0.8, 1.4);
        self.params.palette_span = rng.range(1.0, 1.2);
        self.params.brightness = 1.0;
        self.params.smoothing = 1.0;
        self.params.ground = 0.5;
        self.params.color_mode = if k == 1 || rng.chance(0.2) { ColorMode::Palette } else { ColorMode::Species };

        // Always a uniform scatter: compact layouts (disks, rings, clusters,
        // the vortex) can stay a lone shape in a black frame, and species
        // bands read as a flag.
        self.pending = Population {
            agents: agents_for_density(rng.range(1.5, 2.5), self.size, self.max_agents),
            species: k,
            layout: Layout::Scatter,
            spawn_radius: rng.range(0.9, 1.0),
            clusters: 4 + rng.below(8),
        };
        // Skip the light "Ink" palette: mutations stay luminous on black.
        let dark: Vec<usize> = (0..palette_count()).filter(|&i| palette_at(i).name != "Ink").collect();
        self.lut.set(gpu, *rng.pick(&dark));
        if rng.chance(0.3) {
            // The new species count, not the one still seeded.
            self.colors_from_palette(k);
        }
        self.post = look(1.0, rng.range(0.35, 0.55), rng.range(0.95, 1.1), rng.range(0.45, 0.6), rng.range(0.9, 1.05));
        self.reset(gpu, seed);
    }

    fn step(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder) {
        frame.gpu.write(&self.sim.uniform, &self.sim_uniform(frame));
        let steps = self.params.steps_per_frame.clamp(1, MAX_STEPS);
        let (gx, gy) = gpu::dispatch_linear(self.population.agents, AGENT_WG);
        let cells = [self.size[0].div_ceil(CELL_WG), self.size[1].div_ceil(CELL_WG)];
        {
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("physarum step"), timestamp_writes: None });
            for _ in 0..steps {
                pass.set_pipeline(&self.agent_pipeline);
                pass.set_bind_group(0, &self.sim_groups[self.current], &[]);
                pass.dispatch_workgroups(gx, gy, 1);
                pass.set_pipeline(&self.diffuse_pipeline);
                pass.dispatch_workgroups(cells[0], cells[1], 1);
                self.current = 1 - self.current;
            }
        }
        // The terrain drifts with simulated time, so it is deterministic too.
        let drift = f64::from(self.params.terrain_drift.clamp(0.0, 1.0)) * f64::from(steps);
        for a in 0..2 {
            let period = f64::from(self.size[a]);
            self.terrain.offset[a] = (self.terrain.offset[a] + drift * self.terrain.dir[a]).rem_euclid(period);
        }
    }

    fn render(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        frame.gpu.write(&self.draw_uniform, &self.draw_uniform(frame));
        {
            // One workgroup measures the frame and updates the black point.
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("physarum ground"), timestamp_writes: None });
            pass.set_pipeline(&self.ground_pipeline);
            pass.set_bind_group(0, &self.ground_groups[self.current], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu::fullscreen_pass(
            encoder,
            "physarum draw",
            target,
            Some(wgpu::Color::BLACK),
            &self.draw_pipeline,
            &[&self.draw_groups[self.current]],
        );
    }

    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) {
        // --- population: applies on Apply / reset ------------------------------
        ui.label(egui::RichText::new("Population").strong());
        let max_millions = self.max_agents as f32 / 1e6;
        let mut millions = self.pending.agents as f32 / 1e6;
        let slider = crate::ui::Slider::new(&mut millions, 0.01..=max_millions).logarithmic(true).text("Agents (M)");
        if ui.add(slider.fixed_decimals(2)).changed() {
            self.pending.agents = ((millions * 1e6) as u32).clamp(MIN_AGENTS, self.max_agents);
        }
        ui.add(crate::ui::Slider::new(&mut self.pending.species, 1..=MAX_SPECIES).text("Species"));
        crate::ui::dropdown(ui, "Layout", self.pending.layout.name(), |ui| {
            for l in Layout::ALL {
                ui.selectable_value(&mut self.pending.layout, l, l.name());
            }
        });
        ui.add(crate::ui::Slider::new(&mut self.pending.spawn_radius, 0.05..=1.0).text("Spawn radius"));
        ui.add(crate::ui::Slider::new(&mut self.pending.clusters, 1..=24).text("Rings / clusters"));
        let dirty = self.pending != self.population;
        let mut apply = false;
        ui.horizontal(|ui| {
            apply = ui.add_enabled(dirty, egui::Button::new("Apply")).on_hover_text("Reseed with these settings").clicked();
            if dirty {
                ui.label(egui::RichText::new("pending: applies on Apply or reset").weak().italics());
            }
        });
        if apply {
            self.reset(gpu, self.seed);
        }

        // --- global trail parameters ------------------------------------------------
        ui.separator();
        let p = &mut self.params;
        ui.add(crate::ui::Slider::new(&mut p.decay, 0.5..=0.995).text("Trail persistence"));
        ui.add(crate::ui::Slider::new(&mut p.diffusion, 0.0..=1.0).text("Diffusion"));
        ui.add(crate::ui::Slider::new(&mut p.crowding, 0.0..=8.0).text("Crowding limit (0 = off)"));
        ui.add(crate::ui::Slider::new(&mut p.saturation, 0.0..=20.0).text("Sensing saturation (0 = off)"));
        ui.add(crate::ui::Slider::new(&mut p.renewal, 0.0..=0.02).text("Renewal / step"));
        ui.add(crate::ui::Slider::new(&mut p.steps_per_frame, 1..=MAX_STEPS).text("Steps / frame"));
        ui.add(crate::ui::Slider::new(&mut p.terrain, 0.0..=3.0).text("Terrain"))
            .on_hover_text("How much the trail's decay varies over the torus: dense regions and quiet voids");
        ui.add(crate::ui::Slider::new(&mut p.terrain_scale, 0.15..=1.5).text("Terrain scale"));
        ui.add(crate::ui::Slider::new(&mut p.terrain_drift, 0.0..=0.2).text("Terrain drift"));
        ui.add(crate::ui::Slider::new(&mut p.swirl, -3.0..=3.0).suffix("°").text("Swirl / step"));
        ui.add(crate::ui::Slider::new(&mut p.gravity, 0.0..=30.0).text("Core pull"))
            .on_hover_text("Attraction of every species to the centre");
        ui.horizontal(|ui| {
            ui.label("Centre");
            ui.add(egui::DragValue::new(&mut p.centre[0]).range(0.0..=1.0).speed(0.005).prefix("x "));
            ui.add(egui::DragValue::new(&mut p.centre[1]).range(0.0..=1.0).speed(0.005).prefix("y "));
        })
        .response
        .on_hover_text("Centre of the swirl (and of the centred layouts, on reset)");

        // --- species ------------------------------------------------------------------
        let k = self.population.species;
        let Params { species, interact, .. } = &mut self.params;
        for s in 0..k {
            let sp = &mut species[s];
            egui::CollapsingHeader::new(format!("Species {}", s + 1))
                .id_salt(("physarum species", s))
                .default_open(s == 0)
                .show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label("Colour");
                        ui.color_edit_button_srgb(&mut sp.color);
                    });
                    ui.add(crate::ui::Slider::new(&mut sp.sensor_angle, 2.0..=150.0).suffix("°").text("Sensor angle"));
                    ui.add(crate::ui::Slider::new(&mut sp.sensor_distance, 1.0..=64.0).text("Sensor distance"));
                    ui.add(crate::ui::Slider::new(&mut sp.turn_angle, 1.0..=150.0).suffix("°").text("Turn angle"));
                    ui.add(crate::ui::Slider::new(&mut sp.speed, 0.1..=4.0).text("Speed"));
                    ui.add(crate::ui::Slider::new(&mut sp.deposit, 0.05..=5.0).logarithmic(true).text("Deposit"));
                    ui.add(crate::ui::Slider::new(&mut sp.wander, 0.0..=45.0).suffix("°").text("Wander"));
                    ui.add(crate::ui::Slider::new(&mut sp.curl, -10.0..=10.0).suffix("°").text("Curl"));
                    ui.add(crate::ui::Slider::new(&mut sp.spread, 0.0..=2.0).text("Size variety"));
                    if k > 1 {
                        ui.label(egui::RichText::new("Attraction to trails (negative repels)").weak());
                        for (j, w) in interact[s].iter_mut().enumerate().take(k) {
                            let label = if j == s { "own".to_string() } else { format!("species {}", j + 1) };
                            ui.add(crate::ui::Slider::new(w, -1.5..=1.5).text(label));
                        }
                    }
                });
        }

        // --- colour ---------------------------------------------------------------------
        ui.separator();
        let mode = &mut self.params.color_mode;
        crate::ui::dropdown(ui, "Colouring", mode.name(), |ui| {
            for m in [ColorMode::Palette, ColorMode::Species] {
                ui.selectable_value(mode, m, m.name());
            }
        });
        self.lut.ui(gpu, ui);
        if ui.button("Species colours from palette").clicked() {
            self.colors_from_palette(self.population.species);
        }
        let p = &mut self.params;
        ui.add(crate::ui::Slider::new(&mut p.exposure, 0.005..=0.5).logarithmic(true).text("Density gain"));
        ui.add(crate::ui::Slider::new(&mut p.traffic_weight, 0.0..=3.0).text("Path weight"));
        ui.add(crate::ui::Slider::new(&mut p.trail_weight, 0.0..=1.0).text("Haze weight"));
        ui.add(crate::ui::Slider::new(&mut p.traffic_persistence, 0.5..=0.99).text("Path persistence"));
        ui.add(crate::ui::Slider::new(&mut p.traffic_blur, 0.0..=1.0).text("Path softness"));
        ui.add(crate::ui::Slider::new(&mut p.brightness, 0.1..=3.0).text("Brightness"));
        ui.add(crate::ui::Slider::new(&mut p.filigree, 1.0..=6.0).text("Filigree (decades)"));
        ui.add(crate::ui::Slider::new(&mut p.glow, 0.0..=4.0).text("Vein glow"));
        ui.add(crate::ui::Slider::new(&mut p.palette_span, 1.0..=2.0).text("Palette span"))
            .on_hover_text("Tone that reaches the palette's top colour: higher keeps it for the densest cores");
        ui.add(crate::ui::Slider::new(&mut p.smoothing, 0.0..=1.0).text("Smoothing"));
        ui.add(crate::ui::Slider::new(&mut p.ground, 0.0..=0.9).text("Dark ground"))
            .on_hover_text("Fraction of the frame kept dark by raising the black point automatically (0 = off)");
    }

    fn post_settings(&self) -> PostSettings {
        self.post
    }

    fn stats(&self) -> String {
        format!(
            "{} agents · {}x{} cells · {} species",
            group_digits(self.population.agents),
            self.size[0],
            self.size[1],
            self.population.species
        )
    }

    fn controls_hint(&self) -> &'static str {
        "Left: drop food (agents gather into a hub) · Right: push agents away & erase trails"
    }
}

/// `4194304` -> `"4,194,304"`.
fn group_digits(n: u32) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}
