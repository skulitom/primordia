//! Lenia (Bert Chan): continuous cellular automata whose smooth ring kernels
//! and Gaussian growth grow soft, gliding organisms.
//!
//! This is the multi-channel, multi-kernel ("expanded") form:
//!
//! ```text
//! U_k = K_k * A_src(k)                                  normalised ring kernel
//! G_k = 2 exp(-(U_k - mu_k)^2 / (2 sigma_k^2)) - 1      growth
//! A_c <- clip(A_c + dt * sum_{k->c} h_k G_k / sum_{k->c} h_k, 0, 1),  dt = 1/T
//! ```
//!
//! Convolution is direct. The CPU bakes every kernel into a table of weights,
//! trimmed row by row so empty taps are skipped, and kernels that share a
//! source channel are packed four to a vec4 so that one shared-memory pass
//! evaluates four kernels at once (see `lenia.wgsl`).
//!
//! Rendering maps each channel through its own palette. Bodies use the
//! palette's middle so they keep their hue; only dense nuclei reach its light
//! end and go into HDR, so the bloom picks out the cores alone. Channels mix
//! by their share of the local density (raised to a per-preset power), the
//! growth field adds a faint halo, every body leaves one soft, diffusing wake,
//! and the dark medium is tinted by a coarse field of nearby life that
//! follows the creatures with a lag.
//!
//! Species that only rarely form from random noise (Orbium) are raised in a
//! "nursery" of many small isolated tori, and only the patches that became
//! steady creatures are released into the world. Mutation nudges the kernels
//! of a curated species and screens the result with a short blocking trial
//! run, so it stays a living relative of that species.

use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use super::{Frame, ViewXform, World};
use crate::gpu::{self, layout, Gpu, SCENE_FORMAT};
use crate::palette::{self, PaletteLut, PALETTES};
use crate::post::{PostSettings, Tonemap};
use crate::rng::Rng;

/// State channels held by the GPU buffers (a parameter set may use fewer).
const CHANNELS: usize = 3;
/// Most kernels a parameter set may have.
const MAX_KERNELS: usize = 16;
/// Kernels evaluated together by one convolution pass (one vec4 of weights per tap).
const SLOTS: usize = 4;
/// Worst case of `sum_c ceil(n_c / SLOTS)` for `MAX_KERNELS` kernels over 3 channels.
const MAX_GROUPS: usize = 6;
/// Output cells per convolution workgroup along each axis (8x32 threads, 4 cells each).
const TILE: u32 = 32;
/// Groups whose halo fits this use the smallest shared tile (best occupancy).
const SMALL_HALO: u32 = 16;
/// Groups whose halo fits this use the middle tile (species with long
/// cross-kernels); larger halos use the tile sized for `r_max`.
const MEDIUM_HALO: u32 = 26;
/// Hard cap on the kernel radius in cells, whatever the device allows.
const RADIUS_CAP: u32 = 36;
/// The domain is half the output resolution, capped at this many cells.
const MAX_CELLS: f32 = 640_000.0;
/// Fixed-point scale of the GPU mass counter (see `cs_compose`).
const MASS_SCALE: f32 = 256.0;
/// A world whose mean density falls below this counts as extinct (revival).
const EXTINCT_LEVEL: f32 = 0.0004;
/// Revival drops a new patch every this many frames while the world is extinct.
const REVIVE_EVERY: u64 = 20;
/// Nudges a mutation tries, each 40% smaller than the last, before keeping
/// the species' own kernels.
const MUTATE_ATTEMPTS: usize = 6;
/// The nursery packs as many tori as fit a square of this many cells.
const NURSERY_SPAN: usize = 1024;
/// Hatchlings kept per run (for the release, the brush and respawning).
const MAX_BROOD: usize = 16;

// --- parameters ---------------------------------------------------------------

/// One kernel: reads `source`, drives the growth of `target`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Kernel {
    pub source: usize,
    pub target: usize,
    /// Radius in cells (before the global scale).
    pub radius: f32,
    /// Number of concentric rings (1..=3) and their peak heights.
    pub rings: usize,
    pub b: [f32; 3],
    /// Growth centre and width.
    pub mu: f32,
    pub sigma: f32,
    /// Weight among the kernels feeding `target`.
    pub h: f32,
}

impl Kernel {
    fn new(source: usize, target: usize, radius: f32, b: &[f32], mu: f32, sigma: f32, h: f32) -> Self {
        let rings = b.len().clamp(1, 3);
        let mut peaks = [0.0; 3];
        peaks[..rings].copy_from_slice(&b[..rings]);
        Self { source, target, radius, rings, b: peaks, mu, sigma, h }
    }

    fn rings(&self) -> usize {
        self.rings.clamp(1, 3)
    }

    /// Kernel shell at distance `d` for an effective radius `radius`: ring `i`
    /// of `rings` is the smooth bump exp(4 - 1 / (x (1 - x))) scaled by `b[i]`.
    fn shell(&self, d: f32, radius: f32) -> f32 {
        let r = d / radius;
        if r >= 1.0 {
            return 0.0;
        }
        let rings = self.rings();
        let br = r * rings as f32;
        let i = (br as usize).min(rings - 1);
        let x = br - i as f32;
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        self.b[i].max(0.0) * (4.0 - 1.0 / (x * (1.0 - x))).exp()
    }

    fn sigma(&self) -> f32 {
        self.sigma.max(0.002)
    }

    /// Growth in empty space, G(0).
    fn rest_growth(&self) -> f32 {
        let s = self.sigma();
        2.0 * (-(self.mu * self.mu) / (2.0 * s * s)).exp() - 1.0
    }
}

fn k(source: usize, target: usize, radius: f32, b: &[f32], mu: f32, sigma: f32, h: f32) -> Kernel {
    Kernel::new(source, target, radius, b, mu, sigma, h)
}

/// mu / sigma at which the growth function vanishes in empty space:
/// 2 exp(-mu^2 / (2 sigma^2)) - 1 = 0  <=>  mu = sigma sqrt(2 ln 2).
const NEUTRAL_MU: f32 = 1.177_41;

/// A cross-species kernel that is silent while its source channel is absent
/// (G(0) = 0), so it leaves each species' own dynamics untouched until another
/// species comes close: a faint presence (u < 2 mu) feeds the target, a dense
/// one (u > 2 mu) inhibits it. Small sigma means "keep away", large "feed on".
fn neutral(source: usize, target: usize, radius: f32, sigma: f32, h: f32) -> Kernel {
    k(source, target, radius, &[1.0], sigma * NEUTRAL_MU, sigma, h)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Seeding {
    /// Dozens of kernel-sized discs of random noise (Lenia's classic soup).
    Patches,
    /// Noise over the whole torus, broken into soft clouds.
    Soup,
    /// A few large noisy discs.
    Sparse,
    /// Smooth lopsided domes with noise: random, but glider-like in profile.
    Blobs,
    /// Random patches hatched in isolated nursery tori; only the ones that
    /// became creatures are released (for species that rarely form in a
    /// crowded soup, or whose soup can explode).
    Nursery,
}

impl Seeding {
    const ALL: [Seeding; 5] = [Seeding::Patches, Seeding::Soup, Seeding::Sparse, Seeding::Blobs, Seeding::Nursery];

    fn name(self) -> &'static str {
        match self {
            Seeding::Patches => "Random patches",
            Seeding::Soup => "Primordial soup",
            Seeding::Sparse => "Sparse seeds",
            Seeding::Blobs => "Lopsided blobs",
            Seeding::Nursery => "Nursery hatchlings",
        }
    }
}

/// A known creature, recorded at kernel radius `radius` (one channel).
#[derive(Debug, PartialEq)]
pub struct Template {
    pub side: usize,
    pub radius: f32,
    /// `side * side` cells, row-major.
    pub cells: &'static [f32],
}

/// Orbium unicaudatus (R = 13, T = 10, mu = 0.15, sigma = 0.015), the pattern
/// published with Bert Chan's Lenia (MIT licence). Random soups at these
/// parameters almost never find it: they die out or boil into labyrinths.
const ORBIUM: Template = Template { side: 20, radius: 13.0, cells: &ORBIUM_CELLS };

// Store a stable template name rather than a pointer or hundreds of fixed cells.
mod template_id {
    use super::{Template, ORBIUM};
    use serde::{Deserialize, Serialize};

    pub fn serialize<S: serde::Serializer>(template: &Option<&Template>, serializer: S) -> Result<S::Ok, S::Error> {
        template.map(|_| "Orbium").serialize(serializer)
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Option<&'static Template>, D::Error> {
        match Option::<String>::deserialize(deserializer)?.as_deref() {
            None => Ok(None),
            Some("Orbium") => Ok(Some(&ORBIUM)),
            Some(name) => Err(serde::de::Error::custom(format!("Unknown creature template: {name}"))),
        }
    }
}

#[rustfmt::skip]
const ORBIUM_CELLS: [f32; 400] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.14, 0.1, 0.0, 0.0, 0.03, 0.03, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.24, 0.3, 0.3, 0.18, 0.14, 0.15, 0.16, 0.15, 0.09, 0.2, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.34, 0.44, 0.46, 0.38, 0.18, 0.14, 0.11, 0.13, 0.19, 0.18, 0.45, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.06, 0.13, 0.39, 0.5, 0.5, 0.37, 0.06, 0.0, 0.0, 0.0, 0.02, 0.16, 0.68, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.11, 0.17, 0.17, 0.33, 0.4, 0.38, 0.28, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.18, 0.42, 0.0, 0.0,
    0.0, 0.0, 0.09, 0.18, 0.13, 0.06, 0.08, 0.26, 0.32, 0.32, 0.27, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.82, 0.0, 0.0,
    0.27, 0.0, 0.16, 0.12, 0.0, 0.0, 0.0, 0.25, 0.38, 0.44, 0.45, 0.34, 0.0, 0.0, 0.0, 0.0, 0.0, 0.22, 0.17, 0.0,
    0.0, 0.07, 0.2, 0.02, 0.0, 0.0, 0.0, 0.31, 0.48, 0.57, 0.6, 0.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49, 0.0,
    0.0, 0.59, 0.19, 0.0, 0.0, 0.0, 0.0, 0.2, 0.57, 0.69, 0.76, 0.76, 0.49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.36, 0.0,
    0.0, 0.58, 0.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.67, 0.83, 0.9, 0.92, 0.87, 0.12, 0.0, 0.0, 0.0, 0.0, 0.22, 0.07,
    0.0, 0.0, 0.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.93, 1.0, 1.0, 1.0, 0.61, 0.0, 0.0, 0.0, 0.0, 0.18, 0.11,
    0.0, 0.0, 0.82, 0.0, 0.0, 0.0, 0.0, 0.0, 0.47, 1.0, 1.0, 0.98, 1.0, 0.96, 0.27, 0.0, 0.0, 0.0, 0.19, 0.1,
    0.0, 0.0, 0.46, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 1.0, 1.0, 0.84, 0.92, 0.97, 0.54, 0.14, 0.04, 0.1, 0.21, 0.05,
    0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.09, 0.8, 1.0, 0.82, 0.8, 0.85, 0.63, 0.31, 0.18, 0.19, 0.2, 0.01,
    0.0, 0.0, 0.0, 0.36, 0.1, 0.0, 0.0, 0.0, 0.05, 0.54, 0.86, 0.79, 0.74, 0.72, 0.6, 0.39, 0.28, 0.24, 0.13, 0.0,
    0.0, 0.0, 0.0, 0.01, 0.3, 0.07, 0.0, 0.0, 0.08, 0.36, 0.64, 0.7, 0.64, 0.6, 0.51, 0.39, 0.29, 0.19, 0.04, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.1, 0.24, 0.14, 0.1, 0.15, 0.29, 0.45, 0.53, 0.52, 0.46, 0.4, 0.31, 0.21, 0.08, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.08, 0.21, 0.21, 0.22, 0.29, 0.36, 0.39, 0.37, 0.33, 0.26, 0.18, 0.09, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.13, 0.19, 0.22, 0.24, 0.24, 0.23, 0.18, 0.13, 0.05, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.06, 0.08, 0.09, 0.07, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0,
];

/// Bilinear sample of a `side`-square plane at fractional cell coordinates;
/// zero outside the plane.
fn bilinear(plane: &[f32], side: usize, x: f32, y: f32) -> f32 {
    let last = (side - 1) as f32;
    if side < 2 || !(0.0..=last).contains(&x) || !(0.0..=last).contains(&y) {
        return 0.0;
    }
    let (ix, iy) = ((x as usize).min(side - 2), (y as usize).min(side - 2));
    let (tx, ty) = (x - ix as f32, y - iy as f32);
    let at = |x: usize, y: usize| plane[y * side + x];
    let top = at(ix, iy) * (1.0 - tx) + at(ix + 1, iy) * tx;
    let bottom = at(ix, iy + 1) * (1.0 - tx) + at(ix + 1, iy + 1) * tx;
    top * (1.0 - ty) + bottom * ty
}

/// Visits every destination offset covered by a `side`-square plane that is
/// scaled by `scale`, rotated by `angle` and centred on the origin, passing
/// the offset and the plane's value there (bilinear) to `put`.
fn for_each_rotated(plane: &[f32], side: usize, scale: f32, angle: f32, mut put: impl FnMut(i32, i32, f32)) {
    let (sin, cos) = angle.sin_cos();
    let half = side as f32 * 0.5;
    let reach = (half * scale * std::f32::consts::SQRT_2).ceil() as i32;
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let (fx, fy) = (dx as f32 / scale, dy as f32 / scale);
            let sx = cos * fx + sin * fy + half - 0.5;
            let sy = -sin * fx + cos * fy + half - 0.5;
            let v = bilinear(plane, side, sx, sy);
            if v > 0.0 {
                put(dx, dy, v);
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Params {
    /// Channels in use (1..=3).
    pub channels: usize,
    pub kernels: Vec<Kernel>,
    /// Known creature the nursery hatches from, instead of random patches.
    #[serde(with = "template_id")]
    pub template: Option<&'static Template>,
    /// Channels (bit mask) the template is raised as, each at the radius of
    /// that channel's own kernel. Other channels (colony species living beside
    /// the template's gliders) start from noise.
    pub template_channels: u32,
    /// Relative numbers of each channel's creatures when a brood is released
    /// and respawned (applies on reset).
    pub abundance: [f32; CHANNELS],
    /// Time resolution T: every step advances dt = 1 / T.
    pub time_res: f32,
    pub steps_per_frame: u32,
    /// Multiplies every kernel radius: grows or shrinks the creatures.
    pub scale: f32,
    pub seeding: Seeding,
    /// Seed patches: radius (relative to the largest kernel), noise amplitude
    /// and number (relative to the default density).
    pub seed_size: f32,
    pub seed_amp: f32,
    pub seed_count: f32,
    /// Drop fresh life into a world that has died out or thinned out.
    pub revive: bool,
    /// Respawning keeps the world's mass above this share of the seeded mass
    /// (hatched creatures when there is a brood, noise patches otherwise);
    /// 0 only revives a world that died out completely.
    pub respawn: f32,
    /// Output pixels per cell: how large the creatures appear (applies on reset).
    pub cell_px: f32,
    /// Explosion quench per channel: the mean density a neighbourhood may
    /// reach before it fades (0 = off). Keeps a boiling collision, or the
    /// wreck two colliding gliders leave, from lingering or taking over.
    pub quench: [f32; CHANNELS],
    // Look.
    /// State -> palette position multiplier.
    pub gain: f32,
    /// Per-channel HDR emission of dense nuclei (drives the bloom).
    pub glow: [f32; CHANNELS],
    /// Growth-field halo strength.
    pub halo: f32,
    /// Wake strength and per-frame persistence.
    pub trail: f32,
    pub persistence: f32,
    pub relief: f32,
    pub brightness: f32,
    /// Palette position used to tint halos and wakes.
    pub tint: f32,
    /// Per-channel brightness (sets the species apart in depth).
    pub level: [f32; CHANNELS],
    /// Per-channel luminous membrane along creature edges.
    pub rim: [f32; CHANNELS],
    /// Per-channel density above which nuclei glow into HDR.
    pub core: [f32; CHANNELS],
    /// Per-channel palette position where bodies start (they span 0.4 of
    /// the palette from there): picks the part of a palette a species wears.
    pub hue: [f32; CHANNELS],
    /// Colour mixing between channels: 1 blends them into gradients, higher
    /// lets the locally dominant channel's hue win.
    pub mix_power: f32,
    /// The medium: strength of its own deep hue, and of the light that
    /// nearby life casts into it (saturating, 0-2).
    pub ground: f32,
    pub medium: f32,
    /// Channel whose palette tints the medium.
    pub ground_channel: usize,
    /// Colour reconstruction, a blend of the cubic B-spline (0: soft, hides
    /// a species' ragged, speckled fringe) and Catmull-Rom (1: interpolates
    /// the cells exactly but rings on diagonal edges in close-ups). The
    /// default 2/3 is exactly the Mitchell-Netravali filter (B = C = 1/3).
    pub sharp: f32,
}

impl Params {
    fn new(channels: usize, kernels: Vec<Kernel>, time_res: f32) -> Self {
        Self {
            channels,
            kernels,
            template: None,
            template_channels: 1,
            abundance: [1.0; CHANNELS],
            time_res,
            steps_per_frame: 2,
            scale: 1.0,
            seeding: Seeding::Patches,
            seed_size: 1.0,
            seed_amp: 1.0,
            seed_count: 1.0,
            revive: true,
            respawn: 0.0,
            cell_px: 2.0,
            quench: [0.0; CHANNELS],
            gain: 1.0,
            glow: [1.2; CHANNELS],
            halo: 0.3,
            trail: 0.0,
            persistence: 0.95,
            relief: 0.35,
            brightness: 1.0,
            tint: 0.4,
            level: [1.0; CHANNELS],
            rim: [0.15; CHANNELS],
            core: [0.6; CHANNELS],
            hue: [0.3; CHANNELS],
            mix_power: 3.0,
            ground: 1.0,
            medium: 0.5,
            ground_channel: 0,
            sharp: 2.0 / 3.0,
        }
    }

    fn active_channels(&self) -> usize {
        self.channels.clamp(1, CHANNELS)
    }

    /// Effective radius of kernel `k` in cells.
    fn radius_of(&self, k: &Kernel, r_max: u32) -> f32 {
        (k.radius * self.scale).clamp(2.0, r_max as f32)
    }

    /// Largest effective kernel radius (sets the size of seeds and creatures).
    fn creature_radius(&self, r_max: u32) -> f32 {
        self.kernels.iter().map(|k| self.radius_of(k, r_max)).fold(4.0, f32::max)
    }

    /// Radius of channel `c`'s own creatures: its largest self-kernel (source
    /// and target `c`), or the largest kernel overall if it has none.
    fn species_radius(&self, c: usize, r_max: u32) -> f32 {
        self.kernels
            .iter()
            .filter(|k| k.source == c && k.target == c)
            .map(|k| self.radius_of(k, r_max))
            .reduce(f32::max)
            .unwrap_or_else(|| self.creature_radius(r_max))
    }

    fn dt(&self) -> f32 {
        1.0 / self.time_res.clamp(1.0, 100.0)
    }

    /// Everything the baked kernel tables depend on.
    fn sim_key(&self) -> SimKey {
        SimKey { channels: self.channels, kernels: self.kernels.clone(), scale: self.scale, time_res: self.time_res }
    }
}

/// The parameters the kernel tables on the GPU were baked from.
#[derive(Clone, Debug, PartialEq)]
struct SimKey {
    channels: usize,
    kernels: Vec<Kernel>,
    scale: f32,
    time_res: f32,
}

impl SimKey {
    /// Whether `p` would bake the same tables (compared in place: this runs
    /// twice a frame).
    fn matches(&self, p: &Params) -> bool {
        self.channels == p.channels && self.scale == p.scale && self.time_res == p.time_res && self.kernels == p.kernels
    }
}

// --- presets ------------------------------------------------------------------

const PRESET_NAMES: &[&str] =
    &["Orbium", "Leviathans", "Menagerie", "Pearl Reef", "Necklaces", "Hydrogeminium", "Tessellatium"];

struct PresetDef {
    params: Params,
    palettes: [&'static str; 3],
    look: PostSettings,
}

fn post(exposure: f32, bloom: f32, threshold: f32, vignette: f32, saturation: f32, tonemap: Tonemap) -> PostSettings {
    PostSettings { exposure, bloom, bloom_threshold: threshold, vignette, saturation, grain: 0.0, tonemap }
}

/// The classic Orbium kernel (b = [1], mu = 0.15, sigma = 0.015) on channel
/// `c` at radius `r`.
fn orbium(c: usize, r: f32) -> Kernel {
    k(c, c, r, &[1.0], 0.15, 0.015, 1.0)
}

/// "Pearls": a two-species kernel set found by screening random kernels.
/// Beads string themselves into rings that grow, break into arcs and bud.
/// Occupies channels `c` and `c + 1`.
fn pearl_kernels(c: usize) -> Vec<Kernel> {
    [
        k(0, 0, 14.23, &[1.0, 0.022], 0.3098, 0.059, 0.282),
        k(1, 1, 8.77, &[1.0], 0.2524, 0.0301, 0.556),
        k(1, 1, 9.70, &[1.0], 0.2427, 0.057, 0.689),
        k(1, 0, 14.17, &[1.0], 0.3993, 0.1279, 0.767),
        k(0, 1, 12.98, &[1.0], 0.1829, 0.0354, 0.448),
        k(0, 1, 9.45, &[1.0], 0.1753, 0.0411, 0.418),
        k(0, 1, 10.24, &[0.381, 1.0], 0.1883, 0.0415, 0.882),
    ]
    .iter()
    .map(|kernel| Kernel { source: kernel.source + c, target: kernel.target + c, ..*kernel })
    .collect()
}

/// Settings shared by the presets populated with hatched Orbium: nursery
/// seeding from the published pattern, respawning, and the explosion quench
/// (Orbium collisions leave wrecks or boil over into labyrinths).
fn hatched(channels: usize, kernels: Vec<Kernel>) -> Params {
    let mut p = Params::new(channels, kernels, 10.0);
    p.steps_per_frame = 3;
    p.seeding = Seeding::Nursery;
    p.template = Some(&ORBIUM);
    p.respawn = 0.8;
    p.quench = [0.05; CHANNELS];
    p.cell_px = 2.5;
    p.trail = 1.0;
    p.persistence = 0.96;
    p.gain = 0.7;
    p.glow = [1.6; CHANNELS];
    p.halo = 0.2;
    p
}

fn preset(index: usize) -> PresetDef {
    match index {
        // Orbium unicaudatus (R 13, T 10, mu 0.15, sigma 0.015), the classic
        // glider, hatched from its published pattern: a few large comets on
        // a deep-blue medium, each trailing one soft wake.
        0 => {
            let mut p = hatched(1, vec![orbium(0, 13.0)]);
            p.cell_px = 3.5;
            p.seed_count = 1.3;
            p.rim = [0.12; CHANNELS];
            p.ground = 1.2;
            p.medium = 0.35;
            PresetDef {
                params: p,
                palettes: ["Glacier", "Glacier", "Glacier"],
                look: post(1.0, 0.8, 1.0, 0.35, 1.1, Tonemap::Aces),
            }
        }
        // Leviathans: a few large Orbium (R 24) among shoals of small ones
        // (R 10), kept apart by neutral cross-kernels.
        1 => {
            // Weak cross-kernels: the species sidestep each other, but a
            // passing minnow does not nudge a giant hard enough to break it.
            let kernels = vec![
                orbium(0, 24.0),
                orbium(1, 10.0),
                neutral(0, 1, 26.0, 0.02, 0.12),
                neutral(1, 0, 26.0, 0.02, 0.12),
            ];
            let mut p = hatched(2, kernels);
            p.template_channels = 0b011;
            p.abundance = [1.0, 3.0, 0.0];
            p.seed_count = 1.8;
            p.tint = 0.35;
            p.trail = 0.6;
            p.level = [1.0, 0.9, 1.0];
            p.rim = [0.15, 0.1, 0.0];
            // Coral from its middle: saturated coral bodies, not violet.
            p.hue = [0.42, 0.3, 0.3];
            p.glow = [1.4, 1.6, 0.0];
            p.quench = [0.06, 0.04, 0.0];
            p.ground = 1.2;
            p.medium = 0.3;
            // A deep teal sea.
            p.ground_channel = 1;
            PresetDef {
                params: p,
                palettes: ["Coral", "Bioluminescence", "Glacier"],
                look: post(1.0, 0.8, 1.0, 0.35, 1.1, Tonemap::Aces),
            }
        }
        // Menagerie: a shoal of three Orbium species at three scales (R 9,
        // 13, 20), each at its own brightness, that keep out of each other's
        // way through neutral cross-kernels.
        2 => {
            let mut kernels = vec![orbium(0, 9.0), orbium(1, 13.0), orbium(2, 20.0)];
            for (s, t) in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2)] {
                kernels.push(neutral(s, t, 22.0, 0.02, 0.3));
            }
            let mut p = hatched(3, kernels);
            p.template_channels = 0b111;
            p.abundance = [1.4, 1.0, 0.6];
            p.seed_count = 2.5;
            // Brighter the smaller: the shoal reads in depth.
            p.level = [1.0, 0.8, 0.65];
            p.rim = [0.12; CHANNELS];
            p.hue = [0.3, 0.3, 0.42];
            p.trail = 0.8;
            p.ground = 0.7;
            p.medium = 0.35;
            // A blue-violet sea (Coral's deep end).
            p.ground_channel = 2;
            PresetDef {
                params: p,
                palettes: ["Bioluminescence", "Solar", "Coral"],
                look: post(1.05, 0.7, 1.0, 0.3, 1.15, Tonemap::Aces),
            }
        }
        // Pearl Reef: Orbium gliders weaving between colonies of pearls, each
        // side deflected by the other through neutral cross-kernels.
        3 => {
            let mut kernels = vec![orbium(0, 13.0)];
            kernels.extend(pearl_kernels(1));
            for c in [1, 2] {
                kernels.push(neutral(0, c, 20.0, 0.02, 0.3));
                kernels.push(neutral(c, 0, 20.0, 0.02, 0.3));
            }
            let mut p = hatched(3, kernels);
            p.template_channels = 0b001;
            // The pearls are quenched a little so that open water remains for
            // the gliders, but not so hard that their rings never close.
            p.quench = [0.05, 0.11, 0.11];
            // Pearls are dense throughout, so they do not glow: the gliders'
            // nuclei stay the brightest points, and the teal shell is dimmed.
            p.level = [1.0, 1.0, 0.6];
            p.rim = [0.12, 0.12, 0.05];
            p.core = [0.6, 0.9, 0.9];
            p.glow = [1.4, 0.0, 0.0];
            p.medium = 0.3;
            PresetDef {
                params: p,
                palettes: ["Glacier", "Ember", "Bioluminescence"],
                look: post(1.0, 0.7, 1.0, 0.3, 1.1, Tonemap::Aces),
            }
        }
        // Necklaces: the pearl species alone, growing rings that break into
        // arcs and bud new beads; the quench keeps them from matting.
        4 => {
            let mut p = Params::new(2, pearl_kernels(0), 10.0);
            p.steps_per_frame = 3;
            p.quench = [0.12; CHANNELS];
            p.respawn = 0.3;
            p.gain = 0.9;
            // The beads glow, the coral tube keeps its colour.
            p.glow = [0.5, 0.0, 0.0];
            p.rim = [0.12; CHANNELS];
            p.relief = 0.25;
            p.core = [0.85; CHANNELS];
            p.ground = 0.35;
            p.medium = 0.1;
            PresetDef {
                params: p,
                palettes: ["Ember", "Coral", "Coral"],
                look: post(1.0, 0.45, 1.0, 0.3, 1.1, Tonemap::Aces),
            }
        }
        // Hydrogeminium natans (R 18, T 2, b = [1/2, 1, 2/3]): a three-ring
        // kernel whose colonies of rings and amoebae grow, divide and merge;
        // a mild quench keeps open water between them.
        5 => {
            let mut p = Params::new(1, vec![k(0, 0, 18.0, &[0.5, 1.0, 2.0 / 3.0], 0.26, 0.036, 1.0)], 2.0);
            p.steps_per_frame = 1;
            p.seeding = Seeding::Soup;
            p.quench = [0.16; CHANNELS];
            p.gain = 0.8;
            p.glow = [0.5; CHANNELS];
            p.halo = 0.3;
            p.relief = 0.15;
            p.rim = [0.04; CHANNELS];
            p.core = [0.9; CHANNELS];
            p.ground = 0.6;
            p.medium = 0.08;
            PresetDef {
                params: p,
                palettes: ["Lagoon", "Lagoon", "Lagoon"],
                look: post(1.0, 0.6, 1.0, 0.35, 1.05, Tonemap::Aces),
            }
        }
        // Tessellatium gyrans: three channels and fifteen kernels; every
        // creature is made of all three channels in shifting proportions.
        _ => {
            let r = 12.0;
            let mut p = Params::new(
                3,
                vec![
                    k(0, 0, r * 0.91, &[1.0], 0.272, 0.0595, 0.138),
                    k(0, 0, r * 0.62, &[1.0], 0.349, 0.1585, 0.48),
                    k(0, 0, r * 0.50, &[1.0, 0.25], 0.2, 0.0332, 0.284),
                    k(1, 1, r * 0.97, &[0.0, 1.0], 0.114, 0.0528, 0.256),
                    k(1, 1, r * 0.72, &[1.0], 0.447, 0.0777, 0.5),
                    k(1, 1, r * 0.80, &[5.0 / 6.0, 1.0], 0.247, 0.0342, 0.622),
                    k(2, 2, r * 0.96, &[1.0], 0.21, 0.0617, 0.35),
                    k(2, 2, r * 0.56, &[1.0], 0.462, 0.1192, 0.218),
                    k(2, 2, r * 0.78, &[1.0], 0.446, 0.1793, 0.556),
                    k(0, 1, r * 0.79, &[11.0 / 12.0, 1.0], 0.327, 0.1408, 0.344),
                    k(0, 2, r * 0.50, &[0.75, 1.0], 0.476, 0.0995, 0.456),
                    k(1, 0, r * 0.72, &[11.0 / 12.0, 1.0], 0.379, 0.0697, 0.67),
                    k(1, 2, r * 0.68, &[1.0], 0.262, 0.0877, 0.42),
                    k(2, 0, r * 0.82, &[1.0 / 6.0, 1.0, 0.0], 0.412, 0.1101, 0.43),
                    k(2, 1, r * 0.82, &[1.0], 0.201, 0.0786, 0.278),
                ],
                2.0,
            );
            p.steps_per_frame = 2;
            p.cell_px = 4.0;
            p.seed_count = 2.0;
            // Fresh noise patches (its own seeding) top the population up
            // when half of it has gone.
            p.respawn = 0.5;
            // Long wakes draw each creature's gyrating path.
            p.trail = 0.8;
            p.persistence = 0.97;
            p.tint = 0.45;
            // Dense all over: no nuclei glow, and bodies stay in the saturated
            // middle of their palettes.
            p.gain = 0.65;
            p.glow = [0.0; CHANNELS];
            p.relief = 0.2;
            p.rim = [0.0; CHANNELS];
            p.core = [0.9; CHANNELS];
            p.hue = [0.35, 0.3, 0.3];
            // Channels blend into gradients rather than switching hard, and
            // the species' speckled fringe is softened.
            p.mix_power = 1.5;
            p.sharp = 0.0;
            p.medium = 0.08;
            PresetDef {
                params: p,
                palettes: ["Coral", "Solar", "Ember"],
                look: post(1.0, 0.6, 1.0, 0.3, 1.15, Tonemap::Aces),
            }
        }
    }
}

/// Palette triples that sit well together (used by mutation).
const TRIADS: &[[&str; 3]] = &[
    ["Solar", "Bioluminescence", "Coral"],
    ["Coral", "Bioluminescence", "Glacier"],
    ["Glacier", "Ember", "Bioluminescence"],
    ["Glacier", "Bioluminescence", "Aurora"],
    ["Ember", "Solar", "Coral"],
    ["Lagoon", "Coral", "Solar"],
    ["Nacre", "Glacier", "Aurora"],
];

// --- GPU uniforms ---------------------------------------------------------------

/// Mirrors `Group` in lenia.wgsl (144 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GroupUniform {
    size: [u32; 2],
    src: u32,
    flags: u32,
    halo: u32,
    tile_w: u32,
    row_start: u32,
    row_count: u32,
    dt: f32,
    _pad: u32,
    wrap: [u32; 2],
    mu: [f32; 4],
    inv2s2: [f32; 4],
    to_channel: [[f32; 4]; 4],
}

/// Mirrors `Brush` in lenia_render.wgsl (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BrushUniform {
    size: [u32; 2],
    center: [f32; 2],
    radius: f32,
    mode: u32,
    seed: u32,
    channels: u32,
    amplitude: f32,
    threshold: u32,
    side: u32,
    room: u32,
    watch: u32,
    probe: u32,
    base: u32,
    _pad: u32,
    spots: [[f32; 4]; 2],
}

/// `Brush.mode` values (see lenia_render.wgsl).
const MODE_PAINT: u32 = 1;
const MODE_ERASE: u32 = 2;
const MODE_REVIVE: u32 = 3;
const MODE_STAMP: u32 = 4;

/// Mirrors `Quench` in lenia_render.wgsl (64 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct QuenchUniform {
    size: [u32; 2],
    grid: [u32; 2],
    reach: [u32; 4],
    limit: [u32; 4],
    keep: f32,
    channels: u32,
    _pad: [u32; 2],
}

/// Mirrors `Compose` in lenia_render.wgsl (48 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComposeUniform {
    size: [u32; 2],
    channels: u32,
    decay: f32,
    halo: f32,
    trail: f32,
    spread: f32,
    _pad: f32,
    rest: [f32; 4],
}

/// Mirrors `Light` in lenia_render.wgsl (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightUniform {
    grid: [u32; 2],
    keep: f32,
    scale: f32,
}

/// Mirrors `Draw` in lenia_render.wgsl (160 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    view: ViewXform,
    size: [f32; 2],
    grid: [u32; 2],
    channels: u32,
    relief: f32,
    gain: f32,
    brightness: f32,
    tint: f32,
    mix_power: f32,
    ground: f32,
    medium: f32,
    level: [f32; 4],
    rim: [f32; 4],
    core: [f32; 4],
    glow: [f32; 4],
    hue: [f32; 4],
    ground_channel: u32,
    sharp: f32,
    _pad: [f32; 2],
}

// --- kernel tables ---------------------------------------------------------------

/// One convolution pass: up to four kernels sharing a source channel.
struct GroupPlan {
    uniform: GroupUniform,
    halo: u32,
}

/// Baked kernel tables for a parameter set.
struct Plan {
    groups: Vec<GroupPlan>,
    /// Per row of taps: (first weight, weight count incl. 3 zero pads, tile offset, 0).
    rows: Vec<[u32; 4]>,
    /// Normalised weights, one lane per kernel slot.
    taps: Vec<[f32; 4]>,
    /// Growth rate of each channel in empty space (baseline of the halo).
    rest: [f32; 4],
}

fn build_plan(params: &Params, r_max: u32) -> Plan {
    let channels = params.active_channels();
    // Kernels pointing at channels that are switched off fold onto the last one.
    let kernels: Vec<Kernel> = params
        .kernels
        .iter()
        .take(MAX_KERNELS)
        .map(|k| Kernel { source: k.source.min(channels - 1), target: k.target.min(channels - 1), ..*k })
        .collect();
    let radius: Vec<f32> = kernels.iter().map(|k| params.radius_of(k, r_max)).collect();

    let mut sum_h = [0.0f32; CHANNELS];
    for k in &kernels {
        sum_h[k.target] += k.h.max(0.0);
    }
    let share = |k: &Kernel| if sum_h[k.target] > 0.0 { k.h.max(0.0) / sum_h[k.target] } else { 0.0 };
    let mut rest = [0.0f32; 4];
    for k in &kernels {
        rest[k.target] += share(k) * k.rest_growth();
    }

    // Pack kernels by source channel (smallest radii together), four per group.
    let mut order: Vec<usize> = (0..kernels.len()).collect();
    order.sort_by(|&a, &b| kernels[a].source.cmp(&kernels[b].source).then(radius[a].total_cmp(&radius[b])));
    let mut packs: Vec<Vec<usize>> = Vec::new();
    for i in order {
        match packs.last_mut() {
            Some(pack) if pack.len() < SLOTS && kernels[pack[0]].source == kernels[i].source => pack.push(i),
            _ => packs.push(vec![i]),
        }
    }
    packs.truncate(MAX_GROUPS);

    let mut plan = Plan { groups: Vec::new(), rows: Vec::new(), taps: Vec::new(), rest };
    let last = packs.len().saturating_sub(1);
    for (g, pack) in packs.iter().enumerate() {
        let halo = pack.iter().map(|&i| radius[i].ceil() as u32).max().unwrap_or(1).clamp(1, r_max);
        let span = (2 * halo + 1) as usize;
        let tile_w = TILE + 2 * halo + 1;

        // Sample each kernel on the (2h+1)^2 grid and normalise it to sum 1.
        let mut grid = vec![[0.0f32; 4]; span * span];
        for (slot, &i) in pack.iter().enumerate() {
            let mut kernel = kernels[i];
            let mut sum = 0.0f32;
            for pass in 0..2 {
                sum = 0.0;
                for y in 0..span {
                    for x in 0..span {
                        let (dx, dy) = (x as f32 - halo as f32, y as f32 - halo as f32);
                        let v = kernel.shell((dx * dx + dy * dy).sqrt(), radius[i]);
                        grid[y * span + x][slot] = v;
                        sum += v;
                    }
                }
                // Every ring peak at zero would leave an empty kernel (and a
                // dead world): fall back to a plain single ring.
                if sum > 1e-6 || pass == 1 {
                    break;
                }
                kernel.rings = 1;
                kernel.b = [1.0, 0.0, 0.0];
            }
            for cell in grid.iter_mut() {
                cell[slot] /= sum.max(1e-6);
            }
        }

        let row_start = plan.rows.len() as u32;
        let live = |w: &[f32; 4]| w.iter().any(|&v| v > 1e-7);
        for y in 0..span {
            let row = &grid[y * span..(y + 1) * span];
            let Some(first) = row.iter().position(live) else { continue };
            let end = row.iter().rposition(live).map_or(first, |e| e) + 1;
            plan.rows.push([plan.taps.len() as u32, (end - first + 3) as u32, y as u32 * tile_w + first as u32, 0]);
            plan.taps.extend_from_slice(&row[first..end]);
            plan.taps.extend_from_slice(&[[0.0; 4]; 3]);
        }

        let mut u = GroupUniform::zeroed();
        u.src = kernels[pack[0]].source as u32;
        u.flags = u32::from(g == 0) | (u32::from(g == last) << 1);
        u.halo = halo;
        u.tile_w = tile_w;
        u.row_start = row_start;
        u.row_count = plan.rows.len() as u32 - row_start;
        u.dt = params.dt();
        for (slot, &i) in pack.iter().enumerate() {
            let kernel = &kernels[i];
            let s = kernel.sigma();
            u.mu[slot] = kernel.mu;
            u.inv2s2[slot] = 1.0 / (2.0 * s * s);
            u.to_channel[slot][kernel.target] = share(kernel);
        }
        plan.groups.push(GroupPlan { uniform: u, halo });
    }
    plan
}

// --- simulation buffers --------------------------------------------------------------

/// Convolution pipelines (small-, medium- and large-tile variants) and their layout.
struct StepPipelines {
    layout: wgpu::BindGroupLayout,
    small: wgpu::ComputePipeline,
    medium: wgpu::ComputePipeline,
    large: wgpu::ComputePipeline,
    small_halo: u32,
    medium_halo: u32,
    /// Largest kernel radius the large tile supports on this device.
    r_max: u32,
}

impl StepPipelines {
    fn new(gpu: &Gpu) -> Self {
        let floats = gpu.device.limits().max_compute_workgroup_storage_size / 4;
        // Shared floats for a tile with `halo`: rows of (TILE + 2h) cells, odd stride.
        let cap = |halo: u32| (TILE + 2 * halo) * (TILE + 2 * halo + 1);
        let mut r_max = RADIUS_CAP;
        while r_max > 8 && cap(r_max) > floats {
            r_max -= 1;
        }
        let small_halo = SMALL_HALO.min(r_max);
        let medium_halo = MEDIUM_HALO.clamp(small_halo, r_max);
        let cs = ShaderStages::COMPUTE;
        let layout = gpu.bind_group_layout(
            "lenia step",
            &[
                layout::uniform(0, cs),
                layout::storage(1, cs, true),
                layout::storage(2, cs, false),
                layout::storage(3, cs, false),
                layout::storage(4, cs, true),
                layout::storage(5, cs, true),
            ],
        );
        let pipeline_layout = gpu.pipeline_layout("lenia step", &[&layout]);
        let build = |halo: u32| {
            let source = format!("const TILE_CAP: u32 = {}u;\n{}", cap(halo), include_str!("../shaders/lenia.wgsl"));
            let module = gpu.shader("lenia step", &source);
            gpu.compute_pipeline("lenia step", &pipeline_layout, &module, "cs_step")
        };
        let small = build(small_halo);
        let medium = build(medium_halo);
        let large = build(r_max);
        Self { layout, small, medium, large, small_halo, medium_halo, r_max }
    }

    /// The smallest tile that holds a group with this halo: shared memory per
    /// workgroup limits how many workgroups share a multiprocessor.
    fn pick(&self, halo: u32) -> &wgpu::ComputePipeline {
        if halo <= self.small_halo {
            &self.small
        } else if halo <= self.medium_halo {
            &self.medium
        } else {
            &self.large
        }
    }
}

/// Simulation state for one domain size: the world's own, a mutation trial or
/// the nursery.
struct Sim {
    size: [u32; 2],
    /// Torus size: `size` itself, or one nursery cell (many isolated tori).
    wrap: [u32; 2],
    /// Ping-pong state, three f32 planes each.
    state: [wgpu::Buffer; 2],
    /// Per-channel growth rate of the last step (vec4 per cell); doubles as
    /// the accumulator between the groups of a step.
    growth: wgpu::Buffer,
    rows: wgpu::Buffer,
    taps: wgpu::Buffer,
    uniforms: Vec<wgpu::Buffer>,
    /// `binds[g][i]` runs group `g` reading `state[i]` and writing `state[1 - i]`.
    binds: Vec<[wgpu::BindGroup; 2]>,
    /// Halo of each active group (selects the tile variant).
    halos: Vec<u32>,
    /// Index of the buffer holding the latest state.
    current: usize,
}

impl Sim {
    fn new(gpu: &Gpu, pipes: &StepPipelines, size: [u32; 2]) -> Self {
        Self::with_wrap(gpu, pipes, size, size)
    }

    /// `wrap` must equal `size` or divide it in multiples of `TILE`.
    fn with_wrap(gpu: &Gpu, pipes: &StepPipelines, size: [u32; 2], wrap: [u32; 2]) -> Self {
        let cells = size[0] as u64 * size[1] as u64;
        let none = wgpu::BufferUsages::empty();
        let state = [
            gpu.storage_buffer("lenia state a", cells * 4 * CHANNELS as u64, none),
            gpu.storage_buffer("lenia state b", cells * 4 * CHANNELS as u64, none),
        ];
        let growth = gpu.storage_buffer("lenia growth", cells * 16, none);
        let span = 2 * pipes.r_max as u64 + 1;
        let rows = gpu.storage_buffer("lenia rows", MAX_GROUPS as u64 * span * 16, none);
        let taps = gpu.storage_buffer("lenia taps", MAX_GROUPS as u64 * span * (span + 3) * 16, none);
        let uniforms: Vec<wgpu::Buffer> =
            (0..MAX_GROUPS).map(|_| gpu.uniform_buffer("lenia group", &GroupUniform::zeroed())).collect();
        let binds = uniforms
            .iter()
            .map(|uniform| {
                [0, 1].map(|i| {
                    gpu.bind_group(
                        "lenia step",
                        &pipes.layout,
                        &[
                            uniform.as_entire_binding(),
                            state[i].as_entire_binding(),
                            state[1 - i].as_entire_binding(),
                            growth.as_entire_binding(),
                            rows.as_entire_binding(),
                            taps.as_entire_binding(),
                        ],
                    )
                })
            })
            .collect();
        Self { size, wrap, state, growth, rows, taps, uniforms, binds, halos: Vec::new(), current: 0 }
    }

    fn upload(&mut self, gpu: &Gpu, plan: &Plan) {
        if !plan.rows.is_empty() {
            gpu.queue.write_buffer(&self.rows, 0, bytemuck::cast_slice(&plan.rows));
            gpu.queue.write_buffer(&self.taps, 0, bytemuck::cast_slice(&plan.taps));
        }
        for (group, buffer) in plan.groups.iter().zip(&self.uniforms) {
            let mut u = group.uniform;
            u.size = self.size;
            u.wrap = self.wrap;
            gpu.write(buffer, &u);
        }
        self.halos = plan.groups.iter().map(|g| g.halo).collect();
    }

    fn load(&mut self, gpu: &Gpu, cells: &[f32]) {
        gpu.queue.write_buffer(&self.state[0], 0, bytemuck::cast_slice(cells));
        self.current = 0;
    }

    /// Records `steps` full Lenia steps.
    fn record(&mut self, pass: &mut wgpu::ComputePass<'_>, pipes: &StepPipelines, steps: u32) {
        if self.halos.is_empty() {
            return;
        }
        let groups = [self.size[0].div_ceil(TILE), self.size[1].div_ceil(TILE)];
        for _ in 0..steps {
            for (g, &halo) in self.halos.iter().enumerate() {
                pass.set_pipeline(pipes.pick(halo));
                pass.set_bind_group(0, &self.binds[g][self.current], &[]);
                pass.dispatch_workgroups(groups[0], groups[1], 1);
            }
            self.current = 1 - self.current;
        }
    }

    /// Blocking readback of the first `planes` channels of the current state.
    fn read(&self, gpu: &Gpu, planes: usize) -> Vec<f32> {
        let bytes = self.size[0] as u64 * self.size[1] as u64 * 4 * planes.clamp(1, CHANNELS) as u64;
        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lenia trial readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lenia trial readback") });
        encoder.copy_buffer_to_buffer(&self.state[self.current], 0, &staging, 0, bytes);
        gpu.queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = gpu.device.poll(wgpu::PollType::Wait);
        if !matches!(rx.recv(), Ok(Ok(()))) {
            return Vec::new();
        }
        let values = slice
            .get_mapped_range()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        staging.unmap();
        values
    }
}

// --- seeding ---------------------------------------------------------------------

/// Initial state (three planes) for a domain of `size` cells.
fn seed_state(size: [u32; 2], params: &Params, r_max: u32, seed: u64) -> Vec<f32> {
    let [w, h] = size;
    let n = (w * h) as usize;
    let mut cells = vec![0.0f32; n * CHANNELS];
    let mut rng = Rng::new(seed);
    let channels = params.active_channels();
    let r = params.creature_radius(r_max);
    let area = n as f32;

    // Disc of per-cell uniform noise; each channel gets its own amplitude.
    let disc = |cells: &mut [f32], rng: &mut Rng, radius: f32, amplitude: f32| {
        let (cx, cy) = (rng.range(0.0, w as f32), rng.range(0.0, h as f32));
        let amps: Vec<f32> = (0..channels).map(|_| amplitude * rng.range(0.55, 1.0)).collect();
        let ri = radius.ceil() as i32;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                if (dx * dx + dy * dy) as f32 > radius * radius {
                    continue;
                }
                let x = (cx as i32 + dx).rem_euclid(w as i32) as usize;
                let y = (cy as i32 + dy).rem_euclid(h as i32) as usize;
                for (c, amp) in amps.iter().enumerate() {
                    cells[c * n + y * w as usize + x] = rng.f32() * amp;
                }
            }
        }
    };

    let (size_k, amp_k, count_k) =
        (params.seed_size.clamp(0.2, 4.0), params.seed_amp.clamp(0.05, 1.0), params.seed_count.clamp(0.05, 8.0));
    match params.seeding {
        Seeding::Patches | Seeding::Nursery => {
            let count = (area / (6.0 * r).powi(2) * count_k).clamp(3.0, 2000.0) as u32;
            for _ in 0..count {
                let radius = r * size_k * rng.range(0.7, 1.2);
                let amplitude = amp_k * rng.range(0.7, 1.0);
                disc(&mut cells, &mut rng, radius, amplitude);
            }
        }
        Seeding::Blobs => {
            // Smooth domes whose density ramps up along a random direction,
            // roughly the profile of a glider, roughened with noise.
            let count = (area / (6.0 * r).powi(2) * count_k).clamp(3.0, 2000.0) as u32;
            for _ in 0..count {
                let radius = r * size_k * rng.range(0.8, 1.2);
                let amplitude = amp_k * rng.range(0.8, 1.0);
                let (cx, cy) = (rng.range(0.0, w as f32), rng.range(0.0, h as f32));
                let dirs: Vec<(f32, f32)> = (0..channels)
                    .map(|_| {
                        let a = rng.range(0.0, TAU);
                        (a.cos(), a.sin())
                    })
                    .collect();
                let ri = radius.ceil() as i32;
                for dy in -ri..=ri {
                    for dx in -ri..=ri {
                        let q2 = (dx * dx + dy * dy) as f32 / (radius * radius);
                        if q2 >= 1.0 {
                            continue;
                        }
                        let x = (cx as i32 + dx).rem_euclid(w as i32) as usize;
                        let y = (cy as i32 + dy).rem_euclid(h as i32) as usize;
                        for (c, &(ux, uy)) in dirs.iter().enumerate() {
                            let ramp = 0.5 + 0.5 * (ux * dx as f32 + uy * dy as f32) / radius;
                            let v = amplitude * (1.0 - q2).powi(2) * (0.3 + 0.7 * ramp) * rng.range(0.7, 1.0);
                            cells[c * n + y * w as usize + x] = v;
                        }
                    }
                }
            }
        }
        Seeding::Sparse => {
            let count = (area / (14.0 * r).powi(2) * count_k).clamp(2.0, 400.0) as u32;
            for _ in 0..count {
                let radius = r * size_k * rng.range(1.5, 2.4);
                let amplitude = amp_k * rng.range(0.7, 1.0);
                disc(&mut cells, &mut rng, radius, amplitude);
            }
        }
        Seeding::Soup => {
            // Noise everywhere, modulated by a coarse cloud mask (bilinear
            // lattice that wraps with the torus).
            let spacing = (4.0 * r).max(8.0);
            let gx = ((w as f32 / spacing).round() as usize).max(1);
            let gy = ((h as f32 / spacing).round() as usize).max(1);
            let lattice: Vec<f32> = (0..gx * gy).map(|_| rng.f32()).collect();
            for y in 0..h as usize {
                let fy = y as f32 / h as f32 * gy as f32;
                let (y0, ty) = (fy as usize % gy, fy.fract());
                let y1 = (y0 + 1) % gy;
                for x in 0..w as usize {
                    let fx = x as f32 / w as f32 * gx as f32;
                    let (x0, tx) = (fx as usize % gx, fx.fract());
                    let x1 = (x0 + 1) % gx;
                    let top = lattice[y0 * gx + x0] * (1.0 - tx) + lattice[y0 * gx + x1] * tx;
                    let bottom = lattice[y1 * gx + x0] * (1.0 - tx) + lattice[y1 * gx + x1] * tx;
                    let m = top * (1.0 - ty) + bottom * ty;
                    let mask = ((m - 0.45) / 0.3).clamp(0.0, 1.0);
                    if mask <= 0.0 {
                        continue;
                    }
                    for c in 0..channels {
                        cells[c * n + y * w as usize + x] = rng.f32() * mask;
                    }
                }
            }
        }
    }
    cells
}

/// Index drawn with probability proportional to `weights` (the last positive
/// one on rounding; 0 when every weight is zero).
fn pick_weighted(weights: &[f32], rng: &mut Rng) -> usize {
    let total: f32 = weights.iter().map(|w| w.max(0.0)).sum();
    let mut x = rng.range(0.0, total);
    for (i, &w) in weights.iter().enumerate() {
        if w > 0.0 && x < w {
            return i;
        }
        x -= w.max(0.0);
    }
    weights.iter().rposition(|&w| w > 0.0).unwrap_or(0)
}

// --- nursery -----------------------------------------------------------------------

/// A creature hatched in the nursery, centred in a square window.
struct Hatchling {
    side: usize,
    /// `CHANNELS` planes of `side * side` cells.
    cells: Vec<f32>,
    /// Direction of travel in window coordinates (radians, y down).
    heading: f32,
    /// The channel it belongs to.
    species: usize,
}

impl Hatchling {
    /// The creature turned by `quarter` right angles, which is exact (no
    /// resampling that could hurt a fragile species): its planes and heading.
    fn turned(&self, quarter: u32) -> (Vec<f32>, f32) {
        let s = self.side;
        let mut out = vec![0.0f32; self.cells.len()];
        for (c, plane) in self.cells.chunks_exact(s * s).enumerate() {
            for y in 0..s {
                for x in 0..s {
                    // A quarter turn (y down) sends cell (x, y) to (s-1-y, x).
                    let (tx, ty) = match quarter % 4 {
                        0 => (x, y),
                        1 => (s - 1 - y, x),
                        2 => (s - 1 - x, s - 1 - y),
                        _ => (y, s - 1 - x),
                    };
                    out[c * s * s + ty * s + tx] = plane[y * s + x];
                }
            }
        }
        (out, self.heading + (quarter % 4) as f32 * FRAC_PI_2)
    }

    /// The creature rotated by `angle` within its own window (bilinear
    /// resampling blurs it slightly; it heals within a few steps). Compact
    /// creatures fit any rotation: the window is about four radii across.
    fn rotated(&self, angle: f32) -> Hatchling {
        let s = self.side;
        let half = (s / 2) as i32;
        let mut cells = vec![0.0f32; self.cells.len()];
        for (c, plane) in self.cells.chunks_exact(s * s).enumerate() {
            for_each_rotated(plane, s, 1.0, angle, |dx, dy, v| {
                let (x, y) = (dx + half, dy + half);
                if (0..s as i32).contains(&x) && (0..s as i32).contains(&y) {
                    cells[c * s * s + y as usize * s + x as usize] = v;
                }
            });
        }
        Hatchling { side: s, cells, heading: self.heading + angle, species: self.species }
    }

    /// The channel holding most of the creature's mass.
    fn dominant(&self, channels: usize) -> usize {
        let plane = self.side * self.side;
        let mass = |c: usize| self.cells[c * plane..(c + 1) * plane].iter().sum::<f32>();
        (0..channels.clamp(1, CHANNELS)).max_by(|&a, &b| mass(a).total_cmp(&mass(b))).unwrap_or(0)
    }
}

/// Up to `MAX_BROOD` hatchlings, taken from each species in turn so that
/// every species that hatched is represented.
fn thin_brood(brood: Vec<Hatchling>) -> Vec<Hatchling> {
    let mut buckets: [Vec<Hatchling>; CHANNELS] = Default::default();
    for chick in brood.into_iter().rev() {
        buckets[chick.species.min(CHANNELS - 1)].push(chick);
    }
    let mut kept = Vec::new();
    while kept.len() < MAX_BROOD && buckets.iter().any(|b| !b.is_empty()) {
        for bucket in &mut buckets {
            if kept.len() < MAX_BROOD {
                kept.extend(bucket.pop());
            }
        }
    }
    kept
}

/// Writes `planes` (one `side`-square window per channel, centred on
/// `centre`) into a state of `size` cells with max blending, wrapping around.
fn stamp_max(cells: &mut [f32], size: [u32; 2], planes: &[f32], side: usize, centre: [f32; 2], channels: usize) {
    let (w, h) = (size[0] as i32, size[1] as i32);
    let n = (w * h) as usize;
    let half = side as i32 / 2;
    for (c, plane) in planes.chunks_exact(side * side).enumerate().take(channels) {
        for (i, &v) in plane.iter().enumerate() {
            if v <= 0.0 {
                continue;
            }
            let x = (centre[0] as i32 + (i % side) as i32 - half).rem_euclid(w) as usize;
            let y = (centre[1] as i32 + (i / side) as i32 - half).rem_euclid(h) as usize;
            let dst = &mut cells[c * n + y * w as usize + x];
            *dst = dst.max(v);
        }
    }
}

/// Records `steps` Lenia steps in bounded command buffers and submits them.
fn run_steps(gpu: &Gpu, pipes: &StepPipelines, sim: &mut Sim, steps: u32) {
    let mut done = 0;
    while done < steps {
        let chunk = (steps - done).min(256);
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("lenia offline") });
        {
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lenia offline"), timestamp_writes: None });
            sim.record(&mut pass, pipes, chunk);
        }
        gpu.queue.submit([encoder.finish()]);
        done += chunk;
    }
}

/// Connected blobs (density > 0.15) of one `cell`-sized torus inside `density`
/// (row stride `stride`, corner `origin`): (area, mass, circular-mean centroid).
fn torus_blobs(density: &[f32], stride: usize, origin: (usize, usize), cell: usize) -> Vec<(usize, f32, [f32; 2])> {
    let at = |x: usize, y: usize| density[(origin.1 + y) * stride + origin.0 + x];
    let mut seen = vec![false; cell * cell];
    let mut blobs = Vec::new();
    let mut stack = Vec::new();
    let k = TAU / cell as f32;
    for start in 0..cell * cell {
        if seen[start] || at(start % cell, start / cell) <= 0.15 {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut area, mut mass) = (0usize, 0.0f32);
        let mut trig = [0.0f32; 4];
        while let Some(i) = stack.pop() {
            let (x, y) = (i % cell, i / cell);
            let d = at(x, y);
            area += 1;
            mass += d;
            trig[0] += d * (x as f32 * k).cos();
            trig[1] += d * (x as f32 * k).sin();
            trig[2] += d * (y as f32 * k).cos();
            trig[3] += d * (y as f32 * k).sin();
            for (dx, dy) in [(1, 0), (cell - 1, 0), (0, 1), (0, cell - 1)] {
                let j = ((y + dy) % cell) * cell + (x + dx) % cell;
                if !seen[j] && at(j % cell, j / cell) > 0.15 {
                    seen[j] = true;
                    stack.push(j);
                }
            }
        }
        let angle = |c: f32, s: f32| s.atan2(c).rem_euclid(TAU) / k;
        blobs.push((area, mass, [angle(trig[0], trig[1]), angle(trig[2], trig[3])]));
    }
    blobs
}

/// Hatches creatures from random patches (or the template): one patch per
/// small isolated torus (so failures, explosive ones included, cannot
/// spread), run until settled. Tori left holding a single compact creature
/// of steady mass are kept, and a diverse few of those are returned.
fn hatch(gpu: &Gpu, pipes: &StepPipelines, params: &Params, seed: u64) -> Vec<Hatchling> {
    let r = params.creature_radius(pipes.r_max);
    let cell = nursery_cell(r);
    // As many tori as fit NURSERY_SPAN cells (a few dozen hatchlings are
    // plenty), and never more than the device's largest storage binding,
    // which the growth buffer (16 bytes a cell) must fit.
    let limit = gpu.device.limits().max_storage_buffer_binding_size as u64;
    let mut grid = (NURSERY_SPAN / cell).clamp(4, 16);
    while grid > 1 && ((cell * grid) as u64).pow(2) * 16 > limit {
        grid -= 1;
    }
    let side = cell * grid;
    let n = side * side;
    let channels = params.active_channels();
    // Channels the template is raised as, taken in turn by the tori.
    let hosts: Vec<usize> = (0..channels).filter(|&c| params.template_channels & (1 << c) != 0).collect();
    let host = |t: usize| if hosts.is_empty() { 0 } else { hosts[t % hosts.len()] };
    let mut rng = Rng::new(seed ^ 0x4E55_5253_4552_5921);
    let mut cells = vec![0.0f32; n * CHANNELS];
    // Each torus gets its own random patch: a noise disc, a lopsided dome or a
    // noisy crescent, of random size and strength, so the nursery samples a
    // wide range of starting shapes.
    for t in 0..grid * grid {
        let (ox, oy) = ((t % grid) * cell, (t / grid) * cell);
        let c = (cell / 2) as i32;
        if let Some(template) = params.template {
            // A known creature at a random heading, slightly jittered in size
            // and strength; the settling run polishes it for these parameters.
            // Each torus raises one host species, sized by its own kernel.
            let species = host(t);
            let scale = params.species_radius(species, pipes.r_max) / template.radius * rng.range(0.98, 1.02);
            let amp = rng.range(0.97, 1.03);
            let angle = rng.range(0.0, TAU);
            for_each_rotated(template.cells, template.side, scale, angle, |dx, dy, v| {
                let x = (c + dx).rem_euclid(cell as i32) as usize;
                let y = (c + dy).rem_euclid(cell as i32) as usize;
                cells[species * n + (oy + y) * side + ox + x] = (v * amp).min(1.0);
            });
            continue;
        }
        let shape = rng.below(3);
        let radius = r * params.seed_size.clamp(0.2, 2.0) * rng.range(0.5, 1.3);
        let amp = params.seed_amp.clamp(0.05, 1.0) * rng.range(0.4, 1.0);
        let amps: Vec<f32> = (0..channels).map(|_| amp * rng.range(0.55, 1.0)).collect();
        let heading = rng.range(0.0, TAU);
        let (hx, hy) = (heading.cos(), heading.sin());
        let (ring, width) = (rng.range(0.35, 0.7), rng.range(0.15, 0.35));
        let ri = radius.ceil() as i32;
        for dy in -ri..=ri {
            for dx in -ri..=ri {
                let q = ((dx * dx + dy * dy) as f32).sqrt() / radius;
                if q >= 1.0 {
                    continue;
                }
                let facing = (hx * dx as f32 + hy * dy as f32) / radius;
                let (x, y) = (ox + (c + dx) as usize, oy + (c + dy) as usize);
                for (ch, a) in amps.iter().enumerate() {
                    let v = match shape {
                        0 => rng.f32(),
                        1 => (1.0 - q * q).powi(2) * (0.3 + 0.35 * (1.0 + facing)) * rng.range(0.7, 1.0),
                        _ => {
                            let band = (-((q - ring) / width).powi(2)).exp();
                            band * (0.5 + 0.5 * facing / q.max(1e-3)).max(0.0).powf(1.5) * rng.range(0.8, 1.0)
                        }
                    };
                    cells[ch * n + y * side + x] = v * a;
                }
            }
        }
    }
    let mut sim = Sim::with_wrap(gpu, pipes, [side as u32; 2], [cell as u32; 2]);
    sim.upload(gpu, &build_plan(params, pipes.r_max));
    sim.load(gpu, &cells);
    drop(cells);
    // Settle long enough for a patch to become a creature (or die), then
    // watch a little longer: a creature keeps one compact blob of steady mass.
    let t = params.time_res.clamp(1.0, 100.0);
    run_steps(gpu, pipes, &mut sim, (60.0 * t).clamp(120.0, 1200.0) as u32);
    let early = sim.read(gpu, channels);
    run_steps(gpu, pipes, &mut sim, (8.0 * t).clamp(20.0, 160.0) as u32);
    let late = sim.read(gpu, channels);
    if early.len() < n * channels || late.len() < n * channels {
        return Vec::new();
    }
    let density = |s: &[f32]| -> Vec<f32> { (0..n).map(|i| (0..channels).map(|c| s[c * n + i]).sum()).collect() };
    let (d_early, d_late) = (density(&early), density(&late));

    let mut brood = Vec::new();
    for t in 0..grid * grid {
        let origin = ((t % grid) * cell, (t / grid) * cell);
        let blobs: Vec<_> = torus_blobs(&d_late, side, origin, cell).into_iter().filter(|b| b.0 >= 12).collect();
        let before: Vec<_> = torus_blobs(&d_early, side, origin, cell).into_iter().filter(|b| b.0 >= 12).collect();
        let ([(area, mass, centre)], [(_, mass0, centre0)]) = (blobs.as_slice(), before.as_slice()) else { continue };
        // Judge size against the radius of the species this torus raised.
        let rs = if params.template.is_some() { params.species_radius(host(t), pipes.r_max) } else { r };
        let compact = (*area as f32) > 0.3 * rs * rs && (*area as f32) < (8.0 * rs * rs).min(0.3 * (cell * cell) as f32);
        if !compact || !(0.8..1.25).contains(&(mass / mass0.max(1e-3))) {
            continue;
        }
        let d = torus_delta_cpu(*centre0, *centre, [cell as f32; 2]);
        // Cut a window of about four radii around the creature, centred on it
        // (compact creatures lie well inside: their area is below 8 r^2).
        let w = ((4.0 * rs).ceil() as usize).min(cell);
        let mut window = vec![0.0f32; CHANNELS * w * w];
        let (sx, sy) = (centre[0].round() as usize + cell - w / 2, centre[1].round() as usize + cell - w / 2);
        for y in 0..w {
            for x in 0..w {
                let src = (origin.1 + (sy + y) % cell) * side + origin.0 + (sx + x) % cell;
                for c in 0..channels {
                    window[c * w * w + y * w + x] = late[c * n + src];
                }
            }
        }
        let mut chick = Hatchling { side: w, cells: window, heading: d[1].atan2(d[0]), species: host(t) };
        if params.template.is_none() {
            chick.species = chick.dominant(channels);
        }
        let chick = chick.rotated(rng.range(0.0, TAU));
        log::debug!(
            "lenia nursery: species {} mass {mass:.1} area {area} heading {:.0} deg",
            chick.species,
            chick.heading.to_degrees()
        );
        brood.push(chick);
    }
    log::debug!("lenia nursery: {} of {} tori hatched ({cell}-cell tori)", brood.len(), grid * grid);
    thin_brood(brood)
}

/// Shortest displacement from `a` to `b` on a torus of `size` cells.
fn torus_delta_cpu(a: [f32; 2], b: [f32; 2], size: [f32; 2]) -> [f32; 2] {
    let d = [b[0] - a[0], b[1] - a[1]];
    [d[0] - size[0] * (d[0] / size[0]).round(), d[1] - size[1] * (d[1] / size[1]).round()]
}

/// Side of one nursery torus (a multiple of `TILE`) for creatures of radius `r`.
fn nursery_cell(r: f32) -> usize {
    ((4.5 * r) as u32).div_ceil(TILE).max(2) as usize * TILE as usize
}

/// Initial state made of nursery hatchlings released at well-spaced random
/// spots, each turned by a random number of right angles, in proportion to
/// each species' abundance. Returns the state and the brood (kept for the
/// brush and for respawning); falls back to plain patches when nothing
/// hatched.
fn nursery_state(
    gpu: &Gpu,
    pipes: &StepPipelines,
    size: [u32; 2],
    params: &Params,
    seed: u64,
) -> (Vec<f32>, Vec<Hatchling>) {
    let brood = hatch(gpu, pipes, params, seed);
    if brood.is_empty() {
        let fallback = Params { seeding: Seeding::Patches, ..params.clone() };
        return (seed_state(size, &fallback, pipes.r_max, seed), brood);
    }
    let [w, h] = [size[0] as usize, size[1] as usize];
    let n = w * h;
    let channels = params.active_channels();
    let mut cells = vec![0.0f32; n * CHANNELS];
    let mut rng = Rng::new(seed ^ 0x005E_ED0F_11FE);
    let weights = species_weights(&brood.iter().map(|c| c.species).collect::<Vec<_>>(), params);
    let total: f32 = weights.iter().sum();
    // Creatures are sized by their own species (template species' cross-
    // kernels may reach much further than their bodies).
    let radius = |s: usize| {
        if params.template.is_some() { params.species_radius(s, pipes.r_max) } else { params.creature_radius(pipes.r_max) }
    };
    let area: f32 = (0..CHANNELS).map(|s| weights[s] / total.max(1e-6) * radius(s).powi(2)).sum();
    let count = ((n as f32) / (49.0 * area.max(1.0)) * params.seed_count.clamp(0.05, 8.0)).clamp(1.0, 400.0) as usize;
    let mut spots: Vec<([f32; 2], usize)> = Vec::new();
    for _ in 0..count * 30 {
        if spots.len() == count {
            break;
        }
        let s = pick_weighted(&weights, &mut rng);
        let p = [rng.range(0.0, w as f32), rng.range(0.0, h as f32)];
        let clear = spots.iter().all(|(q, t)| {
            let d = torus_delta_cpu(p, *q, [w as f32, h as f32]);
            let gap = 1.6 * (radius(s) + radius(*t));
            d[0] * d[0] + d[1] * d[1] > gap * gap
        });
        if clear {
            spots.push((p, s));
        }
    }
    for (p, s) in spots {
        let pool: Vec<&Hatchling> = brood.iter().filter(|chick| chick.species == s).collect();
        let Some(&chick) = pool.get(rng.below(pool.len() as u32) as usize) else { continue };
        let (planes, _) = chick.turned(rng.below(4));
        stamp_max(&mut cells, size, &planes, chick.side, p, channels);
    }

    // Channels whose species never hatched (for example colony-forming
    // species living beside template gliders) start from noise patches that
    // cover all of those channels together, as coupled species need.
    let empty: Vec<usize> = (0..channels).filter(|&c| !brood.iter().any(|chick| chick.species == c)).collect();
    if let Some(radius) = empty.iter().map(|&c| params.species_radius(c, pipes.r_max)).reduce(f32::max) {
        let count = ((n as f32) / (6.0 * radius).powi(2) * params.seed_count.clamp(0.05, 8.0)).clamp(2.0, 400.0) as usize;
        for _ in 0..count {
            let (cx, cy) = (rng.range(0.0, w as f32), rng.range(0.0, h as f32));
            let (disc, amp) = (radius * rng.range(0.7, 1.2), rng.range(0.6, 1.0));
            let ri = disc.ceil() as i32;
            for dy in -ri..=ri {
                for dx in -ri..=ri {
                    if (dx * dx + dy * dy) as f32 <= disc * disc {
                        let x = (cx as i32 + dx).rem_euclid(w as i32) as usize;
                        let y = (cy as i32 + dy).rem_euclid(h as i32) as usize;
                        for &c in &empty {
                            cells[c * n + y * w + x] = rng.f32() * amp;
                        }
                    }
                }
            }
        }
    }
    (cells, brood)
}

/// Release weight of each channel's species: its abundance if it hatched
/// (every hatched species evenly when all of those abundances are zero).
fn species_weights(hatched: &[usize], params: &Params) -> [f32; CHANNELS] {
    let mut weights = [0.0f32; CHANNELS];
    for &s in hatched {
        weights[s.min(CHANNELS - 1)] = params.abundance[s.min(CHANNELS - 1)].max(0.0);
    }
    if weights.iter().sum::<f32>() <= 0.0 {
        for &s in hatched {
            weights[s.min(CHANNELS - 1)] = 1.0;
        }
    }
    weights
}

// --- mutation ------------------------------------------------------------------

/// Outcome of a short blocking trial run on a small domain.
#[derive(Clone, Copy, Debug)]
struct Trial {
    /// Fraction of cells occupied (density > 0.1) at the end of the run.
    occupancy: f32,
    /// Occupancy at the end relative to earlier (flooding worlds keep spreading).
    spread: f32,
    /// Normalised L1 change over the last stretch: 0 frozen, ~1 all moving.
    activity: f32,
    /// Smallest channel's share of the mass times the channel count (1 = even).
    balance: f32,
}

impl Trial {
    /// Whether a nudged species (`self`) still behaves like the original
    /// (`reference`): alive, not flooding, still moving and no channel lost.
    fn resembles(&self, reference: &Trial) -> bool {
        let occ = reference.occupancy.max(0.005);
        (0.3 * occ..2.0 * occ + 0.02).contains(&self.occupancy)
            && self.spread < 1.3 * reference.spread.max(1.0)
            && self.activity > 0.5 * reference.activity
            && self.balance > 0.5 * reference.balance
    }
}

fn measure(early: &[f32], late: &[f32], cells: usize, channels: usize) -> Trial {
    if early.len() < cells * channels || late.len() < cells * channels {
        return Trial { occupancy: 0.0, spread: 0.0, activity: 0.0, balance: 0.0 };
    }
    let (mut occ_early, mut occ_late) = (0usize, 0usize);
    let (mut diff, mut total) = (0.0f64, 0.0f64);
    let mut per_channel = [0.0f64; CHANNELS];
    for i in 0..cells {
        let (mut a, mut b) = (0.0f32, 0.0f32);
        for (c, mass) in per_channel.iter_mut().enumerate().take(channels) {
            let (va, vb) = (early[c * cells + i], late[c * cells + i]);
            a += va;
            b += vb;
            *mass += vb as f64;
            diff += (va - vb).abs() as f64;
            total += (va + vb) as f64;
        }
        occ_early += usize::from(a > 0.1);
        occ_late += usize::from(b > 0.1);
    }
    let mass: f64 = per_channel.iter().sum();
    let balance = if channels == 1 || mass <= 0.0 {
        1.0
    } else {
        per_channel[..channels].iter().fold(f64::MAX, |m, &v| m.min(v)) / mass * channels as f64
    };
    Trial {
        occupancy: occ_late as f32 / cells as f32,
        spread: occ_late as f32 / occ_early.max(1) as f32,
        activity: if total > 0.0 { (diff / total) as f32 } else { 0.0 },
        balance: balance as f32,
    }
}

/// Runs `params` for a few hundred steps on a small torus and measures it.
fn run_trial(gpu: &Gpu, pipes: &StepPipelines, params: &Params, seed: u64) -> Trial {
    let r = params.creature_radius(pipes.r_max);
    let side = (((14.0 * r) as u32).clamp(192, 320) / TILE) * TILE;
    let size = [side, side];
    let channels = params.active_channels();
    let mut sim = Sim::new(gpu, pipes, size);
    sim.upload(gpu, &build_plan(params, pipes.r_max));
    sim.load(gpu, &seed_state(size, params, pipes.r_max, seed));
    let steps = (60.0 * params.time_res.clamp(1.0, 100.0)).clamp(150.0, 600.0) as u32;
    run_steps(gpu, pipes, &mut sim, steps * 3 / 4);
    let early = sim.read(gpu, channels);
    run_steps(gpu, pipes, &mut sim, steps / 4);
    let late = sim.read(gpu, channels);
    measure(&early, &late, (side * side) as usize, channels)
}

/// `base` with every kernel nudged by up to `amount` times a species-safe
/// step: close enough to stay alive, far enough to look and move differently.
/// Template species (Orbium) live in a narrow band of mu and sigma and are
/// hatched at their radius (a larger one means a larger, slower nursery), so
/// they get small steps throughout. Neutral cross-kernels stay neutral
/// (silent in empty space).
fn nudged(base: &Params, rng: &mut Rng, amount: f32) -> Params {
    let mut p = base.clone();
    let (dmu, dsigma, dradius) = if p.template.is_some() { (0.015, 0.04, 0.05) } else { (0.02, 0.05, 0.15) };
    let mut vary = |v: f32, step: f32| v * (1.0 + amount * rng.range(-step, step));
    for kernel in &mut p.kernels {
        let neutral = (kernel.mu / kernel.sigma() - NEUTRAL_MU).abs() < 1e-3;
        kernel.sigma = vary(kernel.sigma, dsigma);
        kernel.mu = if neutral { kernel.sigma * NEUTRAL_MU } else { vary(kernel.mu, dmu) };
        kernel.radius = vary(kernel.radius, dradius);
        kernel.h = vary(kernel.h, 0.25);
    }
    p.scale = vary(p.scale, dradius);
    p
}

/// Rust-literal description of a kernel set, logged so good mutations can be kept.
fn describe(params: &Params) -> String {
    let mut s = format!("channels {} T {:.2}\n", params.channels, params.time_res);
    for k in &params.kernels {
        let b: Vec<String> = k.b[..k.rings()].iter().map(|v| format!("{v:.3}")).collect();
        s += &format!(
            "    k({}, {}, {:.2}, &[{}], {:.4}, {:.4}, {:.3}),\n",
            k.source,
            k.target,
            k.radius,
            b.join(", "),
            k.mu,
            k.sigma,
            k.h
        );
    }
    s
}

// --- the world ---------------------------------------------------------------------

/// Format of the per-frame state and glow textures the display samples.
const TEX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Most mass a stamp spot may already hold (in state units).
const SPAWN_ROOM: f32 = 2.0;
/// Rate (per unit of simulated time) at which the quench fades over-crowded
/// blocks: 10% a frame at T 10 and 3 steps a frame, and as fast in simulated
/// time for species that take bigger steps (whose growth would outpace it).
const QUENCH_RATE: f32 = 0.35;
/// Share of the medium's light kept from one frame to the next (its lag).
const LIGHT_KEEP: f32 = 0.96;

/// Layouts, pipelines and uniforms of the brush, quench, compose, light and
/// draw passes (independent of the domain size).
struct Graphics {
    brush_layout: wgpu::BindGroupLayout,
    quench_layout: wgpu::BindGroupLayout,
    compose_layout: wgpu::BindGroupLayout,
    light_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    brush: wgpu::ComputePipeline,
    probe: wgpu::ComputePipeline,
    quench: wgpu::ComputePipeline,
    compose: wgpu::ComputePipeline,
    light: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
    /// Pointer paint and erase.
    brush_uniform: wgpu::Buffer,
    /// Creature stamps (pointer or respawn) and revival.
    spawn_uniform: wgpu::Buffer,
    quench_uniform: wgpu::Buffer,
    compose_uniform: wgpu::Buffer,
    light_uniform: wgpu::Buffer,
    draw_uniform: wgpu::Buffer,
    /// Linear, repeating: the display wraps the torus through it.
    wrap: wgpu::Sampler,
}

impl Graphics {
    fn new(gpu: &Gpu) -> Self {
        let module = gpu.shader("lenia render", include_str!("../shaders/lenia_render.wgsl"));
        let cs = ShaderStages::COMPUTE;
        let fs = ShaderStages::FRAGMENT;

        let brush_layout = gpu.bind_group_layout(
            "lenia brush",
            &[
                layout::uniform(0, cs),
                layout::storage(1, cs, false),
                layout::storage(2, cs, false),
                layout::storage(3, cs, true),
            ],
        );
        let brush_pl = gpu.pipeline_layout("lenia brush", &[&brush_layout]);
        let quench_layout = gpu.bind_group_layout(
            "lenia quench",
            &[layout::uniform(0, cs), layout::storage(1, cs, false), layout::storage(2, cs, true)],
        );
        let write_only = wgpu::StorageTextureAccess::WriteOnly;
        let compose_layout = gpu.bind_group_layout(
            "lenia compose",
            &[
                layout::uniform(0, cs),
                layout::storage(1, cs, true),
                layout::storage(2, cs, true),
                layout::storage(3, cs, true),
                layout::storage(4, cs, false),
                layout::storage(5, cs, false),
                layout::storage_texture(6, cs, TEX_FORMAT, write_only),
                layout::storage_texture(7, cs, TEX_FORMAT, write_only),
                layout::storage(8, cs, false),
            ],
        );
        let light_layout = gpu.bind_group_layout(
            "lenia light",
            &[layout::uniform(0, cs), layout::storage(1, cs, true), layout::storage(2, cs, false)],
        );
        let draw_layout = gpu.bind_group_layout(
            "lenia draw",
            &[
                layout::uniform(0, fs),
                layout::texture(1, fs, true),
                layout::texture(2, fs, true),
                layout::sampler(3, fs, true),
                layout::texture(4, fs, true),
                layout::texture(5, fs, true),
                layout::texture(6, fs, true),
                layout::sampler(7, fs, true),
                layout::storage(8, fs, true),
            ],
        );
        let compute = |label: &str, layout: &wgpu::BindGroupLayout, entry: &str| {
            gpu.compute_pipeline(label, &gpu.pipeline_layout(label, &[layout]), &module, entry)
        };
        Self {
            brush: gpu.compute_pipeline("lenia brush", &brush_pl, &module, "cs_brush"),
            probe: gpu.compute_pipeline("lenia probe", &brush_pl, &module, "cs_probe"),
            quench: compute("lenia quench", &quench_layout, "cs_quench"),
            compose: compute("lenia compose", &compose_layout, "cs_compose"),
            light: compute("lenia light", &light_layout, "cs_light"),
            draw: gpu.fullscreen_pipeline(
                "lenia draw",
                &gpu.pipeline_layout("lenia draw", &[&draw_layout]),
                &module,
                "fs_draw",
                SCENE_FORMAT,
                None,
            ),
            brush_layout,
            quench_layout,
            compose_layout,
            light_layout,
            draw_layout,
            brush_uniform: gpu.uniform_buffer("lenia brush", &BrushUniform::zeroed()),
            spawn_uniform: gpu.uniform_buffer("lenia spawn", &BrushUniform::zeroed()),
            quench_uniform: gpu.uniform_buffer("lenia quench", &QuenchUniform::zeroed()),
            compose_uniform: gpu.uniform_buffer("lenia compose", &ComposeUniform::zeroed()),
            light_uniform: gpu.uniform_buffer("lenia light", &LightUniform::zeroed()),
            draw_uniform: gpu.uniform_buffer("lenia draw", &DrawUniform::zeroed()),
            wrap: gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat),
        }
    }
}

/// Brush bind groups (`[i]` acts on `sim.state[i]`) for one brush uniform.
fn brush_binds(
    gpu: &Gpu,
    gfx: &Graphics,
    uniform: &wgpu::Buffer,
    sim: &Sim,
    mass: &wgpu::Buffer,
    stamp: &wgpu::Buffer,
) -> [wgpu::BindGroup; 2] {
    [0, 1].map(|i| {
        gpu.bind_group(
            "lenia brush",
            &gfx.brush_layout,
            &[
                uniform.as_entire_binding(),
                sim.state[i].as_entire_binding(),
                mass.as_entire_binding(),
                stamp.as_entire_binding(),
            ],
        )
    })
}

/// Everything sized by the simulation domain (rebuilt when a preset changes
/// the cell size).
struct Domain {
    size: [u32; 2],
    sim: Sim,
    /// Wake intensity per channel (vec4 per cell), ping-ponged: each compose
    /// pass diffuses one into the other.
    trails: [wgpu::Buffer; 2],
    /// Which of `trails` holds the latest wakes.
    trail: usize,
    /// Fixed-point bookkeeping: [0] total mass and [1..4] each channel's
    /// (compose), [4..8] the mass under each candidate spot of a stamp (probe).
    mass: wgpu::Buffer,
    /// Every brood window in every quarter turn, copied by `MODE_STAMP`.
    stamp: wgpu::Buffer,
    /// `[i]` acts on `sim.state[i]`.
    brush_binds: [wgpu::BindGroup; 2],
    spawn_binds: [wgpu::BindGroup; 2],
    quench_binds: [wgpu::BindGroup; 2],
    /// `[i][j]` reads `sim.state[i]` and `trails[j]`.
    compose_binds: [[wgpu::BindGroup; 2]; 2],
    light_bind: wgpu::BindGroup,
    draw_bind: wgpu::BindGroup,
}

impl Domain {
    fn new(gpu: &Gpu, pipes: &StepPipelines, gfx: &Graphics, luts: &[PaletteLut; CHANNELS], size: [u32; 2]) -> Self {
        let sim = Sim::new(gpu, pipes, size);
        let cells = size[0] as u64 * size[1] as u64;
        let none = wgpu::BufferUsages::empty();
        let trails = [gpu.storage_buffer("lenia trail a", cells * 16, none), gpu.storage_buffer("lenia trail b", cells * 16, none)];
        let mass = gpu.storage_buffer("lenia mass", 32, none);
        // Mass per 16x16 block (per channel and total), written by compose
        // and read by the quench and the light of the medium.
        let grid = [size[0].div_ceil(16) as u64, size[1].div_ceil(16) as u64];
        let blocks = gpu.storage_buffer("lenia blocks", grid[0] * grid[1] * 16, none);
        let light = gpu.storage_buffer("lenia light", grid[0] * grid[1] * 16, none);
        // Grown to fit the brood on every reset that hatches one.
        let stamp = gpu.storage_buffer("lenia stamp", 16, none);
        let tex_usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
        let (_, state_view) = gpu.texture_2d("lenia state texture", size, TEX_FORMAT, tex_usage);
        let (_, glow_view) = gpu.texture_2d("lenia glow texture", size, TEX_FORMAT, tex_usage);

        let brush = brush_binds(gpu, gfx, &gfx.brush_uniform, &sim, &mass, &stamp);
        let spawn = brush_binds(gpu, gfx, &gfx.spawn_uniform, &sim, &mass, &stamp);
        let quench_binds = [0, 1].map(|i| {
            gpu.bind_group(
                "lenia quench",
                &gfx.quench_layout,
                &[gfx.quench_uniform.as_entire_binding(), sim.state[i].as_entire_binding(), blocks.as_entire_binding()],
            )
        });
        let compose_binds = [0, 1].map(|i| {
            [0, 1].map(|j| {
                gpu.bind_group(
                    "lenia compose",
                    &gfx.compose_layout,
                    &[
                        gfx.compose_uniform.as_entire_binding(),
                        sim.state[i].as_entire_binding(),
                        sim.growth.as_entire_binding(),
                        trails[j].as_entire_binding(),
                        trails[1 - j].as_entire_binding(),
                        mass.as_entire_binding(),
                        wgpu::BindingResource::TextureView(&state_view),
                        wgpu::BindingResource::TextureView(&glow_view),
                        blocks.as_entire_binding(),
                    ],
                )
            })
        });
        let light_bind = gpu.bind_group(
            "lenia light",
            &gfx.light_layout,
            &[gfx.light_uniform.as_entire_binding(), blocks.as_entire_binding(), light.as_entire_binding()],
        );
        let draw_bind = gpu.bind_group(
            "lenia draw",
            &gfx.draw_layout,
            &[
                gfx.draw_uniform.as_entire_binding(),
                wgpu::BindingResource::TextureView(&state_view),
                wgpu::BindingResource::TextureView(&glow_view),
                wgpu::BindingResource::Sampler(&gfx.wrap),
                wgpu::BindingResource::TextureView(&luts[0].view),
                wgpu::BindingResource::TextureView(&luts[1].view),
                wgpu::BindingResource::TextureView(&luts[2].view),
                wgpu::BindingResource::Sampler(&luts[0].sampler),
                light.as_entire_binding(),
            ],
        );
        Self {
            size,
            sim,
            trails,
            trail: 0,
            mass,
            stamp,
            brush_binds: brush,
            spawn_binds: spawn,
            quench_binds,
            compose_binds,
            light_bind,
            draw_bind,
        }
    }

    /// Grows the stamp buffer to at least `bytes` (rebinding the brushes).
    fn ensure_stamp(&mut self, gpu: &Gpu, gfx: &Graphics, bytes: u64) {
        if bytes <= self.stamp.size() {
            return;
        }
        self.stamp = gpu.storage_buffer("lenia stamp", bytes, wgpu::BufferUsages::empty());
        self.brush_binds = brush_binds(gpu, gfx, &gfx.brush_uniform, &self.sim, &self.mass, &self.stamp);
        self.spawn_binds = brush_binds(gpu, gfx, &gfx.spawn_uniform, &self.sim, &self.mass, &self.stamp);
    }

    /// Blocks of 16x16 cells (one per compose workgroup).
    fn grid(&self) -> [u32; 2] {
        [self.size[0].div_ceil(16), self.size[1].div_ceil(16)]
    }

    fn cells(&self) -> f32 {
        self.size[0] as f32 * self.size[1] as f32
    }
}

/// One brood member in one quarter turn, uploaded to the stamp buffer.
struct StampWindow {
    /// First float of its planes in the stamp buffer.
    base: u32,
    side: u32,
    heading: f32,
    species: usize,
}

pub struct Lenia {
    /// Output size in pixels; the domain is derived from it and `cell_px`.
    output: [u32; 2],
    params: Params,
    preset: usize,
    look: PostSettings,
    luts: [PaletteLut; CHANNELS],
    pipes: StepPipelines,
    gfx: Graphics,
    dom: Domain,
    /// Kernel tables currently on the GPU (rebuilt when the parameters change).
    uploaded: Option<SimKey>,
    /// The brood of the current run (empty unless nursery seeding), as
    /// uploaded stamp windows.
    windows: Vec<StampWindow>,
    rest: [f32; 4],
    /// Mass of the seeded state (all channels), the reference for revival.
    target_mass: f32,
    /// Channels (bit mask) whose species hatched into the brood, and each
    /// one's seeded mass: respawning keeps every species near its own (a
    /// shared target would let lost giants be replaced by minnows).
    brood_mask: u32,
    species_mass: [f32; CHANNELS],
    /// Seed of the current run (drives revival and brush noise).
    run_seed: u64,
    /// Seed of a run not yet started: the world seeds itself on its first
    /// frame, so creating it and then loading another preset straight away
    /// (as the command line does) hatches only once.
    pending: Option<u64>,
    /// Frames stepped since the last reset.
    tick: u64,
    /// Brush strokes so far (each stroke paints its own noise).
    strokes: u32,
    painting: bool,
    /// Where the current stroke last released a creature (cells).
    last_stamp: Option<[f32; 2]>,
    /// A step happened since the last render (wakes decay once per frame).
    stepped: bool,
    /// Just reseeded: no growth field yet, wakes and quench blocks belong to
    /// the previous run.
    fresh: bool,
}

pub fn create(gpu: &Gpu, output_size: [u32; 2], seed: u64) -> Box<dyn World> {
    Box::new(Lenia::new(gpu, output_size, seed))
}

/// Domain for an output of `output` pixels at `cell_px` pixels per cell,
/// shrunk on very large outputs to keep the frame time in budget.
fn domain_size(output: [u32; 2], cell_px: f32) -> [u32; 2] {
    let px = cell_px.clamp(1.0, 4.0);
    let mut w = output[0].max(1) as f32 / px;
    let mut h = output[1].max(1) as f32 / px;
    if w * h > MAX_CELLS {
        let s = (MAX_CELLS / (w * h)).sqrt();
        w *= s;
        h *= s;
    }
    [(w as u32).max(64), (h as u32).max(64)]
}

impl Lenia {
    fn new(gpu: &Gpu, output: [u32; 2], seed: u64) -> Self {
        let pipes = StepPipelines::new(gpu);
        let gfx = Graphics::new(gpu);
        let luts = [0, 1, 2].map(|_| PaletteLut::new(gpu, 0));
        let def = preset(0);
        let dom = Domain::new(gpu, &pipes, &gfx, &luts, domain_size(output, def.params.cell_px));
        let mut world = Self {
            output,
            params: def.params,
            preset: 0,
            look: def.look,
            luts,
            pipes,
            gfx,
            dom,
            uploaded: None,
            windows: Vec::new(),
            rest: [0.0; 4],
            target_mass: 0.0,
            brood_mask: 0,
            species_mass: [0.0; CHANNELS],
            run_seed: seed,
            pending: Some(seed),
            tick: 0,
            strokes: 0,
            painting: false,
            last_stamp: None,
            stepped: false,
            fresh: true,
        };
        world.set_palettes(gpu, def.palettes);
        world
    }

    fn set_palettes(&mut self, gpu: &Gpu, names: [&str; 3]) {
        for (lut, name) in self.luts.iter_mut().zip(names) {
            lut.set(gpu, palette::find(name).unwrap_or(0));
        }
    }

    /// Starts the run deferred by `new`, if it has not started yet.
    fn start(&mut self, gpu: &Gpu) {
        if let Some(seed) = self.pending {
            self.reset(gpu, seed);
        }
    }

    /// Re-bakes and uploads the kernel tables when the parameters changed.
    fn sync(&mut self, gpu: &Gpu) {
        if !self.uploaded.as_ref().is_some_and(|key| key.matches(&self.params)) {
            let plan = build_plan(&self.params, self.pipes.r_max);
            self.dom.sim.upload(gpu, &plan);
            self.rest = plan.rest;
            self.uploaded = Some(self.params.sim_key());
        }
    }

    /// Uploads every brood member in all four quarter turns, once per run, so
    /// that stamping one later only takes a uniform write.
    fn upload_brood(&mut self, gpu: &Gpu, brood: &[Hatchling]) {
        self.windows.clear();
        let mut data: Vec<f32> = Vec::new();
        for chick in brood {
            for quarter in 0..4 {
                let (planes, heading) = chick.turned(quarter);
                let base = data.len() as u32;
                self.windows.push(StampWindow { base, side: chick.side as u32, heading, species: chick.species });
                data.extend_from_slice(&planes);
            }
        }
        if !data.is_empty() {
            self.dom.ensure_stamp(gpu, &self.gfx, (data.len() * 4) as u64);
            gpu.queue.write_buffer(&self.dom.stamp, 0, bytemuck::cast_slice(&data));
        }
    }

    /// A species of the brood, drawn by abundance.
    fn pick_species(&self, rng: &mut Rng) -> usize {
        let hatched: Vec<usize> = (0..CHANNELS).filter(|&c| self.brood_mask & (1 << c) != 0).collect();
        pick_weighted(&species_weights(&hatched, &self.params), rng)
    }

    /// A `MODE_STAMP` brush that drops the brood window of `species` whose
    /// heading is closest to `heading` on the first of `spots` holding at
    /// most `room` mass, provided that species' mass is below `threshold`
    /// (fixed point). `None` without a brood.
    fn stamp(
        &self,
        base: BrushUniform,
        species: usize,
        heading: f32,
        spots: [[f32; 2]; 4],
        threshold: u32,
        room: u32,
    ) -> Option<BrushUniform> {
        let off = |a: f32| ((a - heading + PI).rem_euclid(TAU) - PI).abs();
        let closest = |a: &&StampWindow, b: &&StampWindow| off(a.heading).total_cmp(&off(b.heading));
        let window =
            self.windows.iter().filter(|w| w.species == species).min_by(closest).or_else(|| self.windows.iter().min_by(closest))?;
        // Room is judged over the creature plus a margin of about a radius:
        // the whole stamp window is rarely empty in a crowded world.
        let radius = self.params.species_radius(window.species, self.pipes.r_max);
        let probe = ((3.0 * radius).ceil() as u32).min(window.side);
        let [a, b, c, d] = spots;
        Some(BrushUniform {
            mode: MODE_STAMP,
            threshold,
            room,
            watch: 1 << window.species,
            probe,
            side: window.side,
            base: window.base,
            spots: [[a[0], a[1], b[0], b[1]], [c[0], c[1], d[0], d[1]]],
            ..base
        })
    }
}

impl World for Lenia {
    fn settings(&self) -> anyhow::Result<crate::library::WorldSettings> {
        Ok(crate::library::WorldSettings::Lenia {
            params: self.params.clone(), palettes: std::array::from_fn(|i| PALETTES[self.luts[i].index()].name.to_owned()), post: self.look,
        })
    }

    fn restore_settings(&mut self, gpu: &Gpu, settings: &crate::library::WorldSettings, seed: u64) -> anyhow::Result<()> {
        let crate::library::WorldSettings::Lenia { params, palettes, post } = settings else { anyhow::bail!("Wrong world settings"); };
        anyhow::ensure!((1..=CHANNELS).contains(&params.channels) && !params.kernels.is_empty() && params.kernels.len() <= MAX_KERNELS, "Invalid Lenia kernels");
        // Disabled channels retain their kernels; build_plan folds those onto
        // the last active channel. Such recipes must still round-trip.
        anyhow::ensure!((1..=128).contains(&params.steps_per_frame) && params.kernels.iter().all(|k| k.source < CHANNELS && k.target < CHANNELS), "Invalid Lenia channels or step count");
        for name in palettes { anyhow::ensure!(palette::find(name).is_some(), "Unknown palette: {name}"); }
        self.params = params.clone();
        self.look = *post;
        self.set_palettes(gpu, [&palettes[0], &palettes[1], &palettes[2]]);
        self.reset(gpu, seed);
        Ok(())
    }

    fn id(&self) -> &'static str {
        "lenia"
    }

    fn name(&self) -> &'static str {
        "Lenia"
    }

    fn size(&self) -> [u32; 2] {
        self.dom.size
    }

    fn presets(&self) -> &'static [&'static str] {
        PRESET_NAMES
    }

    fn preset(&self) -> usize {
        self.preset
    }

    fn load_preset(&mut self, gpu: &Gpu, index: usize, seed: u64) {
        let index = index.min(PRESET_NAMES.len() - 1);
        let def = preset(index);
        self.preset = index;
        self.params = def.params;
        self.look = def.look;
        self.set_palettes(gpu, def.palettes);
        self.reset(gpu, seed);
    }

    fn reset(&mut self, gpu: &Gpu, seed: u64) {
        self.pending = None;
        let size = domain_size(self.output, self.params.cell_px);
        if size != self.dom.size {
            self.dom = Domain::new(gpu, &self.pipes, &self.gfx, &self.luts, size);
            self.uploaded = None;
        }
        self.windows.clear();
        let cells = if self.params.seeding == Seeding::Nursery {
            let (cells, brood) = nursery_state(gpu, &self.pipes, size, &self.params, seed);
            self.upload_brood(gpu, &brood);
            cells
        } else {
            seed_state(size, &self.params, self.pipes.r_max, seed)
        };
        let n = size[0] as usize * size[1] as usize;
        let channel_mass: Vec<f32> = cells.chunks_exact(n).map(|plane| plane.iter().sum()).collect();
        self.brood_mask = self.windows.iter().fold(0, |mask, w| mask | 1 << w.species);
        let mask = self.brood_mask;
        self.species_mass = std::array::from_fn(|c| if mask & (1 << c) != 0 { channel_mass[c] } else { 0.0 });
        self.target_mass = channel_mass.iter().sum();
        self.dom.sim.load(gpu, &cells);
        // Unknown mass until the next compose pass: never revive on frame one.
        gpu.queue.write_buffer(&self.dom.mass, 0, bytemuck::bytes_of(&u32::MAX));
        self.run_seed = seed;
        self.tick = 0;
        self.fresh = true;
        self.last_stamp = None;
    }

    fn mutate(&mut self, gpu: &Gpu, seed: u64) {
        let mut rng = Rng::new(seed ^ 0x1E41_A5EE_D5A1_7ED5);
        // A relative of a curated species. Random kernel sets are nearly
        // always dead or boiling (about one in twenty is lively), whereas a
        // nudged species lands somewhere interesting almost every time. The
        // nudge shrinks until a short trial run still behaves like the species;
        // template species are screened by the nursery instead (see below).
        let index = rng.below(PRESET_NAMES.len() as u32) as usize;
        let base = preset(index);
        let reference = base.params.template.is_none().then(|| run_trial(gpu, &self.pipes, &base.params, seed));
        let mut params = base.params.clone();
        let mut amount = 1.0;
        for attempt in 0..MUTATE_ATTEMPTS {
            let candidate = nudged(&base.params, &mut rng, amount);
            let fits = match &reference {
                Some(reference) => {
                    let trial = run_trial(gpu, &self.pipes, &candidate, seed.wrapping_add(1 + attempt as u64));
                    log::debug!("lenia mutation trial {attempt}: {trial:?} (species: {reference:?})");
                    trial.resembles(reference)
                }
                None => true,
            };
            if fits {
                params = candidate;
                break;
            }
            amount *= 0.6;
        }
        log::info!("lenia mutation of {}:\n{}", PRESET_NAMES[index], describe(&params));

        // A random but coherent look around the species' own.
        let b = &base.params;
        params.gain = b.gain * rng.range(0.85, 1.15);
        params.glow = b.glow.map(|g| g * rng.range(0.8, 1.25));
        params.halo = rng.range(0.1, 0.4);
        params.tint = rng.range(0.3, 0.55);
        params.trail = if params.template.is_some() || rng.chance(0.3) { rng.range(0.4, 1.0) } else { 0.0 };
        params.rim = b.rim.map(|r| r * rng.range(0.6, 1.4));
        params.ground = b.ground * rng.range(0.7, 1.4);
        params.medium = b.medium * rng.range(0.6, 1.5);
        let palettes = if params.active_channels() == 1 {
            // Any palette on a dark ground (the last one, Ink, is light).
            let name = PALETTES[rng.below(PALETTES.len() as u32 - 1) as usize].name;
            [name; 3]
        } else {
            *rng.pick(TRIADS)
        };
        let mut look = base.look;
        look.exposure *= rng.range(0.9, 1.1);
        look.bloom *= rng.range(0.85, 1.2);
        self.preset = index;
        self.params = params;
        self.look = look;
        self.set_palettes(gpu, palettes);
        self.reset(gpu, seed);
        // A template species nudged out of its niche hatches nothing: fall
        // back to the species' own kernels (keeping the new look).
        if self.params.template.is_some() && self.windows.is_empty() {
            log::info!("lenia mutation hatched nothing; keeping {}'s own kernels", PRESET_NAMES[index]);
            self.params.kernels = base.params.kernels;
            self.params.scale = base.params.scale;
            self.reset(gpu, seed);
        }
    }

    fn step(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder) {
        let gpu = frame.gpu;
        self.start(gpu);
        self.sync(gpu);
        let size = self.dom.size;
        let fsize = [size[0] as f32, size[1] as f32];
        let grid = self.dom.grid();
        let channels = self.params.active_channels() as u32;
        let r_max = self.pipes.r_max;
        let r = self.params.creature_radius(r_max);
        let base = BrushUniform { size, channels, ..BrushUniform::zeroed() };
        let room = (SPAWN_ROOM * MASS_SCALE) as u32;

        // Pointer. The primary button paints noise, or, when the run has a
        // brood of hatched creatures, releases them along the stroke heading
        // the way it moves; the secondary button erases.
        let pointer = frame.pointer.filter(|p| p.primary || p.secondary);
        let primary = pointer.is_some_and(|p| p.primary);
        if primary && !self.painting {
            self.strokes = self.strokes.wrapping_add(1);
            self.last_stamp = None;
        }
        self.painting = primary;
        let mut paint = None;
        // A creature stamp or a revival patch, and whether it needs probing.
        let mut spawn = None;
        if let Some(p) = pointer {
            let centre = [p.pos[0] * fsize[0], p.pos[1] * fsize[1]];
            if primary && !self.windows.is_empty() {
                let spacing = (3.0 * r).max(1.5 * p.radius);
                let heading = match self.last_stamp {
                    None => Some(Rng::new(self.run_seed ^ self.strokes as u64).range(0.0, TAU)),
                    Some(last) => {
                        let d = torus_delta_cpu(last, centre, fsize);
                        (d[0].hypot(d[1]) >= spacing).then(|| d[1].atan2(d[0]))
                    }
                };
                if let Some(heading) = heading {
                    let mut rng = Rng::new(self.run_seed ^ ((self.strokes as u64) << 32) ^ self.tick);
                    let species = self.pick_species(&mut rng);
                    // Candidates: under the cursor, beside the stroke on either
                    // side, then behind it, so a stroke through a crowd finds
                    // room instead of dropping a creature onto another.
                    let reach = 1.5 * self.params.species_radius(species, r_max);
                    let (hx, hy) = (heading.cos() * reach, heading.sin() * reach);
                    let [cx, cy] = centre;
                    let spots = [centre, [cx - hy, cy + hx], [cx + hy, cy - hx], [cx - hx, cy - hy]];
                    spawn = self.stamp(base, species, heading, spots, u32::MAX, room).map(|brush| (brush, true));
                    self.last_stamp = Some(centre);
                }
            } else {
                paint = Some(BrushUniform {
                    center: centre,
                    radius: p.radius.max(2.0),
                    mode: if primary { MODE_PAINT } else { MODE_ERASE },
                    seed: (self.run_seed as u32) ^ self.strokes.wrapping_mul(0x9E37_79B9),
                    amplitude: 1.0,
                    ..base
                });
            }
        }

        // Every few frames: respawn while the world's mass is below its share
        // of the seeded mass (or revive a world that died out). The GPU
        // decides, from the mass summed by the last compose pass.
        if spawn.is_none() && self.params.revive && self.tick % REVIVE_EVERY == REVIVE_EVERY - 1 {
            let mut rng = Rng::new(self.run_seed ^ self.tick.wrapping_mul(0x2545_F491_4F6C_DD1D));
            let mut spot = || [rng.range(0.0, fsize[0]), rng.range(0.0, fsize[1])];
            let spots = [spot(), spot(), spot(), spot()];
            let respawn = self.params.respawn.clamp(0.0, 1.0);
            let extinct = self.dom.cells() * EXTINCT_LEVEL;
            let fixed = |m: f32| (m.max(extinct) * MASS_SCALE).min(u32::MAX as f32 * 0.5) as u32;
            spawn = if self.windows.is_empty() {
                let threshold = fixed(self.target_mass * respawn);
                let mut rng = Rng::new(self.run_seed ^ self.tick);
                let brush = BrushUniform {
                    center: spots[0],
                    radius: r * rng.range(1.0, 1.5),
                    mode: MODE_REVIVE,
                    seed: rng.next_u32(),
                    amplitude: rng.range(0.7, 1.0),
                    threshold,
                    ..base
                };
                Some((brush, false))
            } else {
                let species = self.pick_species(&mut rng);
                let heading = rng.range(0.0, TAU);
                let threshold = fixed(self.species_mass[species] * respawn);
                self.stamp(base, species, heading, spots, threshold, room).map(|brush| (brush, true))
            };
        }

        // The quench judges crowding from the block masses of the last compose
        // pass, which still describe the previous run until this one has been
        // composed once.
        let quench = !self.fresh && self.params.quench.iter().any(|&q| q > 0.0);
        if quench {
            // Each channel's neighbourhood reaches about 1.5 radii of its own
            // species beyond the block; the limit is a mean density over it.
            let frame_time = self.params.steps_per_frame.clamp(1, 16) as f32 * self.params.dt();
            let keep = (-QUENCH_RATE * frame_time).exp();
            let mut uniform = QuenchUniform { size, grid, keep, channels, ..QuenchUniform::zeroed() };
            for c in 0..CHANNELS {
                let reach = (1.5 * self.params.species_radius(c, r_max) / 16.0).ceil().clamp(1.0, 4.0) as u32;
                let area = ((2 * reach + 1) * 16).pow(2) as f32;
                uniform.reach[c] = reach;
                // A channel with the quench off gets a limit it can never reach.
                let q = self.params.quench[c];
                uniform.limit[c] =
                    if q > 0.0 { (q * area * MASS_SCALE).min(u32::MAX as f32 * 0.5) as u32 } else { u32::MAX };
            }
            gpu.write(&self.gfx.quench_uniform, &uniform);
        }

        if spawn.is_some_and(|(_, probe)| probe) {
            encoder.clear_buffer(&self.dom.mass, 16, Some(16));
        }
        if let Some(brush) = &paint {
            gpu.write(&self.gfx.brush_uniform, brush);
        }
        if let Some((brush, _)) = &spawn {
            gpu.write(&self.gfx.spawn_uniform, brush);
        }
        let current = self.dom.sim.current;
        let mut pass =
            encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lenia step"), timestamp_writes: None });
        if paint.is_some() {
            pass.set_pipeline(&self.gfx.brush);
            pass.set_bind_group(0, &self.dom.brush_binds[current], &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        if let Some((brush, probe)) = spawn {
            pass.set_bind_group(0, &self.dom.spawn_binds[current], &[]);
            if probe {
                pass.set_pipeline(&self.gfx.probe);
                pass.dispatch_workgroups(4, 1, 1);
            }
            pass.set_pipeline(&self.gfx.brush);
            if brush.mode == MODE_STAMP {
                pass.dispatch_workgroups(brush.side.div_ceil(16), brush.side.div_ceil(16), 1);
            } else {
                pass.dispatch_workgroups(grid[0], grid[1], 1);
            }
        }
        if quench {
            pass.set_pipeline(&self.gfx.quench);
            pass.set_bind_group(0, &self.dom.quench_binds[current], &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        self.dom.sim.record(&mut pass, &self.pipes, self.params.steps_per_frame.clamp(1, 16));
        drop(pass);
        self.tick += 1;
        self.stepped = true;
    }

    fn render(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let gpu = frame.gpu;
        self.start(gpu);
        self.sync(gpu);
        let size = self.dom.size;
        let grid = self.dom.grid();
        let (fresh, stepped) = (self.fresh, self.stepped);
        if fresh {
            // A new run: forget the previous run's wakes (the medium's light
            // restarts below).
            for trail in &self.dom.trails {
                encoder.clear_buffer(trail, 0, None);
            }
        }
        // Total and per-channel mass; the stamp probes are cleared when used.
        encoder.clear_buffer(&self.dom.mass, 0, Some(16));
        let p = &self.params;
        let channels = p.active_channels() as u32;
        gpu.write(
            &self.gfx.compose_uniform,
            &ComposeUniform {
                size,
                channels,
                decay: if stepped { p.persistence.clamp(0.0, 0.995) } else { 1.0 },
                // The growth field is the previous run's until a step has run.
                halo: if fresh && !stepped { 0.0 } else { p.halo },
                trail: p.trail,
                spread: if stepped { 1.0 } else { 0.0 },
                _pad: 0.0,
                rest: self.rest,
            },
        );
        gpu.write(
            &self.gfx.light_uniform,
            &LightUniform {
                grid,
                keep: if fresh {
                    0.0
                } else if stepped {
                    LIGHT_KEEP
                } else {
                    1.0
                },
                scale: 1.0 / (MASS_SCALE * 256.0),
            },
        );
        let lanes = |v: [f32; CHANNELS]| [v[0], v[1], v[2], 0.0];
        gpu.write(
            &self.gfx.draw_uniform,
            &DrawUniform {
                view: frame.view,
                size: [size[0] as f32, size[1] as f32],
                grid,
                channels,
                relief: p.relief,
                gain: p.gain,
                brightness: p.brightness,
                tint: p.tint,
                mix_power: p.mix_power.clamp(0.5, 6.0),
                ground: p.ground,
                medium: p.medium,
                level: lanes(p.level),
                rim: lanes(p.rim),
                // Below 1, so the shader's smoothstep(core, 1, v) stays defined.
                core: lanes(p.core.map(|c| c.clamp(0.0, 0.98))),
                glow: lanes(p.glow),
                hue: lanes(p.hue.map(|h| h.clamp(0.0, 0.6))),
                ground_channel: p.ground_channel.min(p.active_channels() - 1) as u32,
                sharp: p.sharp.clamp(0.0, 1.0),
                _pad: [0.0; 2],
            },
        );
        {
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("lenia compose"), timestamp_writes: None });
            pass.set_pipeline(&self.gfx.compose);
            pass.set_bind_group(0, &self.dom.compose_binds[self.dom.sim.current][self.dom.trail], &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
            pass.set_pipeline(&self.gfx.light);
            pass.set_bind_group(0, &self.dom.light_bind, &[]);
            pass.dispatch_workgroups(grid[0].div_ceil(8), grid[1].div_ceil(8), 1);
        }
        self.dom.trail = 1 - self.dom.trail;
        gpu::fullscreen_pass(encoder, "lenia draw", target, Some(wgpu::Color::BLACK), &self.gfx.draw, &[&self.dom.draw_bind]);
        if stepped {
            self.fresh = false;
        }
        self.stepped = false;
    }

    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) {
        let r_max = self.pipes.r_max as f32;
        let Self { params: p, luts, .. } = self;

        ui.add(crate::ui::Slider::new(&mut p.time_res, 1.0..=30.0).text("Time resolution T"));
        ui.add(crate::ui::Slider::new(&mut p.steps_per_frame, 1..=8).text("Steps / frame"));
        ui.add(crate::ui::Slider::new(&mut p.scale, 0.5..=2.0).text("Kernel scale"));
        // Kernels of switched-off channels fold onto the last active one
        // while simulating (see `build_plan`); their own settings are kept.
        ui.add(crate::ui::Slider::new(&mut p.channels, 1..=CHANNELS).text("Channels"));
        crate::ui::dropdown(ui, "Seeding (on reset)", p.seeding.name(), |ui| {
            for s in Seeding::ALL {
                ui.selectable_value(&mut p.seeding, s, s.name());
            }
        });
        ui.checkbox(&mut p.revive, "Revive / respawn life");
        ui.add(crate::ui::Slider::new(&mut p.respawn, 0.0..=1.0).text("Respawn below (share of seeded mass)"));
        ui.add(crate::ui::Slider::new(&mut p.cell_px, 1.5..=4.0).text("Cell size in pixels (on reset)"));
        for c in 0..p.active_channels() {
            ui.add(crate::ui::Slider::new(&mut p.quench[c], 0.0..=0.3).text(format!("Explosion quench, channel {}", c + 1)));
        }

        ui.separator();
        ui.label(format!("Kernels ({})", p.kernels.len()));
        let channels = p.active_channels();
        let count = p.kernels.len();
        let mut remove = None;
        for (i, kernel) in p.kernels.iter_mut().enumerate() {
            let title = format!(
                "K{}  {} > {}  R {:.0}  μ {:.3}  σ {:.3}",
                i + 1,
                kernel.source + 1,
                kernel.target + 1,
                kernel.radius,
                kernel.mu,
                kernel.sigma
            );
            egui::CollapsingHeader::new(title).id_salt(("lenia kernel", i)).show(ui, |ui| {
                if channels > 1 {
                    ui.horizontal(|ui| {
                        channel_picker(ui, ("lenia src", i), "From", &mut kernel.source, channels);
                        channel_picker(ui, ("lenia dst", i), "To", &mut kernel.target, channels);
                    });
                }
                ui.add(crate::ui::Slider::new(&mut kernel.radius, 3.0..=r_max).text("Radius R"));
                ui.add(crate::ui::Slider::new(&mut kernel.mu, 0.01..=0.6).text("Growth centre μ").fixed_decimals(3));
                ui.add(
                    crate::ui::Slider::new(&mut kernel.sigma, 0.002..=0.25)
                        .logarithmic(true)
                        .text("Growth width σ")
                        .fixed_decimals(4),
                );
                ui.add(crate::ui::Slider::new(&mut kernel.h, 0.05..=1.0).text("Weight h"));
                ui.add(crate::ui::Slider::new(&mut kernel.rings, 1..=3).text("Rings"));
                for (j, b) in kernel.b.iter_mut().enumerate().take(kernel.rings) {
                    ui.add(crate::ui::Slider::new(b, 0.0..=1.0).text(format!("Ring {} peak", j + 1)));
                }
                if count > 1 && ui.small_button("Remove kernel").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            p.kernels.remove(i);
        }
        if p.kernels.len() < MAX_KERNELS && ui.button("Add kernel").clicked() {
            let mut kernel = p.kernels.last().copied().unwrap_or_else(|| k(0, 0, 13.0, &[1.0], 0.15, 0.015, 1.0));
            kernel.h *= 0.5;
            p.kernels.push(kernel);
        }

        ui.separator();
        for (c, lut) in luts.iter_mut().enumerate().take(channels) {
            let mut index = lut.index();
            ui.label(format!("Channel {}", c + 1));
            if palette::combo(ui, &format!("lenia palette {c}"), &mut index) {
                lut.set(gpu, index);
            }
            ui.add(crate::ui::Slider::new(&mut p.level[c], 0.0..=2.0).text("Brightness"));
            ui.add(crate::ui::Slider::new(&mut p.rim[c], 0.0..=2.0).text("Membrane rim"));
            ui.add(crate::ui::Slider::new(&mut p.hue[c], 0.0..=0.6).text("Body palette start"));
            ui.add(crate::ui::Slider::new(&mut p.core[c], 0.3..=0.98).text("Nuclei glow above"));
            ui.add(crate::ui::Slider::new(&mut p.glow[c], 0.0..=4.0).text("Nucleus glow"));
            if p.seeding == Seeding::Nursery && channels > 1 {
                ui.add(crate::ui::Slider::new(&mut p.abundance[c], 0.0..=8.0).text("Abundance (on reset)"));
            }
        }
        ui.add(crate::ui::Slider::new(&mut p.gain, 0.5..=3.0).text("Palette gain"));
        ui.add(crate::ui::Slider::new(&mut p.halo, 0.0..=2.0).text("Growth halo"));
        ui.add(crate::ui::Slider::new(&mut p.trail, 0.0..=3.0).text("Wakes"));
        ui.add(crate::ui::Slider::new(&mut p.persistence, 0.8..=0.99).text("Wake persistence"));
        ui.add(crate::ui::Slider::new(&mut p.tint, 0.0..=1.0).text("Halo and wake tint"));
        ui.add(crate::ui::Slider::new(&mut p.relief, 0.0..=1.0).text("Relief lighting"));
        ui.add(crate::ui::Slider::new(&mut p.mix_power, 1.0..=4.0).text("Colour mixing (blend … dominant)"));
        ui.add(crate::ui::Slider::new(&mut p.sharp, 0.0..=1.0).text("Sharpness (soft … crisp)"));
        if channels > 1 {
            channel_picker(ui, "lenia ground", "Medium from channel", &mut p.ground_channel, channels);
        }
        ui.add(crate::ui::Slider::new(&mut p.ground, 0.0..=4.0).text("Medium hue"));
        ui.add(crate::ui::Slider::new(&mut p.medium, 0.0..=2.0).text("Light cast into the medium"));
        ui.add(crate::ui::Slider::new(&mut p.brightness, 0.2..=3.0).text("Brightness"));
    }

    fn post_settings(&self) -> PostSettings {
        self.look
    }

    fn stats(&self) -> String {
        let p = &self.params;
        let radii = p.kernels.iter().map(|k| p.radius_of(k, self.pipes.r_max));
        let (lo, hi) = radii.fold((f32::MAX, 0.0f32), |(lo, hi), r| (lo.min(r), hi.max(r)));
        format!(
            "{}x{} cells · {} ch · {} kernels · R {:.0}-{:.0} · dt {:.2}",
            self.dom.size[0],
            self.dom.size[1],
            p.active_channels(),
            p.kernels.len(),
            lo.min(hi),
            hi,
            p.dt()
        )
    }

    fn controls_hint(&self) -> &'static str {
        "Left: release creatures / seed life · Right: erase"
    }
}

/// Channel combo for a kernel end. A kernel on a switched-off channel shows
/// as such (it is folded onto the last active channel while simulating).
fn channel_picker(ui: &mut egui::Ui, id: impl std::hash::Hash, label: &str, value: &mut usize, channels: usize) {
    let text = if *value < channels { format!("{label} {}", *value + 1) } else { format!("{label} {} (off)", *value + 1) };
    egui::ComboBox::from_id_salt(id).selected_text(text).width(72.0).show_ui(ui, |ui| {
        for c in 0..channels {
            ui.selectable_value(value, c, format!("Channel {}", c + 1));
        }
    });
}
