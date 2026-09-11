//! Particle Life: N particles of K species attract and repel each other through
//! an asymmetric matrix (Jeffrey Ventrella's "Clusters", Tom Mohr's force model)
//! and self-assemble into cells, worms, rotating suns and hunting comets on a torus.
//!
//! Simulation (`particle_life.wgsl`), per sub-step:
//! 1. `cs_count`     bins every particle into a uniform grid (cell size >= r_max)
//!    with an atomic counter, remembering its rank inside the cell;
//! 2. `cs_scan`      one workgroup turns the counts into cell start offsets (and
//!    zeroes the counters for the next sub-step);
//! 3. `cs_scatter`   copies the particles into cell order;
//! 4. `cs_force`     nine threads per sorted particle, one per cell of its 3x3
//!    block, sum the pair forces in fixed point;
//! 5. `cs_integrate` adds the nine partials in a fixed order, applies friction,
//!    the brush and the speed limit, and writes back in particle-id order.
//!
//! Fixed-point sums and id-ordered write-back make every run exactly
//! reproducible from its seed, despite the atomic (unordered) binning.
//!
//! Rendering (`particle_life_draw.wgsl`): every particle is an instanced quad,
//! with one instance per torus tile the view can see. Soft glows accumulate in
//! an HDR trail texture that fades once per simulation step; each glow is
//! stretched along the particle's motion since the last frame, so fast movers
//! draw continuous streaks. Crisp heads are drawn into a second buffer each
//! frame, and the composite turns both into light with a logarithmic density
//! roll-off, a chroma floor (overlapping species keep a hue instead of
//! averaging to grey) and a hue-preserving highlight shoulder. When the camera
//! is zoomed far out, the buffers hold a single domain period that the
//! composite repeats, so the cost does not grow with the number of copies.
//!
//! The domain is measured in abstract units: its aspect ratio follows the
//! output and its area follows the particle count and density, so a preset
//! behaves identically at any output resolution. The presets were picked by
//! rendering a few hundred random matrices at this density and keeping the
//! ones that stay alive, distinct and colourful across seeds.

use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use super::{Frame, ViewXform, World};
use crate::gpu::{self, layout, Gpu, SCENE_FORMAT};
use crate::palette::{self, PALETTES};
use crate::post::{PostSettings, Tonemap};
use crate::rng::Rng;

const MAX_KINDS: usize = 8;
const WORKGROUP: u32 = 256;
const MAX_SUBSTEPS: u32 = 8;
/// Particle count limits (a new count applies on restart).
const MIN_COUNT: u32 = 2048;
const MAX_COUNT: u32 = 262_144;
/// Particles per square unit (a new density applies on restart).
const MIN_DENSITY: f32 = 0.006;
const MAX_DENSITY: f32 = 0.06;
/// Interaction radius limits in world units; the grid is allocated for the minimum.
const MIN_RADIUS: f32 = 12.0;
const MAX_RADIUS: f32 = 120.0;
/// The radius is also capped so that about this many neighbours fall within it
/// on average: beyond that the pair loop, not the physics, dominates, and no
/// slider combination should be able to stall the app.
const MAX_NEIGHBOURS: f32 = 600.0;
/// The force is normalised to this many neighbours within r_max on average, so
/// changing the radius or the density rescales structures without destabilising them.
const REFERENCE_NEIGHBOURS: f32 = 30.0;
/// Particle size is given in pixels for a 1080-pixel-high output at zoom 1.
const REFERENCE_HEIGHT: f32 = 1080.0;
/// Speed (in r_max per second) at which a particle is ~63% "hot".
const SPEED_REF: f32 = 1.5;
/// Per-species size multipliers.
const MIN_SPECIES_SIZE: f32 = 0.25;
const MAX_SPECIES_SIZE: f32 = 3.0;
/// Beyond this many domain periods across the view, the canvas holds one
/// period that the composite repeats, instead of instancing every copy.
const WRAP_PERIODS: f32 = 3.0;
/// Safety cap on the copies per axis in the per-copy (screen) canvas mode.
const MAX_TILES: u32 = 8;

type Matrix = [[f32; MAX_KINDS]; MAX_KINDS];

const ONES: Matrix = [[1.0; MAX_KINDS]; MAX_KINDS];
const EVEN: [f32; MAX_KINDS] = [1.0; MAX_KINDS];

/// Pads a smaller square matrix into a `Matrix` (unused entries are zero).
const fn pad<const N: usize>(rows: [[f32; N]; N]) -> Matrix {
    let mut m = [[0.0; MAX_KINDS]; MAX_KINDS];
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            m[i][j] = rows[i][j];
            j += 1;
        }
        i += 1;
    }
    m
}

/// Pads the first entries of a per-species table; the rest are `fill`.
const fn per_species<const N: usize>(values: [f32; N], fill: f32) -> [f32; MAX_KINDS] {
    let mut out = [fill; MAX_KINDS];
    let mut i = 0;
    while i < N {
        out[i] = values[i];
        i += 1;
    }
    out
}

// --- colours -------------------------------------------------------------------

/// A hand-picked set of species colours (sRGB hex). Each preset has its own:
/// one dominant hue family plus an accent, with the space-filling species
/// (foam, lace, networks) held at lower value so the organisms stay the
/// subject. Entries past a preset's species count only show when the user
/// adds species.
struct Scheme {
    name: &'static str,
    colors: [u32; MAX_KINDS],
}

// Scheme indices, so presets refer to them without a name lookup.
const TIDEPOOL: usize = 0;
const CELLS: usize = 1;
const EMBER: usize = 2;
const SOLAR: usize = 3;
const LACE: usize = 4;
const MARBLE: usize = 5;
const HUNT: usize = 6;
const GLACIER: usize = 7;
const ORCHID: usize = 8;
const JEWELS: usize = 9;

const SCHEMES: &[Scheme] = &[
    // Coral nuclei, deep ultramarine foam, aqua eel heads, gold eel bodies.
    Scheme { name: "Tidepool", colors: [0xff6a55, 0x1a3fb0, 0x2cf0c8, 0xffc53d, 0x8a5cff, 0x00b4ff, 0xff8fb1, 0x4dffa6] },
    // Vermilion nuclei, sapphire membranes, turquoise rings, a dim amber network.
    Scheme { name: "Cells", colors: [0xff5a3d, 0x2f6bff, 0x3fe0d0, 0x9a6400, 0x8a5cff, 0x00c8ff, 0xff7ac8, 0xa6ff3d] },
    // Dim crimson and rust blobs, gold beads, violet bodies (then rose, amber, magenta, scarlet).
    Scheme { name: "Ember", colors: [0xb0203f, 0xb8501a, 0xffc23d, 0x9340ff, 0xff4fa3, 0xffa200, 0xd61aff, 0xff4a1c] },
    // A warm-only sunset: gold, orange, red, pink, vermilion (mixtures stay
    // warm), over a dim ember dust.
    Scheme { name: "Solar", colors: [0xffd23f, 0xff8c1a, 0xff3d3d, 0xff4fa8, 0xff5a36, 0x7a2812, 0xffb000, 0xc04dff] },
    // Deep emerald lace, cyan swimmers, violet, gold nuclei.
    Scheme { name: "Lace", colors: [0x12a866, 0x00c8ff, 0xa64dff, 0xffc94d, 0xff4fa3, 0x2f6bff, 0x00ffd0, 0xb8ff3d] },
    // Gold, orange and cream stars over dim violet and indigo gases.
    Scheme { name: "Marble", colors: [0xffd36b, 0xff7a3d, 0xfff2c0, 0x5a2a8a, 0x2a3a9a, 0xff4f8b, 0x19e6c8, 0x3d8bff] },
    // Magenta hunters, cyan prey.
    Scheme { name: "Hunt", colors: [0xff2e88, 0x00d2ff, 0x9dff00, 0x7a3dff, 0xffd000, 0x00ffa2, 0xff5c00, 0x3d7bff] },
    // Ice blue, cobalt, a warm pink accent, lavender, indigo, teal, azure, mint.
    Scheme { name: "Glacier", colors: [0x4da6ff, 0x2f4bff, 0xff6fae, 0xb18cff, 0x7a5cff, 0x00e6d2, 0x00a2ff, 0x62ffd0] },
    // A single cyan accent, rose and magenta beads (then violet, orchid, indigo, pink, gold).
    Scheme { name: "Orchid", colors: [0x00c8ff, 0xff8fb1, 0xff4fa3, 0x8a5cff, 0xc04dff, 0x5a3dff, 0xff6ad5, 0xffc94d] },
    // Ruby, sapphire, emerald, amber, violet, cyan, pink, lime.
    Scheme { name: "Jewels", colors: [0xff3b5c, 0x2f7bff, 0x19e68c, 0xffb000, 0xb44dff, 0x00d4e6, 0xff7ac8, 0xa6ff3d] },
    // Evenly spaced hues.
    Scheme { name: "Spectrum", colors: [0xff4d4d, 0xffa64d, 0xfff04d, 0x6cff4d, 0x4dffd2, 0x4da6ff, 0x9f4dff, 0xff4dd2] },
];

// JEWELS is the highest index referenced by name: this fails the build if the
// table ever loses an entry the presets rely on.
const _: () = assert!(JEWELS < SCHEMES.len());

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Colors {
    /// One of the hand-picked sets in `SCHEMES`.
    Scheme(usize),
    /// Evenly spaced samples of a `palette::PALETTES` gradient between `lo` and `hi`.
    Gradient { palette: usize, lo: f32, hi: f32 },
}

// --- spawning and random matrices ----------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spawn {
    Uniform,
    Clusters,
    Disc,
}

impl Spawn {
    const ALL: [Spawn; 3] = [Spawn::Uniform, Spawn::Clusters, Spawn::Disc];

    fn name(self) -> &'static str {
        match self {
            Spawn::Uniform => "Uniform soup",
            Spawn::Clusters => "Scattered clusters",
            Spawn::Disc => "Central disc",
        }
    }
}

/// A free-form random matrix: most species cling to their own kind, cross
/// terms are arbitrary.
fn random_matrix(rng: &mut Rng, k: usize) -> Matrix {
    let mut m = [[0.0; MAX_KINDS]; MAX_KINDS];
    for (i, row) in m.iter_mut().enumerate().take(k) {
        for (j, value) in row.iter_mut().enumerate().take(k) {
            *value = if i != j {
                rng.range(-1.0, 1.0)
            } else if rng.chance(0.85) {
                rng.range(0.15, 1.0)
            } else {
                rng.range(-0.4, 0.15)
            };
        }
    }
    m
}

/// Rejects the dull outcomes seen while exploring random matrices:
/// * nearly reciprocal matrices settle into a frozen lattice;
/// * mostly attractive ones collapse into a few dense blobs (which are also by
///   far the slowest to simulate);
/// * when every species clings to its own kind, each condenses into isolated
///   dots that end up out of each other's reach. A "spreader" (weak or negative
///   self-attraction) forms a foam, lace or membrane instead, which keeps the
///   world connected and moving.
fn is_lively(m: &Matrix, k: usize) -> bool {
    let mut sum = 0.0;
    let mut asymmetry = 0.0;
    let mut spreader = false;
    for i in 0..k {
        spreader |= m[i][i] < 0.15;
        for j in 0..k {
            sum += m[i][j];
            if j > i {
                asymmetry += (m[i][j] - m[j][i]).abs();
            }
        }
    }
    let pairs = (k * (k - 1) / 2).max(1) as f32;
    spreader && sum / (k * k) as f32 <= 0.25 && asymmetry / pairs >= 0.5
}

/// A random matrix that passes `is_lively` (roughly a third of draws do, so 64
/// attempts practically never run out; the last draw is used if they do).
fn lively_matrix(rng: &mut Rng, k: usize) -> Matrix {
    let mut m = random_matrix(rng, k);
    for _ in 0..64 {
        if is_lively(&m, k) {
            break;
        }
        m = random_matrix(rng, k);
    }
    m
}

// --- presets -------------------------------------------------------------------

/// Suggested post-processing (plus the ground colour) for a preset.
#[derive(Clone, Copy)]
struct Look {
    exposure: f32,
    bloom: f32,
    threshold: f32,
    vignette: f32,
    saturation: f32,
    tonemap: Tonemap,
    ground: u32,
}

impl Look {
    fn post(self) -> PostSettings {
        PostSettings {
            exposure: self.exposure,
            bloom: self.bloom,
            bloom_threshold: self.threshold,
            vignette: self.vignette,
            saturation: self.saturation,
            grain: 0.0,
            tonemap: self.tonemap,
        }
    }
}

#[derive(Clone, Copy)]
struct Preset {
    name: &'static str,
    kinds: usize,
    matrix: Matrix,
    radii: Matrix,
    weights: [f32; MAX_KINDS],
    r_max: f32,
    beta: f32,
    force: f32,
    half_life: f32,
    dt: f32,
    substeps: u32,
    count: u32,
    density: f32,
    spawn: Spawn,
    colors: Colors,
    color_shift: usize,
    sizes: [f32; MAX_KINDS],
    size: f32,
    glow: f32,
    trail: f32,
    trail_gain: f32,
    trail_scale: f32,
    speed_glow: f32,
    knee: f32,
    relief: f32,
    look: Look,
}

/// ACES keeps saturated species colours saturated; bloom only picks up the
/// genuinely hot parts (fast comets and dense knots) above 1.0.
const BASE_LOOK: Look = Look {
    exposure: 1.0,
    bloom: 0.6,
    threshold: 1.0,
    vignette: 0.35,
    saturation: 1.1,
    tonemap: Tonemap::Aces,
    ground: 0x030408,
};

/// Shared settings. At 32k particles and density 0.04 about 200 neighbours fall
/// within r_max, which is where structures grow large enough to read as
/// organisms at 1080p; one 0.02 s step per frame looks the same as two of
/// 0.01 s and costs half as much.
const BASE: Preset = Preset {
    name: "",
    kinds: 4,
    // Every preset brings its own matrix.
    matrix: [[0.0; MAX_KINDS]; MAX_KINDS],
    radii: ONES,
    weights: EVEN,
    r_max: 40.0,
    beta: 0.3,
    force: 10.0,
    half_life: 0.04,
    dt: 0.02,
    substeps: 1,
    count: 32_768,
    density: 0.04,
    spawn: Spawn::Uniform,
    colors: Colors::Scheme(JEWELS),
    color_shift: 0,
    sizes: EVEN,
    size: 3.0,
    glow: 1.0,
    trail: 0.85,
    trail_gain: 0.6,
    trail_scale: 1.8,
    speed_glow: 1.0,
    knee: 0.45,
    relief: 0.0,
    look: BASE_LOOK,
};

const PRESETS: &[Preset] = &[
    // A deep-blue foam of coral-nucleus cells, combed into currents by golden
    // eels with aqua heads that swim straight through it.
    Preset {
        name: "Tidepool",
        kinds: 4,
        matrix: pad([
            [0.85, -0.90, 0.36, 0.03],
            [-0.90, -0.29, -0.59, -0.15],
            [-0.76, 0.09, 0.84, 0.04],
            [-0.26, -0.90, 0.39, 0.79],
        ]),
        weights: per_species([0.77, 0.77, 1.0, 1.0], 1.0),
        sizes: per_species([1.07, 1.0, 1.0, 1.0], 1.0),
        relief: 0.5,
        colors: Colors::Scheme(TIDEPOOL),
        look: Look { ground: 0x02050f, ..BASE_LOOK },
        ..BASE
    },
    // Membrane-bound cells (blue) of different sizes with layered nuclei,
    // packed in an amber intercellular network; they drift, merge and coarsen.
    Preset {
        name: "Living Cells",
        kinds: 4,
        matrix: pad([
            [0.98, -0.35, 0.63, -0.03],
            [-0.64, 0.15, -0.09, 0.90],
            [0.27, 0.26, 0.43, -0.13],
            [0.55, -0.75, -0.68, -0.16],
        ]),
        weights: per_species([1.0, 0.8, 0.55, 1.0], 1.0),
        spawn: Spawn::Clusters,
        relief: 0.5,
        colors: Colors::Scheme(CELLS),
        look: Look { ground: 0x04060e, ..BASE_LOOK },
        ..BASE
    },
    // A writhing soup of segmented violet worms with golden beads.
    Preset {
        name: "Serpents",
        kinds: 4,
        matrix: pad([
            [0.44, -0.40, -0.08, -0.72],
            [0.90, 0.32, -0.77, -0.19],
            [0.54, -0.21, -0.26, 0.94],
            [0.08, -0.56, 0.18, -0.15],
        ]),
        weights: per_species([0.3, 0.35, 1.0, 1.0], 1.0),
        r_max: 56.0,
        trail_gain: 0.45,
        colors: Colors::Scheme(EMBER),
        ..BASE
    },
    // Each species follows the next: spinning yin-yang suns that unroll into
    // comets and roll up again, ploughing dark wakes through a dim ember dust
    // (a sixth species that feels nothing but the repulsion core).
    Preset {
        name: "Rotating Suns",
        kinds: 6,
        matrix: pad([
            [0.92, 0.44, -0.13, -0.74, -0.11, 0.0],
            [0.15, 0.89, 0.80, 0.08, -0.30, 0.0],
            [-0.50, -0.18, 0.83, 0.62, 0.10, 0.0],
            [-0.59, -0.03, 0.22, 0.68, 0.31, 0.0],
            [0.59, -0.31, -0.74, -0.17, 0.64, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        sizes: per_species([1.0, 1.0, 1.0, 1.0, 1.0, 0.8], 1.0),
        r_max: 30.0,
        speed_glow: 0.6,
        trail: 0.75,
        colors: Colors::Scheme(SOLAR),
        look: Look { saturation: 1.2, ..BASE_LOOK },
        ..BASE
    },
    // A spreading emerald lace with sparse gold-nucleus cells and bright
    // swimmers darting between them.
    Preset {
        name: "Lace",
        kinds: 4,
        matrix: pad([
            [-0.33, 0.29, 0.80, -0.95],
            [0.73, 0.48, -0.85, -0.81],
            [-0.46, -0.44, 0.00, -0.45],
            [0.34, 0.79, -0.42, 0.56],
        ]),
        weights: per_species([1.0, 1.0, 1.0, 0.4], 1.0),
        relief: 0.5,
        colors: Colors::Scheme(LACE),
        ..BASE
    },
    // Two gases that feel nothing but the repulsion core, which reaches
    // further between unlike particles than between like ones: they demix
    // into a marbled labyrinth that slowly coarsens, while warm stars and
    // comets (the first three species) drift through it and carve dark rivers.
    Preset {
        name: "Marbling",
        kinds: 5,
        matrix: pad([
            [0.91, 0.72, 0.07, 0.0, 0.0],
            [0.63, 0.50, 0.99, 0.0, 0.0],
            [0.03, -0.52, 1.00, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]),
        radii: pad([
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.3, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.3],
        ]),
        density: 0.05,
        size: 2.0,
        glow: 0.5,
        trail: 0.8,
        trail_gain: 0.8,
        trail_scale: 2.5,
        speed_glow: 0.5,
        colors: Colors::Scheme(MARBLE),
        look: Look { saturation: 1.2, ..BASE_LOOK },
        ..BASE
    },
    // Packs of big magenta hunters drag through sheets of cyan prey. Hunters
    // sense prey from the full radius, prey only notice them up close, and the
    // prey barely cling to each other, so they stream rather than ball up.
    Preset {
        name: "Predator & Prey",
        kinds: 2,
        matrix: pad([[0.3, 0.9], [-0.7, 0.05]]),
        radii: pad([[0.6, 1.0], [0.5, 1.0]]),
        weights: per_species([0.35, 1.0], 1.0),
        sizes: per_species([1.6, 1.0], 1.0),
        colors: Colors::Scheme(HUNT),
        ..BASE
    },
    // A food chain: plankton clump and flee the fish, which school and flee
    // the rare pink jellies, which hunt the fish.
    Preset {
        name: "Plankton",
        kinds: 3,
        matrix: pad([[0.5, -0.8, 0.0], [0.9, 0.3, -0.9], [0.0, 0.9, -0.3]]),
        weights: per_species([1.0, 0.5, 0.15], 1.0),
        trail_gain: 0.45,
        knee: 0.35,
        relief: 0.5,
        colors: Colors::Scheme(GLACIER),
        ..BASE
    },
    // Strings of rose and magenta beads that wander, break and re-thread,
    // with sparse cyan shards between them: each bead pulls the next species
    // along while the third keeps them apart.
    Preset {
        name: "Necklaces",
        kinds: 3,
        matrix: pad([[0.14, 0.12, -0.32], [0.34, -0.27, 0.72], [-0.76, 0.25, 0.21]]),
        weights: per_species([0.25, 1.0, 1.0], 1.0),
        trail_gain: 0.4,
        colors: Colors::Scheme(ORCHID),
        ..BASE
    },
];

fn preset_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| PRESETS.iter().map(|p| p.name).collect())
}

// --- parameters ----------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub kinds: usize,
    /// `matrix[i][j]`: how strongly species i is attracted to species j.
    pub matrix: Matrix,
    /// `radii[i][j]`: interaction radius of the pair as a fraction of `r_max`.
    pub radii: Matrix,
    /// Relative abundance of each species.
    pub weights: [f32; MAX_KINDS],
    pub r_max: f32,
    pub beta: f32,
    pub force: f32,
    /// Seconds for friction to halve a particle's velocity.
    pub half_life: f32,
    pub dt: f32,
    pub substeps: u32,
    /// Spring constant (1/s^2) of the attracting brush; the repelling brush
    /// pushes with three times this times its radius (units/s^2).
    pub pointer_strength: f32,
    /// Particle count, density and spawn pattern apply on restart.
    pub count: u32,
    pub density: f32,
    pub spawn: Spawn,
    pub colors: Colors,
    /// Rotates which colour goes to which species.
    pub color_shift: usize,
    /// Size multiplier of each species (heads and glows).
    pub sizes: [f32; MAX_KINDS],
    /// Particle radius in pixels at 1080p, zoom 1.
    pub size: f32,
    pub glow: f32,
    /// Fraction of the trail kept per simulation step.
    pub trail: f32,
    pub trail_gain: f32,
    /// Trail glow radius relative to the particle radius.
    pub trail_scale: f32,
    pub speed_glow: f32,
    /// Brightness above which overlapping light is rolled off.
    pub knee: f32,
    /// Strength of the density shading that models dense cores as lit domes.
    pub relief: f32,
}

impl Params {
    fn from_preset(p: &Preset) -> Self {
        Self {
            kinds: p.kinds.clamp(2, MAX_KINDS),
            matrix: p.matrix,
            radii: p.radii,
            weights: p.weights,
            r_max: p.r_max,
            beta: p.beta,
            force: p.force,
            half_life: p.half_life,
            dt: p.dt,
            substeps: p.substeps,
            pointer_strength: 60.0,
            count: p.count,
            density: p.density,
            spawn: p.spawn,
            colors: p.colors,
            color_shift: p.color_shift,
            sizes: p.sizes,
            size: p.size,
            glow: p.glow,
            trail: p.trail,
            trail_gain: p.trail_gain,
            trail_scale: p.trail_scale,
            speed_glow: p.speed_glow,
            knee: p.knee,
            relief: p.relief,
        }
    }
}

// --- GPU mirrors -----------------------------------------------------------------

/// Mirrors `Sim` in particle_life.wgsl (640 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimUniform {
    domain: [f32; 2],
    grid: [u32; 2],
    cell: [f32; 2],
    count: u32,
    kinds: u32,
    r_max: f32,
    beta: f32,
    force: f32,
    friction: f32,
    dt: f32,
    max_speed: f32,
    pointer: [f32; 2],
    pointer_radius: f32,
    pointer_mode: u32,
    pointer_strength: f32,
    cells: u32,
    salt: u32,
    _pad: [u32; 3],
    mat: [[f32; 4]; 16],
    rad: [[f32; 4]; 16],
    cdf: [[f32; 4]; 2],
}

/// Mirrors `Draw` in particle_life_draw.wgsl (256 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    view: ViewXform,
    screen: ViewXform,
    domain: [f32; 2],
    canvas: [f32; 2],
    tiles: [u32; 2],
    count: u32,
    wrap: u32,
    head_radius: f32,
    trail_radius: f32,
    head_gain: f32,
    trail_gain: f32,
    speed_ref: f32,
    speed_glow: f32,
    fade: f32,
    knee: f32,
    motion: f32,
    relief: f32,
    zoom1_ppu: f32,
    _pad: f32,
    ground: [f32; 4],
    /// rgb = linear colour, a = size multiplier.
    colors: [[f32; 4]; MAX_KINDS],
}

const _: () = assert!(std::mem::size_of::<SimUniform>() == 640);
const _: () = assert!(std::mem::size_of::<DrawUniform>() == 256);

/// Packs an 8x8 table row-major into 16 vec4s, mapping each entry through `f`.
fn pack_table(m: &Matrix, f: impl Fn(f32) -> f32) -> [[f32; 4]; 16] {
    let mut out = [[0.0; 4]; 16];
    for (i, row) in m.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            let idx = i * MAX_KINDS + j;
            out[idx / 4][idx % 4] = f(v);
        }
    }
    out
}

/// `pcg_hash` from common.wgsl.
fn pcg_hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Species of particle `id`; must match `species()` in particle_life.wgsl.
fn species_of(id: u32, salt: u32, cdf: &[f32; MAX_KINDS], kinds: usize) -> f32 {
    let u = ((pcg_hash(id ^ salt) % 840) as f32 + 0.5) / 840.0;
    (0..kinds.saturating_sub(1)).filter(|&j| u >= cdf[j]).count() as f32
}

/// Domain size in world units for `count` particles at `density` and this aspect ratio.
fn domain_for(count: u32, density: f32, aspect: f32) -> [u32; 2] {
    let area = count as f32 / density;
    let h = (area / aspect).sqrt();
    [((h * aspect).round() as u32).max(64), (h.round() as u32).max(64)]
}

/// Grid cells needed for the smallest interaction radius on this domain.
fn cell_capacity(size: [u32; 2]) -> u32 {
    let gx = ((size[0] as f32 / MIN_RADIUS) as u32).max(3);
    let gy = ((size[1] as f32 / MIN_RADIUS) as u32).max(3);
    gx * gy
}

/// Buffers whose size depends on the particle count and the domain.
struct Buffers {
    particles: u32,
    cells: u32,
    /// (x, y, species, gene) in the order of the last sort.
    pos: wgpu::Buffer,
    vel: wgpu::Buffer,
    counts: wgpu::Buffer,
    sim_group: wgpu::BindGroup,
    draw_group: wgpu::BindGroup,
    /// Sorted copies, cell starts and scratch: only referenced by `sim_group`.
    _scratch: [wgpu::Buffer; 4],
}

impl Buffers {
    fn new(
        gpu: &Gpu,
        sim_layout: &wgpu::BindGroupLayout,
        draw_layout: &wgpu::BindGroupLayout,
        sim_uniform: &wgpu::Buffer,
        draw_uniform: &wgpu::Buffer,
        particles: u32,
        cells: u32,
    ) -> Self {
        let n = particles as u64;
        let none = wgpu::BufferUsages::empty();
        let pos = gpu.storage_buffer("pl pos", n * 16, none);
        let vel = gpu.storage_buffer("pl vel", n * 8, none);
        let sorted_pos = gpu.storage_buffer("pl sorted pos", n * 16, none);
        let sorted_vel = gpu.storage_buffer("pl sorted vel", n * 8, none);
        let counts = gpu.storage_buffer("pl cell counts", cells as u64 * 4, none);
        let starts = gpu.storage_buffer("pl cell starts", (cells as u64 + 1) * 4, none);
        // Slots, then nine partial forces per particle (see particle_life.wgsl).
        let scratch = gpu.storage_buffer("pl scratch", n * 9 * 8, none);
        let sim_group = gpu.bind_group(
            "pl sim",
            sim_layout,
            &[
                sim_uniform.as_entire_binding(),
                pos.as_entire_binding(),
                vel.as_entire_binding(),
                sorted_pos.as_entire_binding(),
                sorted_vel.as_entire_binding(),
                counts.as_entire_binding(),
                starts.as_entire_binding(),
                scratch.as_entire_binding(),
            ],
        );
        let draw_group = gpu.bind_group(
            "pl draw",
            draw_layout,
            &[draw_uniform.as_entire_binding(), pos.as_entire_binding(), vel.as_entire_binding()],
        );
        Self { particles, cells, pos, vel, counts, sim_group, draw_group, _scratch: [sorted_pos, sorted_vel, starts, scratch] }
    }
}

/// Where the particles are drawn this frame.
struct CanvasPlan {
    size: [u32; 2],
    /// The canvas holds one domain period that the composite repeats.
    wrap: bool,
    /// Canvas uv -> world uv for the sprite passes.
    view: ViewXform,
}

/// HDR buffers the particles are drawn into: the fading trail and this frame's
/// particle heads.
struct Canvas {
    size: [u32; 2],
    wrap: bool,
    trail: wgpu::TextureView,
    heads: wgpu::TextureView,
    /// Both textures, sampled by the composite pass.
    group: wgpu::BindGroup,
}

/// `dst * src` blending, used to fade the trail texture.
const BLEND_MULTIPLY: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::Src,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::Src,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Instanced additive quads (4-vertex strips) into `SCENE_FORMAT`.
fn sprite_pipeline(
    gpu: &Gpu,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
) -> wgpu::RenderPipeline {
    gpu.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs_entry),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fs_entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: SCENE_FORMAT,
                blend: Some(gpu::BLEND_ADDITIVE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}

fn color_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    })
}

// --- the world -------------------------------------------------------------------

pub struct ParticleLife {
    /// Output aspect ratio the domain was shaped for.
    aspect: f32,
    /// Domain size in world units.
    size: [u32; 2],
    /// Particles currently simulated (`params.count` applies on restart).
    count: u32,
    /// Spawn pattern of the current run (`params.spawn` applies on restart).
    applied_spawn: Spawn,
    params: Params,
    preset: usize,
    seed: u64,
    /// Per-seed salt of the species hash.
    salt: u32,
    look: PostSettings,
    /// Background colour (sRGB hex).
    ground: u32,
    /// Drives the "Randomise" button.
    rng: Rng,
    /// Simulation steps and simulated seconds since the last render: the trail
    /// fades per step and streaks span the motion.
    pending_steps: u32,
    pending_motion: f32,
    sim_uniform: wgpu::Buffer,
    draw_uniform: wgpu::Buffer,
    sim_layout: wgpu::BindGroupLayout,
    draw_layout: wgpu::BindGroupLayout,
    canvas_layout: wgpu::BindGroupLayout,
    count_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    force_pipeline: wgpu::ComputePipeline,
    integrate_pipeline: wgpu::ComputePipeline,
    fade_pipeline: wgpu::RenderPipeline,
    trail_pipeline: wgpu::RenderPipeline,
    composite_pipeline: wgpu::RenderPipeline,
    head_pipeline: wgpu::RenderPipeline,
    buffers: Buffers,
    canvas: Option<Canvas>,
    clear_trail: bool,
}

pub fn create(gpu: &Gpu, output_size: [u32; 2], seed: u64) -> Box<dyn World> {
    Box::new(ParticleLife::new(gpu, output_size, seed))
}

impl ParticleLife {
    pub fn new(gpu: &Gpu, output_size: [u32; 2], seed: u64) -> Self {
        let aspect = (output_size[0].max(1) as f32 / output_size[1].max(1) as f32).clamp(0.25, 4.0);
        let sim_module = gpu.shader("particle-life sim", include_str!("../shaders/particle_life.wgsl"));
        let draw_module = gpu.shader("particle-life draw", include_str!("../shaders/particle_life_draw.wgsl"));

        let cs = ShaderStages::COMPUTE;
        let sim_layout = gpu.bind_group_layout(
            "pl sim",
            &[
                layout::uniform(0, cs),
                layout::storage(1, cs, false),
                layout::storage(2, cs, false),
                layout::storage(3, cs, false),
                layout::storage(4, cs, false),
                layout::storage(5, cs, false),
                layout::storage(6, cs, false),
                layout::storage(7, cs, false),
            ],
        );
        let sim_pl = gpu.pipeline_layout("pl sim", &[&sim_layout]);
        let count_pipeline = gpu.compute_pipeline("pl count", &sim_pl, &sim_module, "cs_count");
        let scan_pipeline = gpu.compute_pipeline("pl scan", &sim_pl, &sim_module, "cs_scan");
        let scatter_pipeline = gpu.compute_pipeline("pl scatter", &sim_pl, &sim_module, "cs_scatter");
        let force_pipeline = gpu.compute_pipeline("pl force", &sim_pl, &sim_module, "cs_force");
        let integrate_pipeline = gpu.compute_pipeline("pl integrate", &sim_pl, &sim_module, "cs_integrate");

        let vs = ShaderStages::VERTEX;
        let draw_layout = gpu.bind_group_layout(
            "pl draw",
            &[
                layout::uniform(0, vs | ShaderStages::FRAGMENT),
                layout::storage(1, vs, true),
                layout::storage(2, vs, true),
            ],
        );
        let fs = ShaderStages::FRAGMENT;
        let canvas_layout =
            gpu.bind_group_layout("pl canvas", &[layout::texture(0, fs, false), layout::texture(1, fs, false)]);
        let draw_pl = gpu.pipeline_layout("pl draw", &[&draw_layout]);
        let composite_pl = gpu.pipeline_layout("pl composite", &[&draw_layout, &canvas_layout]);
        let fade_pipeline =
            gpu.fullscreen_pipeline("pl fade", &draw_pl, &draw_module, "fs_fade", SCENE_FORMAT, Some(BLEND_MULTIPLY));
        let composite_pipeline =
            gpu.fullscreen_pipeline("pl composite", &composite_pl, &draw_module, "fs_composite", SCENE_FORMAT, None);
        let trail_pipeline = sprite_pipeline(gpu, "pl trail", &draw_pl, &draw_module, "vs_trail", "fs_trail");
        let head_pipeline = sprite_pipeline(gpu, "pl heads", &draw_pl, &draw_module, "vs_head", "fs_head");

        let sim_uniform = gpu.uniform_buffer("pl sim", &SimUniform::zeroed());
        let draw_uniform = gpu.uniform_buffer("pl draw", &DrawUniform::zeroed());

        let params = Params::from_preset(&PRESETS[0]);
        let size = domain_for(params.count, params.density, aspect);
        let buffers = Buffers::new(
            gpu,
            &sim_layout,
            &draw_layout,
            &sim_uniform,
            &draw_uniform,
            params.count,
            cell_capacity(size),
        );

        let mut world = Self {
            aspect,
            size,
            count: params.count,
            applied_spawn: params.spawn,
            params,
            preset: 0,
            seed,
            salt: 0,
            look: PRESETS[0].look.post(),
            ground: PRESETS[0].look.ground,
            rng: Rng::new(seed),
            pending_steps: 0,
            pending_motion: 0.0,
            sim_uniform,
            draw_uniform,
            sim_layout,
            draw_layout,
            canvas_layout,
            count_pipeline,
            scan_pipeline,
            scatter_pipeline,
            force_pipeline,
            integrate_pipeline,
            fade_pipeline,
            trail_pipeline,
            composite_pipeline,
            head_pipeline,
            buffers,
            canvas: None,
            clear_trail: true,
        };
        world.load_preset(gpu, 0, seed);
        world
    }

    /// Effective interaction radius: the requested one, capped so that about
    /// `MAX_NEIGHBOURS` fall within it and so that the grid keeps at least 3
    /// cells per axis (the 3x3 block then never double counts), and never
    /// below `MIN_RADIUS` (the grid is allocated for that).
    fn r_max(&self) -> f32 {
        let [w, h] = [self.size[0] as f32, self.size[1] as f32];
        let density = self.count as f32 / (w * h);
        let crowd_cap = (MAX_NEIGHBOURS / (std::f32::consts::PI * density.max(1e-6))).sqrt();
        self.params.r_max.min(crowd_cap).min(w.min(h) / 3.0).clamp(MIN_RADIUS, MAX_RADIUS)
    }

    fn grid(&self) -> [u32; 2] {
        let r = self.r_max();
        [((self.size[0] as f32 / r) as u32).max(3), ((self.size[1] as f32 / r) as u32).max(3)]
    }

    /// Simulated seconds per sub-step.
    fn sim_dt(&self) -> f32 {
        self.params.dt.clamp(0.001, 0.05)
    }

    /// Speed limit: a safety net only, a quarter of r_max per sub-step. (A
    /// lower, "physical" cap is not Galilean invariant and distorts rotating or
    /// drifting structures.)
    fn max_speed(&self) -> f32 {
        0.25 * self.r_max() / self.sim_dt()
    }

    /// Cumulative species weights (entries past `kinds - 1` are never read).
    fn species_cdf(&self) -> [f32; MAX_KINDS] {
        let k = self.params.kinds;
        let weights = self.params.weights.map(|w| w.clamp(0.0, 1.0));
        let total: f32 = weights[..k].iter().sum();
        let mut cdf = [1.0; MAX_KINDS];
        let mut acc = 0.0;
        for (s, slot) in cdf.iter_mut().enumerate().take(k) {
            acc += if total > 1e-3 { weights[s] / total } else { 1.0 / k as f32 };
            *slot = acc;
        }
        cdf
    }

    /// Linear RGB colour of every species.
    fn species_colors(&self) -> [[f32; 3]; MAX_KINDS] {
        let p = &self.params;
        let k = p.kinds.max(1);
        std::array::from_fn(|s| match p.colors {
            Colors::Scheme(i) => {
                let scheme = &SCHEMES[i.min(SCHEMES.len() - 1)];
                palette::hex_to_linear(scheme.colors[(s + p.color_shift) % MAX_KINDS])
            }
            Colors::Gradient { palette, lo, hi } => {
                let t = lo + (hi - lo) * (((s + p.color_shift) % k) as f32 + 0.5) / k as f32;
                let c = PALETTES[palette.min(PALETTES.len() - 1)].sample(t);
                // Keep dark palette ends visible: lift luminance to a floor.
                let lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
                const FLOOR: f32 = 0.05;
                if lum >= FLOOR {
                    c
                } else if lum > 1e-4 {
                    c.map(|v| v * FLOOR / lum)
                } else {
                    [FLOOR; 3]
                }
            }
        })
    }

    fn species_colors_u8(&self) -> [egui::Color32; MAX_KINDS] {
        self.species_colors().map(|c| {
            let [r, g, b] = c.map(|v| (palette::linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8);
            egui::Color32::from_rgb(r, g, b)
        })
    }

    /// Initial particles: positions from the spawn pattern, a random gene each,
    /// zero velocity.
    fn spawn(&self, seed: u64) -> Vec<[f32; 4]> {
        let mut rng = Rng::new(seed);
        let [w, h] = [self.size[0] as f32, self.size[1] as f32];
        let cdf = self.species_cdf();
        let r = self.r_max();
        let blobs: Vec<[f32; 2]> = match self.params.spawn {
            Spawn::Clusters => {
                let n = ((w * h) / (7.0 * r).powi(2)).clamp(8.0, 400.0) as usize;
                (0..n).map(|_| [rng.range(0.0, w), rng.range(0.0, h)]).collect()
            }
            _ => Vec::new(),
        };
        (0..self.count)
            .map(|id| {
                let (x, y) = match self.params.spawn {
                    Spawn::Uniform => (rng.range(0.0, w), rng.range(0.0, h)),
                    Spawn::Clusters => {
                        let c = blobs[rng.below(blobs.len() as u32) as usize];
                        (c[0] + rng.normal() * 1.2 * r, c[1] + rng.normal() * 1.2 * r)
                    }
                    Spawn::Disc => {
                        let radius = 0.3 * w.min(h) * rng.f32().sqrt();
                        let a = rng.range(0.0, std::f32::consts::TAU);
                        (0.5 * w + radius * a.cos(), 0.5 * h + radius * a.sin())
                    }
                };
                let wrap = |v: f32, size: f32| v.rem_euclid(size).min(size - 0.001);
                [wrap(x, w), wrap(y, h), species_of(id, self.salt, &cdf, self.params.kinds), id as f32]
            })
            .collect()
    }

    fn sim_uniform(&self, frame: &Frame) -> SimUniform {
        let p = &self.params;
        let r_max = self.r_max();
        let grid = self.grid();
        let domain = [self.size[0] as f32, self.size[1] as f32];
        let dt = self.sim_dt();
        let neighbours = self.count as f32 / (domain[0] * domain[1]) * std::f32::consts::PI * r_max * r_max;
        let normalise = (REFERENCE_NEIGHBOURS / neighbours.max(1.0)).clamp(0.05, 4.0);
        let (pointer, pointer_radius, pointer_mode) = match frame.pointer {
            Some(ptr) if ptr.primary || ptr.secondary => (
                [ptr.pos[0] * domain[0], ptr.pos[1] * domain[1]],
                ptr.radius.max(1.0),
                if ptr.primary { 1 } else { 2 },
            ),
            _ => ([0.0; 2], 0.0, 0),
        };
        let cdf = self.species_cdf();
        SimUniform {
            domain,
            grid,
            cell: [domain[0] / grid[0] as f32, domain[1] / grid[1] as f32],
            count: self.count,
            kinds: p.kinds.clamp(2, MAX_KINDS) as u32,
            r_max,
            beta: p.beta.clamp(0.05, 0.9),
            force: p.force.clamp(0.0, 100.0) * r_max * normalise,
            friction: 0.5f32.powf(dt / p.half_life.max(1e-3)),
            dt,
            max_speed: self.max_speed(),
            pointer,
            pointer_radius,
            pointer_mode,
            pointer_strength: p.pointer_strength.clamp(0.0, 1000.0),
            cells: grid[0] * grid[1],
            salt: self.salt,
            _pad: [0; 3],
            mat: pack_table(&p.matrix, |v| v.clamp(-1.0, 1.0)),
            rad: pack_table(&p.radii, |v| v.clamp(0.25, 1.0)),
            cdf: [[cdf[0], cdf[1], cdf[2], cdf[3]], [cdf[4], cdf[5], cdf[6], cdf[7]]],
        }
    }

    /// The canvas follows the output pixels, unless the view spans so many
    /// periods that instancing every copy would be wasteful (far zoom-outs,
    /// only reachable from the CLI): then it holds one period, at about the
    /// resolution that period has on screen, and the composite repeats it.
    fn canvas_plan(&self, frame: &Frame) -> CanvasPlan {
        let target = [frame.target_size[0].max(1), frame.target_size[1].max(1)];
        let scale = frame.view.scale.map(f32::abs);
        if scale[0].max(scale[1]) <= WRAP_PERIODS {
            return CanvasPlan { size: target, wrap: false, view: frame.view };
        }
        let size = [0, 1].map(|a| ((target[a] as f32 / scale[a].max(1e-3)).ceil() as u32).clamp(8, target[a].max(8)));
        CanvasPlan { size, wrap: true, view: ViewXform { scale: [1.0, 1.0], offset: [0.0, 0.0] } }
    }

    /// Draw parameters and the number of sprite instances for this frame.
    fn draw_uniform(&self, frame: &Frame, plan: &CanvasPlan, fade: f32, motion: f32) -> (DrawUniform, u32) {
        let p = &self.params;
        let view = plan.view;
        let domain = [self.size[0] as f32, self.size[1] as f32];
        let canvas = [plan.size[0] as f32, plan.size[1] as f32];
        let head_radius = p.size.clamp(0.25, 8.0) * domain[1] / REFERENCE_HEIGHT;
        let trail_radius = head_radius * p.trail_scale.clamp(1.0, 5.0);
        let sizes = p.sizes.map(|s| s.clamp(MIN_SPECIES_SIZE, MAX_SPECIES_SIZE));
        let biggest = sizes[..p.kinds.clamp(2, MAX_KINDS)].iter().fold(1.0f32, |m, &s| m.max(s));

        // Copies per axis: a quad (plus a pixel of slack) can overlap at most
        // floor(visible span + 2 margins) + 1 tiles. The margin covers the
        // widest glow and the longest streak the speed limit allows.
        let px_per_unit = canvas[0] / (view.scale[0] * domain[0]).max(1e-6);
        let streak_px = self.max_speed() * motion * px_per_unit;
        let half_px = (trail_radius * biggest * px_per_unit).max(0.8) * 3.0 + 2.0 + streak_px;
        let tiles = [0, 1].map(|a| {
            let span = view.scale[a].abs() * (1.0 + 2.0 * half_px / canvas[a]);
            (span.floor() as u32 + 1).clamp(1, MAX_TILES)
        });

        // Glows keep their zoom-1 width in pixels beyond the head when zoomed
        // in. The domain is shaped like the output, so at zoom 1 a unit spans
        // output height / domain height pixels; a wrap canvas is never zoomed
        // in, so its own scale is used there.
        let zoom1_px_per_unit = frame.target_size[1].max(1) as f32 / domain[1];
        let colors = self.species_colors();
        let ground = palette::hex_to_linear(self.ground);
        let uniform = DrawUniform {
            view,
            screen: frame.view,
            domain,
            canvas,
            tiles,
            count: self.count,
            wrap: u32::from(plan.wrap),
            head_radius,
            trail_radius,
            head_gain: p.glow.clamp(0.0, 10.0),
            // The splat is scaled by (1 - fade) so a resting particle converges
            // to the same brightness at any trail length.
            trail_gain: p.trail_gain.clamp(0.0, 10.0) * (1.0 - fade),
            speed_ref: SPEED_REF * self.r_max(),
            speed_glow: p.speed_glow.clamp(0.0, 5.0),
            fade,
            knee: p.knee.clamp(0.1, 20.0),
            motion,
            relief: p.relief.clamp(0.0, 1.0),
            zoom1_ppu: zoom1_px_per_unit,
            _pad: 0.0,
            ground: [ground[0], ground[1], ground[2], 1.0],
            colors: std::array::from_fn(|s| [colors[s][0], colors[s][1], colors[s][2], sizes[s]]),
        };
        (uniform, self.count * tiles[0] * tiles[1])
    }

    /// (Re)creates the trail and heads textures when the canvas changes.
    fn ensure_canvas(&mut self, gpu: &Gpu, size: [u32; 2], wrap: bool) {
        let size = [size[0].max(1), size[1].max(1)];
        if self.canvas.as_ref().is_some_and(|c| c.size == size && c.wrap == wrap) {
            return;
        }
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let (_trail_texture, trail) = gpu.texture_2d("pl trail", size, SCENE_FORMAT, usage);
        let (_heads_texture, heads) = gpu.texture_2d("pl heads", size, SCENE_FORMAT, usage);
        let group = gpu.bind_group(
            "pl canvas",
            &self.canvas_layout,
            &[wgpu::BindingResource::TextureView(&trail), wgpu::BindingResource::TextureView(&heads)],
        );
        self.canvas = Some(Canvas { size, wrap, trail, heads, group });
        self.clear_trail = true;
    }

    /// A new world biased towards lively behaviour. Most of the time (80%) it
    /// is a true mutation: a curated preset's matrix with its species
    /// relabelled and every entry nudged, which keeps the preset's character
    /// (cells, worms, comets, ...) and its rendering style while landing
    /// somewhere new; in tests every such mutation stayed interesting.
    /// Otherwise it is a fresh 3-5 species matrix that passes `is_lively`
    /// (about half of those settle into plain dots, hence the low share).
    /// Colours are random; the population settings and the brush strength
    /// are kept.
    fn randomise(&mut self, seed: u64) {
        let mut rng = Rng::new(seed ^ 0x9A27_1C1E);
        let old = self.params;
        let mut p = Params::from_preset(&BASE);
        if rng.chance(0.8) {
            let src = &PRESETS[rng.below(PRESETS.len() as u32) as usize];
            let k = src.kinds;
            // Fisher-Yates: species i of the mutant is species perm[i] of the source.
            let mut perm: Vec<usize> = (0..k).collect();
            for i in (1..k).rev() {
                perm.swap(i, rng.below(i as u32 + 1) as usize);
            }
            p = Params::from_preset(src);
            for i in 0..k {
                p.weights[i] = src.weights[perm[i]];
                p.sizes[i] = src.sizes[perm[i]];
                for j in 0..k {
                    p.matrix[i][j] = (src.matrix[perm[i]][perm[j]] + rng.range(-0.2, 0.2)).clamp(-1.0, 1.0);
                    p.radii[i][j] = src.radii[perm[i]][perm[j]];
                }
            }
        } else {
            p.kinds = *rng.pick(&[3, 4, 4, 4, 5, 5]);
            p.matrix = lively_matrix(&mut rng, p.kinds);
        }
        p.colors = Colors::Scheme(rng.below(SCHEMES.len() as u32) as usize);
        p.color_shift = rng.below(MAX_KINDS as u32) as usize;
        p.count = old.count;
        p.density = old.density;
        p.spawn = old.spawn;
        p.pointer_strength = old.pointer_strength;
        self.params = p;
        self.look = BASE_LOOK.post();
        self.ground = BASE_LOOK.ground;
    }
}

impl World for ParticleLife {
    fn id(&self) -> &'static str {
        "particle-life"
    }

    fn name(&self) -> &'static str {
        "Particle Life"
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
        self.preset = index;
        self.params = Params::from_preset(&PRESETS[index]);
        self.look = PRESETS[index].look.post();
        self.ground = PRESETS[index].look.ground;
        self.reset(gpu, seed);
    }

    fn reset(&mut self, gpu: &Gpu, seed: u64) {
        self.seed = seed;
        self.salt = Rng::new(seed ^ 0x5A17).next_u32();
        self.rng = Rng::new(seed ^ 0x5EED);
        let p = &mut self.params;
        p.count = p.count.clamp(MIN_COUNT, MAX_COUNT);
        p.density = p.density.clamp(MIN_DENSITY, MAX_DENSITY);
        let size = domain_for(p.count, p.density, self.aspect);
        let cells = cell_capacity(size);
        if p.count != self.buffers.particles || cells > self.buffers.cells {
            self.buffers = Buffers::new(
                gpu,
                &self.sim_layout,
                &self.draw_layout,
                &self.sim_uniform,
                &self.draw_uniform,
                p.count,
                cells,
            );
        }
        self.size = size;
        self.count = p.count;
        self.applied_spawn = p.spawn;

        let pos = self.spawn(seed);
        let vel = vec![[0.0f32; 2]; pos.len()];
        gpu.queue.write_buffer(&self.buffers.pos, 0, bytemuck::cast_slice(&pos));
        gpu.queue.write_buffer(&self.buffers.vel, 0, bytemuck::cast_slice(&vel));
        // The counters are zero after every complete step; clear anyway so a
        // restart never inherits stale counts.
        gpu.queue.write_buffer(&self.buffers.counts, 0, &vec![0u8; self.buffers.cells as usize * 4]);
        self.clear_trail = true;
        self.pending_steps = 0;
        self.pending_motion = 0.0;
    }

    fn mutate(&mut self, gpu: &Gpu, seed: u64) {
        self.randomise(seed);
        self.reset(gpu, seed);
    }

    fn step(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder) {
        frame.gpu.write(&self.sim_uniform, &self.sim_uniform(frame));
        let (gx, gy) = gpu::dispatch_linear(self.count, WORKGROUP);
        let (fx, fy) = gpu::dispatch_linear(self.count * 9, WORKGROUP);
        let substeps = self.params.substeps.clamp(1, MAX_SUBSTEPS);
        {
            let mut pass = encoder
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("pl step"), timestamp_writes: None });
            pass.set_bind_group(0, &self.buffers.sim_group, &[]);
            for _ in 0..substeps {
                pass.set_pipeline(&self.count_pipeline);
                pass.dispatch_workgroups(gx, gy, 1);
                pass.set_pipeline(&self.scan_pipeline);
                pass.dispatch_workgroups(1, 1, 1);
                pass.set_pipeline(&self.scatter_pipeline);
                pass.dispatch_workgroups(gx, gy, 1);
                pass.set_pipeline(&self.force_pipeline);
                pass.dispatch_workgroups(fx, fy, 1);
                pass.set_pipeline(&self.integrate_pipeline);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        self.pending_steps += 1;
        self.pending_motion += self.sim_dt() * substeps as f32;
    }

    fn render(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let plan = self.canvas_plan(frame);
        self.ensure_canvas(frame.gpu, plan.size, plan.wrap);

        // The trail fades once per simulation step, so streaks keep the length
        // they have in the 60 fps renders at any display rate. While paused it
        // fades on the wall clock instead, so it settles after a pan or zoom.
        let steps = std::mem::take(&mut self.pending_steps);
        let motion = std::mem::take(&mut self.pending_motion);
        let trail = self.params.trail.clamp(0.0, 0.99);
        let fade = if steps > 0 {
            trail.powi(steps.min(64) as i32)
        } else {
            trail.powf(frame.dt.clamp(1.0 / 240.0, 0.1) * 60.0)
        };
        let (uniform, instances) = self.draw_uniform(frame, &plan, fade, motion);
        frame.gpu.write(&self.draw_uniform, &uniform);
        let trail_load =
            if std::mem::take(&mut self.clear_trail) { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load };
        let Some(canvas) = &self.canvas else { return };

        {
            let mut pass = color_pass(encoder, "pl trail", &canvas.trail, trail_load);
            pass.set_bind_group(0, &self.buffers.draw_group, &[]);
            pass.set_pipeline(&self.fade_pipeline);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&self.trail_pipeline);
            pass.draw(0..4, 0..instances);
        }
        {
            let mut pass = color_pass(encoder, "pl heads", &canvas.heads, wgpu::LoadOp::Clear(wgpu::Color::BLACK));
            pass.set_bind_group(0, &self.buffers.draw_group, &[]);
            pass.set_pipeline(&self.head_pipeline);
            pass.draw(0..4, 0..instances);
        }
        gpu::fullscreen_pass(
            encoder,
            "pl composite",
            target,
            Some(wgpu::Color::BLACK),
            &self.composite_pipeline,
            &[&self.buffers.draw_group, &canvas.group],
        );
    }

    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) {
        let dots = self.species_colors_u8();
        let applied = (self.count, self.size, self.applied_spawn);
        let aspect = self.aspect;
        let restart;
        {
            let Self { params: p, rng, ground, .. } = self;
            ui.add(egui::Slider::new(&mut p.kinds, 2..=MAX_KINDS).text("Species"));
            ui.label(
                egui::RichText::new("Attraction matrix: each row feels the columns. Drag a cell, right-click to zero.")
                    .weak()
                    .small(),
            );
            matrix_editor(ui, &mut p.matrix, p.kinds, &dots);
            ui.horizontal(|ui| {
                if ui.button("Randomise").on_hover_text("A new random matrix, filtered for lively ones").clicked() {
                    p.matrix = lively_matrix(rng, p.kinds);
                }
                if ui.button("Mirror").on_hover_text("Make the attractions symmetric (calmer)").clicked() {
                    for i in 0..p.kinds {
                        for j in 0..i {
                            let avg = 0.5 * (p.matrix[i][j] + p.matrix[j][i]);
                            p.matrix[i][j] = avg;
                            p.matrix[j][i] = avg;
                        }
                    }
                }
                if ui.button("Transpose").on_hover_text("Swap who chases whom").clicked() {
                    for i in 0..p.kinds {
                        for j in 0..i {
                            let t = p.matrix[i][j];
                            p.matrix[i][j] = p.matrix[j][i];
                            p.matrix[j][i] = t;
                        }
                    }
                }
            });
            ui.add(egui::Slider::new(&mut p.r_max, MIN_RADIUS..=MAX_RADIUS).text("Interaction radius"));
            ui.add(egui::Slider::new(&mut p.beta, 0.1..=0.6).text("Repulsion core β"));
            ui.add(egui::Slider::new(&mut p.force, 1.0..=40.0).logarithmic(true).text("Force"));
            ui.add(egui::Slider::new(&mut p.half_life, 0.005..=0.5).logarithmic(true).text("Friction half-life (s)"));
            ui.add(egui::Slider::new(&mut p.dt, 0.002..=0.03).text("Time step"));
            ui.add(egui::Slider::new(&mut p.substeps, 1..=MAX_SUBSTEPS).text("Steps / frame"));
            ui.add(egui::Slider::new(&mut p.pointer_strength, 5.0..=300.0).logarithmic(true).text("Brush strength"));
            egui::CollapsingHeader::new("Species balance").show(ui, |ui| {
                for (s, w) in p.weights.iter_mut().enumerate().take(p.kinds) {
                    ui.horizontal(|ui| {
                        species_dot(ui, dots[s], 14.0);
                        ui.add(egui::Slider::new(w, 0.05..=1.0).text(format!("species {}", s + 1)));
                    });
                }
            });
            egui::CollapsingHeader::new("Species size").show(ui, |ui| {
                for (s, size) in p.sizes.iter_mut().enumerate().take(p.kinds) {
                    ui.horizontal(|ui| {
                        species_dot(ui, dots[s], 14.0);
                        ui.add(
                            egui::Slider::new(size, MIN_SPECIES_SIZE..=MAX_SPECIES_SIZE)
                                .logarithmic(true)
                                .text(format!("species {}", s + 1)),
                        );
                    });
                }
            });

            ui.separator();
            ui.label(egui::RichText::new("Population (applies on restart)").strong());
            ui.add(egui::Slider::new(&mut p.count, MIN_COUNT..=MAX_COUNT).logarithmic(true).text("Particles"));
            ui.add(egui::Slider::new(&mut p.density, MIN_DENSITY..=MAX_DENSITY).logarithmic(true).text("Density"));
            egui::ComboBox::from_label("Spawn").selected_text(p.spawn.name()).show_ui(ui, |ui| {
                for s in Spawn::ALL {
                    ui.selectable_value(&mut p.spawn, s, s.name());
                }
            });
            let pending = (p.count, domain_for(p.count, p.density, aspect), p.spawn) != applied;
            let label = if pending { "Apply & restart  •" } else { "Apply & restart" };
            restart = ui.button(label).on_hover_text("Reallocate if needed and respawn with the same seed").clicked();

            ui.separator();
            let mut choice = match p.colors {
                Colors::Scheme(i) => i,
                Colors::Gradient { .. } => SCHEMES.len(),
            };
            let before = choice;
            let selected = SCHEMES.get(choice).map_or("Palette gradient", |s| s.name);
            egui::ComboBox::from_label("Colours").selected_text(selected).show_ui(ui, |ui| {
                for (i, scheme) in SCHEMES.iter().enumerate() {
                    ui.horizontal(|ui| {
                        scheme_swatch(ui, &scheme.colors);
                        ui.selectable_value(&mut choice, i, scheme.name);
                    });
                }
                ui.selectable_value(&mut choice, SCHEMES.len(), "Palette gradient");
            });
            if choice != before {
                p.colors = if choice < SCHEMES.len() {
                    Colors::Scheme(choice)
                } else {
                    Colors::Gradient { palette: 1, lo: 0.3, hi: 1.0 }
                };
            }
            if let Colors::Gradient { palette: index, lo, hi } = &mut p.colors {
                palette::combo(ui, "pl palette", index);
                ui.add(egui::Slider::new(lo, 0.0..=1.0).text("Gradient from"));
                ui.add(egui::Slider::new(hi, 0.0..=1.0).text("Gradient to"));
            }
            ui.add(egui::Slider::new(&mut p.color_shift, 0..=MAX_KINDS - 1).text("Colour rotation"));
            ui.horizontal(|ui| {
                let mut rgb = [(*ground >> 16) as u8, (*ground >> 8) as u8, *ground as u8];
                if ui.color_edit_button_srgb(&mut rgb).changed() {
                    *ground = u32::from(rgb[0]) << 16 | u32::from(rgb[1]) << 8 | u32::from(rgb[2]);
                }
                ui.label("Ground");
            });
            ui.add(egui::Slider::new(&mut p.size, 0.5..=5.0).text("Particle size"));
            ui.add(egui::Slider::new(&mut p.glow, 0.1..=4.0).logarithmic(true).text("Brightness"));
            ui.add(egui::Slider::new(&mut p.speed_glow, 0.0..=3.0).text("Speed glow"));
            ui.add(egui::Slider::new(&mut p.trail, 0.0..=0.98).text("Trail length"));
            ui.add(egui::Slider::new(&mut p.trail_gain, 0.0..=4.0).text("Trail brightness"));
            ui.add(egui::Slider::new(&mut p.trail_scale, 1.0..=5.0).text("Glow radius"));
            ui.add(egui::Slider::new(&mut p.knee, 0.2..=8.0).logarithmic(true).text("Highlight roll-off"));
            ui.add(egui::Slider::new(&mut p.relief, 0.0..=1.0).text("Relief shading"));
        }
        if restart {
            self.reset(gpu, self.seed);
        }
    }

    fn post_settings(&self) -> PostSettings {
        self.look
    }

    fn stats(&self) -> String {
        let g = self.grid();
        format!(
            "{} particles · {} species · r {:.0} · grid {}x{}",
            group_digits(self.count),
            self.params.kinds,
            self.r_max(),
            g[0],
            g[1]
        )
    }

    fn controls_hint(&self) -> &'static str {
        "Left: attract · Right: repel"
    }
}

// --- UI helpers ------------------------------------------------------------------

/// 49152 -> "49,152".
fn group_digits(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn species_dot(ui: &mut egui::Ui, color: egui::Color32, size: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), size * 0.32, color);
}

fn scheme_swatch(ui: &mut egui::Ui, colors: &[u32; MAX_KINDS]) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(48.0, 12.0), egui::Sense::hover());
    let w = rect.width() / MAX_KINDS as f32;
    for (i, &hex) in colors.iter().enumerate() {
        let x = rect.left() + i as f32 * w;
        let c = egui::Color32::from_rgb((hex >> 16) as u8, (hex >> 8) as u8, hex as u8);
        let cell = egui::Rect::from_min_max(egui::pos2(x, rect.top()), egui::pos2(x + w + 0.5, rect.bottom()));
        ui.painter().rect_filled(cell, 0.0, c);
    }
}

/// Heat-map colour of a matrix entry: teal for attraction, red for repulsion.
fn value_color(v: f32) -> egui::Color32 {
    let t = v.clamp(-1.0, 1.0).abs();
    let (r, g, b) = if v >= 0.0 { (40.0, 210.0, 170.0) } else { (235.0, 70.0, 85.0) };
    let base = 34.0;
    let mix = |c: f32| (base + (c - base) * t) as u8;
    egui::Color32::from_rgb(mix(r), mix(g), mix(b))
}

/// Compact editor for the attraction matrix. Each cell shows its value as a
/// colour; dragging right/up increases it, right-click resets it to zero.
fn matrix_editor(ui: &mut egui::Ui, m: &mut Matrix, k: usize, dots: &[egui::Color32; MAX_KINDS]) {
    let cell = ((ui.available_width() - 16.0) / (k as f32 + 1.0) - 2.0).clamp(12.0, 30.0);
    egui::Grid::new("pl matrix").spacing([2.0, 2.0]).show(ui, |ui| {
        ui.allocate_exact_size(egui::vec2(cell, cell), egui::Sense::hover());
        for &dot in dots.iter().take(k) {
            species_dot(ui, dot, cell);
        }
        ui.end_row();
        for (i, row) in m.iter_mut().enumerate().take(k) {
            species_dot(ui, dots[i], cell);
            for (j, value) in row.iter_mut().enumerate().take(k) {
                let (rect, response) = ui.allocate_exact_size(egui::vec2(cell, cell), egui::Sense::click_and_drag());
                if response.dragged() {
                    let d = response.drag_delta();
                    *value = (*value + (d.x - d.y) * 0.01).clamp(-1.0, 1.0);
                }
                if response.secondary_clicked() {
                    *value = 0.0;
                }
                let painter = ui.painter();
                painter.rect_filled(rect, 3.0, value_color(*value));
                if cell >= 22.0 {
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        format!("{:+.1}", *value),
                        egui::FontId::proportional(9.0),
                        egui::Color32::from_gray(235),
                    );
                }
                response.on_hover_text(format!("species {} towards species {}: {:+.2}", i + 1, j + 1, *value));
            }
            ui.end_row();
        }
    });
}
