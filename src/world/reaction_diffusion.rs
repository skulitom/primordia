//! Gray-Scott reaction-diffusion: two virtual chemicals react and diffuse on a
//! torus, painting coral, dividing cells, fingerprints, worms, gliders, spiral
//! waves and soap-film foam.
//!
//! This is the full-featured showcase world. It follows every convention a
//! world needs (uniform mirroring, ping-pong compute, palette LUT, pointer
//! brush, presets, mutation, parameter UI, per-preset post look) and layers a
//! good deal of optional art direction on top. For a minimal template, start
//! from `placeholder.rs`; the optional machinery here is fenced by
//! `--- optional: ... ---` section comments so the skeleton stays easy to find.
//!
//! Frame pipeline:
//! 1. `step` applies brush strokes, rain drops and the revive safety net once
//!    (`cs_inject`), then runs N explicit-Euler sub-steps (`cs_step`,
//!    ping-pong buffers).
//! 2. `render` bakes the state into a filterable texture (V, U, blurred V,
//!    |dV/dt|) while histogramming it (`cs_prepare`), turns the histogram into
//!    a smoothed auto-contrast window (`cs_resolve`), and finally lights the
//!    field as a height map with one of five materials (`fs_display`).

use std::f32::consts::TAU;
use std::sync::OnceLock;

use bytemuck::{Pod, Zeroable};
use wgpu::ShaderStages;

use super::{Frame, ViewXform, World};
use crate::gpu::{self, Gpu, SCENE_FORMAT, layout};
use crate::metrics::{self, MetricDesc, Reduction, Unit};
use crate::palette::{self, PaletteLut};

#[cfg(test)]
#[path = "reaction_diffusion_tests.rs"]
mod tests;
use crate::post::{PostSettings, Tonemap};
use crate::rng::Rng;

const WORKGROUP: u32 = 16;
/// Cells per output pixel along each axis (patterns read better slightly enlarged).
const DOMAIN_SCALE: f32 = 2.0 / 3.0;
/// Automatic droplets per frame the shader accepts (`Sim.drops`).
const MAX_DROPS: usize = 4;
/// 128 V bins + 128 U bins of `u32`.
const HIST_BYTES: u64 = 256 * 4;
/// Contrast window + revive state (`array<f32, 8>` in the shader).
const STATS_BYTES: u64 = 8 * 4;
/// V from which a cell counts as alive (the revive logic's threshold in `cs_resolve`).
const ALIVE_V: f32 = 0.03;
/// V from which a cell counts as pattern body rather than fringe.
const BODY_V: f32 = 0.25;
/// V from which a cell counts as flooded (foam, solid fill).
const FILLED_V: f32 = 0.4;
/// Change of V per frame from which a cell counts as still changing (measured
/// on the last step and scaled by the steps per frame).
const ACTIVE_DV: f32 = 1e-3;

/// Lane order of `reaction_diffusion_measure.wgsl`.
const METRICS: &[MetricDesc] = &[
    MetricDesc {
        id: "alive",
        label: "Alive cells",
        unit: Unit::Fraction,
        hint: "Cells whose V exceeds 0.03: the footprint of the pattern.",
    },
    MetricDesc {
        id: "body",
        label: "Body cells",
        unit: Unit::Fraction,
        hint: "Cells whose V exceeds 0.25: the solid interior of spots, stripes and worms.",
    },
    MetricDesc { id: "v_mean", label: "Mean V", unit: Unit::Scalar, hint: "Mean concentration of the activator V." },
    MetricDesc { id: "u_mean", label: "Mean U", unit: Unit::Scalar, hint: "Mean concentration of the substrate U." },
    MetricDesc {
        id: "active",
        label: "Changing cells",
        unit: Unit::Fraction,
        hint: "Cells whose V is changing by more than 0.001 per frame, judged from the last step: zero once a pattern has frozen.",
    },
    MetricDesc {
        id: "v_drift",
        label: "V drift",
        unit: Unit::Scalar,
        hint: "Mean change of V per step: positive while the pattern spreads, negative while it dies back.",
    },
    MetricDesc {
        id: "edge",
        label: "Boundary cells",
        unit: Unit::Fraction,
        hint: "Alive cells with a dead neighbour: the pattern's perimeter, high for fine labyrinths and low for blobs.",
    },
    MetricDesc {
        id: "filled",
        label: "Filled cells",
        unit: Unit::Fraction,
        hint: "Cells whose V exceeds 0.4: flooded ground, the state the revive logic watches for.",
    },
];
/// Drift phase advanced per unit of simulated time at drift speed 1.
const DRIFT_RATE: f64 = 1.0 / 2000.0;
/// The drift phase wraps at this period. Every multiple of the phase the
/// shader uses (1, 0.8, 0.6, 0.35, 0.3, 0.25, 0.24, 0.2, 0.18) completes a whole
/// number of cycles over it, so the wrap is invisible and the f32 upload
/// keeps its precision on arbitrarily long runs.
const DRIFT_PERIOD: f64 = 200.0 * std::f64::consts::PI;
/// Converts |dV| per sub-step (divided by dt) into display units for the growth glow.
const ACTIVITY_SCALE: f32 = 1500.0;
/// Spray brush grid spacing and dot radius, in cells at pattern scale 1.
/// Dots must be big enough to nucleate even the pickiest regime (coral).
const SPRAY_SPACING: f32 = 12.0;
const SPRAY_DOT: f32 = 4.5;
/// Share of spray grid cells that hold a hole when the brush lays down fresh
/// foam: enough holes to read as bubbles, sparse enough that the walls
/// between them stay thick enough to survive.
const FOAM_HOLES: f32 = 0.3;
/// `Sim.pointer_mode` flag: the same button was held on the previous frame,
/// so the stroke continues from `Sim.prev_pointer`.
const STROKE_CONTINUES: u32 = 4;
const FEED_RANGE: std::ops::RangeInclusive<f32> = 0.01..=0.1;
/// Feed from which seeds die on bare U and the filled state becomes the
/// ground (measured with blob seeding across the survival band).
const FILLED_FEED: f32 = 0.077;

/// Saddle-node curve of Gray-Scott: non-trivial homogeneous states exist for
/// k below it, and the interesting morphologies live in a thin band around it.
fn critical_kill(feed: f32) -> f32 {
    feed.max(0.0).sqrt() * 0.5 - feed
}

/// Offsets (low, high) from [`critical_kill`] between which patterns survive,
/// measured at feed 0.01, 0.02, ..., 0.10. The kill slider and `mutate` stay
/// inside it; the atlas sweeps a trimmed copy (`atlas_band` in the shader).
/// At feed 0.03 sparse seeds still grow into slowly budding rafts of spots
/// up to about +0.0080 (the Solitons preset lives there).
const KILL_BAND: [(f32, f32); 10] = [
    (0.0030, 0.0075),
    (-0.0030, 0.0070),
    (-0.0030, 0.0082),
    (-0.0016, 0.0060),
    (-0.0015, 0.0040),
    (-0.0014, 0.0028),
    (-0.0012, 0.0004),
    (-0.0010, 0.0000),
    (-0.0008, -0.0001),
    (-0.0010, -0.0002),
];

fn kill_band(feed: f32) -> (f32, f32) {
    let x = (feed * 100.0 - 1.0).clamp(0.0, 8.999);
    let i = x as usize;
    let t = x - i as f32;
    let (a, b) = (KILL_BAND[i], KILL_BAND[i + 1]);
    (lerp(a.0, b.0, t), lerp(a.1, b.1, t))
}

/// The upper homogeneous steady state (u, v) of Gray-Scott, if it exists.
fn blue_state(feed: f32, kill: f32) -> Option<(f32, f32)> {
    let s = feed + kill;
    let disc = feed * feed - 4.0 * feed * s * s;
    if disc < 0.0 || s <= 0.0 {
        return None;
    }
    let v = (feed + disc.sqrt()) / (2.0 * s);
    Some(((s / v).clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Seeding {
    Sparse,
    Blobs,
    Noise,
    Dense,
    Square,
    Foam,
    Rings,
    Center,
    Fronts,
}

impl Seeding {
    const ALL: [Seeding; 9] = [
        Seeding::Sparse,
        Seeding::Blobs,
        Seeding::Noise,
        Seeding::Dense,
        Seeding::Square,
        Seeding::Foam,
        Seeding::Rings,
        Seeding::Center,
        Seeding::Fronts,
    ];

    fn name(self) -> &'static str {
        match self {
            Seeding::Sparse => "A few colonies",
            Seeding::Blobs => "Scattered blobs",
            Seeding::Noise => "Speckled noise",
            Seeding::Dense => "Dense patches",
            Seeding::Square => "Central square",
            Seeding::Foam => "Foam (filled, with holes)",
            Seeding::Rings => "Concentric rings",
            Seeding::Center => "Single seed",
            Seeding::Fronts => "Broken wave fronts",
        }
    }
}

/// How the field is lit (`Draw.material` in the shader).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Material {
    Lacquer,
    Nacre,
    Luminous,
    Ink,
    DarkField,
}

impl Material {
    const ALL: [Material; 5] = [
        Material::Lacquer,
        Material::Nacre,
        Material::Luminous,
        Material::Ink,
        Material::DarkField,
    ];

    fn name(self) -> &'static str {
        match self {
            Material::Lacquer => "Lacquer (glossy relief)",
            Material::Nacre => "Nacre (thin-film iridescence)",
            Material::Luminous => "Luminous (subsurface glow)",
            Material::Ink => "Ink on paper",
            Material::DarkField => "Dark-field (glowing membranes)",
        }
    }

    fn code(self) -> u32 {
        match self {
            Material::Lacquer => 0,
            Material::Nacre => 1,
            Material::Luminous => 2,
            Material::Ink => 3,
            Material::DarkField => 4,
        }
    }
}

// --- parameters and presets --------------------------------------------------

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct Params {
    pub feed: f32,
    pub kill: f32,
    /// Diffusion of U (pattern size grows with its square root).
    pub scale: f32,
    /// Du / Dv.
    pub ratio: f32,
    pub dt: f32,
    pub steps_per_frame: u32,
    /// Vary feed/kill across the domain (a live map of every morphology).
    pub atlas: bool,
    pub seeding: Seeding,
    /// Amplitude of the slow travelling modulation of k.
    pub drift: f32,
    pub drift_speed: f32,
    /// Log-amplitude of the spatial modulation of pattern size.
    pub scale_var: f32,
    /// Expected automatic droplets per frame.
    pub rain: f32,
    /// Kill added inside a few large, slowly drifting lagoons. Lifted above
    /// the survival band, it keeps open ground that the pattern re-colonises
    /// as the lagoons move on, so dense regimes keep a composition.
    pub ground: f32,
    /// Anisotropy of V's diffusion along a whorled ridge flow, which turns
    /// labyrinths into fingerprint ridges with loops, whorls and deltas.
    pub aniso: f32,

    pub material: Material,
    /// Treat the filled (high V) state as ground so its holes read as bodies.
    pub invert: bool,
    /// Part of the palette the level 0..1 is mapped onto.
    pub pal_lo: f32,
    pub pal_hi: f32,
    /// Level range over which ground turns into body.
    pub contrast_lo: f32,
    pub contrast_hi: f32,
    pub relief: f32,
    pub gloss: f32,
    pub glow: f32,
    /// Strength of the soft U-depletion halo around structure.
    pub halo: f32,
    /// Palette position (before the palette range) of the halo's colour.
    pub aura_tint: f32,
    pub iridescence: f32,
    /// Nacre only: how much the ground faintly reflects the film (0..1).
    pub reflect: f32,
    /// Local contrast of the level against its blurred neighbourhood, so
    /// shallow ripples in flat plateaus read as texture instead of blur.
    pub clarity: f32,
    pub activity: f32,
    pub shadow: f32,
    pub brightness: f32,
}

struct Preset {
    name: &'static str,
    palette: &'static str,
    params: Params,
    post: PostSettings,
}

const BASE: Params = Params {
    feed: 0.0545,
    kill: 0.062,
    scale: 1.0,
    ratio: 2.0,
    dt: 1.0,
    steps_per_frame: 24,
    atlas: false,
    seeding: Seeding::Blobs,
    drift: 0.0,
    drift_speed: 1.0,
    scale_var: 0.0,
    rain: 0.0,
    ground: 0.0,
    aniso: 0.0,
    material: Material::Lacquer,
    invert: false,
    pal_lo: 0.0,
    pal_hi: 0.85,
    contrast_lo: 0.2,
    contrast_hi: 0.45,
    relief: 0.9,
    gloss: 1.0,
    glow: 0.8,
    halo: 0.12,
    aura_tint: 0.4,
    iridescence: 0.0,
    reflect: 0.0,
    clarity: 0.0,
    activity: 0.3,
    shadow: 0.6,
    brightness: 1.0,
};

const LOOK: PostSettings = PostSettings {
    exposure: 1.0,
    bloom: 0.5,
    bloom_threshold: 0.9,
    vignette: 0.4,
    saturation: 1.08,
    grain: 0.0,
    tonemap: Tonemap::Agx,
};

/// Bloom-heavy look for the self-lit materials.
const GLOW_LOOK: PostSettings = PostSettings {
    bloom: 0.85,
    bloom_threshold: 0.7,
    ..LOOK
};

const PRESETS: &[Preset] = &[
    Preset {
        name: "Coral Reef",
        palette: "Coral",
        params: Params {
            feed: 0.0545,
            kill: 0.062,
            scale: 1.2,
            steps_per_frame: 12,
            seeding: Seeding::Sparse,
            drift: 0.0006,
            scale_var: 0.2,
            ground: 0.005,
            pal_lo: 0.12,
            pal_hi: 0.82,
            contrast_lo: 0.2,
            contrast_hi: 0.42,
            halo: 0.08,
            iridescence: 0.15,
            activity: 0.5,
            ..BASE
        },
        post: PostSettings {
            exposure: 1.1,
            saturation: 1.1,
            ..LOOK
        },
    },
    Preset {
        name: "Mitosis",
        palette: "Bioluminescence",
        params: Params {
            feed: 0.0367,
            kill: 0.0649,
            scale: 1.4,
            steps_per_frame: 22,
            seeding: Seeding::Blobs,
            drift: 0.0022,
            scale_var: 0.2,
            ground: 0.003,
            material: Material::DarkField,
            pal_hi: 0.92,
            contrast_lo: 0.12,
            contrast_hi: 0.4,
            relief: 0.7,
            gloss: 0.7,
            glow: 0.9,
            halo: 0.25,
            iridescence: 0.8,
            activity: 0.3,
            shadow: 0.4,
            ..BASE
        },
        post: PostSettings {
            bloom: 0.7,
            bloom_threshold: 0.8,
            vignette: 0.45,
            ..GLOW_LOOK
        },
    },
    Preset {
        name: "Fingerprints",
        palette: "Ink",
        params: Params {
            feed: 0.037,
            kill: 0.06,
            seeding: Seeding::Noise,
            drift: 0.001,
            scale_var: 0.1,
            aniso: 0.25,
            material: Material::Ink,
            pal_hi: 1.0,
            contrast_lo: 0.3,
            contrast_hi: 0.55,
            relief: 0.7,
            gloss: 0.2,
            shadow: 0.3,
            ..BASE
        },
        post: PostSettings {
            bloom: 0.0,
            vignette: 0.25,
            saturation: 1.0,
            grain: 0.1,
            tonemap: Tonemap::Linear,
            ..LOOK
        },
    },
    Preset {
        name: "Worms",
        palette: "Moss",
        params: Params {
            feed: 0.058,
            kill: 0.065,
            scale: 1.1,
            seeding: Seeding::Dense,
            drift: 0.0015,
            scale_var: 0.3,
            ground: 0.002,
            pal_lo: 0.05,
            pal_hi: 0.97,
            contrast_hi: 0.36,
            gloss: 1.1,
            halo: 0.18,
            shadow: 0.7,
            ..BASE
        },
        post: LOOK,
    },
    Preset {
        name: "Solitons",
        palette: "Aurora",
        params: Params {
            // Just below where spots stop budding: a few sparse seeds grow
            // into rafts of glowing orbs that bud slowly at their rims.
            feed: 0.030,
            kill: 0.0644,
            scale: 1.5,
            steps_per_frame: 32,
            seeding: Seeding::Sparse,
            drift: 0.0006,
            // Lagoons keep open water between the rafts once they have
            // budded across most of the torus, and shrink the orbs near
            // their shores.
            ground: 0.006,
            material: Material::Luminous,
            pal_lo: 0.1,
            pal_hi: 0.95,
            contrast_lo: 0.35,
            contrast_hi: 0.6,
            relief: 1.1,
            glow: 0.7,
            gloss: 1.4,
            halo: 0.0,
            iridescence: 0.3,
            activity: 0.0,
            shadow: 0.5,
            ..BASE
        },
        post: PostSettings {
            bloom: 0.45,
            bloom_threshold: 0.8,
            ..GLOW_LOOK
        },
    },
    Preset {
        name: "Crescent Gliders",
        palette: "Nacre",
        params: Params {
            feed: 0.014,
            kill: 0.050,
            seeding: Seeding::Dense,
            // Gliders are mobile, so the lagoons must be strongly lethal to
            // keep clearings the swarms stream around.
            ground: 0.009,
            material: Material::Nacre,
            pal_hi: 0.75,
            contrast_lo: 0.4,
            contrast_hi: 0.65,
            iridescence: 0.85,
            halo: 0.0,
            activity: 0.0,
            ..BASE
        },
        post: LOOK,
    },
    Preset {
        name: "Spiral Waves",
        palette: "Ember",
        params: Params {
            feed: 0.010,
            kill: 0.045,
            // A slightly finer scale fits more fronts, and so more spirals.
            scale: 0.85,
            seeding: Seeding::Fronts,
            material: Material::Luminous,
            pal_hi: 0.9,
            contrast_lo: 0.30,
            contrast_hi: 0.5,
            glow: 1.0,
            gloss: 0.6,
            // A faint, cool halo: the refractory zones read as shadow, and
            // no growth glow, which lit dying blobs as ghosts in the dark.
            halo: 0.05,
            aura_tint: 0.18,
            activity: 0.0,
            ..BASE
        },
        post: PostSettings {
            bloom: 0.7,
            bloom_threshold: 0.8,
            ..GLOW_LOOK
        },
    },
    Preset {
        name: "Bubbles",
        palette: "Glacier",
        params: Params {
            feed: 0.090,
            kill: 0.0597,
            seeding: Seeding::Foam,
            scale_var: 0.4,
            material: Material::Nacre,
            pal_hi: 0.45,
            contrast_lo: 0.25,
            contrast_hi: 0.5,
            gloss: 1.2,
            iridescence: 1.1,
            reflect: 0.4,
            halo: 0.1,
            ..BASE
        },
        post: LOOK,
    },
    Preset {
        name: "Spots & Stripes",
        palette: "Solar",
        params: Params {
            feed: 0.035,
            // A touch above the critical kill: the weather still sweeps
            // stripes into honeycomb and back, but rarely floods regions
            // into flat plateaus of the filled state.
            kill: 0.0589,
            seeding: Seeding::Noise,
            drift: 0.002,
            scale_var: 0.25,
            contrast_hi: 0.52,
            clarity: 0.5,
            // The holes of the honeycomb would each catch a pinpoint glint.
            gloss: 0.6,
            ..BASE
        },
        post: LOOK,
    },
    Preset {
        name: "Morphology Atlas",
        palette: "Lagoon",
        params: Params {
            atlas: true,
            seeding: Seeding::Dense,
            drift: 0.0004,
            pal_hi: 0.95,
            activity: 0.0,
            halo: 0.0,
            ..BASE
        },
        post: LOOK,
    },
];

fn preset_names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| PRESETS.iter().map(|p| p.name).collect())
}

// --- optional: mutation regimes ----------------------------------------------

/// Boxes of parameter space (feed range, kill offset from the critical curve)
/// that stay alive from their seeding; `mutate` samples inside them.
struct Regime {
    feed: (f32, f32),
    offset: (f32, f32),
    seeding: Seeding,
    /// The filled state covers most of the domain: show its holes as bodies
    /// so the ground stays dark.
    invert: bool,
    /// Bodies cover much of the domain, so additive light (iridescence, inner
    /// glow, bloom) borrowed from a sparse preset's look would wash it out,
    /// and lagoons are welcome to open up some negative space.
    dense: bool,
    /// Stripes that may follow a whorled ridge flow.
    stripes: bool,
}

const REGIMES: &[Regime] = &[
    // Coral / labyrinths.
    Regime {
        feed: (0.045, 0.058),
        offset: (-0.0012, 0.0004),
        seeding: Seeding::Blobs,
        invert: false,
        dense: true,
        stripes: true,
    },
    // Mitosis / dividing spots.
    Regime {
        feed: (0.030, 0.040),
        offset: (0.0042, 0.0056),
        seeding: Seeding::Blobs,
        invert: false,
        dense: true,
        stripes: false,
    },
    // Worms.
    Regime {
        feed: (0.050, 0.060),
        offset: (0.0016, 0.0026),
        seeding: Seeding::Dense,
        invert: false,
        dense: true,
        stripes: true,
    },
    // Chaotic self-replicating spots.
    Regime {
        feed: (0.018, 0.026),
        offset: (0.0048, 0.0062),
        seeding: Seeding::Dense,
        invert: false,
        dense: true,
        stripes: false,
    },
    // Crescent gliders and pulsing soliton swarms.
    Regime {
        feed: (0.013, 0.015),
        offset: (0.0045, 0.0070),
        seeding: Seeding::Blobs,
        invert: false,
        dense: false,
        stripes: false,
    },
    // Curling spiral fronts.
    Regime {
        feed: (0.010, 0.012),
        offset: (0.0044, 0.0052),
        seeding: Seeding::Fronts,
        invert: false,
        dense: false,
        stripes: false,
    },
    // Foam: thin walls around empty cells.
    Regime {
        feed: (0.088, 0.096),
        offset: (-0.0006, -0.0002),
        seeding: Seeding::Foam,
        invert: false,
        dense: false,
        stripes: false,
    },
    // Holes and negative stripes (inverted, so the sparse holes are the bodies).
    Regime {
        feed: (0.034, 0.046),
        offset: (-0.0016, -0.0002),
        seeding: Seeding::Noise,
        invert: true,
        dense: false,
        stripes: false,
    },
];

impl Params {
    /// Largest time step that keeps explicit Euler stable for the fastest
    /// diffusion anywhere in the domain. The Laplacian's most negative
    /// eigenvalue is -1.6 (checkerboard) and the reaction adds up to ~0.3 more
    /// damping to U, so the amplification |1 - dt (1.6 D + 0.3)| stays below
    /// ~0.95; at D = 1 this is exactly dt = 1. The ridge-flow anisotropy
    /// raises the fastest diffusion by at most the factor (1 + aniso).
    fn stable_dt(&self) -> f32 {
        let stretch = 1.0 + self.aniso.abs();
        let d_max =
            self.scale.max(self.scale / self.ratio.max(0.1)) * self.scale_var.exp() * stretch;
        self.dt.min(1.9 / (1.6 * d_max + 0.3))
    }

    /// Regimes whose ground is the filled (high V) state: seeds die on bare U
    /// there, so "create" lays down fresh foam (the filled state riddled with
    /// holes), "erase" pops bubbles back to bare U, and the revive lays down a
    /// fresh foam.
    fn filled_ground(&self) -> bool {
        !self.atlas && self.feed >= FILLED_FEED
    }

    fn chemistry_ui(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.atlas, "Morphology atlas (F/k vary across space)");
        if self.atlas {
            ui.label(
                egui::RichText::new(
                    "Feed peaks along the middle row, kill along the middle column.",
                )
                .weak()
                .small(),
            );
        } else {
            ui.add(
                crate::ui::Slider::new(&mut self.feed, FEED_RANGE)
                    .text("Feed F")
                    .fixed_decimals(4),
            );
            // Kill is confined to the band where patterns survive at this feed.
            let kc = critical_kill(self.feed);
            let (lo, hi) = kill_band(self.feed);
            self.kill = self.kill.clamp(kc + lo, kc + hi);
            ui.add(
                crate::ui::Slider::new(&mut self.kill, (kc + lo)..=(kc + hi))
                    .text("Kill k")
                    .fixed_decimals(4),
            );
            ui.label(
                egui::RichText::new(format!("critical kill {kc:.4}"))
                    .weak()
                    .small(),
            );
        }
        ui.add(crate::ui::Slider::new(&mut self.scale, 0.5..=1.6).text("Pattern scale"));
        ui.add(crate::ui::Slider::new(&mut self.ratio, 1.6..=2.6).text("Diffusion ratio U/V"));
        ui.add(crate::ui::Slider::new(&mut self.dt, 0.2..=1.0).text("Time step"));
        ui.add(crate::ui::Slider::new(&mut self.steps_per_frame, 1..=64).text("Steps / frame"));
        ui.add(
            crate::ui::Slider::new(&mut self.drift, 0.0..=0.004)
                .text("Drift (k weather)")
                .fixed_decimals(4),
        );
        ui.add(crate::ui::Slider::new(&mut self.drift_speed, 0.0..=4.0).text("Drift speed"));
        ui.add(crate::ui::Slider::new(&mut self.scale_var, 0.0..=0.6).text("Size variation"));
        ui.add(
            crate::ui::Slider::new(&mut self.ground, 0.0..=0.008)
                .text("Lagoons (kill bias)")
                .fixed_decimals(4),
        );
        ui.add(crate::ui::Slider::new(&mut self.aniso, 0.0..=0.3).text("Ridge flow"));
        ui.add(crate::ui::Slider::new(&mut self.rain, 0.0..=1.0).text("Rain (drops / frame)"));
        crate::ui::dropdown(ui, "Seeding (on reset)", self.seeding.name(), |ui| {
                for s in Seeding::ALL {
                    ui.selectable_value(&mut self.seeding, s, s.name());
                }
            });
    }

    fn look_ui(&mut self, ui: &mut egui::Ui) {
        crate::ui::dropdown(ui, "Material", self.material.name(), |ui| {
                for m in Material::ALL {
                    ui.selectable_value(&mut self.material, m, m.name());
                }
            });
        ui.checkbox(&mut self.invert, "Invert (holes become bodies)");
        ui.add(crate::ui::Slider::new(&mut self.pal_lo, 0.0..=1.0).text("Palette start"));
        ui.add(crate::ui::Slider::new(&mut self.pal_hi, 0.0..=1.0).text("Palette end"));
        ui.add(crate::ui::Slider::new(&mut self.contrast_lo, 0.0..=0.9).text("Edge low"));
        ui.add(crate::ui::Slider::new(&mut self.contrast_hi, 0.1..=1.0).text("Edge high"));
        self.contrast_hi = self.contrast_hi.max(self.contrast_lo + 0.05);
        ui.add(crate::ui::Slider::new(&mut self.relief, 0.0..=1.5).text("Relief"));
        ui.add(crate::ui::Slider::new(&mut self.shadow, 0.0..=1.0).text("Shadow"));
        ui.add(crate::ui::Slider::new(&mut self.gloss, 0.0..=2.0).text("Gloss"));
        ui.add(crate::ui::Slider::new(&mut self.glow, 0.0..=2.0).text("Inner glow"));
        ui.add(crate::ui::Slider::new(&mut self.halo, 0.0..=1.5).text("Halo"));
        ui.add(crate::ui::Slider::new(&mut self.aura_tint, 0.0..=1.0).text("Halo hue"));
        ui.add(crate::ui::Slider::new(&mut self.iridescence, 0.0..=1.5).text("Iridescence"));
        ui.add(crate::ui::Slider::new(&mut self.reflect, 0.0..=1.0).text("Film reflection (nacre)"));
        ui.add(crate::ui::Slider::new(&mut self.clarity, 0.0..=1.5).text("Clarity (local contrast)"));
        ui.add(crate::ui::Slider::new(&mut self.activity, 0.0..=2.0).text("Growth glow"));
        ui.add(crate::ui::Slider::new(&mut self.brightness, 0.2..=3.0).text("Brightness"));
    }
}

// --- GPU mirrors ---------------------------------------------------------------

/// Mirrors `Sim` in reaction_diffusion.wgsl (192 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SimUniform {
    size: [u32; 2],
    feed: f32,
    kill: f32,
    du: f32,
    dv: f32,
    dt: f32,
    atlas: u32,
    drift: f32,
    drift_phase: f32,
    scale_var: f32,
    frame: u32,
    pointer: [f32; 2],
    pointer_radius: f32,
    pointer_mode: u32,
    create_state: [f32; 2],
    erase_state: [f32; 2],
    spray: [f32; 2],
    create_jitter: f32,
    spray_density: f32,
    drop_count: u32,
    filled_ground: u32,
    prev_pointer: [f32; 2],
    ground: f32,
    flow_phase: f32,
    aniso: f32,
    _pad: f32,
    drops: [[f32; 4]; MAX_DROPS],
}

/// Mirrors `Prep` in reaction_diffusion.wgsl (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PrepUniform {
    size: [u32; 2],
    snap: u32,
    _pad0: u32,
    act_scale: f32,
    rate: f32,
    _pad1: [f32; 2],
}

/// Mirrors `Draw` in reaction_diffusion.wgsl (96 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DrawUniform {
    view: ViewXform,
    size: [u32; 2],
    time: f32,
    material: u32,
    contrast: [f32; 2],
    pal_range: [f32; 2],
    relief: f32,
    gloss: f32,
    glow: f32,
    halo: f32,
    iridescence: f32,
    activity: f32,
    shadow: f32,
    brightness: f32,
    invert: u32,
    aura_tint: f32,
    reflect: f32,
    clarity: f32,
}

/// Mirrors `Measure` in reaction_diffusion_measure.wgsl (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct MeasureUniform {
    size: [u32; 2],
    inv_cells: f32,
    _pad: f32,
    /// Alive V, body V, filled V, active |dV|.
    thresholds: [f32; 4],
}

// The WGSL structs must match byte for byte.
const _: () = assert!(std::mem::size_of::<SimUniform>() == 192);
const _: () = assert!(std::mem::size_of::<MeasureUniform>() == 32);
const _: () = assert!(std::mem::size_of::<PrepUniform>() == 32);
const _: () = assert!(std::mem::size_of::<DrawUniform>() == 96);

// --- world ---------------------------------------------------------------------

pub struct ReactionDiffusion {
    size: [u32; 2],
    params: Params,
    post: PostSettings,
    preset: usize,
    lut: PaletteLut,
    /// Ping-pong state buffers of `vec2<f32>(u, v)`.
    buffers: [wgpu::Buffer; 2],
    /// Index of the buffer holding the latest state.
    current: usize,
    /// Contrast window and revive state written by `cs_resolve`.
    stats: wgpu::Buffer,
    sim_uniform: wgpu::Buffer,
    prep_uniform: wgpu::Buffer,
    draw_uniform: wgpu::Buffer,
    inject_pipeline: wgpu::ComputePipeline,
    step_pipeline: wgpu::ComputePipeline,
    /// `step_groups[i]` reads `buffers[i]` and writes `buffers[1 - i]`.
    step_groups: [wgpu::BindGroup; 2],
    prepare_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    /// `prepare_groups[i]` bakes `buffers[i]` (latest) against `buffers[1 - i]`.
    prepare_groups: [wgpu::BindGroup; 2],
    draw_pipeline: wgpu::RenderPipeline,
    draw_group: wgpu::BindGroup,
    _field: wgpu::Texture,
    measure_uniform: wgpu::Buffer,
    measure_pipeline: wgpu::ComputePipeline,
    /// `measure_groups[i]` measures `buffers[i]` against `buffers[1 - i]`.
    measure_groups: [wgpu::BindGroup; 2],
    reduction: Reduction,

    /// Frames stepped since the last reset.
    frame: u64,
    /// Simulated time since the last reset.
    sim_time: f64,
    /// Accumulated drift phase (speed can change at any time without jumps).
    drift_clock: f64,
    /// Droplet stream, reseeded on every reset.
    rain_rng: Rng,
    /// Shape of the ridge-flow potential, drawn from the seed on every reset.
    flow_phase: f32,
    /// Brush position and button of the previous stepped frame while a
    /// button is held, so strokes stamp only their leading edge.
    last_pointer: Option<([f32; 2], u32)>,
    /// Jump the contrast window to its target on the next render (after a reset).
    snap_window: bool,
}


fn domain_size(output_size: [u32; 2]) -> [u32; 2] {
    [
        ((output_size[0] as f32 * DOMAIN_SCALE) as u32).max(64),
        ((output_size[1] as f32 * DOMAIN_SCALE) as u32).max(64),
    ]
}

pub fn validate_output_size(gpu: &Gpu, output_size: [u32; 2]) -> anyhow::Result<()> {
    let size = domain_size(output_size);
    let limits = gpu.device.limits();
    let bytes = (u64::from(size[0]) * u64::from(size[1])).saturating_mul(8);
    anyhow::ensure!(size.iter().all(|&n| n <= limits.max_texture_dimension_2d)
        && bytes <= u64::from(limits.max_storage_buffer_binding_size) && bytes <= limits.max_buffer_size,
        "Reaction-diffusion dimensions exceed this GPU's texture or buffer limits");
    Ok(())
}

pub fn create(gpu: &Gpu, output_size: [u32; 2], seed: u64) -> Box<dyn World> {
    Box::new(ReactionDiffusion::new(gpu, domain_size(output_size), seed))
}

impl ReactionDiffusion {
    pub fn new(gpu: &Gpu, size: [u32; 2], seed: u64) -> Self {
        let module = gpu.shader(
            "reaction-diffusion",
            include_str!("../shaders/reaction_diffusion.wgsl"),
        );
        let cells = size[0] as u64 * size[1] as u64;
        let bytes = cells * 8;
        let buffers = [
            gpu.storage_buffer("rd state a", bytes, wgpu::BufferUsages::empty()),
            gpu.storage_buffer("rd state b", bytes, wgpu::BufferUsages::empty()),
        ];
        let stats = gpu.storage_buffer("rd stats", STATS_BYTES, wgpu::BufferUsages::empty());
        let hist = gpu.storage_buffer("rd histogram", HIST_BYTES, wgpu::BufferUsages::empty());
        let (field, field_view) = gpu.texture_2d(
            "rd field",
            size,
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );

        let sim_uniform = gpu.uniform_buffer("rd sim", &SimUniform::zeroed());
        let prep_uniform = gpu.uniform_buffer("rd prep", &PrepUniform::zeroed());
        let draw_uniform = gpu.uniform_buffer("rd draw", &DrawUniform::zeroed());

        let cs = ShaderStages::COMPUTE;
        let step_layout = gpu.bind_group_layout(
            "rd step",
            &[
                layout::uniform(0, cs),
                layout::storage(1, cs, true),
                layout::storage(2, cs, false),
                layout::storage(3, cs, true),
            ],
        );
        let step_pipeline_layout = gpu.pipeline_layout("rd step", &[&step_layout]);
        let step_pipeline =
            gpu.compute_pipeline("rd step", &step_pipeline_layout, &module, "cs_step");
        let inject_pipeline =
            gpu.compute_pipeline("rd inject", &step_pipeline_layout, &module, "cs_inject");
        let step_groups = [0, 1].map(|i| {
            gpu.bind_group(
                "rd step",
                &step_layout,
                &[
                    sim_uniform.as_entire_binding(),
                    buffers[i].as_entire_binding(),
                    buffers[1 - i].as_entire_binding(),
                    stats.as_entire_binding(),
                ],
            )
        });

        let prep_layout = gpu.bind_group_layout(
            "rd prepare",
            &[
                layout::uniform(0, cs),
                layout::storage(1, cs, true),
                layout::storage(2, cs, true),
                layout::storage_texture(
                    3,
                    cs,
                    wgpu::TextureFormat::Rgba16Float,
                    wgpu::StorageTextureAccess::WriteOnly,
                ),
                layout::storage(4, cs, false),
                layout::storage(5, cs, false),
            ],
        );
        let prep_pipeline_layout = gpu.pipeline_layout("rd prepare", &[&prep_layout]);
        let prepare_pipeline =
            gpu.compute_pipeline("rd prepare", &prep_pipeline_layout, &module, "cs_prepare");
        let resolve_pipeline =
            gpu.compute_pipeline("rd resolve", &prep_pipeline_layout, &module, "cs_resolve");
        let prepare_groups = [0, 1].map(|i| {
            gpu.bind_group(
                "rd prepare",
                &prep_layout,
                &[
                    prep_uniform.as_entire_binding(),
                    buffers[i].as_entire_binding(),
                    buffers[1 - i].as_entire_binding(),
                    wgpu::BindingResource::TextureView(&field_view),
                    hist.as_entire_binding(),
                    stats.as_entire_binding(),
                ],
            )
        });

        let fs = ShaderStages::FRAGMENT;
        let draw_layout = gpu.bind_group_layout(
            "rd draw",
            &[
                layout::uniform(0, fs),
                layout::texture(1, fs, true),
                layout::sampler(2, fs, true),
                layout::texture(3, fs, true),
                layout::sampler(4, fs, true),
                layout::storage(5, fs, true),
            ],
        );
        let draw_pipeline = gpu.fullscreen_pipeline(
            "rd display",
            &gpu.pipeline_layout("rd draw", &[&draw_layout]),
            &module,
            "fs_display",
            SCENE_FORMAT,
            None,
        );
        let lut = PaletteLut::new(gpu, palette::find(PRESETS[0].palette).unwrap_or(0));
        // Repeat addressing is the torus wrap for the filtered field lookups.
        let wrap = gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat);
        let draw_group = gpu.bind_group(
            "rd draw",
            &draw_layout,
            &[
                draw_uniform.as_entire_binding(),
                wgpu::BindingResource::TextureView(&field_view),
                wgpu::BindingResource::Sampler(&wrap),
                wgpu::BindingResource::TextureView(&lut.view),
                wgpu::BindingResource::Sampler(&lut.sampler),
                stats.as_entire_binding(),
            ],
        );

        let measure_uniform = gpu.uniform_buffer("rd measure", &MeasureUniform::zeroed());
        let blocks = [size[0].div_ceil(WORKGROUP), size[1].div_ceil(WORKGROUP)];
        let reduction = Reduction::new(gpu, "rd measure", blocks[0] * blocks[1]);
        let measure_layout = gpu.bind_group_layout(
            "rd measure",
            &[
                layout::uniform(0, cs),
                layout::storage(1, cs, true),
                layout::storage(2, cs, true),
                layout::storage(3, cs, false),
            ],
        );
        let measure_module = gpu.shader(
            "rd measure",
            &format!("{}\n{}", metrics::WGSL, include_str!("../shaders/reaction_diffusion_measure.wgsl")),
        );
        let measure_pipeline = gpu.compute_pipeline(
            "rd measure",
            &gpu.pipeline_layout("rd measure", &[&measure_layout]),
            &measure_module,
            "cs_measure",
        );
        let measure_groups = [0, 1].map(|i| {
            gpu.bind_group(
                "rd measure",
                &measure_layout,
                &[
                    measure_uniform.as_entire_binding(),
                    buffers[i].as_entire_binding(),
                    buffers[1 - i].as_entire_binding(),
                    reduction.partials().as_entire_binding(),
                ],
            )
        });

        let mut world = Self {
            size,
            params: PRESETS[0].params,
            post: PRESETS[0].post,
            preset: 0,
            lut,
            buffers,
            current: 0,
            stats,
            sim_uniform,
            prep_uniform,
            draw_uniform,
            inject_pipeline,
            step_pipeline,
            step_groups,
            prepare_pipeline,
            resolve_pipeline,
            prepare_groups,
            draw_pipeline,
            draw_group,
            _field: field,
            measure_uniform,
            measure_pipeline,
            measure_groups,
            reduction,
            frame: 0,
            sim_time: 0.0,
            drift_clock: 0.0,
            rain_rng: Rng::new(seed),
            flow_phase: 0.0,
            last_pointer: None,
            snap_window: true,
        };
        world.load_preset(gpu, 0, seed);
        world
    }

    /// Normalisation and thresholds of the measurement kernel. The tests
    /// recompute the metrics on the CPU from the same values.
    fn measure_uniform(&self) -> MeasureUniform {
        MeasureUniform {
            size: self.size,
            inv_cells: 1.0 / (self.size[0] as f32 * self.size[1] as f32),
            _pad: 0.0,
            thresholds: [ALIVE_V, BODY_V, FILLED_V, ACTIVE_DV / self.params.steps_per_frame.max(1) as f32],
        }
    }

    /// Reduces the latest field into `reduction`'s totals.
    fn record_measure(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder) {
        gpu.write(&self.measure_uniform, &self.measure_uniform());
        let blocks = [self.size[0].div_ceil(WORKGROUP), self.size[1].div_ceil(WORKGROUP)];
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rd measure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.measure_pipeline);
            pass.set_bind_group(0, &self.measure_groups[self.current], &[]);
            pass.dispatch_workgroups(blocks[0], blocks[1], 1);
        }
        self.reduction.record(gpu, encoder, blocks[0] * blocks[1]);
    }

    /// Builds the initial (u, v) field on the CPU: u = 1, v = 0 with seeded patches.
    fn seed_state(&self, seed: u64) -> Vec<[f32; 2]> {
        let [w, h] = self.size;
        let p = &self.params;
        let mut rng = Rng::new(seed);
        let mut cells = vec![[1.0f32, 0.0f32]; (w * h) as usize];
        let area = (w * h) as f32;
        // Seeds scale with the pattern wavelength, which grows with sqrt(D).
        let unit = p.scale.max(0.1).sqrt();
        let mut scatter =
            |count: u32, radius: (f32, f32), hole: bool, cells: &mut Vec<[f32; 2]>| {
                for _ in 0..count {
                    let c = [rng.range(0.0, w as f32), rng.range(0.0, h as f32)];
                    let r = rng.range(radius.0, radius.1) * unit;
                    stamp(cells, self.size, c, r, &mut rng, hole);
                }
            };
        match p.seeding {
            Seeding::Sparse => scatter(
                (area / 20000.0).max(5.0) as u32,
                (4.0, 8.0),
                false,
                &mut cells,
            ),
            Seeding::Blobs => scatter(
                (area / 9000.0).max(6.0) as u32,
                (2.5, 9.0),
                false,
                &mut cells,
            ),
            Seeding::Noise => scatter(
                (area / 600.0).max(20.0) as u32,
                (1.0, 3.0),
                false,
                &mut cells,
            ),
            Seeding::Dense => {
                // Patches covering roughly a third of the domain nucleate every regime.
                let mean_area = std::f32::consts::PI * 30.0 * unit * unit;
                scatter(
                    (area * 0.3 / mean_area).max(12.0) as u32,
                    (3.0, 8.0),
                    false,
                    &mut cells,
                );
            }
            Seeding::Square => {
                let side = w.min(h) as f32 * 0.22;
                let (x0, y0) = ((w as f32 - side) * 0.5, (h as f32 - side) * 0.5);
                for y in y0 as u32..(y0 + side) as u32 {
                    for x in x0 as u32..(x0 + side) as u32 {
                        cells[(y.min(h - 1) * w + x.min(w - 1)) as usize] = seed_value(&mut rng);
                    }
                }
            }
            Seeding::Foam => {
                // Start from the upper homogeneous state and punch holes into it.
                let (bu, bv) = blue_state(p.feed, p.kill).unwrap_or((0.5, 0.25));
                let mut jitter = Rng::new(seed ^ 0xF0A4);
                for c in cells.iter_mut() {
                    *c = [
                        bu + jitter.range(-0.02, 0.02),
                        bv + jitter.range(-0.02, 0.02),
                    ];
                }
                scatter(
                    (area / 1500.0).max(12.0) as u32,
                    (2.0, 5.0),
                    true,
                    &mut cells,
                );
            }
            Seeding::Rings => {
                let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
                let max_r = w.min(h) as f32 * 0.45;
                let mut r = 12.0;
                let mut ring_rng = Rng::new(seed ^ 0x2146);
                while r < max_r {
                    let steps = (r * 0.8) as u32;
                    for i in 0..steps {
                        let a = i as f32 / steps as f32 * TAU;
                        let at = [cx + a.cos() * r, cy + a.sin() * r];
                        stamp(&mut cells, self.size, at, 2.0 * unit, &mut ring_rng, false);
                    }
                    r += ring_rng.range(28.0, 60.0);
                }
            }
            Seeding::Center => {
                let centre = [w as f32 / 2.0, h as f32 / 2.0];
                stamp(&mut cells, self.size, centre, 10.0 * unit, &mut rng, false);
            }
            Seeding::Fronts => {
                let count = (area / 60000.0).max(4.0) as u32;
                for _ in 0..count {
                    let centre = [rng.range(0.0, w as f32), rng.range(0.0, h as f32)];
                    let angle = rng.range(0.0, TAU);
                    let half_len = rng.range(60.0, 140.0) * unit;
                    front(
                        &mut cells, self.size, centre, angle, half_len, unit, &mut rng,
                    );
                }
            }
        }
        cells
    }

    /// This frame's automatic droplets (deterministic from the seed).
    fn next_drops(&mut self) -> (u32, [[f32; 4]; MAX_DROPS]) {
        let mut drops = [[0.0f32; 4]; MAX_DROPS];
        let rate = self.params.rain.clamp(0.0, MAX_DROPS as f32);
        let mut count = rate.floor() as usize;
        if self.rain_rng.f32() < rate.fract() {
            count += 1;
        }
        let count = count.min(MAX_DROPS);
        let unit = self.params.scale.max(0.1).sqrt();
        for drop in drops.iter_mut().take(count) {
            let x = self.rain_rng.range(0.0, self.size[0] as f32);
            let y = self.rain_rng.range(0.0, self.size[1] as f32);
            let r = self.rain_rng.range(3.0, 6.0) * unit;
            *drop = [x, y, r, 0.0];
        }
        (count as u32, drops)
    }

    /// This frame's simulation uniform. Also advances the stroke tracking,
    /// so it is called exactly once per stepped frame.
    fn sim_uniform(&mut self, frame: &Frame) -> SimUniform {
        let (drop_count, drops) = self.next_drops();
        let (pointer, radius, button) = match frame.pointer {
            Some(ptr) if ptr.primary || ptr.secondary => (
                ptr.pos,
                ptr.radius.max(1.0),
                if ptr.primary { 1 } else { 2 },
            ),
            _ => ([0.0, 0.0], 0.0, 0),
        };
        // A stroke continues while the same button stays held; the shader then
        // stamps only what the brush newly covers.
        let (prev_pointer, pointer_mode) = match self.last_pointer {
            Some((prev, last)) if button != 0 && last == button => {
                (prev, button | STROKE_CONTINUES)
            }
            _ => (pointer, button),
        };
        self.last_pointer = (button != 0).then_some((pointer, button));

        let p = &self.params;
        // "Create" sprays seeds onto the ground and "erase" restores bare U.
        // Where the filled state is the ground (foam), seeds would die on
        // bare U, so "create" lays down fresh foam (the filled state riddled
        // with holes that swell into bubbles) and "erase" pops bubbles.
        let (create_state, create_jitter, spray_density) = if p.filled_ground() {
            let (bu, bv) = blue_state(p.feed, p.kill).unwrap_or((0.5, 0.25));
            ([bu, bv], 0.0, FOAM_HOLES)
        } else {
            ([0.5, 0.25], 0.1, 0.1)
        };
        let unit = p.scale.max(0.1).sqrt();
        SimUniform {
            size: self.size,
            feed: p.feed,
            kill: p.kill,
            du: p.scale,
            dv: p.scale / p.ratio.max(0.1),
            dt: p.stable_dt(),
            atlas: u32::from(p.atlas),
            drift: p.drift,
            drift_phase: self.drift_clock as f32,
            scale_var: p.scale_var,
            frame: self.frame as u32,
            pointer,
            pointer_radius: radius,
            pointer_mode,
            create_state,
            erase_state: [1.0, 0.0],
            spray: [SPRAY_SPACING * unit, SPRAY_DOT * unit],
            create_jitter,
            spray_density,
            drop_count,
            filled_ground: u32::from(p.filled_ground()),
            prev_pointer,
            ground: p.ground,
            flow_phase: self.flow_phase,
            aniso: p.aniso,
            _pad: 0.0,
            drops,
        }
    }
}

/// Seed chemistry: Pearson's perturbed (0.5, 0.25).
fn seed_value(rng: &mut Rng) -> [f32; 2] {
    [0.5 + rng.range(-0.05, 0.05), 0.25 + rng.range(-0.05, 0.05)]
}

// --- optional: seeding shapes --------------------------------------------------

/// Paints a disk of seed chemistry (or, with `hole`, of the trivial state) into `cells`.
fn stamp(
    cells: &mut [[f32; 2]],
    size: [u32; 2],
    centre: [f32; 2],
    r: f32,
    rng: &mut Rng,
    hole: bool,
) {
    let [w, h] = size;
    let ri = r.ceil() as i32;
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            if (dx * dx + dy * dy) as f32 > r * r {
                continue;
            }
            let x = (centre[0] as i32 + dx).rem_euclid(w as i32) as u32;
            let y = (centre[1] as i32 + dy).rem_euclid(h as i32) as u32;
            cells[(y * w + x) as usize] = if hole { [1.0, 0.0] } else { seed_value(rng) };
        }
    }
}

/// Paints a broken wave front: a band of seed chemistry backed by a wider,
/// U-depleted (refractory) band. Such a front can only travel away from its
/// refractory side, and its two free ends curl up into a pair of
/// counter-rotating spirals; the wide refractory band gives them room to wind.
fn front(
    cells: &mut [[f32; 2]],
    size: [u32; 2],
    centre: [f32; 2],
    angle: f32,
    half_len: f32,
    unit: f32,
    rng: &mut Rng,
) {
    let [w, h] = size;
    let along = [angle.cos(), angle.sin()];
    let ahead = 2.5 * unit;
    let behind = 22.0 * unit;
    let reach = (half_len + behind).ceil() as i32;
    for dy in -reach..=reach {
        for dx in -reach..=reach {
            let (fx, fy) = (dx as f32, dy as f32);
            // Coordinates along the front and along its direction of travel.
            let t = fx * along[0] + fy * along[1];
            let s = fy * along[0] - fx * along[1];
            if t.abs() > half_len || s > ahead || s < -behind {
                continue;
            }
            let x = (centre[0] as i32 + dx).rem_euclid(w as i32) as u32;
            let y = (centre[1] as i32 + dy).rem_euclid(h as i32) as u32;
            cells[(y * w + x) as usize] = if s >= 0.0 {
                seed_value(rng)
            } else {
                [0.15, 0.0]
            };
        }
    }
}

impl World for ReactionDiffusion {
    fn settings(&self) -> anyhow::Result<crate::library::WorldSettings> {
        Ok(crate::library::WorldSettings::ReactionDiffusion {
            params: self.params, palette: palette::PALETTES[self.lut.index()].name.to_owned(), post: self.post,
        })
    }

    fn restore_settings(&mut self, gpu: &Gpu, settings: &crate::library::WorldSettings, seed: u64) -> anyhow::Result<()> {
        let crate::library::WorldSettings::ReactionDiffusion { params, palette: name, post } = settings else { anyhow::bail!("Wrong world settings"); };
        let index = palette::find(name).ok_or_else(|| anyhow::anyhow!("Unknown palette: {name}"))?;
        anyhow::ensure!((1..=128).contains(&params.steps_per_frame), "Invalid step count");
        // Scale also controls CPU seed-stamp radii; reject extreme values
        // before they can overflow integer disk bounds during reset.
        anyhow::ensure!((0.1..=4.0).contains(&params.scale)
            && (0.1..=10.0).contains(&params.ratio) && (0.01..=1.0).contains(&params.dt)
            && (0.0..=0.2).contains(&params.feed) && (0.0..=0.2).contains(&params.kill),
            "Invalid reaction-diffusion rates or scale");
        self.params = *params;
        self.post = *post;
        self.lut.set(gpu, index);
        self.reset(gpu, seed);
        Ok(())
    }

    fn id(&self) -> &'static str {
        "reaction-diffusion"
    }

    fn name(&self) -> &'static str {
        "Reaction-Diffusion"
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
        self.params = preset.params;
        self.post = preset.post;
        self.lut
            .set(gpu, palette::find(preset.palette).unwrap_or(0));
        self.reset(gpu, seed);
    }

    fn reset(&mut self, gpu: &Gpu, seed: u64) {
        self.flow_phase = Rng::new(seed ^ 0x7269_6467).range(0.0, TAU);
        let cells = self.seed_state(seed);
        // Both buffers get the seed so the first activity estimate is zero.
        for buffer in &self.buffers {
            gpu.queue
                .write_buffer(buffer, 0, bytemuck::cast_slice(&cells));
        }
        // Clear the revive counter and flag (floats 4..8): a field that went
        // uniform before the reset must not trigger a revive over the new seed.
        gpu.queue
            .write_buffer(&self.stats, 16, bytemuck::cast_slice(&[0.0f32; 4]));
        self.current = 0;
        self.frame = 0;
        self.sim_time = 0.0;
        self.drift_clock = 0.0;
        self.rain_rng = Rng::new(seed ^ 0x5241_494E);
        self.last_pointer = None;
        self.snap_window = true;
    }

    fn mutate(&mut self, gpu: &Gpu, seed: u64) {
        let mut rng = Rng::new(seed ^ 0xC0FFEE);
        // Chemistry: a point inside one of the known-alive regimes.
        let regime = rng.pick(REGIMES);
        let feed = rng.range(regime.feed.0, regime.feed.1);
        let (lo, hi) = kill_band(feed);
        let offset = rng.range(regime.offset.0, regime.offset.1).clamp(lo, hi);
        // Look: borrow a proven dark-ground preset look, then vary it. Ink is
        // a light look, and dark-field membranes are tuned for sparse cells
        // (they glow in the gaps of a dense labyrinth), so neither is lent.
        let donors: Vec<&Preset> = PRESETS
            .iter()
            .filter(|p| {
                !p.params.atlas && !matches!(p.params.material, Material::Ink | Material::DarkField)
            })
            .collect();
        let donor = donors[rng.below(donors.len() as u32) as usize];
        let mut p = donor.params;
        p.feed = feed;
        p.kill = critical_kill(feed) + offset;
        p.atlas = false;
        p.seeding = regime.seeding;
        p.invert = regime.invert;
        p.scale = rng.range(0.8, 1.3);
        p.ratio = 2.0;
        p.dt = 1.0;
        p.steps_per_frame = 24;
        p.drift = rng.range(0.0004, 0.0014);
        p.drift_speed = rng.range(0.6, 1.4);
        p.scale_var = rng.range(0.0, 0.3);
        p.rain = 0.0;
        // Dense regimes often get lagoons for negative space: k inside them
        // is lifted a little above the top of the survival band.
        p.ground = if regime.dense && rng.chance(0.6) {
            (hi - offset).max(0.0) + rng.range(0.0008, 0.0016)
        } else {
            0.0
        };
        // Stripes sometimes follow a whorled ridge flow.
        p.aniso = if regime.stripes && rng.chance(0.3) {
            rng.range(0.15, 0.25)
        } else {
            0.0
        };
        p.relief = (p.relief * rng.range(0.8, 1.2)).clamp(0.4, 1.3);
        p.gloss = (p.gloss * rng.range(0.8, 1.2)).clamp(0.3, 1.6);
        // The donor's contrast window was tuned for its own density; a window
        // this high keeps the ground dark even when the new regime fills the
        // domain (a low window would turn a dense labyrinth into all body).
        p.contrast_lo = rng.range(0.28, 0.4);
        p.contrast_hi = p.contrast_lo + rng.range(0.2, 0.28);
        let mut post = donor.post;
        if regime.dense {
            p.iridescence = p.iridescence.min(0.35);
            p.glow = p.glow.min(0.6);
            p.halo = p.halo.min(0.1);
            post.bloom = post.bloom.min(0.5);
        }
        self.params = p;
        self.post = post;
        let dark: Vec<usize> = (0..palette::PALETTES.len())
            .filter(|&i| palette::PALETTES[i].name != "Ink")
            .collect();
        self.lut
            .set(gpu, dark[rng.below(dark.len() as u32) as usize]);
        self.reset(gpu, seed);
    }

    fn step(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder) {
        let uniform = self.sim_uniform(frame);
        frame.gpu.write(&self.sim_uniform, &uniform);
        let groups = [
            self.size[0].div_ceil(WORKGROUP),
            self.size[1].div_ceil(WORKGROUP),
        ];
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rd step"),
                timestamp_writes: None,
            });
            // `step_groups[1 - current]` writes `buffers[current]`, so the
            // injection edits the latest state in place.
            pass.set_pipeline(&self.inject_pipeline);
            pass.set_bind_group(0, &self.step_groups[1 - self.current], &[]);
            pass.dispatch_workgroups(groups[0], groups[1], 1);
            pass.set_pipeline(&self.step_pipeline);
            for _ in 0..self.params.steps_per_frame {
                pass.set_bind_group(0, &self.step_groups[self.current], &[]);
                pass.dispatch_workgroups(groups[0], groups[1], 1);
                self.current = 1 - self.current;
            }
        }
        let elapsed = self.params.steps_per_frame as f64 * uniform.dt as f64;
        self.frame += 1;
        self.sim_time += elapsed;
        let advance = elapsed * self.params.drift_speed as f64 * DRIFT_RATE;
        self.drift_clock = (self.drift_clock + advance).rem_euclid(DRIFT_PERIOD);
    }

    fn metrics(&self) -> &'static [MetricDesc] {
        METRICS
    }

    fn measure(&mut self, frame: &Frame, encoder: &mut wgpu::CommandEncoder, sink: &mut metrics::Sink<'_>) {
        if !sink.is_live() {
            return;
        }
        self.record_measure(frame.gpu, encoder);
        sink.push(encoder, self.reduction.totals());
    }

    fn render(
        &mut self,
        frame: &Frame,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
    ) {
        let p = &self.params;
        frame.gpu.write(
            &self.prep_uniform,
            &PrepUniform {
                size: self.size,
                snap: u32::from(self.snap_window),
                _pad0: 0,
                act_scale: ACTIVITY_SCALE / p.stable_dt().max(1e-3),
                rate: 0.08,
                _pad1: [0.0; 2],
            },
        );
        self.snap_window = false;
        frame.gpu.write(
            &self.draw_uniform,
            &DrawUniform {
                view: frame.view,
                size: self.size,
                time: frame.time,
                material: p.material.code(),
                contrast: [p.contrast_lo, p.contrast_hi.max(p.contrast_lo + 0.02)],
                pal_range: [p.pal_lo, p.pal_hi],
                relief: p.relief,
                gloss: p.gloss,
                glow: p.glow,
                halo: p.halo,
                iridescence: p.iridescence,
                activity: p.activity,
                shadow: p.shadow,
                brightness: p.brightness,
                invert: u32::from(p.invert),
                aura_tint: p.aura_tint,
                reflect: p.reflect,
                clarity: p.clarity,
            },
        );
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rd prepare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.prepare_pipeline);
            pass.set_bind_group(0, &self.prepare_groups[self.current], &[]);
            pass.dispatch_workgroups(
                self.size[0].div_ceil(WORKGROUP),
                self.size[1].div_ceil(WORKGROUP),
                1,
            );
            pass.set_pipeline(&self.resolve_pipeline);
            pass.dispatch_workgroups(1, 1, 1);
        }
        gpu::fullscreen_pass(
            encoder,
            "rd display",
            target,
            Some(wgpu::Color::BLACK),
            &self.draw_pipeline,
            &[&self.draw_group],
        );
    }

    fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) {
        let Self { params, lut, .. } = self;
        egui::CollapsingHeader::new("Chemistry")
            .default_open(true)
            .show(ui, |ui| params.chemistry_ui(ui));
        egui::CollapsingHeader::new("Look")
            .default_open(true)
            .show(ui, |ui| {
                lut.ui(gpu, ui);
                params.look_ui(ui);
            });
    }

    fn post_settings(&self) -> PostSettings {
        self.post
    }

    fn stats(&self) -> String {
        let p = &self.params;
        let t = self.sim_time / 1000.0;
        if p.atlas {
            format!(
                "{}x{} cells · atlas · t {t:.1}k",
                self.size[0], self.size[1]
            )
        } else {
            format!(
                "{}x{} cells · F {:.4} k {:.4} · t {t:.1}k",
                self.size[0], self.size[1], p.feed, p.kill
            )
        }
    }

    fn controls_hint(&self) -> &'static str {
        if self.params.filled_ground() {
            "Left: blow fresh foam · Right: pop bubbles"
        } else {
            "Left: spray seeds of chemical V · Right: erase back to bare U"
        }
    }
}
