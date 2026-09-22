//! `primordia explore`: novelty-ranked exploration of a world's mutation space.
//!
//! Every candidate parameter set (a `World::mutate` in round 0, a numerically
//! perturbed copy of a kept recipe in refinement rounds) is simulated headlessly
//! while its measurements (`crate::metrics`) are collected. The measurement
//! traces become a behaviour descriptor, the candidates are scaled robustly
//! against each other, and the archive keeps the ones that are most mutually
//! different (or the extremes of one metric). The kept candidates are
//! re-simulated from their recipes for their images, which also proves that
//! the recipes round-trip; the recipes load through the app's Library tab, and
//! `render --recipe` renders them (or the images, which carry them) again.
//!
//! The search starts from a preset, or from a recipe (`--recipe`), and `--set`
//! edits either before the first candidate is evaluated.

use std::fmt;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{bail, Context as _, Result};
use serde_json::Value;

use crate::capture::{self, Provenance, Readback};
use crate::failure::Failure;
use crate::gpu::Gpu;
use crate::headless::{self, contact_sheet, frame_deadline, OUTPUT_FORMAT};
use crate::library::{Library, SavedWorld, WorldSettings};
use crate::metrics::{MetricDesc, Sample, Sampler, MAX_METRICS};
use crate::post::{Post, PostSettings};
use crate::recipe::{self, Recipe, Setting, Source};
use crate::rng::Rng;
use crate::world::{self, guarded, Camera, Frame, ViewXform, World, WORLDS};

/// Stream of candidate seeds and perturbations, separate from the app's seeds.
const SEED_SALT: u64 = 0x4558_504c_4f52_4521;
/// Perturbations tried per child before it is skipped.
const CHILD_ATTEMPTS: u32 = 8;
/// Scaled distance below which two candidates count as the same behaviour.
const DUPLICATE: f32 = 1e-6;
/// Scaled values are bounded so one exploded candidate cannot dominate every distance.
const SCALED_LIMIT: f32 = 6.0;
/// A vital measurement's late mean below this marks a candidate as inert (the
/// vital measurements of each world are registered in `WorldEntry::vital`).
const INERT: f32 = 1e-3;

/// How the archive is chosen from the evaluated candidates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Select {
    /// Mutually most different behaviour (farthest-point selection by novelty).
    Novelty,
    /// The largest late mean of one metric.
    Max(String),
    /// The smallest late mean of one metric.
    Min(String),
}

impl FromStr for Select {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let text = text.trim();
        if text.eq_ignore_ascii_case("novelty") {
            return Ok(Self::Novelty);
        }
        let (kind, id) = text
            .split_once(':')
            .ok_or_else(|| format!("expected 'novelty', 'max:<metric>' or 'min:<metric>' (got '{text}')"))?;
        let id = id.trim();
        if id.is_empty() || !id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_') {
            return Err(format!("'{id}' is not a metric id (lowercase letters, digits and underscores)"));
        }
        match kind.trim().to_ascii_lowercase().as_str() {
            "max" => Ok(Self::Max(id.to_string())),
            "min" => Ok(Self::Min(id.to_string())),
            other => Err(format!("unknown selection '{other}' (use novelty, max:<metric> or min:<metric>)")),
        }
    }
}

impl fmt::Display for Select {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Novelty => f.write_str("novelty"),
            Self::Max(id) => write!(f, "max:{id}"),
            Self::Min(id) => write!(f, "min:{id}"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ExploreJob {
    /// World and preset names; ignored when `recipe` is given.
    pub world: String,
    pub preset: Option<String>,
    /// Start from this recipe (run from its own seed) instead of a preset.
    pub recipe: Option<Recipe>,
    /// `--set` edits of the recipe or preset the search starts from.
    pub sets: Vec<Setting>,
    /// Master seed: candidate seeds and perturbations follow from it, and the
    /// base preset runs from it.
    pub seed: u64,
    /// Mutations evaluated in round 0 (the base preset is evaluated as well).
    pub runs: u32,
    /// Refinement rounds, each perturbing recipes from the current archive.
    pub refine: u32,
    /// Children evaluated per refinement round.
    pub children: u32,
    /// Relative size of a perturbation (multiplicative log-normal noise).
    pub strength: f32,
    /// Candidates kept in the archive.
    pub keep: usize,
    pub frames: u32,
    pub size: [u32; 2],
    /// Frames-per-second ceiling per candidate (0 = unlimited).
    pub max_fps: f32,
    pub out_dir: PathBuf,
    pub select: Select,
    /// Also save the kept recipes into this library folder.
    pub library: Option<PathBuf>,
    pub sheet: bool,
    /// Also write every evaluated candidate's final frame under `all/`.
    pub all: bool,
    /// Let inert candidates (dead, empty or frozen worlds) into the archive.
    pub inert: bool,
}

impl ExploreJob {
    pub fn new(world: &str) -> Self {
        Self {
            world: world.to_string(),
            preset: None,
            recipe: None,
            sets: Vec::new(),
            seed: 1,
            runs: 48,
            refine: 1,
            children: 24,
            strength: 0.15,
            keep: 12,
            frames: 600,
            size: [640, 360],
            max_fps: headless::DEFAULT_MAX_FPS,
            out_dir: PathBuf::from("explore"),
            select: Select::Novelty,
            library: None,
            sheet: true,
            all: false,
            inert: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The base preset, unchanged.
    Preset,
    /// The base recipe (`--recipe`), or the preset edited by `--set`.
    Recipe,
    /// A `World::mutate` from a fresh seed.
    Mutation,
    /// A perturbed copy of the archive member `parent`'s recipe.
    Child { parent: usize },
}

impl Origin {
    /// The `origin` column of candidates.csv and field of `--json`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Preset => "preset",
            Self::Recipe => "recipe",
            Self::Mutation => "mutation",
            Self::Child { .. } => "child",
        }
    }
}

impl fmt::Display for Origin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Child { parent } => write!(f, "child of #{parent}"),
            other => f.write_str(other.name()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub index: usize,
    pub round: u32,
    pub origin: Origin,
    pub seed: u64,
    /// Preset index at the time of capture (restoring a recipe never changes it).
    pub preset: usize,
    pub settings: WorldSettings,
    pub look: PostSettings,
    /// `None` when the run produced too few or non-finite measurements.
    pub descriptor: Option<Vec<f32>>,
    /// A vital measurement stayed at zero: nothing lives or moves.
    pub inert: bool,
    pub novelty: f32,
    pub rank: Option<usize>,
    pub secs: f32,
}

#[derive(Debug)]
pub struct Summary {
    /// Index into `WORLDS`.
    pub world: usize,
    /// 0-based preset the search started from.
    pub preset: usize,
    pub candidates: Vec<Candidate>,
    /// Indices of the kept candidates in rank order.
    pub kept: Vec<usize>,
    /// Image and recipe of each kept candidate, in rank order.
    pub images: Vec<PathBuf>,
    pub recipes: Vec<PathBuf>,
    /// Library files written by `--install`, in rank order.
    pub installed: Vec<PathBuf>,
    /// Every candidate's final frame under `all/` (with `--all`).
    pub all: Vec<PathBuf>,
    pub sheet: Option<PathBuf>,
    pub csv: PathBuf,
    pub secs: f32,
}

impl Summary {
    /// Every file written, in the order they were written.
    pub fn files(&self) -> Vec<&Path> {
        let mut files: Vec<&Path> = self.all.iter().map(PathBuf::as_path).collect();
        for (rank, (image, recipe)) in self.images.iter().zip(&self.recipes).enumerate() {
            files.extend([image.as_path(), recipe.as_path()]);
            files.extend(self.installed.get(rank).map(PathBuf::as_path));
        }
        files.push(&self.csv);
        files.extend(self.sheet.as_deref());
        files
    }
}

// --- behaviour descriptors -----------------------------------------------------

/// Column names of a descriptor: three per metric.
pub(crate) fn dim_names(metrics: &[MetricDesc]) -> Vec<String> {
    metrics
        .iter()
        .flat_map(|m| [format!("{}_mean", m.id), format!("{}_std", m.id), format!("{}_drift", m.id)])
        .collect()
}

/// Per metric: the mean of the last 40% of the samples, the temporal standard
/// deviation over that window, and the drift from the first 20%. Series 0 only.
pub(crate) fn descriptor(samples: &[Sample], metric_count: usize) -> Option<Vec<f32>> {
    let n = samples.len();
    if n < 2 {
        return None;
    }
    let early = (n / 5).max(1);
    let late = (2 * n / 5).max(1);
    let mut out = Vec::with_capacity(metric_count * 3);
    for m in 0..metric_count.min(MAX_METRICS) {
        let values = |range: std::ops::Range<usize>| samples[range].iter().map(|s| f64::from(s.values[0][m]));
        let early_mean = values(0..early).sum::<f64>() / early as f64;
        let late_mean = values(n - late..n).sum::<f64>() / late as f64;
        let variance = values(n - late..n).map(|v| (v - late_mean).powi(2)).sum::<f64>() / late as f64;
        out.extend([late_mean as f32, variance.sqrt() as f32, (late_mean - early_mean) as f32]);
    }
    out.iter().all(|v| v.is_finite()).then_some(out)
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { 0.5 * (values[n / 2 - 1] + values[n / 2]) }
}

/// Centres every dimension on its median and divides by 1.4826 × MAD (the
/// standard deviation when the MAD vanishes). Constant dimensions become zero
/// and are not counted as active. Failed rows stay `None`.
pub(crate) fn scale_robust(rows: &[Option<Vec<f32>>]) -> (Vec<Option<Vec<f32>>>, usize) {
    let dims = rows.iter().flatten().map(Vec::len).max().unwrap_or(0);
    let mut scaled: Vec<Option<Vec<f32>>> = rows.iter().map(|r| r.as_ref().map(|r| vec![0.0; r.len()])).collect();
    let mut active = 0;
    for d in 0..dims {
        let mut column: Vec<f32> = rows.iter().flatten().filter_map(|r| r.get(d).copied()).collect();
        if column.is_empty() {
            continue;
        }
        let centre = median(&mut column);
        let mut deviations: Vec<f32> = column.iter().map(|v| (v - centre).abs()).collect();
        let mut scale = 1.4826 * median(&mut deviations);
        if scale < 1e-9 {
            let mean = column.iter().sum::<f32>() / column.len() as f32;
            scale = (column.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / column.len() as f32).sqrt();
        }
        if scale < 1e-9 {
            continue;
        }
        active += 1;
        for (row, out) in rows.iter().zip(&mut scaled) {
            if let (Some(row), Some(out)) = (row, out) {
                if let Some(v) = row.get(d) {
                    out[d] = ((v - centre) / scale).clamp(-SCALED_LIMIT, SCALED_LIMIT);
                }
            }
        }
    }
    (scaled, active)
}

/// Euclidean distances between scaled rows, divided by the square root of the
/// active dimensions; infinite where either row failed.
pub(crate) fn distances(rows: &[Option<Vec<f32>>], active: usize) -> Vec<Vec<f32>> {
    let norm = (active.max(1) as f32).sqrt();
    let n = rows.len();
    let mut dist = vec![vec![f32::INFINITY; n]; n];
    for i in 0..n {
        dist[i][i] = 0.0;
        for j in i + 1..n {
            if let (Some(a), Some(b)) = (&rows[i], &rows[j]) {
                let d = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() / norm;
                dist[i][j] = d;
                dist[j][i] = d;
            }
        }
    }
    dist
}

/// Mean distance to the `k` nearest other candidates (0 for failed ones).
pub(crate) fn novelty(dist: &[Vec<f32>], ok: &[bool], k: usize) -> Vec<f32> {
    (0..dist.len())
        .map(|i| {
            if !ok[i] {
                return 0.0;
            }
            let mut near: Vec<f32> = (0..dist.len()).filter(|&j| j != i && ok[j]).map(|j| dist[i][j]).collect();
            near.sort_by(f32::total_cmp);
            let k = k.min(near.len());
            if k == 0 { 0.0 } else { near[..k].iter().sum::<f32>() / k as f32 }
        })
        .collect()
}

/// Farthest-point selection: the most novel candidate first, then whichever
/// candidate is farthest from everything already selected, skipping
/// duplicates, until `keep` are chosen or nothing distinct is left.
pub(crate) fn select_novel(dist: &[Vec<f32>], novelty: &[f32], ok: &[bool], keep: usize) -> Vec<usize> {
    let mut selected: Vec<usize> = Vec::new();
    let Some(first) = (0..dist.len()).filter(|&i| ok[i]).max_by(|&a, &b| {
        novelty[a].total_cmp(&novelty[b]).then_with(|| b.cmp(&a))
    }) else {
        return selected;
    };
    selected.push(first);
    while selected.len() < keep {
        let best = (0..dist.len())
            .filter(|&i| ok[i] && !selected.contains(&i))
            .map(|i| (i, selected.iter().map(|&s| dist[i][s]).fold(f32::INFINITY, f32::min)))
            .filter(|(_, gap)| *gap > DUPLICATE)
            .max_by(|(a, ga), (b, gb)| ga.total_cmp(gb).then_with(|| b.cmp(a)));
        match best {
            Some((i, _)) => selected.push(i),
            None => break,
        }
    }
    selected
}

/// The `keep` largest (or smallest) scores, skipping behavioural duplicates.
pub(crate) fn select_extreme(score: &[Option<f32>], dist: &[Vec<f32>], largest: bool, keep: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..score.len()).filter(|&i| score[i].is_some_and(f32::is_finite)).collect();
    order.sort_by(|&a, &b| {
        let (sa, sb) = (score[a].unwrap_or(0.0), score[b].unwrap_or(0.0));
        let by_score = if largest { sb.total_cmp(&sa) } else { sa.total_cmp(&sb) };
        by_score.then_with(|| a.cmp(&b))
    });
    let mut selected: Vec<usize> = Vec::new();
    for i in order {
        if selected.len() == keep {
            break;
        }
        if selected.iter().all(|&s| dist[i][s] > DUPLICATE) {
            selected.push(i);
        }
    }
    selected
}

// --- recipe perturbation ----------------------------------------------------------

/// Nudges the numbers of a recipe: floats by log-normal noise (each with
/// probability one half), large integers likewise, small integers by one and
/// never below zero. Integers inside arrays (colours), booleans, strings, nulls
/// and the `post` look are left alone.
pub(crate) fn perturb(value: &mut Value, rng: &mut Rng, strength: f32) {
    perturb_inner(value, rng, strength, false);
}

fn perturb_inner(value: &mut Value, rng: &mut Rng, strength: f32, in_array: bool) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if key != "post" {
                    perturb_inner(child, rng, strength, false);
                }
            }
        }
        Value::Array(items) => {
            for child in items.iter_mut() {
                perturb_inner(child, rng, strength, true);
            }
        }
        Value::Number(number) => {
            if number.is_f64() {
                if let Some(f) = number.as_f64().filter(|_| rng.chance(0.5)) {
                    let factor = f64::from((rng.normal() * strength).exp());
                    if let Some(nudged) = serde_json::Number::from_f64(f * factor) {
                        *value = Value::Number(nudged);
                    }
                }
            } else if !in_array {
                if let Some(i) = number.as_i64() {
                    let nudged = if i > 32 {
                        if rng.chance(0.5) {
                            let factor = f64::from((rng.normal() * strength).exp());
                            ((i as f64) * factor).round().max(1.0) as i64
                        } else {
                            i
                        }
                    } else if rng.chance(0.25) {
                        if i <= 0 || rng.chance(0.5) { i + 1 } else { i - 1 }
                    } else {
                        i
                    };
                    if nudged != i {
                        *value = if number.is_u64() { Value::from(nudged.max(0) as u64) } else { Value::from(nudged) };
                    }
                }
            }
        }
        _ => {}
    }
}

fn numbers_fit_f32(value: &Value) -> bool {
    match value {
        Value::Object(map) => map.values().all(numbers_fit_f32),
        Value::Array(items) => items.iter().all(numbers_fit_f32),
        Value::Number(n) => n.as_f64().is_some_and(|f| f.is_finite() && f.abs() <= f64::from(f32::MAX) / 2.0),
        _ => true,
    }
}

/// A perturbed copy of `settings`, as a recipe the world can validate.
pub(crate) fn perturbed(settings: &WorldSettings, rng: &mut Rng, strength: f32) -> Result<WorldSettings> {
    let mut value = serde_json::to_value(settings).context("serialising the recipe")?;
    perturb(&mut value, rng, strength);
    if !numbers_fit_f32(&value) {
        bail!("a perturbed value left the f32 range");
    }
    let mut settings: WorldSettings = serde_json::from_value(value).context("the perturbed recipe does not parse")?;
    if let WorldSettings::Symbiosis { params, .. } = &mut settings {
        // A comparison would double the cost and report two series.
        params.compare = false;
    }
    Ok(settings)
}

// --- GPU bench ----------------------------------------------------------------------

/// One world, one output size: everything a candidate evaluation reuses.
struct Bench<'a> {
    gpu: &'a Gpu,
    world: Box<dyn World>,
    post: Post,
    out_texture: wgpu::Texture,
    out_view: wgpu::TextureView,
    readback: Readback,
    sampler: Sampler,
    size: [u32; 2],
    frames: u32,
    max_fps: f32,
}

impl Bench<'_> {
    fn new(gpu: &Gpu, world: Box<dyn World>, size: [u32; 2], frames: u32, max_fps: f32) -> Bench<'_> {
        let (out_texture, out_view) = gpu.texture_2d(
            "explore output",
            size,
            OUTPUT_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        Bench {
            gpu,
            world,
            post: Post::new(gpu, size, OUTPUT_FORMAT),
            out_texture,
            out_view,
            readback: Readback::new(gpu, size, OUTPUT_FORMAT),
            sampler: Sampler::new(gpu, Sampler::HEADLESS_SLOTS),
            size,
            frames,
            max_fps,
        }
    }

    fn mutate(&mut self, seed: u64) -> Result<()> {
        let (gpu, world) = (self.gpu, &mut self.world);
        guarded(gpu, || world.mutate(gpu, seed))
    }

    fn restore(&mut self, settings: &WorldSettings, seed: u64) -> Result<()> {
        let (gpu, world) = (self.gpu, &mut self.world);
        guarded(gpu, || world.restore_settings(gpu, settings, seed))?
    }

    /// Simulates `frames` frames, measuring every one; with `capture` the
    /// final frame is post-processed and read back.
    fn run(&mut self, capture: bool) -> Result<(Vec<Sample>, Option<Vec<u8>>)> {
        let gpu = self.gpu;
        let view = ViewXform::fit(self.world.size(), self.size, &Camera::default());
        let look = self.world.post_settings();
        let dt = 1.0 / 60.0;
        let mut samples = Vec::with_capacity(self.frames as usize);
        let started = Instant::now();
        self.sampler.discard();
        for f in 0..self.frames {
            let frame = Frame {
                gpu,
                time: f as f32 * dt,
                dt,
                frame: u64::from(f),
                view,
                target_size: self.size,
                pointer: None,
            };
            let mut encoder =
                gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("explore frame") });
            self.world.step(&frame, &mut encoder);
            if self.sampler.is_full() {
                samples.extend(self.sampler.flush(gpu).map_err(|e| Failure::Gpu.tag(e))?);
            }
            {
                let mut sink = self.sampler.begin(frame.frame, frame.time);
                self.world.measure(&frame, &mut encoder, &mut sink);
            }
            self.world.render(&frame, &mut encoder, self.post.scene_view());
            let last = f + 1 == self.frames;
            if last && capture {
                self.post.run(gpu, &mut encoder, &look, frame.time, &self.out_view);
                self.readback.copy_from(&mut encoder, &self.out_texture);
            }
            gpu.queue.submit([encoder.finish()]);
            self.sampler.map();
            if let Some(problem) = gpu.fatal_error() {
                return Err(Failure::Gpu.error(format!("GPU error while exploring: {problem}")));
            }
            if f % 4 == 3 && !last {
                // Keep the CPU from queueing hundreds of frames ahead of the GPU.
                gpu.wait_idle();
            }
            samples.extend(self.sampler.collect(gpu));
            if self.max_fps > 0.0 {
                let due = frame_deadline(started, f + 1, self.max_fps)?;
                let now = Instant::now();
                if due > now {
                    std::thread::sleep(due - now);
                }
            }
        }
        samples.extend(self.sampler.flush(gpu).map_err(|e| Failure::Gpu.tag(e))?);
        let pixels = if capture { Some(self.readback.read(gpu).map_err(|e| Failure::Gpu.tag(e))?) } else { None };
        Ok((samples, pixels))
    }
}

// --- the search ----------------------------------------------------------------------

struct Analysis {
    dist: Vec<Vec<f32>>,
    ok: Vec<bool>,
    novelty: Vec<f32>,
}

/// Scales and compares every candidate; `usable` marks the ones selection may pick.
fn analyse(candidates: &[Candidate], usable: &[bool]) -> Analysis {
    let rows: Vec<Option<Vec<f32>>> = candidates.iter().map(|c| c.descriptor.clone()).collect();
    let (scaled, active) = scale_robust(&rows);
    let ok: Vec<bool> = scaled.iter().zip(usable).map(|(s, u)| s.is_some() && *u).collect();
    let dist = distances(&scaled, active);
    let k = ok.iter().filter(|o| **o).count().saturating_sub(1).min(5);
    let novelty = novelty(&dist, &ok, k);
    Analysis { dist, ok, novelty }
}

/// Indices of the archive for `select`, in rank order.
fn choose(candidates: &[Candidate], analysis: &Analysis, select: &Select, lane: Option<usize>, keep: usize) -> Vec<usize> {
    match (select, lane) {
        (Select::Novelty, _) => select_novel(&analysis.dist, &analysis.novelty, &analysis.ok, keep),
        (Select::Max(_), Some(lane)) | (Select::Min(_), Some(lane)) => {
            let scores: Vec<Option<f32>> =
                candidates.iter().map(|c| c.descriptor.as_ref().and_then(|d| d.get(lane * 3).copied())).collect();
            select_extreme(&scores, &analysis.dist, matches!(select, Select::Max(_)), keep)
        }
        _ => Vec::new(),
    }
}

fn file_stem(rank: usize, seed: u64) -> String {
    format!("{rank:02}-seed{seed}")
}

fn recipe(world_name: &str, candidate: &Candidate, rank: usize, size: [u32; 2]) -> SavedWorld {
    SavedWorld {
        version: crate::library::RECIPE_VERSION,
        name: format!("{world_name} explore #{rank:02} (seed {})", candidate.seed),
        seed: candidate.seed,
        output_size: size,
        preset: candidate.preset,
        modified: true,
        settings: candidate.settings.clone(),
        look: candidate.look,
        camera: Camera::default(),
    }
}

/// What an explore image says about itself: its recipe, and that `render
/// --recipe <image> --frames <frames>` renders it again.
fn image_provenance(saved: &SavedWorld, gpu: &Gpu, image: &Path, frames: u32) -> Provenance {
    let name = image.file_name().map_or_else(|| image.display().to_string(), |n| n.to_string_lossy().into_owned());
    let command = format!("primordia render --recipe {} --frames {frames}", headless::shell_word(&name));
    Provenance::of(saved, gpu).with_command(command)
}

/// `candidates.csv`: one row per candidate. `preset` is 1-based like `--preset`
/// (recipes store it 0-based), followed by the preset's name.
fn write_csv(path: &Path, dims: &[String], candidates: &[Candidate], presets: &[&str]) -> Result<()> {
    let mut text = String::from("index,round,origin,parent,seed,preset,preset_name,rank,novelty,status,secs");
    for dim in dims {
        text.push(',');
        text.push_str(dim);
    }
    text.push('\n');
    for c in candidates {
        let origin = c.origin.name();
        let parent = match c.origin {
            Origin::Child { parent } => parent.to_string(),
            _ => String::new(),
        };
        let rank = c.rank.map(|r| r.to_string()).unwrap_or_default();
        let status = match (&c.descriptor, c.inert) {
            (None, _) => "failed",
            (Some(_), true) => "inert",
            (Some(_), false) => "ok",
        };
        // Preset names never contain commas or quotes (checked in `world::tests`).
        let preset_name = presets.get(c.preset).copied().unwrap_or("custom");
        let _ = write!(
            text,
            "{},{},{origin},{parent},{},{},{preset_name},{rank},{:.6},{status},{:.3}",
            c.index,
            c.round,
            c.seed,
            c.preset + 1,
            c.novelty,
            c.secs
        );
        for d in 0..dims.len() {
            match c.descriptor.as_ref().and_then(|v| v.get(d)) {
                Some(v) => {
                    let _ = write!(text, ",{v:.6}");
                }
                None => text.push(','),
            }
        }
        text.push('\n');
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// An exploration's names resolved and its inputs checked, before any GPU work.
struct Plan {
    /// Index into `WORLDS`.
    world: usize,
    /// Metric lane of a `max:`/`min:` selection.
    lane: Option<usize>,
    /// The recipe (edited by `--set`) or the preset the search starts from.
    source: Source,
    /// Seed of the base candidate: the recipe's own, or the master seed.
    base_seed: u64,
}

/// Resolves and checks `job` without a GPU: world, preset and metric names,
/// the recipe and its `--set` edits, size and pacing, and that the output (and
/// library) folders can be written.
fn plan(job: &ExploreJob) -> Result<Plan> {
    frame_deadline(Instant::now(), job.frames.max(1), job.max_fps).map_err(|e| Failure::Usage.tag(e))?;
    headless::check_size(job.size)?;
    let base_seed = job.recipe.as_ref().map_or(job.seed, |r| r.saved.seed);
    let source = Source::resolve(job.recipe.as_ref(), &job.world, job.preset.as_deref(), base_seed, &job.sets)?;
    let world = source.world();
    let entry = &WORLDS[world];
    if entry.metrics.is_empty() {
        return Err(Failure::Usage.error(format!("{} does not publish measurements", entry.name)));
    }
    let lane = match &job.select {
        Select::Novelty => None,
        Select::Max(id) | Select::Min(id) => Some(world::resolve_metric(world, id)?),
    };
    headless::check_writable_dir(&job.out_dir)?;
    if let Some(library) = &job.library {
        headless::check_writable_dir(library)?;
    }
    Ok(Plan { world, lane, source, base_seed })
}

pub fn explore(job: &ExploreJob) -> Result<Summary> {
    plan(job)?;
    let gpu = headless::open_gpu()?;
    let summary = explore_with(&gpu, job)?;
    log::info!(
        "kept {} of {} candidates: {} images and {} recipes in {}, {}{}",
        summary.kept.len(),
        summary.candidates.len(),
        summary.images.len(),
        summary.recipes.len(),
        job.out_dir.display(),
        summary.csv.display(),
        summary.sheet.as_ref().map(|s| format!(" and {}", s.display())).unwrap_or_default()
    );
    Ok(summary)
}

pub fn explore_with(gpu: &Gpu, job: &ExploreJob) -> Result<Summary> {
    let Plan { world: world_index, lane, source, base_seed } = plan(job)?;
    let size = job.size;
    let max = gpu.device.limits().max_texture_dimension_2d;
    if size[0] > max || size[1] > max {
        return Err(Failure::Usage.error(format!(
            "{}x{} exceeds this GPU's maximum texture size of {max}",
            size[0], size[1]
        )));
    }
    let keep = job.keep.max(1);
    let world = source.create(gpu, size, base_seed, &job.sets)?;
    let entry = &WORLDS[world_index];
    let world_name = entry.name;
    let presets = world.presets();
    let base_preset = world.preset();
    let metrics = world.metrics();
    let dims = dim_names(metrics);
    let metric_count = metrics.len();
    let vital: Vec<usize> =
        entry.vital.iter().filter_map(|id| metrics.iter().position(|m| m.id == *id)).collect();
    std::fs::create_dir_all(&job.out_dir).with_context(|| format!("creating {}", job.out_dir.display()))?;
    let started = Instant::now();
    let total = 1 + job.runs + job.refine * job.children;
    let frames = job.frames.max(1);
    let base_origin = if job.recipe.is_some() || !job.sets.is_empty() { Origin::Recipe } else { Origin::Preset };
    let recipe_path = job.recipe.as_ref().map(|r| format!(" from {}", r.path.display())).unwrap_or_default();
    let base = format!(
        "{}{recipe_path}{}",
        presets.get(base_preset).copied().unwrap_or("custom"),
        recipe::changed(&job.sets)
    );
    log::info!(
        "exploring {world_name} / {base} at {}x{}: {} mutations, {} rounds of {} children, {} frames each, keeping {keep} by {}",
        size[0],
        size[1],
        job.runs,
        job.refine,
        job.children,
        frames,
        job.select
    );

    let mut bench = Bench::new(gpu, world, size, frames, job.max_fps);
    let mut rng = Rng::new(job.seed ^ SEED_SALT);
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut all = Vec::new();
    let mut evaluate = |bench: &mut Bench, candidates: &mut Vec<Candidate>, round: u32, origin: Origin, seed: u64| -> Result<()> {
        let index = candidates.len();
        let clock = Instant::now();
        let settings = bench.world.settings()?;
        let preset = bench.world.preset();
        let look = bench.world.post_settings();
        let (samples, pixels) = bench.run(job.all)?;
        let descriptor = descriptor(&samples, metric_count);
        let inert = descriptor.as_ref().is_some_and(|d| vital.iter().any(|&lane| d[lane * 3] < INERT));
        if let Some(pixels) = pixels {
            let path = job.out_dir.join("all").join(format!("{index:03}-r{round}-seed{seed}.png"));
            let saved = SavedWorld {
                version: crate::library::RECIPE_VERSION,
                name: format!("{world_name} explore candidate {index} (seed {seed})"),
                seed,
                output_size: size,
                preset,
                modified: true,
                settings: settings.clone(),
                look,
                camera: Camera::default(),
            };
            capture::write_png(&path, size, &pixels, &image_provenance(&saved, gpu, &path, frames))?;
            all.push(path);
        }
        let secs = clock.elapsed().as_secs_f32();
        log::info!(
            "[{:>3}/{total}] round {round} · seed {seed} ({origin}) · {secs:.1}s{}",
            index + 1,
            match (&descriptor, inert) {
                (None, _) => " · no usable measurements",
                (Some(_), true) => " · inert",
                (Some(_), false) => "",
            }
        );
        candidates.push(Candidate {
            index,
            round,
            origin,
            seed,
            preset,
            settings,
            look,
            descriptor,
            inert,
            novelty: 0.0,
            rank: None,
            secs,
        });
        Ok(())
    };
    // Round 0: the base preset (or recipe), then fresh mutations.
    evaluate(&mut bench, &mut candidates, 0, base_origin, base_seed)?;
    for _ in 0..job.runs {
        let seed = rng.next_seed();
        match bench.mutate(seed) {
            Ok(()) => evaluate(&mut bench, &mut candidates, 0, Origin::Mutation, seed)?,
            Err(e) => log::warn!("mutation with seed {seed} skipped: {e:#}"),
        }
    }

    let usable = |candidates: &[Candidate]| -> Vec<bool> { candidates.iter().map(|c| job.inert || !c.inert).collect() };

    // Refinement: perturb the recipes of the current archive.
    for round in 1..=job.refine {
        let analysis = analyse(&candidates, &usable(&candidates));
        let archive = choose(&candidates, &analysis, &job.select, lane, keep);
        if archive.is_empty() {
            log::warn!("round {round}: no usable candidates to refine");
            break;
        }
        log::info!(
            "round {round} archive: {}",
            archive.iter().map(|&i| format!("#{i} (seed {})", candidates[i].seed)).collect::<Vec<_>>().join(", ")
        );
        for _ in 0..job.children {
            let parent = archive[rng.below(archive.len() as u32) as usize];
            let mut strength = job.strength.max(0.0);
            let mut placed = false;
            for _ in 0..CHILD_ATTEMPTS {
                let child = match perturbed(&candidates[parent].settings, &mut rng, strength) {
                    Ok(child) => child,
                    Err(e) => {
                        log::debug!("perturbation of #{parent} rejected: {e:#}");
                        strength *= 0.6;
                        continue;
                    }
                };
                match bench.restore(&child, candidates[parent].seed) {
                    Ok(()) => {
                        let seed = candidates[parent].seed;
                        evaluate(&mut bench, &mut candidates, round, Origin::Child { parent }, seed)?;
                        placed = true;
                        break;
                    }
                    Err(e) => {
                        log::debug!("perturbation of #{parent} rejected: {e:#}");
                        strength *= 0.6;
                    }
                }
            }
            if !placed {
                log::warn!("round {round}: no valid perturbation of #{parent} after {CHILD_ATTEMPTS} attempts");
            }
        }
    }

    // Final archive and outputs.
    let analysis = analyse(&candidates, &usable(&candidates));
    for (c, novelty) in candidates.iter_mut().zip(&analysis.novelty) {
        c.novelty = *novelty;
    }
    let kept = choose(&candidates, &analysis, &job.select, lane, keep);
    if kept.is_empty() {
        bail!("no candidate produced usable measurements (--keep-inert admits dead and frozen ones)");
    }
    let inert_count = candidates.iter().filter(|c| c.inert).count();
    if inert_count > 0 && !job.inert {
        log::info!("{inert_count} inert candidates (dead, empty or frozen) were left out");
    }
    if kept.len() < keep {
        log::info!("only {} distinct candidates to keep", kept.len());
    }
    for (rank, &i) in kept.iter().enumerate() {
        candidates[i].rank = Some(rank + 1);
    }

    let mut images = Vec::new();
    let mut recipes = Vec::new();
    let mut installed = Vec::new();
    let recipe_dir = job.out_dir.join("recipes");
    let mut library = job.library.clone().map(Library::open);
    for (rank, &i) in kept.iter().enumerate() {
        let rank = rank + 1;
        let candidate = &candidates[i];
        let stem = file_stem(rank, candidate.seed);
        // Re-simulating from the recipe both renders the image and proves the recipe round-trips.
        bench.restore(&candidate.settings, candidate.seed).with_context(|| format!("restoring the recipe of #{i}"))?;
        let (_, pixels) = bench.run(true)?;
        let image = job.out_dir.join(format!("{stem}.png"));
        let saved = recipe(world_name, candidate, rank, size);
        capture::write_png(&image, size, &pixels.expect("captured"), &image_provenance(&saved, gpu, &image, frames))?;
        let path = recipe_dir.join(format!("{stem}.json"));
        recipe::write(&path, &saved)?;
        if let Some(library) = &mut library {
            installed.push(library.save_as_new(saved).with_context(|| format!("installing {}", path.display()))?);
        }
        log::info!(
            "#{rank:02} = candidate {i} ({}), seed {}, novelty {:.3} -> {}",
            candidate.origin,
            candidate.seed,
            candidate.novelty,
            image.display()
        );
        images.push(image);
        recipes.push(path);
    }
    if let Some(library) = &library {
        log::info!("installed {} recipes into {}", kept.len(), library.directory.display());
    }
    let csv = job.out_dir.join("candidates.csv");
    write_csv(&csv, &dims, &candidates, presets)?;
    let sheet = if job.sheet {
        let tiles: Vec<(&str, PathBuf)> = images.iter().map(|p| ("kept", p.clone())).collect();
        let path = job.out_dir.join("contact-sheet.png");
        contact_sheet(&tiles, &path, kept.len().clamp(1, 4), 2)?;
        Some(path)
    } else {
        None
    };
    let secs = started.elapsed().as_secs_f32();
    log::info!("explore done in {secs:.1}s");
    Ok(Summary {
        world: world_index,
        preset: base_preset,
        candidates,
        kept,
        images,
        recipes,
        installed,
        all,
        sheet,
        csv,
        secs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MAX_SERIES;
    use serde_json::json;

    fn sample(frame: u64, values: &[f32]) -> Sample {
        let mut all = [[0.0; MAX_METRICS]; MAX_SERIES];
        for (lane, v) in all[0].iter_mut().zip(values) {
            *lane = *v;
        }
        // Series 1 carries garbage that must never leak into a descriptor.
        all[1] = [f32::NAN; MAX_METRICS];
        Sample { frame, time: frame as f32 / 60.0, series: 2, values: all }
    }

    #[test]
    fn select_parses_novelty_and_metric_extremes() {
        assert_eq!("novelty".parse::<Select>(), Ok(Select::Novelty));
        assert_eq!(" Novelty ".parse::<Select>(), Ok(Select::Novelty));
        assert_eq!("max:growth_cover".parse::<Select>(), Ok(Select::Max("growth_cover".into())));
        assert_eq!("min:alive".parse::<Select>(), Ok(Select::Min("alive".into())));
        assert_eq!(Select::Max("veins".into()).to_string(), "max:veins");
        for invalid in ["max:", "best", "max:Bad Id", "top:alive", ""] {
            assert!(invalid.parse::<Select>().is_err(), "{invalid:?} should be rejected");
        }
    }

    #[test]
    fn descriptor_uses_late_mean_std_and_drift_of_series_zero() {
        let samples: Vec<Sample> = (0..10).map(|f| sample(f, &[f as f32, 0.25])).collect();
        let d = descriptor(&samples, 2).unwrap();
        assert_eq!(d.len(), 6);
        assert!((d[0] - 7.5).abs() < 1e-6, "late mean {}", d[0]);
        assert!((d[1] - 1.25f32.sqrt()).abs() < 1e-6, "late std {}", d[1]);
        assert!((d[2] - 7.0).abs() < 1e-6, "drift {}", d[2]);
        assert_eq!(&d[3..], &[0.25, 0.0, 0.0]);

        assert!(descriptor(&samples[..1], 2).is_none(), "one sample is not a trace");
        let mut broken = samples.clone();
        broken[7].values[0][1] = f32::NAN;
        assert!(descriptor(&broken, 2).is_none(), "non-finite measurements fail the candidate");
        assert!(descriptor(&samples, 0).unwrap().is_empty());
    }

    #[test]
    fn robust_scaling_drops_constant_dims_and_ignores_failed_and_extreme_rows() {
        let rows: Vec<Option<Vec<f32>>> = vec![
            Some(vec![1.0, 5.0, 0.0]),
            Some(vec![2.0, 5.0, 1.0]),
            None,
            Some(vec![3.0, 5.0, 2.0]),
            Some(vec![4.0, 5.0, 3.0]),
            Some(vec![5000.0, 5.0, 4.0]),
        ];
        let (scaled, active) = scale_robust(&rows);
        assert_eq!(active, 2, "the constant dimension is inactive");
        assert!(scaled[2].is_none());
        for row in scaled.iter().flatten() {
            assert_eq!(row[1], 0.0);
        }
        // The outlier is bounded and the others keep their spread.
        assert_eq!(scaled[5].as_ref().unwrap()[0], SCALED_LIMIT);
        let ordinary: Vec<f32> = [0, 1, 3, 4].iter().map(|&i| scaled[i].as_ref().unwrap()[0]).collect();
        assert!(ordinary.iter().all(|v| v.abs() <= 2.0), "{ordinary:?}");
        assert!(ordinary[0] < ordinary[1] && ordinary[1] < ordinary[2] && ordinary[2] < ordinary[3]);
        assert_eq!(scale_robust(&[]).1, 0);
    }

    #[test]
    fn novelty_is_the_mean_distance_to_the_nearest_candidates() {
        let rows: Vec<Option<Vec<f32>>> = vec![Some(vec![0.0]), Some(vec![1.0]), Some(vec![10.0]), None];
        let dist = distances(&rows, 1);
        assert_eq!(dist[0][2], 10.0);
        assert!(dist[0][3].is_infinite());
        let ok = [true, true, true, false];
        let novelty = novelty(&dist, &ok, 2);
        assert_eq!(novelty, vec![5.5, 5.0, 9.5, 0.0]);
        assert_eq!(super::novelty(&dist, &ok, 0), vec![0.0; 4]);
    }

    #[test]
    fn farthest_point_selection_takes_one_member_from_each_cluster() {
        let mut rng = Rng::new(7);
        let centres = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
        let mut rows: Vec<Option<Vec<f32>>> = Vec::new();
        for (cx, cy) in centres {
            for _ in 0..5 {
                rows.push(Some(vec![cx + rng.range(-0.01, 0.01), cy + rng.range(-0.01, 0.01)]));
            }
        }
        let (scaled, active) = scale_robust(&rows);
        let dist = distances(&scaled, active);
        let ok = vec![true; rows.len()];
        let novelty = novelty(&dist, &ok, 5);
        let picked = select_novel(&dist, &novelty, &ok, 3);
        assert_eq!(picked.len(), 3);
        let clusters: std::collections::BTreeSet<usize> = picked.iter().map(|i| i / 5).collect();
        assert_eq!(clusters.len(), 3, "{picked:?}");
        assert_eq!(picked[0], (0..15).max_by(|&a, &b| novelty[a].total_cmp(&novelty[b])).unwrap());
        assert_eq!(select_novel(&dist, &novelty, &ok, 4).len(), 4);
        assert_eq!(select_novel(&dist, &novelty, &ok, 100).len(), 15, "keep beyond the pool keeps everything distinct");

        // Duplicates are skipped and failed candidates are never selected.
        let rows: Vec<Option<Vec<f32>>> = vec![Some(vec![1.0]); 5].into_iter().chain([Some(vec![9.0]), None]).collect();
        let (scaled, active) = scale_robust(&rows);
        let dist = distances(&scaled, active);
        let ok: Vec<bool> = scaled.iter().map(Option::is_some).collect();
        let novelty = super::novelty(&dist, &ok, 3);
        assert_eq!(select_novel(&dist, &novelty, &ok, 6).len(), 2);
        assert!(select_novel(&dist, &novelty, &[false; 7], 6).is_empty());
    }

    #[test]
    fn extreme_selection_sorts_by_score_and_skips_duplicates() {
        let rows: Vec<Option<Vec<f32>>> =
            vec![Some(vec![3.0]), Some(vec![1.0]), None, Some(vec![3.0]), Some(vec![2.0])];
        let dist = distances(&rows, 1);
        let score = vec![Some(3.0), Some(1.0), None, Some(3.0), Some(2.0)];
        assert_eq!(select_extreme(&score, &dist, true, 2), vec![0, 4]);
        assert_eq!(select_extreme(&score, &dist, false, 2), vec![1, 4]);
        assert_eq!(select_extreme(&score, &dist, true, 10), vec![0, 4, 1]);
    }

    #[test]
    fn perturbation_is_deterministic_and_touches_only_scalar_numbers() {
        let original = json!({
            "color": [255, 0, 10],
            "n": 100,
            "k": 3,
            "z": 0,
            "f": 0.5,
            "negative": -0.25,
            "s": "Orbium",
            "b": true,
            "t": null,
            "post": { "exposure": 1.0, "bloom": 0.6 },
            "nested": { "list": [0.5, 1.5], "count": 40 }
        });
        let mut changed_float = false;
        let mut changed_int = false;
        for seed in 0..200u64 {
            let mut a = original.clone();
            perturb(&mut a, &mut Rng::new(seed), 1.0);
            let mut b = original.clone();
            perturb(&mut b, &mut Rng::new(seed), 1.0);
            assert_eq!(a, b, "the same seed must give the same perturbation");
            assert_eq!(a["color"], original["color"], "integer arrays stay untouched");
            assert_eq!(a["s"], original["s"]);
            assert_eq!(a["b"], original["b"]);
            assert_eq!(a["t"], original["t"]);
            assert_eq!(a["post"], original["post"], "the look is not perturbed");
            assert!(a["n"].is_u64() && a["n"].as_u64().unwrap() >= 1);
            assert!(a["nested"]["count"].is_u64() && a["nested"]["count"].as_u64().unwrap() >= 1);
            assert!([2, 3, 4].contains(&a["k"].as_u64().unwrap()));
            assert!([0, 1].contains(&a["z"].as_u64().unwrap()));
            let f = a["f"].as_f64().unwrap();
            assert!(a["f"].is_f64() && f.is_finite() && f > 0.0);
            assert!(a["negative"].as_f64().unwrap() < 0.0, "multiplicative noise keeps the sign");
            assert!(a["nested"]["list"].as_array().unwrap().iter().all(|v| v.is_f64()));
            changed_float |= f != 0.5;
            changed_int |= a["n"] != original["n"];
        }
        assert!(changed_float && changed_int);
    }

    #[test]
    fn perturbed_settings_round_trip_through_the_recipe_format() {
        let text = std::fs::read_to_string("tests/fixtures/reaction-diffusion.json").unwrap();
        let saved: SavedWorld = serde_json::from_str(&text).unwrap();
        let mut rng = Rng::new(3);
        for _ in 0..20 {
            let child = perturbed(&saved.settings, &mut rng, 0.15).unwrap();
            let WorldSettings::ReactionDiffusion { params, palette, .. } = &child else { panic!("world changed") };
            let WorldSettings::ReactionDiffusion { params: base, palette: base_palette, .. } = &saved.settings else {
                unreachable!()
            };
            assert_eq!(palette, base_palette);
            assert!((1..=128).contains(&params.steps_per_frame));
            assert!(params.feed.is_finite() && params.feed >= 0.0);
            assert!((params.scale / base.scale).ln().abs() < 1.5);
        }
    }

    #[test]
    fn explore_keeps_the_most_novel_symbiosis_candidates_and_writes_loadable_recipes() {
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let dir = tempfile::tempdir().unwrap();
        let job = ExploreJob {
            seed: 42,
            runs: 3,
            refine: 1,
            children: 2,
            keep: 2,
            frames: 10,
            size: [96, 64],
            max_fps: 0.0,
            out_dir: dir.path().join("out"),
            library: Some(dir.path().join("lib")),
            all: true,
            ..ExploreJob::new("symbiosis")
        };
        let summary = explore_with(&gpu, &job).unwrap();
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
        assert_eq!(summary.candidates.len(), 6, "preset + 3 mutations + 2 children");
        assert_eq!(summary.kept.len(), 2);
        let ranks: Vec<usize> = summary.candidates.iter().filter_map(|c| c.rank).collect();
        assert_eq!(ranks.iter().sum::<usize>(), 3, "ranks 1 and 2 were assigned: {ranks:?}");
        assert!(summary.candidates.iter().all(|c| c.descriptor.is_some()));
        assert!(summary.candidates.iter().any(|c| matches!(c.origin, Origin::Child { .. })));
        assert_eq!(summary.candidates[0].origin, Origin::Preset);

        for image in &summary.images {
            let png = image::open(image).unwrap();
            assert_eq!((png.width(), png.height()), (96, 64));
        }
        assert!(summary.sheet.as_ref().unwrap().exists());
        assert_eq!(std::fs::read_dir(job.out_dir.join("all")).unwrap().count(), 6);
        assert_eq!(summary.all.len(), 6);
        assert_eq!((WORLDS[summary.world].id, summary.preset), ("symbiosis", 0));
        let files = summary.files();
        assert_eq!(files.len(), 6 + 2 * 3 + 2, "all/, image + recipe + install per kept candidate, csv, sheet");
        assert!(files.iter().all(|f| f.exists()), "{files:?}");

        let csv = std::fs::read_to_string(&summary.csv).unwrap();
        let mut lines = csv.lines();
        let header = lines.next().unwrap();
        let expected_dims = dim_names(world::create(&gpu, "symbiosis", [96, 64], None, 1).unwrap().1.metrics());
        assert_eq!(
            header,
            format!(
                "index,round,origin,parent,seed,preset,preset_name,rank,novelty,status,secs,{}",
                expected_dims.join(",")
            )
        );
        let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
        assert_eq!(rows.len(), 6);
        // The preset column is 1-based like --preset; mutations may come from any preset.
        assert_eq!((rows[0][5], rows[0][6]), ("1", "Living Reef"));
        for row in &rows {
            assert_eq!(row.len(), 11 + expected_dims.len());
            let preset: usize = row[5].parse().unwrap();
            assert_eq!(row[6], world::symbiosis::preset_names()[preset - 1], "{row:?}");
            assert!(row[9] == "ok" || row[9] == "inert", "{row:?}");
            assert!(row[10..].iter().all(|v| v.parse::<f32>().unwrap().is_finite()), "{row:?}");
        }
        assert!(summary.kept.iter().all(|&i| !summary.candidates[i].inert), "inert candidates are never kept");
        assert_eq!(summary.recipes.len(), 2);
        assert_eq!(summary.installed.len(), 2);
        assert!(summary.installed.iter().all(|p| p.starts_with(dir.path().join("lib"))));

        let recipes = Library::open(job.out_dir.join("recipes"));
        assert!(recipes.warnings.is_empty(), "{:?}", recipes.warnings);
        assert_eq!(recipes.entries.len(), 2);
        for entry in &recipes.entries {
            assert!(entry.saved.name.contains("explore #"), "{}", entry.saved.name);
            let (index, restored) = entry.saved.instantiate(&gpu).unwrap();
            assert_eq!(WORLDS[index].id, "symbiosis");
            assert_eq!(
                serde_json::to_value(restored.settings().unwrap()).unwrap(),
                serde_json::to_value(&entry.saved.settings).unwrap()
            );
        }
        let installed = Library::open(dir.path().join("lib"));
        assert_eq!(installed.entries.len(), 2);

        // Extremes of one metric, and an unknown metric is rejected with the list of ids.
        let extreme = ExploreJob {
            select: Select::Max("growth_cover".into()),
            refine: 0,
            out_dir: dir.path().join("max"),
            library: None,
            all: false,
            ..job.clone()
        };
        let summary = explore_with(&gpu, &extreme).unwrap();
        let mut covers: Vec<(f32, usize)> =
            summary.candidates.iter().map(|c| (c.descriptor.as_ref().unwrap()[0], c.index)).collect();
        covers.sort_by(|a, b| b.0.total_cmp(&a.0));
        assert_eq!(summary.kept, vec![covers[0].1, covers[1].1]);
        let bad = ExploreJob { select: Select::Max("nope".into()), ..extreme };
        let error = match explore_with(&gpu, &bad) {
            Ok(_) => panic!("an unknown metric must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(error.contains("growth_cover"), "{error}");

        // A search can start from a kept recipe, edited by --set: it is evaluated first, from its own seed.
        let path = summary_recipe(&job);
        let from_recipe = ExploreJob {
            recipe: Some(Recipe::load(&path).unwrap()),
            sets: vec!["params.steps=3".parse().unwrap()],
            world: "ignored".into(),
            runs: 1,
            refine: 0,
            keep: 1,
            out_dir: dir.path().join("from-recipe"),
            library: None,
            all: false,
            inert: true,
            ..job.clone()
        };
        let summary = explore_with(&gpu, &from_recipe).unwrap();
        let base = &summary.candidates[0];
        let saved = crate::library::load(&path).unwrap();
        assert_eq!((base.origin, base.seed, base.preset), (Origin::Recipe, saved.seed, saved.preset));
        let WorldSettings::Symbiosis { params, .. } = &base.settings else { panic!("world changed") };
        assert_eq!(params.steps, 3, "--set applies to the base");
        let csv = std::fs::read_to_string(&summary.csv).unwrap();
        assert!(csv.lines().nth(1).unwrap().starts_with(&format!("0,0,recipe,,{},", saved.seed)), "{csv}");
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    }

    /// The first recipe `job` kept.
    fn summary_recipe(job: &ExploreJob) -> PathBuf {
        let recipes = job.out_dir.join("recipes");
        let mut files: Vec<PathBuf> = std::fs::read_dir(recipes).unwrap().map(|f| f.unwrap().path()).collect();
        files.sort();
        files.remove(0)
    }

    #[test]
    fn plans_reject_bad_names_and_sizes_before_any_gpu_work() {
        let dir = tempfile::tempdir().unwrap();
        let job = ExploreJob { out_dir: dir.path().join("out"), ..ExploreJob::new("symbiosis") };
        let select = Select::Max("growth_cover".into());
        let plan = plan(&ExploreJob { preset: Some("coral".into()), select, ..job.clone() }).unwrap();
        assert_eq!((WORLDS[plan.world].id, plan.source.preset(), plan.lane), ("symbiosis", Some(2), Some(0)));
        assert!(job.out_dir.is_dir(), "the output folder is created and probed up front");
        let cases = [
            (ExploreJob { world: "symbiosys".into(), ..job.clone() }, "did you mean 'symbiosis'?"),
            (ExploreJob { preset: Some("9".into()), ..job.clone() }, "Symbiosis has presets 1-6"),
            (ExploreJob { select: Select::Min("growth_cove".into()), ..job.clone() }, "did you mean 'growth_cover'?"),
            (ExploreJob { size: [8, 64], ..job.clone() }, "8x64 is not a supported size"),
            (ExploreJob { size: [640, 20000], ..job.clone() }, "each side must be 16-16384 pixels"),
        ];
        for (bad, expected) in cases {
            let error = super::plan(&bad).err().expect("the plan must fail");
            assert!(format!("{error:#}").contains(expected), "{error:#}");
            assert_eq!(crate::failure::exit_code(&error), 2, "{error:#}");
        }

        // A recipe decides the world and the base seed; its --set edits are checked here too.
        let recipe = Recipe::load(Path::new("tests/fixtures/reaction-diffusion.json")).unwrap();
        let from_recipe = ExploreJob { recipe: Some(recipe), world: "symbiosis".into(), ..job.clone() };
        let plan = super::plan(&from_recipe).unwrap();
        let base = (WORLDS[plan.world].id, plan.base_seed, plan.source.preset());
        assert_eq!(base, ("reaction-diffusion", u64::MAX, Some(0)));
        let bad = ExploreJob { sets: vec!["params.fed=0.03".parse().unwrap()], ..from_recipe };
        let error = super::plan(&bad).err().expect("an unknown setting");
        assert!(error.to_string().contains("did you mean 'params.feed'?"), "{error}");
        assert_eq!(crate::failure::exit_code(&error), 2);
    }
}
