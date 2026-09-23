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
//! edits either before the first candidate is evaluated. Refinement never
//! perturbs a world's appearance settings (`WorldEntry::appearance`).
//!
//! Each run writes into a folder of its own (`explore/<world>-<preset>-s<seed>`
//! by default) and leaves `run.json` there ([`MANIFEST`]): what ran, what every
//! column of `candidates.csv` holds, and every file it wrote. The next run into
//! the folder removes exactly those files first, so runs never mix; outputs
//! that no manifest lists make it refuse the folder unless `--overwrite`.

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
use crate::headless::{self, contact_sheet, frame_deadline, Tile, OUTPUT_FORMAT};
use crate::library::{Library, SavedWorld, WorldSettings};
use crate::metrics::{MetricDesc, Sample, Sampler, MAX_METRICS};
use crate::post::{Post, PostSettings};
use crate::recipe::{self, Recipe, Setting, Source};
use crate::report;
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
/// Frames per simulated second of every candidate: each frame advances 1/60 s.
const FPS: u32 = 60;

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
    /// `None` = `explore/<world>-<preset>-s<seed>` ([`default_out_dir`]).
    pub out_dir: Option<PathBuf>,
    /// Remove explore outputs in `out_dir` that no `run.json` lists instead of
    /// refusing the folder (a previous run's listed outputs are always removed).
    pub overwrite: bool,
    pub select: Select,
    /// Also save the kept recipes into this library folder.
    pub library: Option<PathBuf>,
    pub sheet: bool,
    /// Also write every evaluated candidate's final frame under `all/`.
    pub all: bool,
    /// Let inert candidates (dead, empty or frozen worlds) into the archive.
    pub inert: bool,
    /// The command line, recorded in `run.json` (empty when not run from one).
    pub argv: Vec<String>,
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
            out_dir: None,
            overwrite: false,
            select: Select::Novelty,
            library: None,
            sheet: true,
            all: false,
            inert: false,
            argv: Vec::new(),
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

impl Candidate {
    /// The `status` column of candidates.csv: `ok`, `inert` or `failed`.
    pub fn status(&self) -> &'static str {
        match (&self.descriptor, self.inert) {
            (None, _) => "failed",
            (Some(_), true) => "inert",
            (Some(_), false) => "ok",
        }
    }
}

#[derive(Debug)]
pub struct Summary {
    /// Index into `WORLDS`.
    pub world: usize,
    /// 0-based preset the search started from.
    pub preset: usize,
    /// Seed of the base candidate (the recipe's own, or the master seed).
    pub base_seed: u64,
    pub candidates: Vec<Candidate>,
    /// Indices of the kept candidates in rank order.
    pub kept: Vec<usize>,
    /// The folder every output but the library files went into.
    pub out_dir: PathBuf,
    /// Image and recipe of each kept candidate, in rank order.
    pub images: Vec<PathBuf>,
    pub recipes: Vec<PathBuf>,
    /// Library files written by `--install`, in rank order.
    pub installed: Vec<PathBuf>,
    /// Every candidate's final frame under `all/` (with `--all`).
    pub all: Vec<PathBuf>,
    pub sheet: Option<PathBuf>,
    pub csv: PathBuf,
    /// `run.json`, written last.
    pub manifest: PathBuf,
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
        files.push(&self.manifest);
        files
    }

    /// The files the manifest lists: every file written into `out_dir` before
    /// it. The library's never are, wherever the library is.
    fn outputs(&self) -> Vec<&Path> {
        let library = |file: &&Path| self.installed.iter().any(|i| i == file);
        self.files().into_iter().filter(|f| !library(f) && *f != self.manifest).collect()
    }
}

// --- behaviour descriptors -----------------------------------------------------

/// Column names of a descriptor: three per metric ([`DESCRIPTOR_STATS`]).
pub(crate) fn dim_names(metrics: &[MetricDesc]) -> Vec<String> {
    metrics.iter().flat_map(|m| DESCRIPTOR_STATS.map(|(stat, _)| format!("{}_{stat}", m.id))).collect()
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

/// Nudges the numbers of a recipe's settings. Floats are multiplied by
/// log-normal noise (each with probability one half), so a float at zero stays
/// at zero; integers above 32 likewise, and smaller ones move by one, never
/// below zero (so a zero can become one). Integers directly inside lists,
/// booleans, text (and so named variants), nulls and every setting in `frozen`
/// (dotted paths, `*` for every item of a list) are left alone.
pub(crate) fn perturb(value: &mut Value, rng: &mut Rng, strength: f32, frozen: &[&str]) {
    let frozen: Vec<Vec<&str>> = frozen.iter().map(|path| path.split('.').collect()).collect();
    perturb_inner(value, rng, strength, &mut Vec::new(), &frozen, false);
}

/// Whether the setting at `path` is one of `frozen`.
fn is_frozen(path: &[String], frozen: &[Vec<&str>]) -> bool {
    frozen.iter().any(|f| f.len() == path.len() && f.iter().zip(path).all(|(f, p)| *f == "*" || f == p))
}

fn perturb_inner(
    value: &mut Value,
    rng: &mut Rng,
    strength: f32,
    path: &mut Vec<String>,
    frozen: &[Vec<&str>],
    in_array: bool,
) {
    match value {
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                path.push(key.clone());
                if !is_frozen(path, frozen) {
                    perturb_inner(child, rng, strength, path, frozen, false);
                }
                path.pop();
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter_mut().enumerate() {
                path.push(index.to_string());
                if !is_frozen(path, frozen) {
                    perturb_inner(child, rng, strength, path, frozen, true);
                }
                path.pop();
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

/// A perturbed copy of `settings`, as a recipe the world can validate. The
/// world's appearance settings ([`world::WorldEntry::appearance`]) keep their values.
pub(crate) fn perturbed(settings: &WorldSettings, rng: &mut Rng, strength: f32) -> Result<WorldSettings> {
    let mut value = serde_json::to_value(settings).context("serialising the recipe")?;
    let appearance = world::find(settings.world_id()).map_or(&[][..], |w| WORLDS[w].appearance);
    perturb(&mut value, rng, strength, appearance);
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
        let dt = 1.0 / FPS as f32;
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

/// Indices of the archive for `select`, in rank order. Only the candidates
/// `analysis` marks usable (measured, and alive unless `--keep-inert`) take part.
fn choose(candidates: &[Candidate], analysis: &Analysis, select: &Select, lane: Option<usize>, keep: usize) -> Vec<usize> {
    match (select, lane) {
        (Select::Novelty, _) => select_novel(&analysis.dist, &analysis.novelty, &analysis.ok, keep),
        (Select::Max(_), Some(lane)) | (Select::Min(_), Some(lane)) => {
            let scores: Vec<Option<f32>> = candidates
                .iter()
                .zip(&analysis.ok)
                .map(|(c, ok)| c.descriptor.as_ref().filter(|_| *ok).and_then(|d| d.get(lane * 3).copied()))
                .collect();
            select_extreme(&scores, &analysis.dist, matches!(select, Select::Max(_)), keep)
        }
        _ => Vec::new(),
    }
}

fn file_stem(rank: usize, seed: u64) -> String {
    format!("{rank:02}-seed{seed}")
}

/// The contact sheet's caption of a kept candidate: "#01 · seed 1".
fn sheet_caption(rank: usize, candidate: &Candidate) -> String {
    format!("#{rank:02} · seed {}", candidate.seed)
}

/// The recipe of kept candidate `rank`. Its name gives the size it was
/// explored at, which is the size it opens at in the app.
fn recipe(world_name: &str, candidate: &Candidate, rank: usize, size: [u32; 2]) -> SavedWorld {
    SavedWorld {
        version: crate::library::RECIPE_VERSION,
        name: format!("{world_name} explore #{rank:02} (seed {}, {}x{})", candidate.seed, size[0], size[1]),
        seed: candidate.seed,
        output_size: size,
        preset: candidate.preset,
        modified: true,
        settings: candidate.settings.clone(),
        look: candidate.look,
        camera: Camera::default(),
    }
}

/// What an explore image says about itself: its recipe, the frames it ran
/// for, and that `render --recipe <image>` renders it again.
fn image_provenance(saved: &SavedWorld, gpu: &Gpu, image: &Path, frames: u32) -> Provenance {
    let name = image.file_name().map_or_else(|| image.display().to_string(), |n| n.to_string_lossy().into_owned());
    let command = format!("primordia render --recipe {}", headless::shell_word(&name));
    Provenance::of(saved, gpu).with_command(command).with_run(frames, FPS)
}

/// The columns of `candidates.csv` before the descriptor, with what they hold
/// (`run.json` repeats them).
pub(crate) const CSV_COLUMNS: &[(&str, &str)] = &[
    ("index", "candidate number from 0, in the order the candidates were evaluated"),
    ("round", "0 for the base and the mutations, then the refinement round that made the child"),
    ("origin", "preset, recipe (a --recipe, or a preset edited by --set), mutation, or child"),
    ("parent", "a child's parent: the index of the kept candidate it perturbs, whose seed it reuses; empty otherwise"),
    ("seed", "the seed the candidate runs from"),
    ("preset", "preset number from 1, as --preset takes it (recipes store it from 0)"),
    ("preset_name", "the preset's name"),
    ("rank", "place among the kept candidates from 1, the NN of NN-seedS.png; empty when not kept"),
    ("novelty", "mean scaled distance to the (up to) five nearest other candidates; 0 for failed and left-out ones"),
    ("status", "ok, inert (a vital measurement stayed near zero) or failed (no usable measurements)"),
    ("secs", "seconds the evaluation took"),
];

/// The descriptor's three columns per measurement, `<id>_<stat>`, with what they hold.
pub(crate) const DESCRIPTOR_STATS: [(&str, &str); 3] = [
    ("mean", "mean over the last 40% of the frames"),
    ("std", "standard deviation over the last 40% of the frames"),
    ("drift", "the mean over the last 40% of the frames minus the mean over the first 20%"),
];

/// `candidates.csv`: one row per candidate, in [`CSV_COLUMNS`] then the
/// descriptor's columns ([`dim_names`]).
fn write_csv(path: &Path, dims: &[String], candidates: &[Candidate], presets: &[&str]) -> Result<()> {
    let header: Vec<&str> = CSV_COLUMNS.iter().map(|(name, _)| *name).chain(dims.iter().map(String::as_str)).collect();
    let mut text = header.join(",");
    text.push('\n');
    for c in candidates {
        let origin = c.origin.name();
        let parent = match c.origin {
            Origin::Child { parent } => parent.to_string(),
            _ => String::new(),
        };
        let rank = c.rank.map(|r| r.to_string()).unwrap_or_default();
        // Preset names never contain commas or quotes (checked in `world::tests`).
        let preset_name = presets.get(c.preset).copied().unwrap_or("custom");
        let _ = write!(
            text,
            "{},{},{origin},{parent},{},{},{preset_name},{rank},{:.6},{},{:.3}",
            c.index,
            c.round,
            c.seed,
            c.preset + 1,
            c.novelty,
            c.status(),
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

// --- the output folder ---------------------------------------------------------------

/// The manifest each run writes into its folder, last: what ran, and every
/// file it wrote there ([`crate::report::explore_manifest`]).
pub const MANIFEST: &str = "run.json";
/// The manifest's `tool`, which marks a folder as explore's.
pub(crate) const MANIFEST_TOOL: &str = "primordia explore";

/// `explore/<world>-<preset>-s<seed>`, or `explore/<world>-<recipe file
/// name>-s<seed>` for a run from `--recipe`; the seed is the master seed.
pub(crate) fn default_out_dir(job: &ExploreJob, source: &Source) -> PathBuf {
    let entry = &WORLDS[source.world()];
    let recipe = job.recipe.as_ref().and_then(|r| r.path.file_stem()).map(|s| headless::slug(&s.to_string_lossy()));
    let base = recipe.filter(|s| !s.is_empty()).unwrap_or_else(|| {
        headless::slug((entry.presets)().get(source.preset().unwrap_or(0)).copied().unwrap_or("custom"))
    });
    Path::new("explore").join(format!("{}-{base}-s{}", entry.id, job.seed))
}

/// Whether a file called `name` in the subfolder `sub` of an explore folder
/// ("" for the folder itself) is named like one of explore's outputs.
fn is_output_name(sub: &str, name: &str) -> bool {
    let digits = |s: &str, at_least: usize| s.len() >= at_least && s.bytes().all(|b| b.is_ascii_digit());
    // NN-seedS: a rank and a seed.
    let kept = |stem: &str| stem.split_once("-seed").is_some_and(|(rank, seed)| digits(rank, 2) && digits(seed, 1));
    // NNN-rR-seedS: a candidate, its round and its seed.
    let candidate = |stem: &str| {
        stem.split_once("-r").is_some_and(|(index, rest)| {
            let round_and_seed = rest.split_once("-seed");
            digits(index, 3) && round_and_seed.is_some_and(|(round, seed)| digits(round, 1) && digits(seed, 1))
        })
    };
    match sub {
        "" => name == "candidates.csv" || name == "contact-sheet.png" || name.strip_suffix(".png").is_some_and(kept),
        "recipes" => name.strip_suffix(".json").is_some_and(kept),
        "all" => name.strip_suffix(".png").is_some_and(candidate),
        _ => false,
    }
}

/// `file` relative to `dir`, with `/` between its parts, or `None` outside `dir`.
pub(crate) fn relative(dir: &Path, file: &Path) -> Option<String> {
    let parts: Vec<String> =
        file.strip_prefix(dir).ok()?.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// A manifest's `relative` path inside `dir`, or `None` when it would leave
/// `dir` (a manifest never makes a run remove anything elsewhere).
fn inside(dir: &Path, relative: &str) -> Option<PathBuf> {
    let parts: Vec<&str> = relative.split('/').collect();
    let plain = |part: &&str| !part.is_empty() && *part != "." && *part != ".." && !part.contains(['\\', ':']);
    parts.iter().all(plain).then(|| parts.iter().fold(dir.to_path_buf(), |path, part| path.join(part)))
}

/// The outputs an earlier run left in an explore folder.
#[derive(Debug, Default)]
struct Previous {
    /// The files its `run.json` lists, which a new run removes.
    listed: Vec<PathBuf>,
    /// Files named like explore's outputs that no `run.json` lists: another
    /// run's, one from before manifests, or one from a run that was killed.
    unlisted: Vec<PathBuf>,
    /// The `run.json` itself.
    manifest: Option<PathBuf>,
}

/// What an earlier run left in `dir`. A `run.json` that explore did not write
/// is invalid input: nothing in that folder is explore's to remove.
fn previous_outputs(dir: &Path) -> Result<Previous> {
    let manifest = dir.join(MANIFEST);
    let mut previous = Previous::default();
    if manifest.exists() {
        let foreign = || {
            Failure::Usage.error(format!(
                "{} was not written by primordia explore, so this folder is not explore's: choose another --out-dir",
                manifest.display()
            ))
        };
        let text = std::fs::read_to_string(&manifest).map_err(|_| foreign())?;
        let value: Value = serde_json::from_str(&text).map_err(|_| foreign())?;
        if value["tool"] != MANIFEST_TOOL {
            return Err(foreign());
        }
        let files = value["files"].as_array().into_iter().flatten().filter_map(Value::as_str);
        previous.listed = files.filter_map(|file| inside(dir, file)).filter(|path| path.is_file()).collect();
        previous.manifest = Some(manifest);
    }
    for sub in ["", "recipes", "all"] {
        let Ok(entries) = std::fs::read_dir(dir.join(sub)) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let named = is_output_name(sub, &entry.file_name().to_string_lossy());
            if named && path.is_file() && !previous.listed.contains(&path) {
                previous.unlisted.push(path);
            }
        }
    }
    previous.unlisted.sort();
    Ok(previous)
}

/// "a, b, c and 4 more": the first few of `files`, relative to `dir`.
fn listing(dir: &Path, files: &[PathBuf]) -> String {
    let names: Vec<String> =
        files.iter().take(3).map(|f| relative(dir, f).unwrap_or_else(|| f.display().to_string())).collect();
    match files.len().saturating_sub(3) {
        0 => names.join(", "),
        more => format!("{} and {more} more", names.join(", ")),
    }
}

/// Removes what an earlier run left in `dir`: the files its manifest lists and
/// the manifest, and with `overwrite` the unlisted outputs as well; then
/// `recipes/` and `all/` when they are left empty. Nothing else is touched.
/// Returns the number of files removed.
fn clear_previous(dir: &Path, previous: &Previous, overwrite: bool) -> Result<usize> {
    let unlisted = if overwrite { &previous.unlisted[..] } else { &[] };
    let mut removed = 0;
    for file in previous.listed.iter().chain(unlisted).chain(&previous.manifest) {
        match std::fs::remove_file(file) {
            Ok(()) => removed += 1,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(anyhow::Error::new(e).context(format!("removing {} of an earlier run", file.display())));
            }
        }
    }
    for sub in ["recipes", "all"] {
        // Fails, and keeps the folder, unless it is empty.
        let _ = std::fs::remove_dir(dir.join(sub));
    }
    Ok(removed)
}

/// What `run.json` records about a run besides its job ([`crate::report::explore_manifest`]).
pub(crate) struct Record<'a> {
    /// Index into `WORLDS`.
    pub world: usize,
    /// 0-based preset of the base candidate.
    pub preset: usize,
    pub base_seed: u64,
    /// When the run started, in UTC (RFC 3339).
    pub started: String,
    /// The GPU's name and backend.
    pub gpu: String,
    pub secs: f32,
    /// Every file written into the output folder, relative to it.
    pub files: Vec<String>,
    /// The finished run, or why it failed.
    pub outcome: std::result::Result<&'a Summary, String>,
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
    /// The output folder, the default filled in.
    out_dir: PathBuf,
    /// What an earlier run left there.
    previous: Previous,
}

/// Resolves and checks `job` without a GPU: world, preset and metric names,
/// the recipe and its `--set` edits, size and pacing, that the output (and
/// library) folders can be written, and that the output folder holds no
/// explore outputs that no `run.json` lists (unless `overwrite`).
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
    let out_dir = job.out_dir.clone().unwrap_or_else(|| default_out_dir(job, &source));
    headless::check_writable_dir(&out_dir)?;
    if let Some(library) = &job.library {
        headless::check_writable_dir(library)?;
    }
    let previous = previous_outputs(&out_dir)?;
    if !previous.unlisted.is_empty() && !job.overwrite {
        return Err(Failure::Usage.error(format!(
            "{} already holds explore outputs that no {MANIFEST} lists ({}): pass --overwrite to replace them, or \
             choose another --out-dir",
            out_dir.display(),
            listing(&out_dir, &previous.unlisted)
        )));
    }
    Ok(Plan { world, lane, source, base_seed, out_dir, previous })
}

pub fn explore(job: &ExploreJob) -> Result<Summary> {
    plan(job)?;
    let gpu = headless::open_gpu()?;
    let summary = explore_with(&gpu, job)?;
    let name = |path: &Path| path.file_name().unwrap_or_default().to_string_lossy().into_owned();
    log::info!(
        "kept {} of {} candidates in {}: images, recipes, {}{} and {MANIFEST}",
        summary.kept.len(),
        summary.candidates.len(),
        summary.out_dir.display(),
        name(&summary.csv),
        summary.sheet.as_deref().map(|s| format!(", {}", name(s))).unwrap_or_default()
    );
    Ok(summary)
}

/// Runs `job` on `gpu`. The output folder is cleared of the earlier run's
/// files first, and `run.json` is written last; when the run fails after
/// writing files, it is written too, marked incomplete, so that the next run
/// into the folder removes them.
pub fn explore_with(gpu: &Gpu, job: &ExploreJob) -> Result<Summary> {
    let plan = plan(job)?;
    let size = job.size;
    let max = gpu.device.limits().max_texture_dimension_2d;
    if size[0] > max || size[1] > max {
        return Err(Failure::Usage.error(format!(
            "{}x{} exceeds this GPU's maximum texture size of {max}",
            size[0], size[1]
        )));
    }
    let removed = clear_previous(&plan.out_dir, &plan.previous, job.overwrite)?;
    if removed > 0 {
        log::info!("removed {removed} files of an earlier run from {}", plan.out_dir.display());
    }
    let started = report::utc_time(std::time::SystemTime::now());
    let clock = Instant::now();
    let result = search(gpu, job, &plan);
    let written = {
        let (preset, files, outcome) = match &result {
            Ok(summary) => {
                let files = summary.outputs().into_iter().filter_map(|f| relative(&plan.out_dir, f)).collect();
                (summary.preset, files, Ok(summary))
            }
            // The folder held no outputs when the run began, so every one in it now is this run's.
            Err(e) => {
                let files = previous_outputs(&plan.out_dir).map(|p| p.unlisted).unwrap_or_default();
                let files = files.iter().filter_map(|f| relative(&plan.out_dir, f)).collect();
                (plan.source.preset().unwrap_or(0), files, Err(format!("{e:#}")))
            }
        };
        let record = Record {
            world: plan.world,
            preset,
            base_seed: plan.base_seed,
            started,
            gpu: capture::gpu_name(gpu),
            secs: clock.elapsed().as_secs_f32(),
            files,
            outcome,
        };
        if record.outcome.is_err() && record.files.is_empty() {
            Ok(())
        } else {
            recipe::write_json(&plan.out_dir.join(MANIFEST), &report::explore_manifest(job, &record))
        }
    };
    match result {
        Ok(summary) => written.map(|()| summary),
        Err(e) => {
            if let Err(problem) = written {
                log::warn!("{problem:#}");
            }
            Err(e)
        }
    }
}

/// The search itself: evaluates the candidates, chooses the archive and
/// writes every output but the manifest.
fn search(gpu: &Gpu, job: &ExploreJob, plan: &Plan) -> Result<Summary> {
    let (world_index, lane, base_seed, out_dir) = (plan.world, plan.lane, plan.base_seed, &plan.out_dir);
    let size = job.size;
    let keep = job.keep.max(1);
    let world = plan.source.create(gpu, size, base_seed, &job.sets)?;
    let entry = &WORLDS[world_index];
    let world_name = entry.name;
    let presets = world.presets();
    let base_preset = world.preset();
    let metrics = world.metrics();
    let dims = dim_names(metrics);
    let metric_count = metrics.len();
    let vital: Vec<usize> =
        entry.vital.iter().filter_map(|id| metrics.iter().position(|m| m.id == *id)).collect();
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
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
    log::info!("writing into {}", out_dir.display());

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
            let path = out_dir.join("all").join(format!("{index:03}-r{round}-seed{seed}.png"));
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
    let recipe_dir = out_dir.join("recipes");
    let mut library = job.library.clone().map(Library::open);
    for (rank, &i) in kept.iter().enumerate() {
        let rank = rank + 1;
        let candidate = &candidates[i];
        let stem = file_stem(rank, candidate.seed);
        // Re-simulating from the recipe both renders the image and proves the recipe round-trips.
        bench.restore(&candidate.settings, candidate.seed).with_context(|| format!("restoring the recipe of #{i}"))?;
        let (_, pixels) = bench.run(true)?;
        let image = out_dir.join(format!("{stem}.png"));
        let saved = recipe(world_name, candidate, rank, size);
        capture::write_png(&image, size, &pixels.expect("captured"), &image_provenance(&saved, gpu, &image, frames))?;
        let path = recipe_dir.join(format!("{stem}.json"));
        // Builds that do not know `provenance` ignore it; the library's copy (--install) goes without it.
        let mut value = serde_json::to_value(&saved).context("serialising the recipe")?;
        value["provenance"] = report::explore_provenance(job, &candidates, i);
        recipe::write_json(&path, &value)?;
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
        log::info!(
            "installed {} recipes into {}; they open at the {}x{} they were explored at (explore with --width and \
             --height for larger ones, or render one larger with render --recipe)",
            kept.len(),
            library.directory.display(),
            size[0],
            size[1]
        );
    }
    let csv = out_dir.join("candidates.csv");
    write_csv(&csv, &dims, &candidates, presets)?;
    let sheet = if job.sheet {
        let tiles: Vec<Tile> = kept
            .iter()
            .zip(&images)
            .enumerate()
            .map(|(rank, (&i, path))| Tile { group: "kept", path, caption: sheet_caption(rank + 1, &candidates[i]) })
            .collect();
        let path = out_dir.join("contact-sheet.png");
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
        base_seed,
        candidates,
        kept,
        out_dir: out_dir.clone(),
        images,
        recipes,
        installed,
        all,
        sheet,
        csv,
        manifest: out_dir.join(MANIFEST),
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

    /// A Reaction-Diffusion mutation whose descriptor starts with `alive` (`None`: it failed).
    fn candidate(index: usize, alive: Option<f32>, inert: bool) -> Candidate {
        let text = std::fs::read_to_string("tests/fixtures/reaction-diffusion.json").unwrap();
        let saved: SavedWorld = serde_json::from_str(&text).unwrap();
        Candidate {
            index,
            round: 0,
            origin: Origin::Mutation,
            seed: index as u64,
            preset: 0,
            settings: saved.settings,
            look: saved.look,
            descriptor: alive.map(|a| vec![a, 0.0, 0.0, index as f32, 0.0, 0.0]),
            inert,
            novelty: 0.0,
            rank: None,
            secs: 0.0,
        }
    }

    #[test]
    fn metric_extremes_leave_out_inert_and_failed_candidates() {
        let candidates = vec![
            candidate(0, Some(0.5), false),
            candidate(1, Some(0.0), true),
            candidate(2, Some(0.9), false),
            candidate(3, None, false),
        ];
        for (inert_allowed, smallest) in [(false, vec![0, 2]), (true, vec![1, 0, 2])] {
            let usable: Vec<bool> = candidates.iter().map(|c| inert_allowed || !c.inert).collect();
            let analysis = analyse(&candidates, &usable);
            let min = choose(&candidates, &analysis, &Select::Min("alive".into()), Some(0), 3);
            assert_eq!(min, smallest, "--keep-inert {inert_allowed}");
            let max = choose(&candidates, &analysis, &Select::Max("alive".into()), Some(0), 3);
            assert_eq!(max[0], 2);
            assert!(!max.contains(&3), "a failed candidate is never kept");
        }
    }

    #[test]
    fn perturbation_is_deterministic_and_touches_only_scalar_numbers() {
        let original = json!({
            "color": [255, 0, 10],
            "n": 100,
            "k": 3,
            "z": 0,
            "f": 0.5,
            "off": 0.0,
            "negative": -0.25,
            "s": "Orbium",
            "b": true,
            "t": null,
            "post": { "exposure": 1.0, "bloom": 0.6 },
            "nested": { "list": [0.5, 1.5], "count": 40, "tint": 0.3 },
            "species": [{ "speed": 1.0, "tint": 0.5 }, { "speed": 2.0, "tint": 0.25 }]
        });
        let frozen = ["post", "nested.tint", "species.*.tint"];
        let mut changed_float = false;
        let mut changed_int = false;
        let mut changed_in_list = false;
        for seed in 0..200u64 {
            let mut a = original.clone();
            perturb(&mut a, &mut Rng::new(seed), 1.0, &frozen);
            let mut b = original.clone();
            perturb(&mut b, &mut Rng::new(seed), 1.0, &frozen);
            assert_eq!(a, b, "the same seed must give the same perturbation");
            assert_eq!(a["color"], original["color"], "integer arrays stay untouched");
            assert_eq!(a["s"], original["s"]);
            assert_eq!(a["b"], original["b"]);
            assert_eq!(a["t"], original["t"]);
            assert_eq!(a["post"], original["post"], "the look is not perturbed");
            assert_eq!(a["nested"]["tint"], original["nested"]["tint"], "frozen settings keep their value");
            for (item, before) in a["species"].as_array().unwrap().iter().zip(original["species"].as_array().unwrap()) {
                assert_eq!(item["tint"], before["tint"], "`*` freezes the setting in every item");
                changed_in_list |= item["speed"] != before["speed"];
            }
            assert_eq!(a["off"], 0.0, "a float at zero stays at zero");
            assert!(a["n"].is_u64() && a["n"].as_u64().unwrap() >= 1);
            assert!(a["nested"]["count"].is_u64() && a["nested"]["count"].as_u64().unwrap() >= 1);
            assert!([2, 3, 4].contains(&a["k"].as_u64().unwrap()));
            assert!([0, 1].contains(&a["z"].as_u64().unwrap()), "a small integer at zero can become one");
            let f = a["f"].as_f64().unwrap();
            assert!(a["f"].is_f64() && f.is_finite() && f > 0.0);
            assert!(a["negative"].as_f64().unwrap() < 0.0, "multiplicative noise keeps the sign");
            assert!(a["nested"]["list"].as_array().unwrap().iter().all(|v| v.is_f64()));
            changed_float |= f != 0.5;
            changed_int |= a["n"] != original["n"];
        }
        assert!(changed_float && changed_int && changed_in_list);
    }

    /// The JSON values at a dotted path (`*` = every list item).
    fn at_path<'a>(value: &'a Value, path: &str) -> Vec<&'a Value> {
        let mut found = vec![value];
        for part in path.split('.') {
            found = found
                .into_iter()
                .flat_map(|v| match (v, part) {
                    (Value::Array(items), "*") => items.iter().collect(),
                    (Value::Array(items), index) => index.parse().ok().and_then(|i: usize| items.get(i)).into_iter().collect(),
                    (Value::Object(fields), key) => fields.get(key).into_iter().collect(),
                    _ => Vec::new(),
                })
                .collect();
        }
        found
    }

    /// Asserts that `child` kept every appearance setting of `parent`, and returns whether anything else changed.
    fn keeps_the_look(parent: &Value, child: &Value, world: usize) -> bool {
        for path in WORLDS[world].appearance {
            let before = at_path(parent, path);
            assert!(!before.is_empty(), "{}: '{path}' is not a setting", WORLDS[world].id);
            assert_eq!(before, at_path(child, path), "{}: '{path}' was perturbed", WORLDS[world].id);
        }
        parent != child
    }

    #[test]
    fn perturbation_never_touches_colours_palettes_or_the_look() {
        // Reaction-Diffusion, through the recipe format: its `params.ground` is chemistry and may move.
        let text = std::fs::read_to_string("tests/fixtures/reaction-diffusion.json").unwrap();
        let saved: SavedWorld = serde_json::from_str(&text).unwrap();
        let rd = world::resolve("reaction-diffusion").unwrap();
        let mut value = serde_json::to_value(&saved.settings).unwrap();
        value["params"]["ground"] = json!(0.25);
        let settings: WorldSettings = serde_json::from_value(value.clone()).unwrap();
        let mut rng = Rng::new(11);
        let (mut changed, mut ground_moved) = (false, false);
        for _ in 0..40 {
            let child = serde_json::to_value(perturbed(&settings, &mut rng, 1.0).unwrap()).unwrap();
            changed |= keeps_the_look(&value, &child, rd);
            ground_moved |= child["params"]["ground"] != value["params"]["ground"];
        }
        assert!(changed && ground_moved);

        // Particle Life: `ground` is the background colour, `color_shift` and `colors` pick the species' colours.
        let pl = world::resolve("particle-life").unwrap();
        let particles = json!({
            "world": "particle-life",
            "ground": 0x02050f,
            "params": {
                "kinds": 4, "force": 12.0, "substeps": 4, "count": 60000, "colors": { "Scheme": 3 },
                "color_shift": 0, "sizes": [1.2, 1.0, 0.8, 1.0], "size": 1.5, "glow": 1.0, "speed_glow": 0.4,
                "trail": 0.8, "trail_gain": 1.0, "trail_scale": 2.0, "knee": 2.0, "relief": 0.3,
                "matrix": [[0.5, -0.2], [0.1, 0.3]]
            },
            "post": { "exposure": 1.0, "bloom": 0.6 }
        });
        let mut changed = false;
        for seed in 0..40 {
            let mut child = particles.clone();
            perturb(&mut child, &mut Rng::new(seed), 1.0, WORLDS[pl].appearance);
            changed |= keeps_the_look(&particles, &child, pl);
        }
        assert!(changed, "the dynamics still move");
    }

    /// Every world's appearance paths name real settings, in every preset and
    /// in mutations, and refinement keeps them all.
    #[test]
    fn gpu_perturbed_recipes_keep_every_worlds_look() {
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let mut rng = Rng::new(5);
        for (index, entry) in WORLDS.iter().enumerate() {
            let (_, mut world) = world::create(&gpu, entry.id, [96, 64], None, 3).unwrap();
            let mut recipes = Vec::new();
            for preset in 0..world.presets().len() {
                world.load_preset(&gpu, preset, 3);
                recipes.push(world.settings().unwrap());
            }
            for seed in [7, 8] {
                world.mutate(&gpu, seed);
                recipes.push(world.settings().unwrap());
            }
            let mut changed = false;
            for settings in &recipes {
                let parent = serde_json::to_value(settings).unwrap();
                for _ in 0..4 {
                    let child = serde_json::to_value(perturbed(settings, &mut rng, 0.5).unwrap()).unwrap();
                    changed |= keeps_the_look(&parent, &child, index);
                }
            }
            assert!(changed, "{}: nothing was perturbed", entry.id);
            gpu.wait_idle();
            assert!(gpu.fatal_error().is_none(), "{}: {:?}", entry.id, gpu.fatal_error());
        }
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
        let out = dir.path().join("out");
        let job = ExploreJob {
            seed: 42,
            runs: 3,
            refine: 1,
            children: 2,
            keep: 2,
            frames: 10,
            size: [96, 64],
            max_fps: 0.0,
            out_dir: Some(out.clone()),
            library: Some(dir.path().join("lib")),
            all: true,
            argv: ["primordia", "explore", "-w", "symbiosis", "--seed", "42"].map(String::from).to_vec(),
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
        assert_eq!(std::fs::read_dir(out.join("all")).unwrap().count(), 6);
        assert_eq!(summary.all.len(), 6);
        assert_eq!((WORLDS[summary.world].id, summary.preset, summary.base_seed), ("symbiosis", 0, 42));
        let files = summary.files();
        let count = 6 + 2 * 3 + 3;
        assert_eq!(files.len(), count, "all/, image + recipe + install per kept candidate, csv, sheet, run.json");
        assert!(files.iter().all(|f| f.exists()), "{files:?}");
        assert_eq!(files.last().copied(), Some(out.join(MANIFEST).as_path()), "the manifest is written last");

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

        // run.json lists every file written into the folder, relative to it (the library's are elsewhere).
        let manifest: Value = serde_json::from_str(&std::fs::read_to_string(&summary.manifest).unwrap()).unwrap();
        assert_eq!((manifest["tool"].as_str(), manifest["complete"].as_bool()), (Some(MANIFEST_TOOL), Some(true)));
        let listed: Vec<PathBuf> =
            manifest["files"].as_array().unwrap().iter().map(|f| inside(&out, f.as_str().unwrap()).unwrap()).collect();
        let in_folder: Vec<&Path> = files.iter().copied().filter(|f| !f.starts_with(dir.path().join("lib"))).collect();
        assert_eq!(listed.iter().map(PathBuf::as_path).collect::<Vec<_>>(), in_folder[..in_folder.len() - 1]);
        assert_eq!(manifest["argv"], json!(job.argv));
        assert_eq!(manifest["gpu"], capture::gpu_name(&gpu));
        assert_eq!(manifest["csv"]["columns"].as_array().unwrap().len(), header.split(',').count());
        for (rank, kept) in manifest["kept"].as_array().unwrap().iter().enumerate() {
            let candidate = &summary.candidates[summary.kept[rank]];
            assert_eq!(kept["seed"], candidate.seed.to_string(), "seeds are strings");
            assert_eq!(kept["image"], relative(&out, &summary.images[rank]).unwrap());
            let recipe = format!("recipes/{}.json", file_stem(rank + 1, candidate.seed));
            assert_eq!(kept["recipe"].as_str(), Some(recipe.as_str()));
        }

        // Each kept recipe says how explore found it; the library ignores that and loads it.
        for (rank, path) in summary.recipes.iter().enumerate() {
            let value: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let provenance = &value["provenance"];
            let origin = (provenance["tool"].as_str(), provenance["run_seed"].as_str());
            assert_eq!(origin, (Some(MANIFEST_TOOL), Some("42")));
            let place = (provenance["rank"].as_u64(), provenance["candidate"].as_u64());
            assert_eq!(place, (Some(rank as u64 + 1), Some(summary.kept[rank] as u64)));
        }
        let recipes = Library::open(out.join("recipes"));
        assert!(recipes.warnings.is_empty(), "{:?}", recipes.warnings);
        assert_eq!(recipes.entries.len(), 2);
        for entry in &recipes.entries {
            let name = &entry.saved.name;
            assert!(name.contains("explore #") && name.ends_with(", 96x64)"), "the size explored at: {name}");
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
            out_dir: Some(dir.path().join("max")),
            library: None,
            all: false,
            ..job.clone()
        };
        let summary = explore_with(&gpu, &extreme).unwrap();
        // Inert candidates are left out of extremes too (a dead world has the smallest growth of all).
        let alive = summary.candidates.iter().filter(|c| !c.inert);
        let mut covers: Vec<(f32, usize)> = alive.map(|c| (c.descriptor.as_ref().unwrap()[0], c.index)).collect();
        covers.sort_by(|a, b| b.0.total_cmp(&a.0));
        assert_eq!(summary.kept, covers.iter().take(2).map(|c| c.1).collect::<Vec<_>>());
        let bad = ExploreJob { select: Select::Max("nope".into()), ..extreme };
        let error = match explore_with(&gpu, &bad) {
            Ok(_) => panic!("an unknown metric must be rejected"),
            Err(e) => e.to_string(),
        };
        assert!(error.contains("growth_cover"), "{error}");

        // A search can start from a kept recipe, edited by --set: it is evaluated first, from its own seed.
        let path = summary_recipe(&out);
        let from_recipe = ExploreJob {
            recipe: Some(Recipe::load(&path).unwrap()),
            sets: vec!["params.steps=3".parse().unwrap()],
            world: "ignored".into(),
            runs: 1,
            refine: 0,
            keep: 1,
            out_dir: Some(dir.path().join("from-recipe")),
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

        // A second run into the first folder replaces that run's files, and only those.
        std::fs::write(out.join("notes.txt"), "mine").unwrap();
        let again = ExploreJob { seed: 43, library: None, all: false, sheet: false, ..job.clone() };
        let summary = explore_with(&gpu, &again).unwrap();
        assert!(!out.join("all").exists() && !out.join("contact-sheet.png").exists(), "the first run's extras went");
        let names = |folder: &Path| {
            let files = std::fs::read_dir(folder).unwrap();
            let mut names: Vec<String> = files.map(|f| f.unwrap().file_name().to_string_lossy().into_owned()).collect();
            names.sort();
            names
        };
        let mut expected: Vec<String> = summary.images.iter().map(|p| relative(&out, p).unwrap()).collect();
        expected.extend(["candidates.csv", "notes.txt", "recipes", MANIFEST].map(String::from));
        expected.sort();
        assert_eq!(names(&out), expected);
        assert_eq!(names(&out.join("recipes")).len(), summary.recipes.len());
        assert_eq!(std::fs::read_to_string(out.join("notes.txt")).unwrap(), "mine");
        assert_eq!(Library::open(dir.path().join("lib")).entries.len(), 2, "installed recipes are never removed");
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    }

    /// The first recipe kept in `out`.
    fn summary_recipe(out: &Path) -> PathBuf {
        let recipes = out.join("recipes");
        let mut files: Vec<PathBuf> = std::fs::read_dir(recipes).unwrap().map(|f| f.unwrap().path()).collect();
        files.sort();
        files.remove(0)
    }

    #[test]
    fn default_folders_name_the_world_the_base_and_the_seed() {
        let job = ExploreJob::new("symbiosis");
        let source = |job: &ExploreJob| {
            Source::resolve(job.recipe.as_ref(), &job.world, job.preset.as_deref(), job.seed, &job.sets).unwrap()
        };
        assert_eq!(default_out_dir(&job, &source(&job)), Path::new("explore").join("symbiosis-living-reef-s1"));
        let coral = ExploreJob { world: "rd".into(), preset: Some("mito".into()), seed: 7, ..job.clone() };
        let expected = Path::new("explore").join("reaction-diffusion-mitosis-s7");
        assert_eq!(default_out_dir(&coral, &source(&coral)), expected);
        let recipe = Recipe::load(Path::new("tests/fixtures/reaction-diffusion.json")).unwrap();
        let from_recipe = ExploreJob { recipe: Some(recipe), ..job.clone() };
        let expected = Path::new("explore").join("reaction-diffusion-reaction-diffusion-s1");
        assert_eq!(default_out_dir(&from_recipe, &source(&from_recipe)), expected);
    }

    #[test]
    fn output_names_are_recognised_exactly() {
        for (sub, name) in [
            ("", "01-seed1.png"),
            ("", "12-seed8010643033089386035.png"),
            ("", "candidates.csv"),
            ("", "contact-sheet.png"),
            ("recipes", "03-seed42.json"),
            ("all", "000-r0-seed1.png"),
            ("all", "123-r12-seed99.png"),
        ] {
            assert!(is_output_name(sub, name), "{sub}/{name}");
        }
        for (sub, name) in [
            ("", "1-seed1.png"),
            ("", "01-seed.png"),
            ("", "01-seed1.json"),
            ("", "notes.txt"),
            ("", MANIFEST),
            ("recipes", "01-seed1.png"),
            ("recipes", "world-abc.json"),
            ("all", "00-r0-seed1.png"),
            ("all", "000-seed1.png"),
            ("other", "01-seed1.png"),
        ] {
            assert!(!is_output_name(sub, name), "{sub}/{name}");
        }
        let dir = Path::new("out");
        assert_eq!(relative(dir, &dir.join("recipes").join("01-seed1.json")).as_deref(), Some("recipes/01-seed1.json"));
        assert_eq!(relative(dir, Path::new("elsewhere/x.png")), None);
        assert_eq!(inside(dir, "recipes/01-seed1.json"), Some(dir.join("recipes").join("01-seed1.json")));
        for escape in ["../x.png", "/etc/x", "C:/x.png", "a//b", "./x", "a\\..\\b", ""] {
            assert_eq!(inside(dir, escape), None, "{escape}");
        }
    }

    #[test]
    fn a_run_removes_only_what_the_earlier_manifest_lists_and_refuses_unlisted_outputs() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let write = |relative: &str| {
            let path = inside(&out, relative).unwrap();
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, "x").unwrap();
            path
        };
        // An earlier run's outputs and its manifest, next to files of the user's own.
        let names =
            ["01-seed5.png", "recipes/01-seed5.json", "all/000-r0-seed5.png", "candidates.csv", "contact-sheet.png"];
        let earlier: Vec<PathBuf> = names.iter().map(|f| write(f)).collect();
        let mine = [write("notes.txt"), write("recipes/favourite.json"), write("all/sketch.png")];
        let outside = dir.path().join("keep.png");
        std::fs::write(&outside, "x").unwrap();
        let mut listed: Vec<&str> = names.to_vec();
        listed.extend(["../keep.png", outside.to_str().unwrap(), "gone.png"]);
        let manifest = json!({ "tool": MANIFEST_TOOL, "schema": 1, "files": listed });
        std::fs::write(out.join(MANIFEST), manifest.to_string()).unwrap();

        let job = ExploreJob { out_dir: Some(out.clone()), ..ExploreJob::new("symbiosis") };
        let plan = plan(&job).unwrap();
        assert_eq!(plan.out_dir, out);
        assert_eq!((plan.previous.listed.len(), plan.previous.unlisted.len()), (5, 0), "{:?}", plan.previous);
        assert_eq!(clear_previous(&out, &plan.previous, false).unwrap(), 6, "five outputs and the manifest");
        assert!(earlier.iter().all(|f| !f.exists()) && !out.join(MANIFEST).exists());
        assert!(mine.iter().all(|f| f.exists()) && outside.exists(), "nothing the manifest does not list is touched");
        assert!(out.join("recipes").is_dir(), "a folder that still holds a file of the user's stays");

        // Outputs that no manifest lists: refused, unless --overwrite, which removes them and nothing else.
        let stray = [write("02-seed9.png"), write("recipes/02-seed9.json"), write("all/001-r1-seed9.png")];
        let error = super::plan(&job).err().expect("a folder with unlisted outputs");
        let message = format!("{error:#}");
        let names = "no run.json lists (02-seed9.png, all/001-r1-seed9.png, recipes/02-seed9.json)";
        assert!(message.contains(names), "{message}");
        assert!(message.contains("pass --overwrite to replace them, or choose another --out-dir"), "{message}");
        assert_eq!(crate::failure::exit_code(&error), 2);
        let overwrite = ExploreJob { overwrite: true, ..job.clone() };
        let plan = super::plan(&overwrite).unwrap();
        assert_eq!(clear_previous(&out, &plan.previous, true).unwrap(), 3);
        assert!(stray.iter().all(|f| !f.exists()) && mine.iter().all(|f| f.exists()));
        write("03-seed1.png");
        assert_eq!(listing(&out, &[out.join("a"), out.join("b"), out.join("c"), out.join("d")]), "a, b, c and 1 more");

        // A run.json that explore did not write puts the folder off limits, even with --overwrite.
        std::fs::write(out.join(MANIFEST), "{\"name\": \"a different tool's run\"}").unwrap();
        let error = super::plan(&overwrite).err().expect("a foreign run.json");
        assert!(format!("{error:#}").contains("was not written by primordia explore"), "{error:#}");
        assert_eq!(crate::failure::exit_code(&error), 2);
        assert!(out.join("03-seed1.png").exists());
    }

    #[test]
    fn manifests_describe_the_run_every_column_and_every_file() {
        let out = Path::new("explore").join("reaction-diffusion-coral-reef-s7");
        let mut candidates = vec![candidate(0, Some(0.5), false), candidate(1, Some(0.9), false)];
        candidates[0].origin = Origin::Preset;
        candidates.push(Candidate { origin: Origin::Child { parent: 1 }, seed: 1, ..candidate(2, Some(0.7), true) });
        candidates[2].rank = Some(1);
        candidates[0].rank = Some(2);
        let stem = |rank: usize, c: usize| file_stem(rank, candidates[c].seed);
        let (first, second) = (stem(1, 2), stem(2, 0));
        let summary = Summary {
            world: 3,
            preset: 0,
            base_seed: 7,
            kept: vec![2, 0],
            out_dir: out.clone(),
            images: vec![out.join(format!("{first}.png")), out.join(format!("{second}.png"))],
            recipes: [&first, &second].map(|stem| out.join("recipes").join(format!("{stem}.json"))).to_vec(),
            // A library inside the folder: its files are still not the run's to list (or remove).
            installed: ["world-a.json", "world-b.json"].map(|name| out.join("library").join(name)).to_vec(),
            all: Vec::new(),
            sheet: Some(out.join("contact-sheet.png")),
            csv: out.join("candidates.csv"),
            manifest: out.join(MANIFEST),
            secs: 1.5,
            candidates,
        };
        let argv: Vec<String> = ["primordia", "explore", "-w", "rd", "--seed", "7"].map(String::from).to_vec();
        let job = ExploreJob { world: "rd".into(), seed: 7, argv, ..ExploreJob::new("rd") };
        let record = |outcome| Record {
            world: 3,
            preset: 0,
            base_seed: 7,
            started: "2026-09-22T14:59:49Z".into(),
            gpu: "Test GPU (Vulkan)".into(),
            secs: 2.0,
            files: summary.outputs().into_iter().filter_map(|f| relative(&out, f)).collect(),
            outcome,
        };
        // Through text and back, as a script would read it.
        let text = report::explore_manifest(&job, &record(Ok(&summary))).to_string();
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!((value["tool"].as_str(), value["schema"].as_u64()), (Some(MANIFEST_TOOL), Some(1)));
        assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!((value["complete"].as_bool(), value["error"].is_null()), (Some(true), true));
        assert_eq!(value["argv"], json!(job.argv));
        assert_eq!((&value["started"], &value["gpu"]), (&json!("2026-09-22T14:59:49Z"), &json!("Test GPU (Vulkan)")));
        assert_eq!(value["world"], json!({ "index": 4, "id": "reaction-diffusion", "name": "Reaction-Diffusion" }));
        assert_eq!(value["base"]["origin"], "preset");
        assert_eq!(value["base"]["preset"], json!({ "index": 1, "name": "Coral Reef", "slug": "coral-reef" }));
        assert_eq!((value["base"]["seed"].as_str(), value["settings"]["seed"].as_str()), (Some("7"), Some("7")));
        let settings = &value["settings"];
        assert_eq!((settings["runs"].as_u64(), settings["size"].clone()), (Some(48), json!([640, 360])));
        assert_eq!((settings["select"].as_str(), settings["install"].is_null()), (Some("novelty"), true));

        // Every column of candidates.csv, in order, with what it holds; descriptor columns name their measurement.
        let columns = value["csv"]["columns"].as_array().unwrap();
        let metrics = WORLDS[3].metrics;
        assert_eq!(columns.len(), CSV_COLUMNS.len() + 3 * metrics.len());
        let names: Vec<&str> = columns.iter().map(|c| c["name"].as_str().unwrap()).collect();
        assert_eq!(&names[..3], ["index", "round", "origin"]);
        assert_eq!(names[CSV_COLUMNS.len()..], dim_names(metrics));
        assert!(columns.iter().all(|c| !c["meaning"].as_str().unwrap().is_empty()));
        let alive = &columns[CSV_COLUMNS.len()];
        assert_eq!((alive["name"].as_str(), alive["stat"].as_str()), (Some("alive_mean"), Some("mean")));
        assert_eq!(alive["metric"]["id"], "alive");
        assert_eq!((&alive["metric"]["unit"], &alive["metric"]["vital"]), (&json!("fraction"), &json!(true)));

        // Files are relative to the folder; the library's are never among them.
        let files: Vec<&str> = value["files"].as_array().unwrap().iter().map(|f| f.as_str().unwrap()).collect();
        let expected = [
            format!("{first}.png"),
            format!("recipes/{first}.json"),
            format!("{second}.png"),
            format!("recipes/{second}.json"),
            "candidates.csv".into(),
            "contact-sheet.png".into(),
        ];
        assert_eq!(files, expected);
        let counts = [&value["evaluated"], &value["inert"], &value["failed"]].map(Value::as_u64);
        assert_eq!(counts, [Some(3), Some(1), Some(0)]);
        let kept = &value["kept"][0];
        assert_eq!((kept["rank"].as_u64(), kept["candidate"].as_u64()), (Some(1), Some(2)));
        assert_eq!((kept["origin"].as_str(), kept["parent"].as_u64()), (Some("child"), Some(1)));
        assert_eq!(kept["seed"], "1", "a child runs from its parent's seed, written as a string");
        assert_eq!((&kept["image"], &kept["recipe"]), (&json!(expected[0]), &json!(expected[1])));
        assert_eq!(kept["installed"], summary.installed[0].display().to_string(), "as written, not relative");
        assert!(value["kept"][1]["parent"].is_null());

        // A failed run is marked incomplete, with its error and no kept candidates.
        let failed = report::explore_manifest(&job, &record(Err("GPU error while exploring".into())));
        let outcome = (failed["complete"].as_bool(), failed["error"].as_str());
        assert_eq!(outcome, (Some(false), Some("GPU error while exploring")));
        assert!(failed["kept"].is_null() && failed["files"].is_array());

        // A kept recipe's provenance: the run, the candidate and its parent, seeds as strings.
        let provenance = report::explore_provenance(&job, &summary.candidates, 2);
        assert_eq!(
            provenance,
            json!({
                "tool": MANIFEST_TOOL, "version": env!("CARGO_PKG_VERSION"), "run_seed": "7", "select": "novelty",
                "candidate": 2, "round": 0, "rank": 1, "novelty": 0.0, "origin": "child", "parent": 1,
                "parent_seed": "1",
            })
        );
    }

    #[test]
    fn plans_reject_bad_names_and_sizes_before_any_gpu_work() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let job = ExploreJob { out_dir: Some(out.clone()), ..ExploreJob::new("symbiosis") };
        let select = Select::Max("growth_cover".into());
        let plan = plan(&ExploreJob { preset: Some("coral".into()), select, ..job.clone() }).unwrap();
        assert_eq!((WORLDS[plan.world].id, plan.source.preset(), plan.lane), ("symbiosis", Some(2), Some(0)));
        assert!(out.is_dir(), "the output folder is created and probed up front");
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
