//! Offscreen rendering: final-frame PNGs, frame sequences, videos (via ffmpeg)
//! and a gallery of every preset.
//!
//! Headless renders call `step` and `render` for every frame at a fixed
//! timestep of `1 / fps` seconds (1/60 unless `--fps` says otherwise; explore
//! always uses 1/60), exactly like the interactive app, so worlds that
//! accumulate display-only state in `render` (e.g. motion trails) look the same.
//!
//! Every name and output path is checked before the GPU is opened ([`plan`]),
//! so a typo or an unwritable folder fails in milliseconds, not after the run.
//!
//! A render starts from a preset or a recipe (`--recipe`), edited by `--set`
//! ([`crate::recipe`]). Every PNG it writes carries the recipe it shows and a
//! command that renders it again ([`capture::write_png`]).

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use ab_glyph::{Font as _, FontRef, ScaleFont as _};
use anyhow::{anyhow, bail, Context as _, Result};

use crate::capture::{self, Provenance, Readback};
use crate::failure::Failure;
use crate::gpu::Gpu;
use crate::library::SavedWorld;
use crate::metrics::{CsvLog, Sample, Sampler};
use crate::post::{Post, Tonemap};
use crate::recipe::{self, Recipe, Setting, Source};
use crate::world::{self, Camera, Frame, Pointer, ViewXform, WORLDS};

/// Headless output format: 8-bit sRGB so readback bytes can be written straight to PNG.
pub(crate) const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[derive(Clone, Debug)]
pub struct RenderJob {
    /// World and preset names; ignored when `recipe` is given.
    pub world: String,
    pub preset: Option<String>,
    /// Render this recipe (its seed replaced by `seed`) instead of a preset.
    pub recipe: Option<Recipe>,
    /// `--set` edits of the recipe or preset, in order.
    pub sets: Vec<Setting>,
    pub seed: u64,
    pub size: [u32; 2],
    pub frames: u32,
    pub fps: u32,
    /// PNG of the final frame. `None` means `renders/<world>-<preset>-s<seed>.png`
    /// (`renders/<recipe file name>.png` for a recipe), unless a video is
    /// requested, in which case no PNG is written.
    pub out: Option<PathBuf>,
    /// Also write the recipe that was rendered to this `.json` file.
    pub save_recipe: Option<PathBuf>,
    /// Video of every frame, one of [`VIDEO_EXTENSIONS`] (.mp4/.mov/.mkv get H.264).
    pub video: Option<PathBuf>,
    /// Save a PNG every `every` frames into `frames_dir` (0 = never).
    pub every: u32,
    pub frames_dir: PathBuf,
    pub exposure: Option<f32>,
    pub bloom: Option<f32>,
    pub bloom_threshold: Option<f32>,
    pub tonemap: Option<Tonemap>,
    pub camera: Camera,
    /// Scripted mouse input, to exercise a world's interaction without a window.
    pub brush: Option<Brush>,
    /// Frames-per-second ceiling (0 = unlimited). Headless renders would
    /// otherwise pin the GPU at 100%; the cap keeps long sessions cool.
    pub max_fps: f32,
    /// Suppress per-10% progress lines (the gallery prints one line per render).
    pub quiet: bool,
    /// Write every frame's measurements to this CSV file (`frame,time,series,<metrics>`).
    pub metrics: Option<PathBuf>,
}

/// Default headless frame-rate ceiling (see [`RenderJob::max_fps`]).
pub const DEFAULT_MAX_FPS: f32 = 240.0;

/// A mouse button held for the whole render, either at a fixed spot or orbiting
/// the centre of the world (one lap every four seconds of simulated time).
#[derive(Clone, Copy, Debug)]
pub struct Brush {
    /// Hold the secondary (right) button instead of the primary (left) one.
    pub secondary: bool,
    /// Fixed position in world uv; `None` orbits the centre.
    pub at: Option<[f32; 2]>,
    /// Radius in domain cells.
    pub radius: f32,
}

impl Brush {
    fn pointer(&self, time: f32) -> Pointer {
        let pos = self.at.unwrap_or_else(|| {
            let a = time * std::f32::consts::TAU / 4.0;
            [0.5 + 0.25 * a.cos(), 0.5 + 0.25 * a.sin()]
        });
        Pointer { pos, primary: !self.secondary, secondary: self.secondary, radius: self.radius }
    }
}

impl RenderJob {
    pub fn new(world: &str) -> Self {
        Self {
            world: world.to_string(),
            preset: None,
            recipe: None,
            sets: Vec::new(),
            seed: 1,
            size: [1920, 1080],
            frames: 600,
            fps: 60,
            out: None,
            save_recipe: None,
            video: None,
            every: 0,
            frames_dir: PathBuf::from("frames"),
            exposure: None,
            bloom: None,
            bloom_threshold: None,
            tonemap: None,
            camera: Camera::default(),
            brush: None,
            max_fps: DEFAULT_MAX_FPS,
            quiet: false,
            metrics: None,
        }
    }
}

/// Encoder speed/quality trade-off for [`spawn_ffmpeg`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encode {
    /// Offline renders: slow preset, near-transparent quality.
    Quality,
    /// Live recording: fast enough for real time at high resolutions.
    Realtime,
}

/// Own the encoder until it is finalized, including early returns on GPU or file errors.
struct VideoEncoder {
    child: Option<Child>,
}

impl VideoEncoder {
    fn write(&mut self, pixels: &[u8]) -> Result<()> {
        let child = self.child.as_mut().expect("encoder is running");
        let written = match child.stdin.as_mut() {
            Some(stdin) => stdin.write_all(pixels),
            None => Err(std::io::Error::other("stdin closed")),
        };
        if let Err(e) = written {
            return Err(ffmpeg_failure(self.child.take().expect("encoder is running"), e));
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        let child = self.child.as_mut().expect("encoder is running");
        drop(child.stdin.take());
        let status = child.wait().context("waiting for ffmpeg").map_err(|e| Failure::Ffmpeg.tag(e))?;
        self.child.take();
        if !status.success() {
            return Err(Failure::Ffmpeg.error(format!("ffmpeg exited with {status}")));
        }
        Ok(())
    }
}

impl Drop for VideoEncoder {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            drop(child.stdin.take());
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

/// The smallest image side any command renders.
pub const MIN_SIZE: u32 = 16;
/// The largest image side any command renders (and a saved recipe may ask for).
pub const MAX_SIZE: u32 = 16384;
/// File types `--video` accepts.
pub const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "mkv", "webm", "gif"];

/// A render job with its names resolved and its outputs checked, before any GPU work.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Index into `WORLDS`.
    pub world: usize,
    /// Output size (rounded down to even numbers for a video).
    pub size: [u32; 2],
    /// Final-frame PNG, with the default name filled in.
    pub out: Option<PathBuf>,
    /// The recipe (edited by `--set`) or the preset to render.
    pub source: Source,
}

/// What a finished render wrote.
#[derive(Clone, Debug)]
pub struct RenderSummary {
    /// Index into `WORLDS`.
    pub world: usize,
    /// 0-based index of the preset rendered.
    pub preset: usize,
    pub size: [u32; 2],
    pub frames: u32,
    /// The rendered recipe differs from its preset (a mutation or `--set`).
    pub modified: bool,
    pub png: Option<PathBuf>,
    pub video: Option<PathBuf>,
    /// The `--every` frame PNGs, in order.
    pub frame_files: Vec<PathBuf>,
    pub metrics: Option<MetricsLog>,
    /// The recipe written by `--save-recipe`.
    pub recipe: Option<PathBuf>,
    pub secs: f32,
}

/// The measurement log a render wrote.
#[derive(Clone, Debug)]
pub struct MetricsLog {
    pub path: PathBuf,
    /// Data rows: one per frame and series.
    pub rows: u64,
    /// Labels of series 0 and 1 when the world compares two habitats.
    pub series: Option<[String; 2]>,
    /// The final frame's measurements.
    pub last: Option<Sample>,
}

impl RenderSummary {
    /// Every file written, in the order they were written.
    pub fn files(&self) -> Vec<&Path> {
        let mut files: Vec<&Path> = self.frame_files.iter().map(PathBuf::as_path).collect();
        files.extend(self.png.as_deref());
        files.extend(self.video.as_deref());
        files.extend(self.metrics.as_ref().map(|m| m.path.as_path()));
        files.extend(self.recipe.as_deref());
        files
    }
}

/// Opens the GPU for a headless command; failing to is a [`Failure::Gpu`].
pub fn open_gpu() -> Result<Gpu> {
    pollster::block_on(Gpu::new(Gpu::create_instance(), None)).map_err(|e| Failure::Gpu.tag(e))
}

pub fn render(job: &RenderJob) -> Result<RenderSummary> {
    let plan = plan(job)?;
    if job.video.is_some() {
        probe_ffmpeg()?;
    }
    let gpu = open_gpu()?;
    execute(&gpu, job, plan)
}

/// Renders `job` on `gpu`.
pub fn render_with(gpu: &Gpu, job: &RenderJob) -> Result<RenderSummary> {
    let plan = plan(job)?;
    execute(gpu, job, plan)
}

/// An image size outside `MIN_SIZE..=MAX_SIZE` is invalid input.
pub fn check_size(size: [u32; 2]) -> Result<()> {
    if size.iter().any(|n| !(MIN_SIZE..=MAX_SIZE).contains(n)) {
        return Err(Failure::Usage.error(format!(
            "{}x{} is not a supported size: each side must be {MIN_SIZE}-{MAX_SIZE} pixels",
            size[0], size[1]
        )));
    }
    Ok(())
}

fn has_extension(path: &Path, extensions: &[&str]) -> bool {
    path.extension().and_then(|e| e.to_str()).is_some_and(|e| extensions.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Creates `dir` if needed and proves that a file can be written into it, so a
/// bad output path fails before the simulation instead of after it.
pub fn check_writable_dir(dir: &Path) -> Result<()> {
    let dir = if dir.as_os_str().is_empty() { Path::new(".") } else { dir };
    let probe = std::fs::create_dir_all(dir).and_then(|()| tempfile::NamedTempFile::new_in(dir).map(drop));
    probe.map_err(|e| Failure::Usage.tag(anyhow!("cannot write to {}: {e}", dir.display())))
}

/// [`check_writable_dir`] for the folder of the file `path`, which must not be a folder itself.
pub fn check_writable_file(path: &Path) -> Result<()> {
    if path.is_dir() {
        return Err(Failure::Usage.error(format!("{} is a folder, not a file name", path.display())));
    }
    check_writable_dir(path.parent().unwrap_or(Path::new("")))
}

/// Resolves and checks `job` without a GPU: the world and preset names, the
/// size and pacing, the output file types and that every output folder can be
/// written. Every failure here is invalid input.
pub fn plan(job: &RenderJob) -> Result<Plan> {
    // Validate the entire pacing interval before rendering or writing any files.
    frame_deadline(Instant::now(), job.frames.max(1), job.max_fps).map_err(|e| Failure::Usage.tag(e))?;
    check_size(job.size)?;
    // yuv420p video needs even dimensions; stills keep the exact size.
    let size = if job.video.is_some() { [job.size[0] & !1, job.size[1] & !1] } else { job.size };
    if size != job.size {
        log::warn!(
            "a video needs even dimensions: rendering {}x{} instead of {}x{}",
            size[0],
            size[1],
            job.size[0],
            job.size[1]
        );
    }
    let source = Source::resolve(job.recipe.as_ref(), &job.world, job.preset.as_deref(), job.seed, &job.sets)?;
    let (world, preset) = (source.world(), source.preset());
    let entry = &WORLDS[world];
    let out = match (&job.out, &job.video) {
        (Some(path), _) => Some(path.clone()),
        (None, None) => Some(default_png(job, world, preset)?),
        (None, Some(_)) => None,
    };
    if let Some(path) = &out {
        if !has_extension(path, &["png"]) {
            return Err(Failure::Usage.error(format!("output image must be a .png file (got '{}')", path.display())));
        }
        check_writable_file(path)?;
    }
    if let Some(path) = &job.video {
        if !has_extension(path, VIDEO_EXTENSIONS) {
            let (last, rest) = VIDEO_EXTENSIONS.split_last().expect("video extensions");
            return Err(Failure::Usage.error(format!(
                "--video must name a .{} or .{last} file (got '{}')",
                rest.join(", ."),
                path.display()
            )));
        }
        check_writable_file(path)?;
    }
    if job.every > 0 {
        check_writable_dir(&job.frames_dir)?;
    }
    if let Some(path) = &job.metrics {
        if entry.metrics.is_empty() {
            return Err(Failure::Usage.error(format!("{} does not publish measurements", entry.name)));
        }
        check_writable_file(path)?;
    }
    if let Some(path) = &job.save_recipe {
        recipe::check_recipe_path(path, "--save-recipe")?;
    }
    Ok(Plan { world, size, out, source })
}

/// `renders/<world>-<preset>-s<seed>.png`, or `renders/<recipe file name>.png`
/// for a recipe (`-s<seed>` added when `--seed` replaced its seed).
fn default_png(job: &RenderJob, world: usize, preset: Option<usize>) -> Result<PathBuf> {
    let entry = &WORLDS[world];
    let Some(recipe) = &job.recipe else {
        let name = (entry.presets)()[preset.unwrap_or(0)];
        return Ok(PathBuf::from("renders").join(format!("{}-{}-s{}.png", entry.id, slug(name), job.seed)));
    };
    let stem = recipe.path.file_stem().map_or_else(|| entry.id.to_string(), |s| s.to_string_lossy().into_owned());
    let seed = if job.seed == recipe.saved.seed { String::new() } else { format!("-s{}", job.seed) };
    let path = PathBuf::from("renders").join(format!("{stem}{seed}.png"));
    let canonical = |p: &Path| std::fs::canonicalize(p).ok();
    if canonical(&path).is_some_and(|out| canonical(&recipe.path) == Some(out)) {
        return Err(Failure::Usage.error(format!(
            "the default output {} is the recipe itself: name the image with -o",
            path.display()
        )));
    }
    Ok(path)
}

/// Quotes `word` for a shell when it needs it.
pub(crate) fn shell_word(word: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || "-_.,:=/+@%".contains(c);
    if !word.is_empty() && word.chars().all(plain) {
        word.to_string()
    } else {
        format!("\"{}\"", word.replace('\\', "\\\\").replace('"', "\\\""))
    }
}

/// A command that renders `png` again after `frames` frames: the preset form
/// when the render started from a preset without `--set`, otherwise
/// `--recipe` with the PNG itself, which carries the recipe with its seed,
/// size, look and camera.
fn reproduce_command(job: &RenderJob, rendered: &SavedWorld, frames: u32, png: &Path) -> String {
    let mut words: Vec<String> = vec!["primordia".into(), "render".into()];
    if job.recipe.is_none() && job.sets.is_empty() {
        let world = world::find(rendered.settings.world_id()).unwrap_or(0);
        let preset = (WORLDS[world].presets)().get(rendered.preset).map_or_else(|| "1".to_string(), |name| slug(name));
        let [width, height] = rendered.output_size;
        words.extend(["-w", WORLDS[world].id, "-p", preset.as_str()].map(String::from));
        words.extend(["--seed".into(), rendered.seed.to_string(), "--width".into(), width.to_string()]);
        words.extend(["--height".into(), height.to_string()]);
        let look = [("--exposure", job.exposure), ("--bloom", job.bloom), ("--bloom-threshold", job.bloom_threshold)];
        for (flag, value) in look {
            words.extend(value.map(|v| [flag.to_string(), v.to_string()]).into_iter().flatten());
        }
        if let Some(tonemap) = job.tonemap {
            words.extend(["--tonemap".into(), format!("{tonemap:?}").to_lowercase()]);
        }
        let camera = rendered.camera;
        if camera.zoom != 1.0 {
            words.extend(["--zoom".into(), camera.zoom.to_string()]);
        }
        if camera.center != [0.5, 0.5] {
            words.push(format!("--center={},{}", camera.center[0], camera.center[1]));
        }
    } else {
        let name = png.file_name().map_or_else(|| png.display().to_string(), |n| n.to_string_lossy().into_owned());
        words.extend(["--recipe".into(), name]);
    }
    words.extend(["--frames".into(), frames.to_string()]);
    if job.fps != 60 {
        words.extend(["--fps".into(), job.fps.to_string()]);
    }
    if let Some(brush) = job.brush {
        words.extend(["--brush", if brush.secondary { "secondary" } else { "primary" }].map(String::from));
        if brush.radius != 40.0 {
            words.extend(["--brush-radius".into(), brush.radius.to_string()]);
        }
        if let Some([x, y]) = brush.at {
            words.push(format!("--brush-at={x},{y}"));
        }
    }
    words.iter().map(|w| shell_word(w)).collect::<Vec<_>>().join(" ")
}

fn execute(gpu: &Gpu, job: &RenderJob, plan: Plan) -> Result<RenderSummary> {
    let size = plan.size;
    let max = gpu.device.limits().max_texture_dimension_2d;
    if size[0] > max || size[1] > max {
        return Err(Failure::Usage.error(format!(
            "{}x{} exceeds this GPU's maximum texture size of {max}",
            size[0], size[1]
        )));
    }

    let mut world = plan.source.create(gpu, size, job.seed, &job.sets)?;
    let preset = world.preset();
    let preset_name = world.presets().get(preset).copied().unwrap_or("custom");
    let out = plan.out;
    let progress_level = if job.quiet { log::Level::Debug } else { log::Level::Info };
    let recipe_path = job.recipe.as_ref().map(|r| format!(" from {}", r.path.display())).unwrap_or_default();
    let from = format!("{recipe_path}{}", recipe::changed(&job.sets));
    log::log!(
        progress_level,
        "rendering {} / {}{from} at {}x{} for {} frames (seed {})",
        world.name(),
        preset_name,
        size[0],
        size[1],
        job.frames.max(1),
        job.seed
    );
    let series = world.comparison_labels();
    let mut metrics = match &job.metrics {
        Some(path) => {
            if let Some(labels) = &series {
                log::log!(progress_level, "measurement series 0: {} · series 1: {}", labels[0], labels[1]);
            }
            Some((CsvLog::create(path, world.metrics())?, Sampler::new(gpu, Sampler::HEADLESS_SLOTS)))
        }
        None => None,
    };
    let mut last_sample: Option<Sample> = None;

    let mut look = plan.source.look(&*world);
    if let Some(e) = job.exposure {
        look.exposure = e;
    }
    if let Some(b) = job.bloom {
        look.bloom = b;
    }
    if let Some(t) = job.bloom_threshold {
        look.bloom_threshold = t;
    }
    if let Some(t) = job.tonemap {
        look.tonemap = t;
    }
    // What is rendered, as a recipe: every PNG carries it and --save-recipe writes it.
    let rendered = match plan.source.snapshot(&*world, job.seed, size, look, job.camera, &job.sets) {
        Ok(saved) => Some(saved),
        Err(e) if job.save_recipe.is_none() => {
            log::debug!("the images carry no recipe: {e:#}");
            None
        }
        Err(e) => return Err(e),
    };
    let provenance = |frames: u32, png: &Path| match &rendered {
        Some(saved) => Provenance::of(saved, gpu).with_command(reproduce_command(job, saved, frames, png)),
        None => Provenance::default(),
    };

    let post = Post::new(gpu, size, OUTPUT_FORMAT);
    let (out_texture, out_view) = gpu.texture_2d(
        "headless output",
        size,
        OUTPUT_FORMAT,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let readback = Readback::new(gpu, size, OUTPUT_FORMAT);
    let view = ViewXform::fit(world.size(), size, &job.camera);
    let mut ffmpeg = match &job.video {
        Some(path) => Some(VideoEncoder { child: Some(spawn_ffmpeg(path, size, job.fps, Encode::Quality)?) }),
        None => None,
    };

    let dt = 1.0 / job.fps.max(1) as f32;
    let frames = job.frames.max(1);
    let started = Instant::now();
    let mut next_report = 0.1;
    let mut frame_files = Vec::new();
    for f in 0..frames {
        let frame = Frame {
            gpu,
            time: f as f32 * dt,
            dt,
            frame: f as u64,
            view,
            target_size: size,
            pointer: job.brush.map(|b| b.pointer(f as f32 * dt)),
        };
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("headless frame") });
        world.step(&frame, &mut encoder);
        if let Some((log, sampler)) = &mut metrics {
            // Never skip a frame of the log: wait for the ring when it is full.
            if sampler.is_full() {
                for sample in sampler.flush(gpu).map_err(|e| Failure::Gpu.tag(e))? {
                    log.write(&sample)?;
                    last_sample = Some(sample);
                }
            }
            let mut sink = sampler.begin(frame.frame, frame.time);
            world.measure(&frame, &mut encoder, &mut sink);
        }
        world.render(&frame, &mut encoder, post.scene_view());

        let last = f + 1 == frames;
        let save_frame = job.every > 0 && (f + 1) % job.every == 0;
        let capture = last || save_frame || ffmpeg.is_some();
        if capture {
            post.run(gpu, &mut encoder, &look, frame.time, &out_view);
            readback.copy_from(&mut encoder, &out_texture);
        }
        gpu.queue.submit([encoder.finish()]);
        if let Some((_, sampler)) = &mut metrics {
            sampler.map();
        }
        if let Some(problem) = gpu.fatal_error() {
            return Err(Failure::Gpu.error(format!("GPU error while rendering: {problem}")));
        }

        if capture {
            let pixels = readback.read(gpu).map_err(|e| Failure::Gpu.tag(e))?;
            if let Some(encoder) = ffmpeg.as_mut() {
                encoder.write(&pixels)?;
            }
            if save_frame {
                let path = job.frames_dir.join(format!("{}_{:05}.png", world.id(), f + 1));
                capture::write_png(&path, size, &pixels, &provenance(f + 1, &path))?;
                frame_files.push(path);
            }
            if last {
                if let Some(path) = &out {
                    capture::write_png(path, size, &pixels, &provenance(frames, path))?;
                    log::log!(progress_level, "wrote {}", path.display());
                }
            }
        } else if f % 4 == 3 {
            // Keep the CPU from queueing hundreds of frames ahead of the GPU.
            gpu.wait_idle();
        }
        if let Some((log, sampler)) = &mut metrics {
            for sample in sampler.collect(gpu) {
                log.write(&sample)?;
                last_sample = Some(sample);
            }
        }
        if job.max_fps > 0.0 {
            // Pace submissions so the GPU idles between frames instead of running flat out.
            let due = frame_deadline(started, f + 1, job.max_fps)?;
            let now = Instant::now();
            if due > now {
                std::thread::sleep(due - now);
            }
        }

        let progress = (f + 1) as f32 / frames as f32;
        if progress >= next_report && !last {
            log::log!(
                progress_level,
                "  {:>3.0}%  ({:.1} fps)",
                progress * 100.0,
                (f + 1) as f32 / started.elapsed().as_secs_f32()
            );
            next_report += 0.1;
        }
    }

    if let Some(encoder) = ffmpeg {
        encoder.finish()?;
        log::log!(progress_level, "wrote {}", job.video.as_ref().map(|p| p.display().to_string()).unwrap_or_default());
    }
    let metrics = match metrics {
        Some((mut log, mut sampler)) => {
            for sample in sampler.flush(gpu).map_err(|e| Failure::Gpu.tag(e))? {
                log.write(&sample)?;
                last_sample = Some(sample);
            }
            let (path, rows) = log.finish()?;
            log::log!(progress_level, "wrote {} ({rows} rows)", path.display());
            Some(MetricsLog { path, rows, series, last: last_sample })
        }
        None => None,
    };
    let recipe = match (&job.save_recipe, &rendered) {
        (Some(path), Some(saved)) => {
            recipe::write(path, saved)?;
            log::log!(progress_level, "wrote {}", path.display());
            Some(path.clone())
        }
        _ => None,
    };
    let secs = started.elapsed().as_secs_f32();
    log::log!(progress_level, "done: {frames} frames in {secs:.1}s ({:.1} fps)", frames as f32 / secs.max(1e-3));
    Ok(RenderSummary {
        world: plan.world,
        preset,
        size,
        frames,
        modified: rendered.as_ref().is_some_and(|saved| saved.modified),
        png: out,
        video: job.video.clone(),
        frame_files,
        metrics,
        recipe,
        secs,
    })
}

pub(crate) fn frame_deadline(started: Instant, frames: u32, max_fps: f32) -> Result<Instant> {
    if !max_fps.is_finite() || max_fps < 0.0 {
        bail!("--max-fps must be a finite, non-negative number (0 = unlimited)");
    }
    if max_fps == 0.0 {
        return Ok(started);
    }
    let interval = std::time::Duration::try_from_secs_f64(f64::from(frames) / f64::from(max_fps))
        .context("--max-fps is too small for the requested frame count")?;
    started.checked_add(interval).context("--max-fps is too small for the system clock")
}

pub struct GalleryJob {
    pub out_dir: PathBuf,
    pub size: [u32; 2],
    pub frames: u32,
    pub seed: u64,
    /// Restrict to one world (id, name, alias or number).
    pub world: Option<String>,
    /// Also write `contact-sheet.png`, every image tiled into one overview.
    pub sheet: bool,
    /// Frames-per-second ceiling per render (0 = unlimited).
    pub max_fps: f32,
}

/// What a finished gallery wrote.
#[derive(Clone, Debug)]
pub struct GallerySummary {
    /// `(world index, 0-based preset, image)` in render order.
    pub images: Vec<(usize, usize, PathBuf)>,
    pub sheet: Option<PathBuf>,
    pub secs: f32,
}

impl GallerySummary {
    /// Every file written, in the order they were written.
    pub fn files(&self) -> Vec<&Path> {
        let mut files: Vec<&Path> = self.images.iter().map(|(_, _, path)| path.as_path()).collect();
        files.extend(self.sheet.as_deref());
        files
    }
}

/// Renders the final frame of every preset of every world into `out_dir`.
pub fn gallery(job: &GalleryJob) -> Result<GallerySummary> {
    let only = job.world.as_deref().map(world::resolve).transpose()?;
    check_size(job.size)?;
    frame_deadline(Instant::now(), job.frames.max(1), job.max_fps).map_err(|e| Failure::Usage.tag(e))?;
    check_writable_dir(&job.out_dir)?;
    let mut todo = Vec::new();
    for (index, entry) in WORLDS.iter().enumerate() {
        if only.is_some_and(|o| o != index) {
            continue;
        }
        todo.extend((entry.presets)().iter().enumerate().map(|(i, name)| (index, i, *name)));
    }
    let gpu = open_gpu()?;

    let total = todo.len();
    let started = Instant::now();
    let mut images = Vec::with_capacity(total);
    for (n, (index, i, name)) in todo.into_iter().enumerate() {
        let entry = &WORLDS[index];
        let out = job.out_dir.join(format!("{}-{:02}-{}.png", entry.id, i + 1, slug(name)));
        let t = Instant::now();
        let render = RenderJob {
            preset: Some((i + 1).to_string()),
            seed: job.seed,
            size: job.size,
            frames: job.frames,
            out: Some(out.clone()),
            max_fps: job.max_fps,
            quiet: true,
            ..RenderJob::new(entry.id)
        };
        render_with(&gpu, &render)?;
        log::info!(
            "[{:>2}/{total}] {} / {} -> {} ({:.1}s)",
            n + 1,
            entry.name,
            name,
            out.display(),
            t.elapsed().as_secs_f32()
        );
        images.push((index, i, out));
    }
    let sheet = if job.sheet && !images.is_empty() {
        let several = images.iter().any(|(index, _, _)| *index != images[0].0);
        let tiles: Vec<Tile> = images
            .iter()
            .map(|(index, i, path)| {
                Tile { group: WORLDS[*index].id, path, caption: gallery_caption(*index, *i, several) }
            })
            .collect();
        let sheet = job.out_dir.join("contact-sheet.png");
        contact_sheet(&tiles, &sheet, 5, 4)?;
        log::info!("wrote {}", sheet.display());
        Some(sheet)
    } else {
        None
    };
    let secs = started.elapsed().as_secs_f32();
    log::info!("gallery done: {total} images in {secs:.1}s");
    Ok(GallerySummary { images, sheet, secs })
}

/// The gallery sheet's caption of preset `preset` (0-based) of world `world`:
/// "2 · Mitosis", followed by the world's name on a sheet of several worlds
/// ("5 · Symbiosis (Physarum)"), where a narrow tile cuts it first.
fn gallery_caption(world: usize, preset: usize, several: bool) -> String {
    let entry = &WORLDS[world];
    let name = format!("{} · {}", preset + 1, (entry.presets)().get(preset).copied().unwrap_or("custom"));
    if several { format!("{name} ({})", entry.name) } else { name }
}

/// One image of a contact sheet.
#[derive(Clone, Debug)]
pub(crate) struct Tile<'a> {
    /// Tiles of a group share rows; a new row starts where the group changes
    /// (the gallery's world, for example).
    pub group: &'a str,
    pub path: &'a Path,
    /// Printed under the image, cut short with "…" to its width; empty for none.
    pub caption: String,
}

/// Background of contact sheets.
const SHEET_BACKGROUND: image::Rgba<u8> = image::Rgba([11, 13, 18, 255]);
/// Colour of contact-sheet captions.
const CAPTION_COLOUR: [u8; 3] = [206, 211, 222];

/// Tiles `tiles` into one overview image at `1 / divisor` of the first image's
/// size, at most `cols` to a row, with each caption in a strip under its image.
pub(crate) fn contact_sheet(tiles: &[Tile], out: &Path, cols: usize, divisor: u32) -> Result<()> {
    const GAP: u32 = 8;

    let mut rows: Vec<Vec<&Tile>> = Vec::new();
    let mut last_group = "";
    for tile in tiles {
        if tile.group != last_group || rows.last().is_some_and(|r| r.len() == cols) {
            rows.push(Vec::new());
            last_group = tile.group;
        }
        if let Some(row) = rows.last_mut() {
            row.push(tile);
        }
    }

    let first = tiles.first().context("a contact sheet needs at least one image")?.path;
    let first = image::open(first).with_context(|| format!("reading {}", first.display()))?;
    let divisor = divisor.max(1);
    let (tw, th) = ((first.width() / divisor).max(1), (first.height() / divisor).max(1));
    // Captions sit in a strip under each row, their size following the tiles'.
    let font = caption_font();
    let px = (th as f32 * 0.12).clamp(10.0, 16.0);
    let (ascent, line) = (font.as_scaled(px).ascent().ceil() as u32, font.as_scaled(px).height().ceil() as u32);
    let band = if tiles.iter().any(|t| !t.caption.is_empty()) { line + 8 } else { 0 };
    let width = cols as u32 * tw + (cols as u32 + 1) * GAP;
    let height = rows.len() as u32 * (th + band) + (rows.len() as u32 + 1) * GAP;
    let mut sheet = image::RgbaImage::from_pixel(width, height, SHEET_BACKGROUND);
    for (r, row) in rows.iter().enumerate() {
        for (c, tile) in row.iter().enumerate() {
            let img = image::open(tile.path).with_context(|| format!("reading {}", tile.path.display()))?.to_rgba8();
            let thumb = image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle);
            let x = GAP + c as u32 * (tw + GAP);
            let y = GAP + r as u32 * (th + band + GAP);
            image::imageops::replace(&mut sheet, &thumb, i64::from(x), i64::from(y));
            if !tile.caption.is_empty() {
                draw_caption(&mut sheet, &font, &tile.caption, [x, y + th + 4 + ascent], px, tw as f32);
            }
        }
    }
    capture::save_png(out, [width, height], sheet.into_raw())
}

/// The captions' typeface: egui's Ubuntu Light, compiled in (no font file is read).
fn caption_font() -> FontRef<'static> {
    FontRef::try_from_slice(epaint_default_fonts::UBUNTU_LIGHT).expect("egui's bundled font parses")
}

/// Width of `text` set in `font` at `px` pixels, kerning included.
fn text_width(font: &FontRef, px: f32, text: &str) -> f32 {
    let scaled = font.as_scaled(px);
    let mut width = 0.0;
    let mut last = None;
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(previous) = last {
            width += scaled.kern(previous, id);
        }
        width += scaled.h_advance(id);
        last = Some(id);
    }
    width
}

/// `text`, cut short with "…" where it would be wider than `width` pixels.
fn fit_text(font: &FontRef, px: f32, text: &str, width: f32) -> String {
    if text_width(font, px, text) <= width {
        return text.to_string();
    }
    let chars: Vec<char> = text.chars().collect();
    (0..chars.len())
        .rev()
        .map(|n| format!("{}…", chars[..n].iter().collect::<String>().trim_end()))
        .find(|cut| text_width(font, px, cut) <= width)
        .unwrap_or_default()
}

/// Draws `text` in [`CAPTION_COLOUR`] with its left edge and baseline at
/// `origin`, `px` pixels high and at most `width` pixels wide.
fn draw_caption(image: &mut image::RgbaImage, font: &FontRef, text: &str, origin: [u32; 2], px: f32, width: f32) {
    let text = fit_text(font, px, text, width);
    let scaled = font.as_scaled(px);
    let (mut caret, mut last) = (origin[0] as f32, None);
    for c in text.chars() {
        let id = scaled.glyph_id(c);
        if let Some(previous) = last {
            caret += scaled.kern(previous, id);
        }
        let glyph = id.with_scale_and_position(px, ab_glyph::point(caret, origin[1] as f32));
        caret += scaled.h_advance(id);
        last = Some(id);
        let Some(outline) = font.outline_glyph(glyph) else { continue };
        let bounds = outline.px_bounds();
        outline.draw(|gx, gy, coverage| {
            let (x, y) = (bounds.min.x as i64 + i64::from(gx), bounds.min.y as i64 + i64::from(gy));
            if x < 0 || y < 0 || x >= i64::from(image.width()) || y >= i64::from(image.height()) {
                return;
            }
            let pixel = image.get_pixel_mut(x as u32, y as u32);
            let alpha = coverage.clamp(0.0, 1.0);
            for (channel, ink) in pixel.0.iter_mut().zip(CAPTION_COLOUR) {
                *channel = (f32::from(*channel) * (1.0 - alpha) + f32::from(ink) * alpha).round() as u8;
            }
        });
    }
}

pub fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

/// Starts ffmpeg reading raw RGBA frames of `size` from stdin and encoding to `path`.
/// H.264 outputs are converted with the BT.709 matrix and tagged accordingly, so
/// players show the same colours as the PNGs.
pub fn spawn_ffmpeg(path: &Path, size: [u32; 2], fps: u32, encode: Encode) -> Result<Child> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let exe = ffmpeg_exe();
    let mut cmd = Command::new(&exe);
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba"])
        .args(["-s", &format!("{}x{}", size[0], size[1])])
        .args(["-framerate", &fps.max(1).to_string(), "-i", "-"]);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if matches!(ext.as_str(), "mp4" | "mov" | "mkv") {
        let speed: [&str; 4] = match encode {
            Encode::Quality => ["-preset", "slow", "-crf", "16"],
            Encode::Realtime => ["-preset", "veryfast", "-crf", "18"],
        };
        cmd.args(["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p", "-c:v", "libx264"])
            .args(speed)
            .args(["-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709"])
            .args(["-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart"]);
    }
    cmd.arg(path).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::inherit());
    cmd.spawn().map_err(|e| missing_ffmpeg(&exe, &e))
}

/// The ffmpeg executable: `PRIMORDIA_FFMPEG`, or `ffmpeg` on the `PATH`.
fn ffmpeg_exe() -> String {
    std::env::var("PRIMORDIA_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_string())
}

/// How to get ffmpeg on this platform.
pub fn ffmpeg_install_hint() -> &'static str {
    if cfg!(target_os = "windows") {
        "install it with `winget install Gyan.FFmpeg` or set PRIMORDIA_FFMPEG to ffmpeg.exe"
    } else if cfg!(target_os = "macos") {
        "install it with `brew install ffmpeg` or set PRIMORDIA_FFMPEG to its path"
    } else {
        "install it with your package manager (e.g. `sudo apt install ffmpeg`) or set PRIMORDIA_FFMPEG to its path"
    }
}

fn missing_ffmpeg(exe: &str, err: &std::io::Error) -> anyhow::Error {
    Failure::Ffmpeg.error(format!("cannot start ffmpeg ('{exe}': {err}); {}", ffmpeg_install_hint()))
}

/// Checks that ffmpeg runs, so a missing encoder is reported before any frame is simulated.
pub fn probe_ffmpeg() -> Result<()> {
    let exe = ffmpeg_exe();
    let status = Command::new(&exe)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| missing_ffmpeg(&exe, &e))?;
    if !status.success() {
        return Err(Failure::Ffmpeg.error(format!("'{exe} -version' exited with {status}; {}", ffmpeg_install_hint())));
    }
    Ok(())
}

/// Turns a failed write into an error that includes ffmpeg's exit status.
pub fn ffmpeg_failure(mut child: Child, err: std::io::Error) -> anyhow::Error {
    drop(child.stdin.take());
    Failure::Ffmpeg.tag(match child.wait() {
        Ok(status) => anyhow!("ffmpeg exited with {status} while encoding ({err})"),
        Err(wait_err) => anyhow!("ffmpeg failed while encoding ({err}; {wait_err})"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pacing_rejects_unrepresentable_and_nonfinite_intervals() {
        let start = Instant::now();
        for fps in [1e-40, f32::NAN, f32::INFINITY, -1.0] {
            assert!(frame_deadline(start, 600, fps).is_err(), "fps: {fps}");
        }
        assert_eq!(frame_deadline(start, 600, 0.0).unwrap(), start);
        assert_eq!(frame_deadline(start, 600, 60.0).unwrap() - start, std::time::Duration::from_secs(10));
    }

    #[test]
    fn plans_check_names_file_types_and_output_folders_before_the_gpu() {
        let dir = tempfile::tempdir().unwrap();
        let job = RenderJob { out: Some(dir.path().join("a").join("b").join("final.png")), ..RenderJob::new("rd") };
        let plan = plan(&RenderJob { preset: Some("mito".into()), ..job.clone() }).unwrap();
        let resolved = (WORLDS[plan.world].id, plan.source.preset(), plan.size);
        assert_eq!(resolved, ("reaction-diffusion", Some(1), [1920, 1080]));
        let folder = dir.path().join("a").join("b");
        assert!(folder.is_dir(), "output folders are created up front");
        assert_eq!(std::fs::read_dir(&folder).unwrap().count(), 0, "the write probe leaves nothing behind");

        // A video needs even sizes and a known file type; without -o no still is written.
        let video = super::plan(&RenderJob { out: None, video: Some(dir.path().join("v.MP4")), size: [321, 181], ..job.clone() })
            .unwrap();
        assert_eq!((video.size, video.out), ([320, 180], None));

        let file = dir.path().join("afile");
        std::fs::write(&file, "").unwrap();
        std::fs::create_dir(dir.path().join("folder.png")).unwrap();
        let cases = [
            (RenderJob { world: "physarm".into(), ..job.clone() }, "did you mean 'physarum'?"),
            (RenderJob { world: "p".into(), ..job.clone() }, "it could be physarum or particle-life"),
            (RenderJob { preset: Some("99".into()), ..job.clone() }, "Reaction-Diffusion has presets 1-10"),
            (RenderJob { out: Some(dir.path().join("x.jpg")), ..job.clone() }, "must be a .png file"),
            (RenderJob { out: Some(dir.path().join("folder.png")), ..job.clone() }, "is a folder"),
            (RenderJob { video: Some(dir.path().join("clip")), ..job.clone() }, "--video must name a .mp4, .mov"),
            (RenderJob { out: Some(file.join("x.png")), ..job.clone() }, "cannot write to"),
            (RenderJob { every: 5, frames_dir: file.join("frames"), ..job.clone() }, "cannot write to"),
            (RenderJob { metrics: Some(file.join("m.csv")), ..job.clone() }, "cannot write to"),
            (RenderJob { size: [8, 1080], ..job.clone() }, "8x1080 is not a supported size"),
            (RenderJob { max_fps: -1.0, ..job.clone() }, "--max-fps"),
        ];
        for (bad, expected) in cases {
            let error = super::plan(&bad).expect_err("the plan must fail");
            assert!(format!("{error:#}").contains(expected), "{error:#}");
            assert_eq!(crate::failure::exit_code(&error), 2, "{error:#}");
        }

        let missing = missing_ffmpeg("no-such-ffmpeg", &std::io::Error::from(std::io::ErrorKind::NotFound));
        assert_eq!(crate::failure::exit_code(&missing), 4);
        assert!(missing.to_string().contains("'no-such-ffmpeg'") && missing.to_string().contains("PRIMORDIA_FFMPEG"));
    }

    #[test]
    fn recipes_name_their_image_and_their_settings_are_checked_before_the_gpu() {
        let recipe = Recipe::load(Path::new("tests/fixtures/reaction-diffusion.json")).unwrap();
        let job = RenderJob { recipe: Some(recipe), seed: u64::MAX, ..RenderJob::new("ignored") };
        let renders = Path::new("renders");
        assert_eq!(default_png(&job, 3, Some(0)).unwrap(), renders.join("reaction-diffusion.png"));
        let reseeded = RenderJob { seed: 5, ..job.clone() };
        assert_eq!(default_png(&reseeded, 3, Some(0)).unwrap(), renders.join("reaction-diffusion-s5.png"));

        let dir = tempfile::tempdir().unwrap();
        let sets = vec!["params.kill=0.05".parse().unwrap(), "post.bloom=0.2".parse().unwrap()];
        let job = RenderJob { out: Some(dir.path().join("r.png")), sets, ..job };
        let plan = plan(&job).unwrap();
        let Source::Recipe(saved) = &plan.source else { panic!("expected the recipe") };
        let crate::library::WorldSettings::ReactionDiffusion { params, .. } = &saved.settings else { panic!() };
        assert_eq!((params.kill, saved.look.bloom, saved.modified), (0.05, 0.2, true));
        let resolved = (WORLDS[plan.world].id, plan.source.preset(), saved.seed);
        assert_eq!(resolved, ("reaction-diffusion", Some(0), u64::MAX));
        let cases = [
            (RenderJob { save_recipe: Some(dir.path().join("r.txt")), ..job.clone() }, "--save-recipe must name"),
            (RenderJob { sets: vec!["params.kil=1".parse().unwrap()], ..job.clone() }, "did you mean 'params.kill'?"),
            (RenderJob { sets: vec!["palette=Frost".parse().unwrap()], ..job.clone() }, "unknown palette 'Frost'"),
        ];
        for (bad, expected) in cases {
            let error = super::plan(&bad).expect_err("the plan must fail");
            assert!(format!("{error:#}").contains(expected), "{error:#}");
            assert_eq!(crate::failure::exit_code(&error), 2, "{error:#}");
        }
    }

    #[test]
    fn png_comments_hold_a_command_that_renders_the_image_again() {
        let saved = Recipe::load(Path::new("tests/fixtures/reaction-diffusion.json")).unwrap().saved;
        let brush = Brush { secondary: true, at: Some([0.25, -0.5]), radius: 12.0 };
        let job = RenderJob { fps: 30, exposure: Some(1.5), brush: Some(brush), ..RenderJob::new("rd") };
        assert_eq!(
            reproduce_command(&job, &saved, 90, Path::new("out/shot.png")),
            "primordia render -w reaction-diffusion -p coral-reef --seed 18446744073709551615 --width 1920 \
             --height 1008 --exposure 1.5 --zoom 0.575 --center=0.4186335,0.944689 --frames 90 --fps 30 --brush secondary \
             --brush-radius 12 --brush-at=0.25,-0.5"
        );
        let edited = RenderJob { sets: vec!["params.feed=0.03".parse().unwrap()], brush: None, ..job };
        assert_eq!(
            reproduce_command(&edited, &saved, 90, Path::new("out/my shot.png")),
            "primordia render --recipe \"my shot.png\" --frames 90 --fps 30"
        );
        assert_eq!(shell_word("C:\\renders\\a.png"), "\"C:\\\\renders\\\\a.png\"");
        assert_eq!(shell_word(""), "\"\"");
        assert_eq!(shell_word("--center=-0.5,1"), "--center=-0.5,1");
    }

    #[test]
    fn contact_sheets_caption_every_tile_within_its_width() {
        let dir = tempfile::tempdir().unwrap();
        let tile = |name: &str, rgb: [u8; 3]| {
            let path = dir.path().join(name);
            let pixels: Vec<u8> = (0..120 * 90).flat_map(|_| [rgb[0], rgb[1], rgb[2], 255]).collect();
            capture::save_png(&path, [120, 90], pixels).unwrap();
            path
        };
        let (red, green, blue) = (tile("r.png", [200, 0, 0]), tile("g.png", [0, 200, 0]), tile("b.png", [0, 0, 200]));
        let long = "#02 · seed 9007199254740991 and a caption far too long for its tile".to_string();
        let tiles = [
            Tile { group: "a", path: &red, caption: "1 · Mitosis".into() },
            Tile { group: "a", path: &green, caption: long.clone() },
            Tile { group: "b", path: &blue, caption: String::new() },
        ];
        let out = dir.path().join("sheet.png");
        contact_sheet(&tiles, &out, 2, 1).unwrap();
        let sheet = image::open(&out).unwrap().to_rgba8();
        const GAP: u32 = 8;
        assert_eq!(sheet.width(), 2 * 120 + 3 * GAP);
        let band = (sheet.height() - 2 * 90 - 3 * GAP) / 2;
        assert!((12..=40).contains(&band), "a strip for the captions under each row: {band}");

        // The images are intact, and ink sits only in the strips under captioned tiles.
        assert_eq!(sheet.get_pixel(GAP + 60, GAP + 45).0, [200, 0, 0, 255]);
        assert_eq!(sheet.get_pixel(2 * GAP + 120 + 60, GAP + 45).0, [0, 200, 0, 255]);
        let ink = |x0: u32, x1: u32, y0: u32, y1: u32| {
            let lit = |x: u32, y: u32| sheet.get_pixel(x, y).0[..3].iter().any(|&c| c > 60);
            (x0..x1).flat_map(|x| (y0..y1).map(move |y| (x, y))).filter(|&(x, y)| lit(x, y)).count()
        };
        let strip = |row: u32| (GAP + row * (90 + band + GAP) + 90, GAP + row * (90 + band + GAP) + 90 + band);
        let (top, bottom) = strip(0);
        assert!(ink(GAP, GAP + 120, top, bottom) > 20, "the first caption is drawn");
        assert!(ink(2 * GAP + 120, 2 * GAP + 240, top, bottom) > 20, "the long caption is drawn");
        assert_eq!(ink(GAP + 120, 2 * GAP + 120, top, bottom), 0, "nothing spills into the gap");
        assert_eq!(ink(2 * GAP + 240, sheet.width(), top, bottom), 0, "the long caption is cut to its tile");
        let (top, bottom) = strip(1);
        assert_eq!(ink(0, sheet.width(), top, bottom), 0, "an empty caption draws nothing");

        let font = caption_font();
        let cut = fit_text(&font, 14.0, &long, 120.0);
        assert!(cut.ends_with('…') && long.starts_with(cut.trim_end_matches('…')), "{cut}");
        assert!(text_width(&font, 14.0, &cut) <= 120.0);
        assert_eq!(fit_text(&font, 14.0, "1 · Mitosis", 120.0), "1 · Mitosis");

        // Without captions there is no strip.
        let plain: Vec<Tile> = tiles.iter().map(|t| Tile { caption: String::new(), ..t.clone() }).collect();
        contact_sheet(&plain, &out, 2, 1).unwrap();
        assert_eq!(image::open(&out).unwrap().height(), 2 * 90 + 3 * GAP);

        assert_eq!(gallery_caption(3, 1, false), "2 · Mitosis");
        assert_eq!(gallery_caption(0, 4, true), "5 · Symbiosis (Physarum)", "the world, where it could be mistaken");
    }

    #[test]
    fn render_logs_one_measurement_row_per_frame_and_series() {
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let dir = tempfile::tempdir().unwrap();
        let csv = dir.path().join("logs").join("run.csv");
        let job = RenderJob {
            size: [96, 64],
            frames: 8,
            every: 3,
            frames_dir: dir.path().join("frames"),
            out: Some(dir.path().join("final.png")),
            max_fps: 0.0,
            quiet: true,
            metrics: Some(csv.clone()),
            ..RenderJob::new("symbiosis")
        };
        let summary = render_with(&gpu, &job).unwrap();
        assert_eq!((WORLDS[summary.world].id, summary.preset, summary.frames), ("symbiosis", 0, 8));
        assert_eq!(summary.frame_files, [3, 6].map(|f| dir.path().join("frames").join(format!("symbiosis_{f:05}.png"))));
        assert_eq!(summary.png.as_deref(), Some(dir.path().join("final.png").as_path()));
        let log = summary.metrics.as_ref().unwrap();
        assert_eq!((log.rows, log.series.is_none()), (8, true));
        assert_eq!(log.last.unwrap().frame, 7, "the last row is the last frame");
        assert!(summary.files().iter().all(|f| f.exists()) && summary.files().len() == 4);

        let text = std::fs::read_to_string(&csv).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        let ids: Vec<&str> = header.split(',').collect();
        assert_eq!(&ids[..3], &["frame", "time", "series"]);
        let (_, world) = world::create(&gpu, "symbiosis", [96, 64], None, 1).unwrap();
        let expected: Vec<&str> = world.metrics().iter().map(|m| m.id).collect();
        assert_eq!(&ids[3..], &expected[..]);

        let rows: Vec<Vec<&str>> = lines.map(|line| line.split(',').collect()).collect();
        assert_eq!(rows.len(), 8, "one row per frame for a single habitat");
        for (f, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), ids.len(), "{row:?}");
            assert_eq!(row[0], f.to_string());
            assert!((row[1].parse::<f32>().unwrap() - f as f32 / 60.0).abs() < 1e-5);
            assert_eq!(row[2], "0");
            for value in &row[3..] {
                assert!(value.parse::<f32>().unwrap().is_finite(), "{row:?}");
            }
        }
        // The frame after seeding already has growth, and it keeps evolving.
        assert!(rows[0][3].parse::<f32>().unwrap() > 0.0);
        assert!(rows.iter().any(|row| row[5].parse::<f32>().unwrap() > 0.0), "changing cells");

        // A comparison logs two rows per frame, in the order of the pane labels.
        let (_, mut paired) = world::create(&gpu, "symbiosis", [96, 64], None, 1).unwrap();
        let mut recipe = paired.settings().unwrap();
        if let crate::library::WorldSettings::Symbiosis { params, .. } = &mut recipe {
            params.compare = true;
        }
        paired.restore_settings(&gpu, &recipe, 1).unwrap();
        assert!(paired.comparison_labels().is_some());
        let csv = dir.path().join("paired.csv");
        let mut log = CsvLog::create(&csv, paired.metrics()).unwrap();
        let mut sampler = Sampler::new(&gpu, Sampler::HEADLESS_SLOTS);
        for f in 0..3u64 {
            let frame = Frame {
                gpu: &gpu,
                time: f as f32 / 60.0,
                dt: 1.0 / 60.0,
                frame: f,
                view: ViewXform::fit(paired.size(), [96, 64], &Camera::default()),
                target_size: [96, 64],
                pointer: None,
            };
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            paired.step(&frame, &mut encoder);
            let mut sink = sampler.begin(f, frame.time);
            paired.measure(&frame, &mut encoder, &mut sink);
            gpu.queue.submit([encoder.finish()]);
            sampler.map();
        }
        for sample in sampler.flush(&gpu).unwrap() {
            log.write(&sample).unwrap();
        }
        assert_eq!(log.finish().unwrap().1, 6);
        let text = std::fs::read_to_string(&csv).unwrap();
        let series: Vec<&str> = text.lines().skip(1).map(|line| line.split(',').nth(2).unwrap()).collect();
        assert_eq!(series, ["0", "1", "0", "1", "0", "1"]);
    }
}
