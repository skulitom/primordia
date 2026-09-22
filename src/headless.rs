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

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _, Result};

use crate::capture::{self, Readback};
use crate::failure::Failure;
use crate::gpu::Gpu;
use crate::metrics::{CsvLog, Sample, Sampler};
use crate::post::{Post, Tonemap};
use crate::world::{self, Camera, Frame, Pointer, ViewXform, WORLDS};

/// Headless output format: 8-bit sRGB so readback bytes can be written straight to PNG.
pub(crate) const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

#[derive(Clone, Debug)]
pub struct RenderJob {
    pub world: String,
    pub preset: Option<String>,
    pub seed: u64,
    pub size: [u32; 2],
    pub frames: u32,
    pub fps: u32,
    /// PNG of the final frame. `None` means `renders/<world>-<preset>-s<seed>.png`,
    /// unless a video is requested, in which case no PNG is written.
    pub out: Option<PathBuf>,
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
            seed: 1,
            size: [1920, 1080],
            frames: 600,
            fps: 60,
            out: None,
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
    /// 0-based preset to load; `None` keeps the world's first.
    pub preset: Option<usize>,
    /// Output size (rounded down to even numbers for a video).
    pub size: [u32; 2],
    /// Final-frame PNG, with the default name filled in.
    pub out: Option<PathBuf>,
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
    pub png: Option<PathBuf>,
    pub video: Option<PathBuf>,
    /// The `--every` frame PNGs, in order.
    pub frame_files: Vec<PathBuf>,
    pub metrics: Option<MetricsLog>,
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
    let world = world::resolve(&job.world)?;
    let preset = job.preset.as_deref().map(|p| world::resolve_preset(world, p)).transpose()?;
    let entry = &WORLDS[world];
    let out = match (&job.out, &job.video) {
        (Some(path), _) => Some(path.clone()),
        (None, None) => {
            let name = (entry.presets)()[preset.unwrap_or(0)];
            Some(PathBuf::from("renders").join(format!("{}-{}-s{}.png", entry.id, slug(name), job.seed)))
        }
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
    Ok(Plan { world, preset, size, out })
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

    let mut world = world::create_at(gpu, plan.world, size, plan.preset, job.seed)?;
    let preset = world.preset();
    let preset_name = world.presets().get(preset).copied().unwrap_or("custom");
    let out = plan.out;
    let progress_level = if job.quiet { log::Level::Debug } else { log::Level::Info };
    log::log!(
        progress_level,
        "rendering {} / {} at {}x{} for {} frames (seed {})",
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

    let mut look = world.post_settings();
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
                capture::save_png(&path, size, pixels.clone())?;
                frame_files.push(path);
            }
            if last {
                if let Some(path) = &out {
                    capture::save_png(path, size, pixels)?;
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
    let secs = started.elapsed().as_secs_f32();
    log::log!(progress_level, "done: {frames} frames in {secs:.1}s ({:.1} fps)", frames as f32 / secs.max(1e-3));
    Ok(RenderSummary { world: plan.world, preset, size, frames, png: out, video: job.video.clone(), frame_files, metrics, secs })
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
        let tiles: Vec<(&str, PathBuf)> = images.iter().map(|(index, _, path)| (WORLDS[*index].id, path.clone())).collect();
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

// Tiles gallery images (quarter size) into one overview image. Each world
/// starts a new row; rows hold at most `cols` images.
/// Tiles `images` (each paired with a grouping key: a new row starts when the
/// key changes) at `1 / divisor` of the first image's size.
pub(crate) fn contact_sheet(images: &[(&str, PathBuf)], out: &Path, cols: usize, divisor: u32) -> Result<()> {
    const GAP: u32 = 8;
    const BACKGROUND: image::Rgba<u8> = image::Rgba([11, 13, 18, 255]);

    let mut rows: Vec<Vec<&Path>> = Vec::new();
    let mut last_world = "";
    for (world, path) in images {
        if *world != last_world || rows.last().is_some_and(|r| r.len() == cols) {
            rows.push(Vec::new());
            last_world = world;
        }
        if let Some(row) = rows.last_mut() {
            row.push(path);
        }
    }

    let first = image::open(&images[0].1).with_context(|| format!("reading {}", images[0].1.display()))?;
    let divisor = divisor.max(1);
    let (tw, th) = ((first.width() / divisor).max(1), (first.height() / divisor).max(1));
    let width = cols as u32 * tw + (cols as u32 + 1) * GAP;
    let height = rows.len() as u32 * th + (rows.len() as u32 + 1) * GAP;
    let mut sheet = image::RgbaImage::from_pixel(width, height, BACKGROUND);
    for (r, row) in rows.iter().enumerate() {
        for (c, path) in row.iter().enumerate() {
            let img = image::open(path).with_context(|| format!("reading {}", path.display()))?.to_rgba8();
            let thumb = image::imageops::resize(&img, tw, th, image::imageops::FilterType::Triangle);
            let x = GAP + c as u32 * (tw + GAP);
            let y = GAP + r as u32 * (th + GAP);
            image::imageops::replace(&mut sheet, &thumb, i64::from(x), i64::from(y));
        }
    }
    capture::save_png(out, [width, height], sheet.into_raw())
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
        assert_eq!((WORLDS[plan.world].id, plan.preset, plan.size), ("reaction-diffusion", Some(1), [1920, 1080]));
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
