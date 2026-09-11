//! Offscreen rendering: final-frame PNGs, frame sequences, videos (via ffmpeg)
//! and a gallery of every preset.
//!
//! Headless renders call `step` and `render` for every frame at a fixed 60 fps
//! timestep, exactly like the interactive app, so worlds that accumulate
//! display-only state in `render` (e.g. motion trails) look the same.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Instant;

use anyhow::{anyhow, bail, Context as _, Result};

use crate::capture::{self, Readback};
use crate::gpu::Gpu;
use crate::post::{Post, Tonemap};
use crate::world::{self, Camera, Frame, Pointer, ViewXform, WORLDS};

/// Headless output format: 8-bit sRGB so readback bytes can be written straight to PNG.
const OUTPUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

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
    /// Video (anything ffmpeg understands; .mp4/.mov/.mkv get H.264) of every frame.
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

pub fn render(job: &RenderJob) -> Result<()> {
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None))?;
    render_with(&gpu, job).map(|_| ())
}

/// Renders `job`; returns the path of the PNG written, if any.
pub fn render_with(gpu: &Gpu, job: &RenderJob) -> Result<Option<PathBuf>> {
    // yuv420p video needs even dimensions; stills keep the exact size.
    let size = if job.video.is_some() {
        [job.size[0].max(2) & !1, job.size[1].max(2) & !1]
    } else {
        [job.size[0].max(1), job.size[1].max(1)]
    };
    let max = gpu.device.limits().max_texture_dimension_2d;
    if size[0] > max || size[1] > max {
        bail!("{}x{} exceeds this GPU's maximum texture size of {max}", size[0], size[1]);
    }

    let (_, mut world) = world::create(gpu, &job.world, size, job.preset.as_deref(), job.seed)?;
    let preset_name = world.presets().get(world.preset()).copied().unwrap_or("custom");
    let out = match (&job.out, &job.video) {
        (Some(path), _) => Some(path.clone()),
        (None, None) => {
            Some(PathBuf::from("renders").join(format!("{}-{}-s{}.png", world.id(), slug(preset_name), job.seed)))
        }
        (None, Some(_)) => None,
    };
    if let Some(path) = &out {
        if !path.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")) {
            bail!("output image must be a .png file (got '{}')", path.display());
        }
    }
    let progress_level = if job.quiet { log::Level::Debug } else { log::Level::Info };
    log::log!(
        progress_level,
        "rendering {} / {} at {}x{} for {} frames (seed {})",
        world.name(),
        preset_name,
        size[0],
        size[1],
        job.frames,
        job.seed
    );

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
        Some(path) => Some(spawn_ffmpeg(path, size, job.fps, Encode::Quality)?),
        None => None,
    };

    let dt = 1.0 / job.fps.max(1) as f32;
    let frames = job.frames.max(1);
    let started = Instant::now();
    let mut next_report = 0.1;
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
        world.render(&frame, &mut encoder, post.scene_view());

        let last = f + 1 == frames;
        let save_frame = job.every > 0 && (f + 1) % job.every == 0;
        let capture = last || save_frame || ffmpeg.is_some();
        if capture {
            post.run(gpu, &mut encoder, &look, frame.time, &out_view);
            readback.copy_from(&mut encoder, &out_texture);
        }
        gpu.queue.submit([encoder.finish()]);
        if let Some(problem) = gpu.fatal_error() {
            bail!("GPU error while rendering: {problem}");
        }

        if capture {
            let pixels = readback.read(gpu)?;
            if let Some(child) = ffmpeg.as_mut() {
                let written = match child.stdin.as_mut() {
                    Some(stdin) => stdin.write_all(&pixels),
                    None => Err(std::io::Error::other("stdin closed")),
                };
                if let Err(e) = written {
                    return Err(ffmpeg_failure(ffmpeg.take().expect("ffmpeg is running"), e));
                }
            }
            if save_frame {
                let path = job.frames_dir.join(format!("{}_{:05}.png", world.id(), f + 1));
                capture::save_png(&path, size, pixels.clone())?;
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
        if job.max_fps > 0.0 {
            // Pace submissions so the GPU idles between frames instead of running flat out.
            let due = started + std::time::Duration::from_secs_f32((f + 1) as f32 / job.max_fps);
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

    if let Some(mut child) = ffmpeg {
        drop(child.stdin.take());
        let status = child.wait().context("waiting for ffmpeg")?;
        if !status.success() {
            bail!("ffmpeg exited with {status}");
        }
        log::info!("wrote {}", job.video.as_ref().map(|p| p.display().to_string()).unwrap_or_default());
    }
    let secs = started.elapsed().as_secs_f32();
    log::log!(progress_level, "done: {frames} frames in {secs:.1}s ({:.1} fps)", frames as f32 / secs.max(1e-3));
    Ok(out)
}

pub struct GalleryJob {
    pub out_dir: PathBuf,
    pub size: [u32; 2],
    pub frames: u32,
    pub seed: u64,
    /// Restrict to one world (id, name or alias).
    pub world: Option<String>,
    /// Also write `contact-sheet.png`, every image tiled into one overview.
    pub sheet: bool,
    /// Frames-per-second ceiling per render (0 = unlimited).
    pub max_fps: f32,
}

/// Renders the final frame of every preset of every world into `out_dir`.
pub fn gallery(job: &GalleryJob) -> Result<()> {
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None))?;
    let only = match &job.world {
        Some(q) => Some(world::find(q).with_context(|| format!("unknown world '{q}'"))?),
        None => None,
    };
    let mut todo = Vec::new();
    for (index, entry) in WORLDS.iter().enumerate() {
        if only.is_some_and(|o| o != index) {
            continue;
        }
        let presets = (entry.create)(&gpu, [64, 64], job.seed).presets();
        todo.extend(presets.iter().enumerate().map(|(i, name)| (entry, i, *name)));
    }

    let total = todo.len();
    let started = Instant::now();
    let mut rendered: Vec<(&'static str, PathBuf)> = Vec::with_capacity(total);
    for (n, (entry, i, name)) in todo.into_iter().enumerate() {
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
        rendered.push((entry.id, out));
    }
    if job.sheet && !rendered.is_empty() {
        let sheet = job.out_dir.join("contact-sheet.png");
        contact_sheet(&rendered, &sheet, 5)?;
        log::info!("wrote {}", sheet.display());
    }
    log::info!("gallery done: {total} images in {:.1}s", started.elapsed().as_secs_f32());
    Ok(())
}

/// Tiles gallery images (quarter size) into one overview image. Each world
/// starts a new row; rows hold at most `cols` images.
fn contact_sheet(images: &[(&str, PathBuf)], out: &Path, cols: usize) -> Result<()> {
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
    let (tw, th) = ((first.width() / 4).max(1), (first.height() / 4).max(1));
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
            std::fs::create_dir_all(parent)?;
        }
    }
    let exe = std::env::var("PRIMORDIA_FFMPEG").unwrap_or_else(|_| "ffmpeg".to_string());
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
    cmd.spawn()
        .with_context(|| format!("failed to start '{exe}' - is ffmpeg installed? (set PRIMORDIA_FFMPEG to its path)"))
}

/// Turns a failed write into an error that includes ffmpeg's exit status.
pub fn ffmpeg_failure(mut child: Child, err: std::io::Error) -> anyhow::Error {
    drop(child.stdin.take());
    match child.wait() {
        Ok(status) => anyhow!("ffmpeg exited with {status} while encoding ({err})"),
        Err(wait_err) => anyhow!("ffmpeg failed while encoding ({err}; {wait_err})"),
    }
}
