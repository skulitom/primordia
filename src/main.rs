//! Primordia — a GPU artificial-life laboratory.

mod app;
mod capture;
mod explore;
mod gpu;
mod headless;
mod library;
mod metrics;
mod palette;
mod post;
mod rng;
mod selftest;
mod stall;
mod ui;
mod world;

use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "primordia",
    version,
    about = "GPU artificial-life laboratory: slime moulds, particle life, Lenia, reaction-diffusion and Symbiosis",
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    run: RunArgs,
}

#[derive(Subcommand)]
// Parsed once at startup: the size of the render arguments is irrelevant.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Render a world offscreen to a PNG (and optionally a video via ffmpeg)
    Render(RenderArgs),
    /// Render the final frame of every preset into a folder
    Gallery(GalleryArgs),
    /// Search a world's mutations for the most novel behaviour and keep them as recipes
    Explore(ExploreArgs),
    /// List worlds, presets and palettes
    List,
    /// Verify on this GPU the shader maths the simulations rely on
    Selftest,
}

#[derive(Args)]
struct RunArgs {
    /// World to open: physarum, particle-life, lenia, reaction-diffusion, symbiosis (or 1-5)
    #[arg(short, long, default_value = "physarum")]
    world: String,
    /// Preset name or 1-based index
    #[arg(short, long)]
    preset: Option<String>,
    /// Random seed (default: time based)
    #[arg(long)]
    seed: Option<u64>,
    /// Window width in logical pixels [default: 1600, shrunk to fit the screen]
    #[arg(long)]
    width: Option<u32>,
    /// Window height in logical pixels [default: 900, shrunk to fit the screen]
    #[arg(long)]
    height: Option<u32>,
    /// Start in borderless fullscreen
    #[arg(long)]
    fullscreen: bool,
    /// Disable vsync (uncapped frame rate)
    #[arg(long)]
    no_vsync: bool,
    /// Simulation resolution relative to the window's pixel size
    #[arg(long, default_value_t = 1.0, value_parser = finite_float)]
    sim_scale: f32,
    /// Start with the control panel hidden (H or Tab shows it)
    #[arg(long)]
    hide_ui: bool,
    /// Quit automatically after this many seconds (useful for smoke tests)
    #[arg(long, value_name = "SECS", value_parser = finite_float)]
    exit_after: Option<f32>,
    /// Folder for screenshots and recordings
    #[arg(long, default_value = "screenshots")]
    screenshot_dir: PathBuf,
    /// Screensaver mode: fade to the next preset (then world) every SECS seconds
    #[arg(long, value_name = "SECS", value_parser = finite_float)]
    tour: Option<f32>,
    /// Start recording an MP4 (into --screenshot-dir) immediately; V toggles it
    #[arg(long)]
    record: bool,
}

#[derive(Clone, Copy, ValueEnum)]
enum TonemapArg {
    Agx,
    Aces,
    Reinhard,
    Linear,
}

impl From<TonemapArg> for post::Tonemap {
    fn from(t: TonemapArg) -> Self {
        match t {
            TonemapArg::Agx => post::Tonemap::Agx,
            TonemapArg::Aces => post::Tonemap::Aces,
            TonemapArg::Reinhard => post::Tonemap::Reinhard,
            TonemapArg::Linear => post::Tonemap::Linear,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum BrushArg {
    /// Left button: the world's create / attract / paint action
    Primary,
    /// Right button: the world's destroy / repel / erase action
    Secondary,
}

#[derive(Args)]
struct RenderArgs {
    /// World to render
    #[arg(short, long, default_value = "physarum")]
    world: String,
    /// Preset name or 1-based index [default: the world's first preset]
    #[arg(short, long)]
    preset: Option<String>,
    /// Random seed (the same seed always gives the same image)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Output width in pixels
    #[arg(long, default_value_t = 1920)]
    width: u32,
    /// Output height in pixels
    #[arg(long, default_value_t = 1080)]
    height: u32,
    /// Frames to simulate before the final image
    #[arg(short, long, default_value_t = 600)]
    frames: u32,
    /// Frame rate for simulated time and video
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=1000))]
    fps: u32,
    /// Output PNG of the final frame [default: renders/<world>-<preset>-s<seed>.png;
    /// skipped when only --video is given]
    #[arg(short, long)]
    out: Option<PathBuf>,
    /// Encode every frame into a video (e.g. clip.mp4) with ffmpeg
    #[arg(long)]
    video: Option<PathBuf>,
    /// Save a PNG every N frames into --frames-dir
    #[arg(long, default_value_t = 0)]
    every: u32,
    /// Folder for the --every frame PNGs
    #[arg(long, default_value = "frames")]
    frames_dir: PathBuf,
    /// Override the preset's exposure
    #[arg(long, value_parser = finite_float)]
    exposure: Option<f32>,
    /// Override the preset's bloom strength (0 disables bloom)
    #[arg(long, value_parser = finite_float)]
    bloom: Option<f32>,
    /// Override the preset's bloom threshold
    #[arg(long, value_parser = finite_float)]
    bloom_threshold: Option<f32>,
    /// Override the preset's tonemapper
    #[arg(long, value_enum)]
    tonemap: Option<TonemapArg>,
    /// Camera zoom (1 = whole world, >1 = close-up, <1 = show the torus tiling)
    #[arg(long, default_value_t = 1.0, value_parser = finite_float)]
    zoom: f32,
    /// Camera centre in world uv, e.g. --center 0.25,0.5
    #[arg(long, value_delimiter = ',', default_values_t = [0.5, 0.5], allow_negative_numbers = true, value_parser = finite_float)]
    center: Vec<f32>,
    /// Hold a scripted mouse button for the whole render (tests interaction)
    #[arg(long, value_enum)]
    brush: Option<BrushArg>,
    /// Where to hold the brush, in world uv (e.g. 0.3,0.6) [default: orbit the centre]
    #[arg(long, value_delimiter = ',', allow_negative_numbers = true, value_parser = finite_float)]
    brush_at: Vec<f32>,
    /// Brush radius in world cells
    #[arg(long, default_value_t = 40.0, value_parser = finite_float)]
    brush_radius: f32,
    /// Frame-rate ceiling that keeps the GPU from running flat out (0 = unlimited)
    #[arg(long, default_value_t = headless::DEFAULT_MAX_FPS, value_parser = finite_float)]
    max_fps: f32,
    /// Write every frame's measurements to a CSV file (frame, time, series, one column per metric)
    #[arg(long, value_name = "PATH")]
    metrics: Option<PathBuf>,
}

#[derive(Args)]
struct GalleryArgs {
    /// Folder to write the images into
    #[arg(short, long, default_value = "gallery")]
    out_dir: PathBuf,
    /// Only this world
    #[arg(short, long)]
    world: Option<String>,
    /// Image width in pixels
    #[arg(long, default_value_t = 1280)]
    width: u32,
    /// Image height in pixels
    #[arg(long, default_value_t = 720)]
    height: u32,
    /// Frames to simulate per preset
    #[arg(short, long, default_value_t = 600)]
    frames: u32,
    /// Random seed used for every preset
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Don't write contact-sheet.png (all images tiled into one overview)
    #[arg(long)]
    no_sheet: bool,
    /// Frame-rate ceiling per render, keeps the GPU from running flat out (0 = unlimited)
    #[arg(long, default_value_t = headless::DEFAULT_MAX_FPS, value_parser = finite_float)]
    max_fps: f32,
}

#[derive(Args)]
struct ExploreArgs {
    /// World to explore
    #[arg(short, long, default_value = "physarum")]
    world: String,
    /// Preset to start from (mutations of some worlds keep parts of it)
    #[arg(short, long)]
    preset: Option<String>,
    /// Master seed: every candidate seed and perturbation follows from it
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Mutations to evaluate in the first round (the preset itself is evaluated too)
    #[arg(long, default_value_t = 48, value_parser = clap::value_parser!(u32).range(1..))]
    runs: u32,
    /// Refinement rounds, each perturbing recipes of the current archive
    #[arg(long, default_value_t = 1)]
    refine: u32,
    /// Children to evaluate per refinement round
    #[arg(long, default_value_t = 24, value_parser = clap::value_parser!(u32).range(1..))]
    children: u32,
    /// Relative size of a perturbation (log-normal noise on the recipe's numbers)
    #[arg(long, default_value_t = 0.15, value_parser = finite_float)]
    strength: f32,
    /// Candidates to keep
    #[arg(long, default_value_t = 12, value_parser = clap::value_parser!(u32).range(1..))]
    keep: u32,
    /// Frames to simulate per candidate
    #[arg(short, long, default_value_t = 600)]
    frames: u32,
    /// Image width in pixels
    #[arg(long, default_value_t = 640)]
    width: u32,
    /// Image height in pixels
    #[arg(long, default_value_t = 360)]
    height: u32,
    /// Frame-rate ceiling per candidate, keeps the GPU from running flat out (0 = unlimited)
    #[arg(long, default_value_t = headless::DEFAULT_MAX_FPS, value_parser = finite_float)]
    max_fps: f32,
    /// Folder for the images, contact sheet, candidates.csv and recipes/
    #[arg(short, long, default_value = "explore")]
    out_dir: PathBuf,
    /// What to keep: "novelty" (mutually most different), "max:<metric>" or "min:<metric>"
    #[arg(long, default_value = "novelty")]
    select: explore::Select,
    /// Also save the kept recipes into the app's library
    #[arg(long)]
    install: bool,
    /// Don't write contact-sheet.png
    #[arg(long)]
    no_sheet: bool,
    /// Also write every evaluated candidate's final frame under all/
    #[arg(long)]
    all: bool,
    /// Let dead, empty or frozen candidates into the archive
    #[arg(long)]
    keep_inert: bool,
}

fn finite_float(value: &str) -> std::result::Result<f32, String> {
    let number = value.parse::<f32>().map_err(|e| e.to_string())?;
    if !number.is_finite() {
        return Err("expected a finite number (NaN and infinity are not supported)".to_string());
    }
    Ok(number)
}

fn main() {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("primordia=info,wgpu_core=warn,wgpu_hal=error,naga=warn"),
    )
    .format_timestamp(None)
    .format_target(false)
    .init();

    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        None => {
            let r = cli.run;
            let window_size = match (r.width, r.height) {
                (None, None) => None,
                (w, h) => Some([w.unwrap_or(1600), h.unwrap_or(900)]),
            };
            app::run(app::AppOptions {
                world: r.world,
                preset: r.preset,
                seed: r.seed,
                window_size,
                fullscreen: r.fullscreen,
                vsync: !r.no_vsync,
                sim_scale: r.sim_scale,
                hide_ui: r.hide_ui,
                exit_after: r.exit_after,
                screenshot_dir: r.screenshot_dir,
                tour: r.tour,
                record: r.record,
            })
        }
        Some(Command::Render(r)) => {
            if r.center.len() != 2 {
                bail!("--center expects two comma-separated numbers, e.g. --center 0.25,0.5");
            }
            if !r.brush_at.is_empty() && r.brush_at.len() != 2 {
                bail!("--brush-at expects two comma-separated numbers, e.g. --brush-at 0.3,0.6");
            }
            let brush = r.brush.map(|b| headless::Brush {
                secondary: matches!(b, BrushArg::Secondary),
                at: (r.brush_at.len() == 2).then(|| [r.brush_at[0], r.brush_at[1]]),
                radius: r.brush_radius.max(1.0),
            });
            headless::render(&headless::RenderJob {
                world: r.world,
                preset: r.preset,
                seed: r.seed,
                size: [r.width, r.height],
                frames: r.frames,
                fps: r.fps,
                out: r.out,
                video: r.video,
                every: r.every,
                frames_dir: r.frames_dir,
                exposure: r.exposure,
                bloom: r.bloom,
                bloom_threshold: r.bloom_threshold,
                tonemap: r.tonemap.map(Into::into),
                camera: world::Camera { center: [r.center[0], r.center[1]], zoom: r.zoom.clamp(0.05, 256.0) },
                brush,
                max_fps: r.max_fps,
                quiet: false,
                metrics: r.metrics,
            })
        }
        Some(Command::Gallery(g)) => headless::gallery(&headless::GalleryJob {
            out_dir: g.out_dir,
            size: [g.width, g.height],
            frames: g.frames,
            seed: g.seed,
            world: g.world,
            sheet: !g.no_sheet,
            max_fps: g.max_fps,
        }),
        Some(Command::Explore(e)) => {
            let mut job = explore::ExploreJob::new(&e.world);
            job.preset = e.preset;
            job.seed = e.seed;
            job.runs = e.runs;
            job.refine = e.refine;
            job.children = e.children;
            job.strength = e.strength;
            job.keep = e.keep as usize;
            job.frames = e.frames;
            job.size = [e.width, e.height];
            job.max_fps = e.max_fps;
            job.out_dir = e.out_dir;
            job.select = e.select;
            job.library = e.install.then(library::default_directory);
            job.sheet = !e.no_sheet;
            job.all = e.all;
            job.inert = e.keep_inert;
            explore::explore(&job)
        }
        Some(Command::List) => list(),
        Some(Command::Selftest) => selftest::run(),
    }
}

fn list() -> Result<()> {
    let gpu = pollster::block_on(gpu::Gpu::new(gpu::Gpu::create_instance(), None))?;
    println!("Worlds (GPU: {}):", gpu.adapter_name());
    for (i, entry) in world::WORLDS.iter().enumerate() {
        let w = (entry.create)(&gpu, [64, 64], 1);
        println!("\n  {}. {} ({})\n     {}", i + 1, entry.name, entry.id, entry.tagline);
        for (j, p) in w.presets().iter().enumerate() {
            println!("       {:>2}. {}", j + 1, p);
        }
    }
    println!("\nPalettes:");
    for p in palette::PALETTES {
        println!("  {}", p.name);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_rejects_nonfinite_numbers_in_every_float_option() {
        for invalid in ["NaN", "inf", "-inf", "1e100"] {
            for flag in ["--sim-scale", "--exit-after", "--tour"] {
                assert!(Cli::try_parse_from(["primordia", &format!("{flag}={invalid}")]).is_err());
            }
            for flag in ["--exposure", "--bloom", "--bloom-threshold", "--zoom", "--brush-radius", "--max-fps"] {
                assert!(Cli::try_parse_from(["primordia", "render", &format!("{flag}={invalid}")]).is_err());
            }
            for flag in ["--center", "--brush-at"] {
                assert!(Cli::try_parse_from(["primordia", "render", &format!("{flag}=0.5,{invalid}")]).is_err());
            }
            assert!(Cli::try_parse_from(["primordia", "gallery", &format!("--max-fps={invalid}")]).is_err());
            for flag in ["--strength", "--max-fps"] {
                assert!(Cli::try_parse_from(["primordia", "explore", &format!("{flag}={invalid}")]).is_err());
            }
        }
    }

    #[test]
    fn cli_parses_an_exploration_and_rejects_bad_selections() {
        let cli = Cli::try_parse_from([
            "primordia", "explore", "--world", "symbiosis", "--select", "max:growth_cover", "--install", "--keep", "4",
            "--runs", "8", "--no-sheet", "--all",
        ])
        .unwrap();
        let Some(Command::Explore(args)) = cli.command else { panic!("expected explore") };
        assert_eq!(args.world, "symbiosis");
        assert_eq!(args.select, explore::Select::Max("growth_cover".into()));
        assert!(args.install && args.no_sheet && args.all);
        assert_eq!((args.keep, args.runs, args.refine, args.children), (4, 8, 1, 24));
        assert_eq!(args.strength, 0.15);
        assert_eq!(args.out_dir, PathBuf::from("explore"));
        assert!(Cli::try_parse_from(["primordia", "explore", "--select", "best"]).is_err());
        assert!(Cli::try_parse_from(["primordia", "explore", "--keep", "0"]).is_err());
        assert!(Cli::try_parse_from(["primordia", "explore", "--runs", "0"]).is_err());
    }

    #[test]
    fn cli_accepts_finite_coordinates_and_unlimited_rendering() {
        assert!(Cli::try_parse_from(["primordia"]).is_ok());
        let cli = Cli::try_parse_from([
            "primordia", "render", "--center=-0.25,1.5", "--brush-at=2,-1", "--max-fps=0", "--bloom=0",
        ]).unwrap();
        let Some(Command::Render(args)) = cli.command else { panic!("expected render") };
        assert_eq!(args.center, [-0.25, 1.5]);
        assert_eq!(args.brush_at, [2.0, -1.0]);
        assert_eq!(args.max_fps, 0.0);
        assert_eq!(args.bloom, Some(0.0));
        assert_eq!(args.metrics, None);
    }

    #[test]
    fn cli_accepts_a_measurement_log_path() {
        let cli = Cli::try_parse_from(["primordia", "render", "--metrics", "runs/reef.csv"]).unwrap();
        let Some(Command::Render(args)) = cli.command else { panic!("expected render") };
        assert_eq!(args.metrics, Some(PathBuf::from("runs/reef.csv")));
        assert!(Cli::try_parse_from(["primordia", "render", "--metrics"]).is_err());
    }
}
