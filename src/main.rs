//! Primordia — a GPU artificial-life laboratory.

mod app;
mod capture;
mod explore;
mod failure;
mod gpu;
mod headless;
mod library;
mod metrics;
mod palette;
mod post;
mod report;
mod rng;
mod selftest;
mod stall;
mod ui;
mod world;

use std::ffi::OsString;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::{Args, Parser, Subcommand, ValueEnum};

use failure::Failure;
use world::WORLDS;

const ABOUT: &str = "GPU artificial-life laboratory: slime moulds, particle life, Lenia, reaction-diffusion and Symbiosis";

const LONG_ABOUT: &str = "\
GPU artificial-life laboratory: slime moulds, particle life, Lenia, reaction-diffusion and Symbiosis.

Without a command, primordia opens the interactive window and blocks until it is closed; add --exit-after SECS \
to quit on its own (smoke tests, scripts). The commands list, render, gallery, explore and selftest run headless \
and exit when they are done.";

const EXAMPLES: &str = "\
Examples:
  primordia --world lenia --preset necklaces           open a world (blocks until the window is closed)
  primordia list --world rd                            presets, measurements and palettes of one world
  primordia render -w rd -p mitosis -o mitosis.png     render a still
  primordia render -w physarum --video network.mp4     render a video (needs ffmpeg)
  primordia gallery -o gallery                         every preset of every world, plus a contact sheet
  primordia explore -w symbiosis --runs 24 --json      search for novel behaviour and keep recipes";

const OUTPUT: &str = "\
Output:
  Logs and progress go to stderr (-q: warnings and errors only, -v: debug). stdout carries results only: the
  path of every file written, one per line, or with --json one JSON object at the end: {\"ok\":true,
  \"command\":\"render\",...} on success, {\"ok\":false,\"error\":{\"kind\":\"usage|gpu|ffmpeg|io|other\",
  \"message\":...}} on failure. Seeds are JSON strings. `primordia list --json` describes every world.";

const EXIT_STATUS: &str = "\
Exit status:
  0  success
  1  another failure (for example reading or writing a file)
  2  invalid input: bad arguments, an unknown or ambiguous world, preset or measurement, a size out of
     range, an output path that cannot be written or has the wrong extension
  3  no usable GPU, or the GPU failed during the run
  4  ffmpeg is missing or failed";

/// Short help for every `--world` option.
const WORLD_HELP: &str = "World: physarum, particle-life, lenia, reaction-diffusion, symbiosis, 1-5, an alias \
    or a unique prefix (see `primordia list`)";

const WORLD_LONG_HELP: &str = "\
World to use: an id, a number or an alias, ignoring case and punctuation; a prefix works when only one id starts \
with it (\"re\" = reaction-diffusion).
  1  physarum            slime, slime-mold, slime-mould, mold, mould
  2  particle-life       particles, particlelife, pl, life
  3  lenia               smoothlife, continuous-ca
  4  reaction-diffusion  rd, gray-scott, grayscott, turing, coral
  5  symbiosis           coupled, hybrid, ecosystem";

const PRESET_HELP: &str = "Preset: a name, a unique prefix or a number from 1 (see `primordia list --world W`)";

/// Every subcommand's long help ends with this pointer.
const MORE_HELP: &str = "Output, environment variables and exit status: see `primordia --help`.";

/// Everything after the options in `primordia --help`.
fn after_long_help() -> String {
    format!(
        "{EXAMPLES}\n\n{OUTPUT}\n\n\
Environment:
  RUST_LOG               Log filter [default: primordia=info,wgpu_core=error,wgpu_hal=error,naga=warn]
  WGPU_BACKEND           GPU backends to try: vulkan, dx12, metal or gl (comma-separated)
  WGPU_POWER_PREF        Which GPU to prefer when there are several: high (default) or low
  WGPU_ADAPTER_NAME      Use the GPU whose name contains this text
  PRIMORDIA_FFMPEG       ffmpeg executable for --video, --record and V [default: ffmpeg on the PATH]
  PRIMORDIA_LIBRARY_DIR  Folder of saved recipes [now: {}]

{EXIT_STATUS}",
        library::default_directory().display()
    )
}

fn install_help() -> String {
    format!(
        "Also save the kept recipes into the app's library ({}; PRIMORDIA_LIBRARY_DIR overrides)",
        library::default_directory().display()
    )
}

#[derive(Parser)]
#[command(
    name = "primordia",
    bin_name = "primordia",
    version,
    about = ABOUT,
    long_about = LONG_ABOUT,
    override_usage = "primordia [OPTIONS]\n       primordia [--json] [-q | -v] <COMMAND> [ARGS]",
    after_help = EXAMPLES,
    after_long_help = after_long_help()
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    #[command(flatten)]
    run: RunArgs,
    #[command(flatten)]
    global: GlobalArgs,
}

#[derive(Args)]
#[command(next_help_heading = "Global options")]
struct GlobalArgs {
    /// Print one JSON object with the result (or the error) on stdout
    #[arg(long, global = true)]
    json: bool,
    /// Only print warnings and errors on stderr
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    quiet: bool,
    /// Also print debug messages on stderr (-vv: trace)
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Subcommand)]
// Parsed once at startup: the size of the render arguments is irrelevant.
#[allow(clippy::large_enum_variant)]
enum Command {
    /// List worlds with their presets, measurements and palettes (no GPU needed)
    #[command(after_long_help = MORE_HELP)]
    List(ListArgs),
    /// Render a world offscreen to a PNG (and optionally a video via ffmpeg)
    #[command(after_help = RENDER_EXAMPLES, after_long_help = concat!(
        "Examples:\n",
        "  primordia render -w rd -p mitosis -o mitosis.png\n",
        "  primordia render -w lenia -p 5 --frames 1200 --video necklaces.mp4 --metrics necklaces.csv\n",
        "  primordia render -w physarum --zoom 3 --center 0.25,0.5 --json\n\n",
        "Output, environment variables and exit status: see `primordia --help`."
    ))]
    Render(RenderArgs),
    /// Render the final frame of every preset into a folder
    #[command(after_long_help = MORE_HELP)]
    Gallery(GalleryArgs),
    /// Search a world's mutations for the most novel behaviour and keep them as recipes
    #[command(after_long_help = concat!(
        "candidates.csv has one row per candidate: index, round, origin, parent, seed, preset (from 1, like --preset), ",
        "preset_name, rank, novelty, status, secs, then three columns per measurement: <id>_mean (mean of the last ",
        "40% of the frames), <id>_std (its standard deviation) and <id>_drift (that mean minus the mean of the first ",
        "20%).\n\n",
        "Output, environment variables and exit status: see `primordia --help`."
    ))]
    Explore(ExploreArgs),
    /// Verify on this GPU the shader maths the simulations rely on
    #[command(after_long_help = MORE_HELP)]
    Selftest,
}

const RENDER_EXAMPLES: &str = "\
Examples:
  primordia render -w rd -p mitosis -o mitosis.png
  primordia render -w lenia -p 5 --frames 1200 --video necklaces.mp4 --metrics necklaces.csv";

#[derive(Args)]
struct RunArgs {
    #[arg(short, long, default_value = "physarum", help = WORLD_HELP, long_help = WORLD_LONG_HELP)]
    world: String,
    #[arg(short, long, help = PRESET_HELP)]
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
    /// Start recording an MP4 (into --screenshot-dir) immediately; V toggles it (needs ffmpeg)
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

/// The `--tonemap` names, for `list`.
fn tonemap_names() -> Vec<String> {
    TonemapArg::value_variants().iter().filter_map(|t| t.to_possible_value()).map(|v| v.get_name().to_string()).collect()
}

#[derive(Clone, Copy, ValueEnum)]
enum BrushArg {
    /// Left button: the world's create / attract / paint action
    Primary,
    /// Right button: the world's destroy / repel / erase action
    Secondary,
}

/// `16..=16384` pixels, the sizes every command (and a saved recipe) supports.
fn image_side() -> clap::builder::RangedI64ValueParser<u32> {
    clap::value_parser!(u32).range(i64::from(headless::MIN_SIZE)..=i64::from(headless::MAX_SIZE))
}

#[derive(Args)]
struct ListArgs {
    /// Only this world (an id, 1-5, an alias or a unique prefix)
    #[arg(short, long)]
    world: Option<String>,
}

#[derive(Args)]
struct RenderArgs {
    #[arg(short, long, default_value = "physarum", help = WORLD_HELP, long_help = WORLD_LONG_HELP)]
    world: String,
    /// Preset: a name, a unique prefix or a number from 1 [default: the world's first preset]
    #[arg(short, long)]
    preset: Option<String>,
    /// Random seed (the same seed always gives the same image on the same GPU)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Output width in pixels (16-16384; a video rounds it down to an even number)
    #[arg(long, default_value_t = 1920, value_parser = image_side())]
    width: u32,
    /// Output height in pixels (16-16384; a video rounds it down to an even number)
    #[arg(long, default_value_t = 1080, value_parser = image_side())]
    height: u32,
    /// Frames to simulate before the final image
    #[arg(short, long, default_value_t = 600, value_parser = at_least_one)]
    frames: u32,
    /// Frames per simulated second: sets the time step (1/fps) and the video frame rate
    #[arg(long, default_value_t = 60, value_parser = clap::value_parser!(u32).range(1..=1000))]
    fps: u32,
    /// Output PNG of the final frame [default: renders/<world>-<preset>-s<seed>.png;
    /// skipped when only --video is given]
    #[arg(short, long)]
    out: Option<PathBuf>,
    /// Encode every frame into a video with ffmpeg: .mp4, .mov, .mkv (H.264), .webm or .gif
    #[arg(long)]
    video: Option<PathBuf>,
    /// Save a PNG every N frames into --frames-dir (0 = off)
    #[arg(long, value_name = "N", default_value_t = 0)]
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
    /// Camera zoom, 0.05-256 (1 = whole world, >1 = close-up, <1 = show the torus tiling)
    #[arg(long, default_value_t = 1.0, value_parser = zoom_factor)]
    zoom: f32,
    /// Camera centre in world uv, as X,Y
    #[arg(long, value_name = "X,Y", value_delimiter = ',', default_value = "0.5,0.5", allow_negative_numbers = true, value_parser = finite_float)]
    center: Vec<f32>,
    /// Hold a scripted mouse button for the whole render (tests interaction)
    #[arg(long, value_enum)]
    brush: Option<BrushArg>,
    /// Where to hold the brush, in world uv (e.g. 0.3,0.6) [default: orbit the centre]
    #[arg(long, value_name = "X,Y", value_delimiter = ',', allow_negative_numbers = true, value_parser = finite_float)]
    brush_at: Vec<f32>,
    /// Brush radius in world cells
    #[arg(long, default_value_t = 40.0, value_parser = finite_float)]
    brush_radius: f32,
    /// Frame-rate ceiling that keeps the GPU from running flat out (0 = unlimited)
    #[arg(long, default_value_t = headless::DEFAULT_MAX_FPS, value_parser = finite_float)]
    max_fps: f32,
    /// Write every frame's measurements to a CSV file: frame, time, series, then one column per measurement
    /// (ids, units and meanings: `primordia list --world W`)
    #[arg(long, value_name = "PATH")]
    metrics: Option<PathBuf>,
}

#[derive(Args)]
struct GalleryArgs {
    /// Folder to write the images into
    #[arg(short, long, default_value = "gallery")]
    out_dir: PathBuf,
    /// Only this world (an id, 1-5, an alias or a unique prefix) [default: every world]
    #[arg(short, long)]
    world: Option<String>,
    /// Image width in pixels (16-16384)
    #[arg(long, default_value_t = 1280, value_parser = image_side())]
    width: u32,
    /// Image height in pixels (16-16384)
    #[arg(long, default_value_t = 720, value_parser = image_side())]
    height: u32,
    /// Frames to simulate per preset
    #[arg(short, long, default_value_t = 600, value_parser = at_least_one)]
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
    #[arg(short, long, default_value = "physarum", help = WORLD_HELP, long_help = WORLD_LONG_HELP)]
    world: String,
    /// Preset to start from, a name or a number from 1 (mutations of some worlds keep parts of it)
    #[arg(short, long)]
    preset: Option<String>,
    /// Master seed: every candidate seed and perturbation follows from it
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Mutations to evaluate in the first round (the preset itself is evaluated too)
    #[arg(long, default_value_t = 48, value_parser = at_least_one)]
    runs: u32,
    /// Refinement rounds, each perturbing recipes of the current archive
    #[arg(long, default_value_t = 1)]
    refine: u32,
    /// Children to evaluate per refinement round
    #[arg(long, default_value_t = 24, value_parser = at_least_one)]
    children: u32,
    /// Relative size of a perturbation (log-normal noise on the recipe's numbers)
    #[arg(long, default_value_t = 0.15, value_parser = finite_float)]
    strength: f32,
    /// Candidates to keep
    #[arg(long, default_value_t = 12, value_parser = at_least_one)]
    keep: u32,
    /// Frames to simulate per candidate (at 60 frames per simulated second)
    #[arg(short, long, default_value_t = 600, value_parser = at_least_one)]
    frames: u32,
    /// Image width in pixels (16-16384)
    #[arg(long, default_value_t = 640, value_parser = image_side())]
    width: u32,
    /// Image height in pixels (16-16384)
    #[arg(long, default_value_t = 360, value_parser = image_side())]
    height: u32,
    /// Frame-rate ceiling per candidate, keeps the GPU from running flat out (0 = unlimited)
    #[arg(long, default_value_t = headless::DEFAULT_MAX_FPS, value_parser = finite_float)]
    max_fps: f32,
    /// Folder for the images, contact sheet, candidates.csv and recipes/
    #[arg(short, long, default_value = "explore")]
    out_dir: PathBuf,
    /// What to keep: "novelty" (mutually most different), "max:<metric>" or "min:<metric>"
    /// (metric ids: `primordia list --world W`)
    #[arg(long, default_value = "novelty")]
    select: explore::Select,
    #[arg(long, help = install_help())]
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

fn at_least_one(value: &str) -> std::result::Result<u32, String> {
    match value.parse::<u32>() {
        Ok(0) => Err("must be at least 1".to_string()),
        Ok(n) => Ok(n),
        Err(e) => Err(e.to_string()),
    }
}

fn zoom_factor(value: &str) -> std::result::Result<f32, String> {
    let zoom = finite_float(value)?;
    if !(0.05..=256.0).contains(&zoom) {
        return Err(format!("{zoom} is not in 0.05..=256"));
    }
    Ok(zoom)
}

/// Primordia's own messages at info; wgpu only when something is wrong (its
/// DX12 "missing downlevel flags" report is a multi-line warning nobody can act on).
const LOG_FILTER: &str = "primordia=info,wgpu_core=error,wgpu_hal=error,naga=warn";

/// Plain stderr logs: info lines as they are, then "warning: " and "error: "
/// prefixes, with the source named only for other crates' messages.
fn init_logging(quiet: bool, verbose: u8) {
    let mut builder = env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(LOG_FILTER));
    // -q and -v decide how much Primordia itself says, even when RUST_LOG is set.
    match (quiet, verbose) {
        (true, _) => builder.filter_module("primordia", log::LevelFilter::Warn),
        (false, 0) => &mut builder,
        (false, 1) => builder.filter_module("primordia", log::LevelFilter::Debug),
        (false, _) => builder.filter_module("primordia", log::LevelFilter::Trace),
    };
    builder
        .format(|buf, record| {
            let prefix = match record.level() {
                log::Level::Error => "error: ",
                log::Level::Warn => "warning: ",
                log::Level::Info => "",
                log::Level::Debug => "debug: ",
                log::Level::Trace => "trace: ",
            };
            let target = record.target();
            if target == "primordia" || target.starts_with("primordia::") {
                writeln!(buf, "{prefix}{}", record.args())
            } else {
                writeln!(buf, "{prefix}[{target}] {}", record.args())
            }
        })
        .init();
}

/// Parses the command line. The window options (`--world`, `--seed`, ...)
/// belong to the interactive app and are refused next to a command, while the
/// global options (`--json`, `-q`, `-v`) work before or after it.
fn parse<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    use clap::parser::ValueSource;
    use clap::{CommandFactory as _, FromArgMatches as _};
    let mut command = Cli::command();
    let matches = command.try_get_matches_from_mut(args)?;
    if let Some((name, _)) = matches.subcommand() {
        let window_options: Vec<String> = command
            .get_arguments()
            .filter(|arg| !arg.is_global_set())
            .filter(|arg| matches.value_source(arg.get_id().as_str()) == Some(ValueSource::CommandLine))
            .map(|arg| arg.get_long().map_or_else(|| arg.get_id().to_string(), |long| format!("--{long}")))
            .collect();
        if !window_options.is_empty() {
            let message = format!(
                "{} {} an option of the interactive window, not of `primordia {name}` (put the command's own \
                 options after `{name}`)",
                window_options.join(", "),
                if window_options.len() == 1 { "is" } else { "are each" }
            );
            return Err(command.error(clap::error::ErrorKind::ArgumentConflict, message));
        }
    }
    Cli::from_arg_matches(&matches).map_err(|e| e.format(&mut command))
}

fn main() {
    let args: Vec<OsString> = std::env::args_os().collect();
    let cli = match parse(&args) {
        Ok(cli) => cli,
        Err(error) => clap_failure(error, &args),
    };
    init_logging(cli.global.quiet, cli.global.verbose);
    let json = cli.global.json;
    let command = command_name(cli.command.as_ref());
    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        if json {
            print_stdout(&failure::json(command, &e).to_string());
        }
        std::process::exit(failure::exit_code(&e));
    }
}

fn command_name(command: Option<&Command>) -> &'static str {
    match command {
        None => "app",
        Some(Command::List(_)) => "list",
        Some(Command::Render(_)) => "render",
        Some(Command::Gallery(_)) => "gallery",
        Some(Command::Explore(_)) => "explore",
        Some(Command::Selftest) => "selftest",
    }
}

/// Reports an argument error like clap does (exit status 2), plus a JSON
/// object on stdout with `--json` and a hint when a world was typed as a command.
fn clap_failure(error: clap::Error, args: &[OsString]) -> ! {
    use clap::error::{ContextKind, ContextValue, ErrorKind};
    if matches!(error.kind(), ErrorKind::DisplayHelp | ErrorKind::DisplayVersion) {
        error.exit();
    }
    let _ = error.print();
    if let Some(ContextValue::String(name)) = error.get(ContextKind::InvalidSubcommand) {
        if world::find(name).is_some() {
            eprintln!("\ntip: to open a world, use `primordia --world {name}`");
        }
    }
    if args.iter().any(|a| a == "--json") {
        let rendered = error.render().to_string();
        // The error and its tips, without the usage line and the pointer to --help.
        let message: Vec<&str> = rendered
            .split("\n\n")
            .filter(|block| !block.starts_with("Usage:") && !block.starts_with("For more information"))
            .map(|block| block.trim().trim_start_matches("error: "))
            .collect();
        let command = args.iter().skip(1).filter_map(|a| a.to_str()).find(|a| {
            ["list", "render", "gallery", "explore", "selftest"].contains(a)
        });
        let value = serde_json::json!({
            "ok": false,
            "command": command,
            "error": { "kind": "usage", "message": message.join("\n"), "exit_code": 2 },
        });
        print_stdout(&value.to_string());
    }
    std::process::exit(2);
}

/// One line of results on stdout. A closed pipe (`primordia list | head`) is not an error.
fn print_stdout(line: &str) {
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{line}").and_then(|()| stdout.flush());
}

/// A headless command's result: its JSON object, or the files it wrote, one per line.
fn emit(json: bool, value: serde_json::Value, files: &[&Path]) {
    if json {
        print_stdout(&value.to_string());
    } else {
        for file in files {
            print_stdout(&file.display().to_string());
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    let json = cli.global.json;
    match cli.command {
        None => {
            if json {
                return Err(Failure::Usage.error(
                    "--json needs a command (list, render, gallery or explore); without one primordia opens its window",
                ));
            }
            let r = cli.run;
            // Resolve names before a window or GPU exists, so a typo fails at once.
            let world = world::resolve(&r.world)?;
            let preset = r.preset.as_deref().map(|p| world::resolve_preset(world, p)).transpose()?;
            if r.record {
                headless::probe_ffmpeg()?;
            }
            let window_size = match (r.width, r.height) {
                (None, None) => None,
                (w, h) => Some([w.unwrap_or(1600), h.unwrap_or(900)]),
            };
            app::run(app::AppOptions {
                world: WORLDS[world].id.to_string(),
                preset: preset.map(|p| (p + 1).to_string()),
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
                return Err(Failure::Usage.error("--center expects two comma-separated numbers, e.g. --center 0.25,0.5"));
            }
            if !r.brush_at.is_empty() && r.brush_at.len() != 2 {
                return Err(Failure::Usage.error("--brush-at expects two comma-separated numbers, e.g. --brush-at 0.3,0.6"));
            }
            let brush = r.brush.map(|b| headless::Brush {
                secondary: matches!(b, BrushArg::Secondary),
                at: (r.brush_at.len() == 2).then(|| [r.brush_at[0], r.brush_at[1]]),
                radius: r.brush_radius.max(1.0),
            });
            let job = headless::RenderJob {
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
                camera: world::Camera { center: [r.center[0], r.center[1]], zoom: r.zoom },
                brush,
                max_fps: r.max_fps,
                quiet: false,
                metrics: r.metrics,
            };
            let summary = headless::render(&job)?;
            emit(json, report::render(&summary, &job), &summary.files());
            Ok(())
        }
        Some(Command::Gallery(g)) => {
            let job = headless::GalleryJob {
                out_dir: g.out_dir,
                size: [g.width, g.height],
                frames: g.frames,
                seed: g.seed,
                world: g.world,
                sheet: !g.no_sheet,
                max_fps: g.max_fps,
            };
            let summary = headless::gallery(&job)?;
            emit(json, report::gallery(&summary, job.seed, job.size, job.frames), &summary.files());
            Ok(())
        }
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
            let summary = explore::explore(&job)?;
            emit(json, report::explore(&summary, &job), &summary.files());
            Ok(())
        }
        Some(Command::List(l)) => {
            let only = l.world.as_deref().map(world::resolve).transpose()?;
            let tonemaps = tonemap_names();
            let tonemaps: Vec<&str> = tonemaps.iter().map(String::as_str).collect();
            if json {
                print_stdout(&report::list_json(only, &tonemaps).to_string());
            } else {
                print_stdout(report::list_text(only, &tonemaps).trim_end());
            }
            Ok(())
        }
        Some(Command::Selftest) => {
            if json {
                return Err(Failure::Usage.error("selftest has no JSON output; its exit status says whether it passed"));
            }
            // Every selftest failure is the GPU's: no adapter, a device error or wrong maths.
            selftest::run().map_err(|e| Failure::Gpu.tag(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory as _;

    #[test]
    fn cli_rejects_nonfinite_numbers_in_every_float_option() {
        for invalid in ["NaN", "inf", "-inf", "1e100"] {
            for flag in ["--sim-scale", "--exit-after", "--tour"] {
                assert!(parse(["primordia", &format!("{flag}={invalid}")]).is_err());
            }
            for flag in ["--exposure", "--bloom", "--bloom-threshold", "--zoom", "--brush-radius", "--max-fps"] {
                assert!(parse(["primordia", "render", &format!("{flag}={invalid}")]).is_err());
            }
            for flag in ["--center", "--brush-at"] {
                assert!(parse(["primordia", "render", &format!("{flag}=0.5,{invalid}")]).is_err());
            }
            assert!(parse(["primordia", "gallery", &format!("--max-fps={invalid}")]).is_err());
            for flag in ["--strength", "--max-fps"] {
                assert!(parse(["primordia", "explore", &format!("{flag}={invalid}")]).is_err());
            }
        }
    }

    #[test]
    fn cli_parses_an_exploration_and_rejects_bad_selections() {
        let cli = parse([
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
        assert!(parse(["primordia", "explore", "--select", "best"]).is_err());
        assert!(parse(["primordia", "explore", "--keep", "0"]).is_err());
        assert!(parse(["primordia", "explore", "--runs", "0"]).is_err());
    }

    #[test]
    fn cli_accepts_finite_coordinates_and_unlimited_rendering() {
        assert!(parse(["primordia"]).is_ok());
        let cli = parse([
            "primordia", "render", "--center=-0.25,1.5", "--brush-at=2,-1", "--max-fps=0", "--bloom=0",
        ]).unwrap();
        let Some(Command::Render(args)) = cli.command else { panic!("expected render") };
        assert_eq!(args.center, [-0.25, 1.5]);
        assert_eq!(args.brush_at, [2.0, -1.0]);
        assert_eq!(args.max_fps, 0.0);
        assert_eq!(args.bloom, Some(0.0));
        assert_eq!(args.metrics, None);
        // The default centre is shown, and parsed, in the syntax the option takes.
        let Some(Command::Render(args)) = parse(["primordia", "render"]).unwrap().command else {
            panic!("expected render")
        };
        assert_eq!(args.center, [0.5, 0.5]);
        let help = Cli::command().find_subcommand_mut("render").unwrap().render_long_help().to_string();
        assert!(help.contains("[default: 0.5,0.5]"), "{help}");
    }

    #[test]
    fn cli_accepts_a_measurement_log_path() {
        let cli = parse(["primordia", "render", "--metrics", "runs/reef.csv"]).unwrap();
        let Some(Command::Render(args)) = cli.command else { panic!("expected render") };
        assert_eq!(args.metrics, Some(PathBuf::from("runs/reef.csv")));
        assert!(parse(["primordia", "render", "--metrics"]).is_err());
    }

    #[test]
    fn cli_rejects_sizes_frames_and_zoom_out_of_range_instead_of_clamping() {
        for command in ["render", "gallery", "explore"] {
            for (flag, value) in [("--width", "15"), ("--height", "0"), ("--width", "16385"), ("--frames", "0")] {
                let parsed = parse(["primordia", command, flag, value]);
                assert!(parsed.is_err(), "{command} {flag} {value}");
            }
            assert!(parse(["primordia", command, "--width", "16", "--height", "16384"]).is_ok());
        }
        for zoom in ["0.01", "300", "0"] {
            assert!(parse(["primordia", "render", "--zoom", zoom]).is_err(), "{zoom}");
        }
        let Some(Command::Render(args)) = parse(["primordia", "render", "--zoom", "256"]).unwrap().command
        else {
            panic!("expected render")
        };
        assert_eq!(args.zoom, 256.0);
    }

    #[test]
    fn global_flags_work_before_and_after_the_command() {
        for argv in [
            &["primordia", "--json", "-q", "render", "-w", "rd"][..],
            &["primordia", "render", "-w", "rd", "--json", "--quiet"],
            &["primordia", "-q", "list", "--json"],
        ] {
            let cli = parse(argv).unwrap_or_else(|e| panic!("{argv:?}: {e}"));
            assert!(cli.global.json && cli.global.quiet, "{argv:?}");
            assert!(cli.command.is_some(), "{argv:?}");
        }
        let cli = parse(["primordia", "explore", "-vv"]).unwrap();
        assert_eq!(cli.global.verbose, 2);
        assert!(parse(["primordia", "list", "-q", "-v"]).is_err(), "-q and -v conflict");
        // Window options still conflict with commands.
        assert!(parse(["primordia", "--world", "lenia", "render"]).is_err());
        let error = parse(["primordia", "--seed", "3", "--json", "render"]).err().expect("a window option").to_string();
        assert!(error.contains("--seed is an option of the interactive window, not of `primordia render`"), "{error}");
        let cli = parse(["primordia", "list", "--world", "rd"]).unwrap();
        let Some(Command::List(args)) = cli.command else { panic!("expected list") };
        assert_eq!(args.world.as_deref(), Some("rd"));
    }

    #[test]
    fn help_documents_the_blocking_window_environment_and_exit_status() {
        let help = Cli::command().render_long_help().to_string();
        assert!(help.contains("Usage: primordia [OPTIONS]"), "the usage line names `primordia`, not primordia.exe");
        for text in ["blocks until it is closed", "--exit-after", "Examples:", "Environment:", "Exit status:"] {
            assert!(help.contains(text), "missing {text:?}");
        }
        for var in ["RUST_LOG", "WGPU_BACKEND", "WGPU_POWER_PREF", "PRIMORDIA_FFMPEG", "PRIMORDIA_LIBRARY_DIR"] {
            assert!(help.contains(var), "missing {var}");
        }
        for code in ["  0  success", "  2  invalid input", "  3  no usable GPU", "  4  ffmpeg"] {
            assert!(help.contains(code), "missing {code:?}");
        }
        // Every --world help lists every id, and the long help every alias.
        for entry in WORLDS {
            assert!(WORLD_HELP.contains(entry.id), "{}", entry.id);
            for alias in entry.aliases {
                assert!(WORLD_LONG_HELP.contains(alias), "{alias}");
            }
        }
        let mut cli = Cli::command();
        for command in ["render", "explore"] {
            let help = cli.find_subcommand_mut(command).unwrap().render_long_help().to_string();
            assert!(help.contains("reaction-diffusion") && help.contains("gray-scott"), "{command}: {help}");
            assert!(help.contains("primordia list --world W"), "{command} points at the measurement list");
        }
        let explore = cli.find_subcommand_mut("explore").unwrap().render_long_help().to_string();
        assert!(explore.contains("PRIMORDIA_LIBRARY_DIR overrides") && explore.contains("<id>_drift"));
    }

    #[test]
    fn tonemap_names_match_the_option_values() {
        assert_eq!(tonemap_names(), ["agx", "aces", "reinhard", "linear"]);
    }
}
