//! Interactive window: winit event loop, wgpu surface, egui control panel.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, anyhow};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Fullscreen, Icon, Window, WindowAttributes, WindowId};

use crate::capture::{self, Readback};
use crate::gpu::Gpu;
use crate::headless::{self, Encode};
use crate::library::{Library, SavedWorld};
use crate::metrics;
use crate::post::{Post, PostSettings};
use crate::rng;
use crate::ui::{self as theme, ACCENT, Inspector, WARN};
use crate::world::{self, Camera, Frame, Pointer, ViewXform, WORLDS, World};

/// Default window size in logical pixels (shrunk to fit small screens).
const DEFAULT_WINDOW: [u32; 2] = [1600, 900];
/// Seconds of fade-out / fade-in around each tour transition.
const TOUR_FADE: f32 = 1.2;
/// Allowed seconds per preset in tour mode (CLI and slider share it).
const TOUR_RANGE: std::ops::RangeInclusive<f32> = 3.0..=600.0;
/// Frame rate of live recordings.
const RECORD_FPS: f32 = 60.0;
/// A second Esc within this many seconds quits.
const QUIT_CONFIRM_SECS: f32 = 1.5;
/// Simulation scale on integrated, virtual and software GPUs when `--sim-scale`
/// is not given: the heaviest presets run at a few frames per second at full size.
const WEAK_GPU_SIM_SCALE: f32 = 0.5;

pub struct AppOptions {
    pub world: String,
    pub preset: Option<String>,
    pub seed: Option<u64>,
    /// Logical window size; `None` picks a default that fits the screen.
    pub window_size: Option<[u32; 2]>,
    pub fullscreen: bool,
    pub vsync: bool,
    /// Simulation resolution relative to the window; `None` picks one for the GPU.
    pub sim_scale: Option<f32>,
    pub hide_ui: bool,
    pub exit_after: Option<f32>,
    pub screenshot_dir: PathBuf,
    /// Start in tour mode, advancing to the next preset every N seconds.
    pub tour: Option<f32>,
    /// Start recording an MP4 immediately.
    pub record: bool,
}

pub fn run(options: AppOptions) -> Result<()> {
    if world::find(&options.world).is_none() {
        let ids: Vec<&str> = WORLDS.iter().map(|w| w.id).collect();
        return Err(anyhow!("unknown world '{}' (available: {})", options.world, ids.join(", ")));
    }
    let event_loop = EventLoop::new().context("creating event loop")?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { options, state: None, error: None };
    event_loop.run_app(&mut app).context("event loop failed")?;
    // Dropping the state finalises any recording before we report errors.
    drop(app.state.take());
    match app.error.take() {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

struct App {
    options: AppOptions,
    state: Option<State>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        match State::new(event_loop, &self.options) {
            Ok(state) => self.state = Some(state),
            Err(e) => {
                self.error = Some(e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };
        if let Err(e) = state.handle_event(event_loop, event) {
            self.error = Some(e);
            event_loop.exit();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        let Some(state) = self.state.as_mut() else { return };
        if let Some(problem) = state.gpu.fatal_error() {
            self.error = Some(anyhow!(
                "{problem}\nIf another program is using most of the GPU or its memory, close it and try again."
            ));
            event_loop.exit();
            return;
        }
        if state.exit_due() {
            event_loop.exit();
            return;
        }
        if state.minimized {
            // Nothing is presented, so vsync no longer paces the loop: sleep instead of spinning.
            event_loop.set_control_flow(ControlFlow::wait_duration(Duration::from_millis(100)));
        } else if !state.gpu_ready() {
            // The GPU is still working on the previous frame: look again shortly
            // rather than spinning (and rather than acquiring the next image).
            event_loop.set_control_flow(ControlFlow::wait_duration(Duration::from_millis(2)));
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
            state.window.request_redraw();
        }
    }
}

enum Action {
    SwitchWorld(usize),
    LoadPreset(usize),
    Reset,
    Mutate,
    Screenshot,
    ToggleRecording,
    ToggleMetricsLog,
    SaveWorld,
    LoadWorld(usize),
    RenameWorld(usize, String),
    DeleteWorld(usize),
    /// Rebuild the current world at this simulation scale.
    SetSimScale(f32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shortcut {
    TogglePause,
    Escape,
    ToggleFullscreen,
    Screenshot,
    ResetView,
    TogglePanel,
    Reset,
    Mutate,
    ToggleRecording,
    ToggleTour,
    Step,
    PrevPreset,
    NextPreset,
    BrushSmaller,
    BrushBigger,
    SelectWorld(usize),
}

impl Shortcut {
    /// Keys that should keep acting while held down.
    fn repeats(self) -> bool {
        matches!(
            self,
            Shortcut::Step
                | Shortcut::PrevPreset
                | Shortcut::NextPreset
                | Shortcut::BrushSmaller
                | Shortcut::BrushBigger
        )
    }
}

/// Maps a key press to a shortcut. Letters, digits and punctuation use the
/// physical key (its US-layout position), so shortcuts also work on non-Latin
/// keyboard layouts. Tab is handled before egui sees it (see `handle_event`).
fn shortcut(event: &KeyEvent) -> Option<Shortcut> {
    use Shortcut::*;
    if let Key::Named(named) = &event.logical_key {
        match named {
            NamedKey::Space => return Some(TogglePause),
            NamedKey::Escape => return Some(Escape),
            NamedKey::F11 => return Some(ToggleFullscreen),
            NamedKey::F12 => return Some(Screenshot),
            NamedKey::Home => return Some(ResetView),
            _ => {}
        }
    }
    let PhysicalKey::Code(code) = event.physical_key else { return None };
    let digit = |n: usize| Some(SelectWorld(n - 1));
    match code {
        KeyCode::KeyR => Some(Reset),
        KeyCode::KeyM => Some(Mutate),
        KeyCode::KeyH => Some(TogglePanel),
        KeyCode::KeyS => Some(Screenshot),
        KeyCode::KeyV => Some(ToggleRecording),
        KeyCode::KeyT => Some(ToggleTour),
        KeyCode::KeyF => Some(ToggleFullscreen),
        KeyCode::KeyN => Some(Step),
        KeyCode::Comma => Some(PrevPreset),
        KeyCode::Period => Some(NextPreset),
        KeyCode::BracketLeft => Some(BrushSmaller),
        KeyCode::BracketRight => Some(BrushBigger),
        KeyCode::Digit0 | KeyCode::Numpad0 => Some(ResetView),
        KeyCode::Digit1 | KeyCode::Numpad1 => digit(1),
        KeyCode::Digit2 | KeyCode::Numpad2 => digit(2),
        KeyCode::Digit3 | KeyCode::Numpad3 => digit(3),
        KeyCode::Digit4 | KeyCode::Numpad4 => digit(4),
        KeyCode::Digit5 | KeyCode::Numpad5 => digit(5),
        KeyCode::Digit6 | KeyCode::Numpad6 => digit(6),
        KeyCode::Digit7 | KeyCode::Numpad7 => digit(7),
        KeyCode::Digit8 | KeyCode::Numpad8 => digit(8),
        KeyCode::Digit9 | KeyCode::Numpad9 => digit(9),
        _ => None,
    }
}

#[derive(Default)]
struct Input {
    /// Cursor position in physical pixels.
    cursor: Option<[f32; 2]>,
    left: bool,
    right: bool,
    middle: bool,
}

struct Toast {
    text: String,
    shown: Instant,
    secs: f32,
}

/// Why the GPU is slow to finish frames, as the log and the title bar tell it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum SlowGpu {
    /// Too early in the run to compare with earlier frames.
    Unknown,
    /// A frame suddenly took far longer than the ones before it: most likely
    /// another program is competing for the GPU.
    Contended,
    /// Frames have been about this slow all along: the GPU needs this many
    /// milliseconds per frame at the current simulation size.
    Overloaded { ms: f32 },
}

impl SlowGpu {
    /// A frame just kept us waiting `waited`; `typical` is the average frame
    /// time before it (`None` in the first frames of a run).
    fn classify(waited: Duration, typical: Option<Duration>) -> Self {
        match typical {
            None => SlowGpu::Unknown,
            Some(typical) if waited > typical * 3 => SlowGpu::Contended,
            Some(typical) => SlowGpu::Overloaded { ms: waited.max(typical).as_secs_f32() * 1000.0 },
        }
    }

    /// Suffix for the window title.
    fn title_note(self) -> String {
        match self {
            SlowGpu::Unknown => " · GPU busy".to_string(),
            SlowGpu::Contended => " · GPU busy (another program may be using it heavily)".to_string(),
            SlowGpu::Overloaded { ms } => format!(" · GPU needs ~{ms:.0} ms per frame at this size"),
        }
    }
}

/// Live H.264 capture of the composited frame (without UI). Frames are read
/// back on the render thread and handed to a writer thread that feeds ffmpeg.
struct Recorder {
    path: PathBuf,
    /// Surface size when the recording started; a resize ends the recording.
    source_size: [u32; 2],
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// Even-sized (yuv420p) top-left crop of `texture`.
    readback: Readback,
    /// (RGBA pixels, how many times to write them). Dropping it ends the writer.
    sender: Option<mpsc::SyncSender<(Vec<u8>, u32)>>,
    writer: Option<JoinHandle<std::io::Result<u64>>>,
    child: std::process::Child,
    started: Instant,
}

struct State {
    window: Arc<Window>,
    gpu: Gpu,
    gpu_name: String,
    /// Name, backend and kind of the GPU, for the About section.
    gpu_detail: String,
    /// How to run this exe from a terminal, for the suggested commands.
    program: String,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    minimized: bool,
    post: Post,
    world: Box<dyn World>,
    world_index: usize,
    look: PostSettings,
    camera: Camera,
    sim_scale: f32,
    /// Parameters no longer match the named preset (after Mutate).
    modified: bool,
    seed: u64,
    world_output_size: [u32; 2],
    saved_name: Option<String>,
    library: Library,
    library_name: String,
    library_edit: Option<(usize, String)>,
    library_delete: Option<usize>,

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    popup_was_open: bool,

    input: Input,
    show_panel: bool,
    inspector: Inspector,
    paused: bool,
    single_step: bool,
    /// Brush radius in logical points, so it looks the same at any DPI.
    brush_pts: f32,

    time: f32,
    frame: u64,
    last_instant: Instant,
    started: Instant,
    frames_since_start: u64,
    /// Frames drawn since the scene last changed (world, preset, resolution),
    /// when the frame-rate average started over.
    frames_in_scene: u64,
    fps: f32,
    last_title: Instant,
    exit_after: Option<f32>,
    output_dir: PathBuf,
    pending_screenshot: bool,
    toast: Option<Toast>,
    quit_armed: Option<Instant>,

    /// Tour mode: walk through every preset of every world like a screensaver.
    tour_enabled: bool,
    tour_secs: f32,
    tour_clock: f32,

    recorder: Option<Recorder>,
    record_frames: u64,
    /// Frames to wait before honouring `--record`, so the window can settle
    /// (its first resize events would otherwise end the recording at once).
    record_start_in: Option<u32>,

    /// Set once the GPU has finished the previous frame. The next swapchain
    /// image is only acquired after that: when another program saturates the
    /// GPU and a frame takes over a second, wgpu 25 would otherwise reuse a
    /// swapchain semaphore that is still in flight
    /// (VUID-vkAcquireNextImageKHR-semaphore-01779) and the driver can hang.
    frame_done: Arc<AtomicBool>,
    /// When we started waiting for a busy GPU.
    gpu_wait_since: Option<Instant>,
    /// Until when the title shows why the GPU is slow (extended by every slow frame).
    gpu_busy_until: Option<Instant>,
    gpu_slow: SlowGpu,
    /// Last time a slow-GPU warning was logged (they are rate-limited).
    gpu_warned_at: Option<Instant>,
    /// Whether the toast suggesting a lower simulation resolution was shown.
    suggested_lower_resolution: bool,
    /// Development aid: extra GPU work per frame (`PRIMORDIA_DEBUG_GPU_STALL_MS`).
    stall: Option<crate::stall::Stall>,

    /// Live measurements: the asynchronous readback ring, the run's traces
    /// for the sparklines, and the optional CSV log.
    sampler: metrics::Sampler,
    history: metrics::History,
    metrics_log: Option<metrics::CsvLog>,
}

impl Drop for State {
    fn drop(&mut self) {
        // Close ffmpeg's input so an in-progress recording is finalised on exit.
        self.stop_recording(None);
        self.stop_metrics_log(None);
    }
}

impl State {
    fn new(event_loop: &ActiveEventLoop, opts: &AppOptions) -> Result<Self> {
        let mut size = opts.window_size.unwrap_or(DEFAULT_WINDOW);
        if opts.window_size.is_none() {
            if let Some(monitor) = event_loop.primary_monitor() {
                let screen = monitor.size().to_logical::<f64>(monitor.scale_factor());
                size = [
                    size[0].min((screen.width * 0.9) as u32).max(320),
                    size[1].min((screen.height * 0.85) as u32).max(240),
                ];
            }
        }
        let mut attrs = Window::default_attributes()
            .with_title("Primordia")
            .with_inner_size(LogicalSize::new(size[0], size[1]))
            .with_min_inner_size(LogicalSize::new(320, 240));
        if opts.fullscreen {
            attrs = attrs.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        let scale_factor = event_loop.primary_monitor().map_or(1.0, |monitor| monitor.scale_factor());
        let window = Arc::new(event_loop.create_window(with_icons(attrs, scale_factor)).context("creating window")?);

        let instance = Gpu::create_instance();
        let surface = instance.create_surface(window.clone()).context("creating surface")?;
        let gpu = pollster::block_on(Gpu::new(instance, Some(&surface)))?;
        let gpu_name = gpu.adapter_name();
        let adapter = gpu.adapter.get_info();
        // Unless told otherwise, a GPU that shares the CPU's memory starts smaller.
        let sim_scale = opts.sim_scale.unwrap_or_else(|| default_sim_scale(adapter.device_type));
        let reduced = opts.sim_scale.is_none() && sim_scale < 1.0;
        if reduced {
            log::info!(
                "{} is {}: simulating at {:.0}% of the window's resolution \
                 (Tools > Simulation resolution, or --sim-scale 1 for full)",
                adapter.name,
                adapter_kind(adapter.device_type),
                sim_scale * 100.0
            );
        }

        let inner = window.inner_size();
        let (w, h) = (inner.width.max(1), inner.height.max(1));
        let caps = surface.get_capabilities(&gpu.adapter);
        // A non-sRGB swapchain: post.wgsl encodes sRGB (with dithering) itself, and
        // egui blends in gamma space as it is designed to.
        use wgpu::TextureFormat as F;
        let format = [F::Bgra8Unorm, F::Rgba8Unorm, F::Bgra8UnormSrgb, F::Rgba8UnormSrgb]
            .into_iter()
            .find(|f| caps.formats.contains(f))
            .context("the window surface offers no 8-bit RGBA/BGRA format")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: w,
            height: h,
            present_mode: if opts.vsync { wgpu::PresentMode::AutoVsync } else { wgpu::PresentMode::AutoNoVsync },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes.first().copied().unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &config);

        let post = Post::new(&gpu, [w, h], format);
        let seed = opts.seed.unwrap_or_else(rng::time_seed);
        let sim_size = scaled([w, h], sim_scale);
        let (world_index, world) = world::create(&gpu, &opts.world, sim_size, opts.preset.as_deref(), seed)?;
        let look = world.post_settings();

        let egui_ctx = egui::Context::default();
        theme::configure(&egui_ctx);
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&gpu.device, format, None, 1, false);
        let output_dir = std::path::absolute(&opts.screenshot_dir).unwrap_or_else(|_| opts.screenshot_dir.clone());

        log::info!("opened {} at {}x{} (world domain {}x{})", world.name(), w, h, world.size()[0], world.size()[1]);
        let now = Instant::now();
        let stall = crate::stall::Stall::from_env(&gpu);
        let sampler = metrics::Sampler::new(&gpu, metrics::Sampler::APP_SLOTS);
        let mut state = Self {
            window,
            gpu,
            gpu_name,
            gpu_detail: describe_adapter(&adapter),
            program: program_name(),
            surface,
            config,
            minimized: false,
            post,
            world,
            world_index,
            look,
            camera: Camera::default(),
            sim_scale,
            modified: false,
            seed,
            world_output_size: sim_size,
            saved_name: None,
            library: Library::open(crate::library::default_directory()),
            library_name: String::new(),
            library_edit: None,
            library_delete: None,
            egui_ctx,
            egui_state,
            egui_renderer,
            popup_was_open: false,
            input: Input::default(),
            show_panel: !opts.hide_ui,
            inspector: Inspector::default(),
            paused: false,
            single_step: false,
            brush_pts: 30.0,
            time: 0.0,
            frame: 0,
            last_instant: now,
            started: now,
            frames_since_start: 0,
            frames_in_scene: 0,
            fps: 0.0,
            last_title: now,
            exit_after: opts.exit_after,
            output_dir,
            pending_screenshot: false,
            toast: None,
            quit_armed: None,
            tour_enabled: opts.tour.is_some(),
            tour_secs: opts.tour.unwrap_or(20.0).clamp(*TOUR_RANGE.start(), *TOUR_RANGE.end()),
            tour_clock: 0.0,
            recorder: None,
            record_frames: 0,
            record_start_in: opts.record.then_some(3),
            frame_done: Arc::new(AtomicBool::new(true)),
            gpu_wait_since: None,
            gpu_busy_until: None,
            gpu_slow: SlowGpu::Unknown,
            gpu_warned_at: None,
            suggested_lower_resolution: reduced,
            stall,
            sampler,
            history: metrics::History::default(),
            metrics_log: None,
        };
        let mut hint =
            "Drag to interact · wheel zooms · middle-drag pans · H hides the panel · Space pauses".to_string();
        if reduced {
            hint.push_str(&format!(
                "\nThis is {}, so worlds run at {:.0}% resolution. Tools › Simulation resolution changes it.",
                adapter_kind(adapter.device_type),
                sim_scale * 100.0
            ));
        }
        state.toast_for(hint, if reduced { 10.0 } else { 6.0 });
        Ok(state)
    }

    fn target_size(&self) -> [u32; 2] {
        [self.config.width, self.config.height]
    }

    fn view(&self) -> ViewXform {
        ViewXform::fit(self.world.size(), self.target_size(), &self.camera)
    }

    fn cursor_uv(&self) -> Option<[f32; 2]> {
        let c = self.input.cursor?;
        let [w, h] = self.target_size();
        Some([c[0] / w.max(1) as f32, c[1] / h.max(1) as f32])
    }

    fn preset_name(&self) -> &'static str {
        self.world.presets().get(self.world.preset()).copied().unwrap_or("Custom")
    }

    fn preset_label(&self) -> String {
        if let Some(name) = &self.saved_name { return name.clone(); }
        if self.modified { format!("{} (mutated)", self.preset_name()) } else { self.preset_name().to_string() }
    }

    /// True once `--exit-after` has elapsed (checked even while minimised).
    fn exit_due(&mut self) -> bool {
        let Some(limit) = self.exit_after else { return false };
        let elapsed = self.started.elapsed().as_secs_f32();
        if elapsed < limit {
            return false;
        }
        log::info!(
            "exit-after {limit}s reached: {} frames, {:.1} fps average",
            self.frames_since_start,
            self.frames_since_start as f32 / elapsed
        );
        self.exit_after = None;
        true
    }

    fn handle_event(&mut self, event_loop: &ActiveEventLoop, event: WindowEvent) -> Result<()> {
        // Tab toggles the panel. Keep it away from egui, which would use it to move
        // keyboard focus onto a widget and then swallow every other shortcut.
        if let WindowEvent::KeyboardInput { event: key, .. } = &event {
            if key.logical_key == Key::Named(NamedKey::Tab) && !self.egui_ctx.wants_keyboard_input() {
                if key.state == ElementState::Pressed && !key.repeat {
                    self.show_panel = !self.show_panel;
                }
                return Ok(());
            }
        }

        let response = self.egui_state.on_window_event(&self.window, &event);
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => self.resize(size),
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = self.window.inner_size();
                self.resize(size);
            }
            WindowEvent::RedrawRequested => self.redraw()?,
            WindowEvent::Focused(false) => {
                // Button releases that happen while another window has focus never
                // reach us; don't let the brush stick on.
                self.input.left = false;
                self.input.right = false;
                self.input.middle = false;
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new = [position.x as f32, position.y as f32];
                if self.input.middle {
                    if let Some(old) = self.input.cursor {
                        let [w, h] = self.target_size();
                        let view = self.view();
                        self.camera.center[0] -= (new[0] - old[0]) / w.max(1) as f32 * view.scale[0];
                        self.camera.center[1] -= (new[1] - old[1]) / h.max(1) as f32 * view.scale[1];
                        self.camera.center =
                            [self.camera.center[0].rem_euclid(1.0), self.camera.center[1].rem_euclid(1.0)];
                    }
                }
                self.input.cursor = Some(new);
            }
            WindowEvent::CursorLeft { .. } => {
                self.input.cursor = None;
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let allow = !pressed || !response.consumed;
                match button {
                    MouseButton::Left if allow => self.input.left = pressed,
                    MouseButton::Right if allow => self.input.right = pressed,
                    MouseButton::Middle if allow => self.input.middle = pressed,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !response.consumed => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                if let Some(uv) = self.cursor_uv() {
                    self.zoom_at(uv, 1.15f32.powf(lines));
                }
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed
                    && !response.consumed
                    && !self.egui_ctx.wants_keyboard_input() =>
            {
                self.on_key(event_loop, &event)?;
            }
            _ => {}
        }
        Ok(())
    }

    fn on_key(&mut self, event_loop: &ActiveEventLoop, event: &KeyEvent) -> Result<()> {
        let Some(shortcut) = shortcut(event) else { return Ok(()) };
        if event.repeat && !shortcut.repeats() {
            return Ok(());
        }
        match shortcut {
            Shortcut::TogglePause => self.paused = !self.paused,
            Shortcut::Escape => {
                if self.window.fullscreen().is_some() {
                    self.window.set_fullscreen(None);
                } else if self.popup_was_open {
                    // egui closes the open dropdown itself.
                } else if self.quit_armed.is_some_and(|t| t.elapsed().as_secs_f32() < QUIT_CONFIRM_SECS) {
                    event_loop.exit();
                } else {
                    self.quit_armed = Some(Instant::now());
                    self.toast_for("Press Esc again to quit".to_string(), QUIT_CONFIRM_SECS);
                }
            }
            Shortcut::ToggleFullscreen => self.toggle_fullscreen(),
            Shortcut::Screenshot => self.pending_screenshot = true,
            Shortcut::ResetView => self.camera = Camera::default(),
            Shortcut::TogglePanel => self.show_panel = !self.show_panel,
            Shortcut::Reset => self.apply(Action::Reset)?,
            Shortcut::Mutate => self.apply(Action::Mutate)?,
            Shortcut::ToggleRecording => self.apply(Action::ToggleRecording)?,
            Shortcut::ToggleTour => self.toggle_tour(),
            Shortcut::Step => {
                self.paused = true;
                self.single_step = true;
            }
            Shortcut::PrevPreset => self.cycle_preset(-1)?,
            Shortcut::NextPreset => self.cycle_preset(1)?,
            Shortcut::BrushSmaller => self.brush_pts = (self.brush_pts / 1.2).max(2.0),
            Shortcut::BrushBigger => self.brush_pts = (self.brush_pts * 1.2).min(400.0),
            Shortcut::SelectWorld(i) => {
                if i < WORLDS.len() {
                    self.apply(Action::SwitchWorld(i))?;
                }
            }
        }
        Ok(())
    }

    fn toggle_fullscreen(&mut self) {
        let next = if self.window.fullscreen().is_some() { None } else { Some(Fullscreen::Borderless(None)) };
        self.window.set_fullscreen(next);
    }

    fn toggle_tour(&mut self) {
        self.tour_enabled = !self.tour_enabled;
        self.tour_clock = TOUR_FADE;
        let state = if self.tour_enabled { "on" } else { "off" };
        self.toast(format!("Tour {state} ({:.0}s per preset)", self.tour_secs));
    }

    fn cycle_preset(&mut self, delta: i64) -> Result<()> {
        let n = self.world.presets().len() as i64;
        if n == 0 {
            return Ok(());
        }
        let next = (self.world.preset() as i64 + delta).rem_euclid(n) as usize;
        self.apply(Action::LoadPreset(next))
    }

    fn zoom_at(&mut self, cursor_uv: [f32; 2], factor: f32) {
        let anchor = self.world.map_position(self.view(), cursor_uv);
        self.camera.zoom = (self.camera.zoom * factor).clamp(0.5, 64.0);
        let after = self.world.map_position(self.view(), cursor_uv);
        self.camera.center = [
            (self.camera.center[0] + anchor[0] - after[0]).rem_euclid(1.0),
            (self.camera.center[1] + anchor[1] - after[1]).rem_euclid(1.0),
        ];
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            self.minimized = true;
            return;
        }
        self.minimized = false;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.gpu.device, &self.config);
        self.post.resize(&self.gpu, [size.width, size.height]);
        let recording_size = self.recorder.as_ref().map(|r| r.source_size);
        if let Some(from) = recording_size {
            if from != [size.width, size.height] {
                log::info!(
                    "window resized from {}x{} to {}x{} while recording; stopping",
                    from[0],
                    from[1],
                    size.width,
                    size.height
                );
                self.stop_recording(Some("the window was resized"));
            }
        }
    }

    /// Re-applies the surface configuration from the window's current size.
    fn reconfigure(&mut self) {
        // Surface recovery can discover a resize before the OS resize event.
        // Use the same path so minimization and recording size stay in sync.
        self.resize(self.window.inner_size());
    }

    fn apply(&mut self, action: Action) -> Result<()> {
        let seed = rng::time_seed();
        if matches!(action, Action::SwitchWorld(_) | Action::LoadPreset(_) | Action::Mutate) {
            // Any change of scene restarts the tour countdown (fading in).
            self.tour_clock = 0.0;
        }
        match action {
            Action::SwitchWorld(index) => {
                if index != self.world_index {
                    let size = scaled(self.target_size(), self.sim_scale);
                    self.gpu.wait_idle();
                    self.gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
                    let created = world::create(&self.gpu, WORLDS[index].id, size, None, seed);
                    let oom = pollster::block_on(self.gpu.device.pop_error_scope());
                    let (i, world) = created?;
                    if let Some(err) = oom {
                        // Keep the current world rather than switching to a half-built one.
                        drop(world);
                        self.report_oom(&err);
                    } else {
                        self.world = world;
                        self.world_index = i;
                        self.look = self.world.post_settings();
                        self.camera = Camera::default();
                        self.modified = false;
                        self.seed = seed;
                        self.world_output_size = size;
                        self.saved_name = None;
                        self.reset_measurements(true);
                        self.toast(format!("{} — {}", WORLDS[i].name, WORLDS[i].tagline));
                    }
                }
            }
            Action::LoadPreset(index) => {
                if let Some(err) = self.catch_oom(|s| s.world.load_preset(&s.gpu, index, seed)) {
                    self.report_oom(&err);
                    return Ok(());
                }
                self.look = self.world.post_settings();
                self.modified = false;
                self.seed = seed;
                self.saved_name = None;
                self.reset_measurements(false);
                self.toast(self.preset_name().to_string());
            }
            Action::Reset => {
                if let Some(err) = self.catch_oom(|s| s.world.reset(&s.gpu, seed)) {
                    self.report_oom(&err);
                } else {
                    self.seed = seed;
                    self.saved_name = None;
                    self.reset_measurements(false);
                }
            }
            Action::Mutate => {
                if let Some(err) = self.catch_oom(|s| s.world.mutate(&s.gpu, seed)) {
                    self.report_oom(&err);
                    return Ok(());
                }
                self.look = self.world.post_settings();
                self.modified = true;
                self.seed = seed;
                self.saved_name = None;
                self.reset_measurements(false);
                self.toast("Mutated".to_string());
            }
            Action::ToggleMetricsLog => {
                if self.metrics_log.is_some() {
                    self.stop_metrics_log(None);
                } else if let Err(e) = self.start_metrics_log() {
                    self.toast_for(format!("Could not start the measurement log: {e:#}"), 6.0);
                }
            }
            Action::Screenshot => self.pending_screenshot = true,
            Action::SaveWorld => {
                let result = self.save_world();
                match result {
                    Ok(()) => {
                        self.library_edit = None;
                        self.library_delete = None;
                        self.toast(format!("Saved {} to your library", self.library_name.trim()));
                    }
                    Err(e) => self.toast_for(format!("Could not save world: {e:#}"), 6.0),
                }
            }
            Action::LoadWorld(index) => {
                if let Err(e) = self.load_world(index) { self.toast_for(format!("Could not load world: {e:#}"), 6.0); }
            }
            Action::RenameWorld(index, name) => {
                match self.library.rename(index, &name) {
                    Ok(()) => { self.library_edit = None; self.library_delete = None; self.toast("Save renamed".into()); }
                    Err(e) => self.toast_for(format!("Could not rename save: {e:#}"), 6.0),
                }
            }
            Action::DeleteWorld(index) => {
                match self.library.delete(index) {
                    Ok(()) => { self.library_edit = None; self.library_delete = None; self.toast("Save removed".into()); }
                    Err(e) => self.toast_for(format!("Could not remove save: {e:#}"), 6.0),
                }
            }
            Action::SetSimScale(scale) => {
                let previous = std::mem::replace(&mut self.sim_scale, scale);
                match self.rebuild_world() {
                    Ok(()) => {
                        let [w, h] = self.world_output_size;
                        self.toast(format!("Simulating at {:.0}% resolution ({w}×{h})", scale * 100.0));
                    }
                    Err(e) => {
                        self.sim_scale = previous;
                        self.toast_for(format!("Could not change the resolution: {e:#}"), 6.0);
                    }
                }
            }
            Action::ToggleRecording => {
                // A manual choice overrides a pending --record startup request.
                self.record_start_in = None;
                if self.recorder.is_some() {
                    self.stop_recording(None);
                } else if let Err(e) = self.start_recording() {
                    log::error!("could not start recording: {e:#}");
                    self.toast(format!("Recording failed: {e}"));
                }
            }
        }
        Ok(())
    }

    /// Rebuilds the current world for the current window and simulation scale,
    /// keeping its preset, parameters, seed, appearance and camera. The
    /// simulation restarts; the current world stays if the new one fails.
    fn rebuild_world(&mut self) -> Result<()> {
        let size = scaled(self.target_size(), self.sim_scale);
        self.gpu.wait_idle();
        self.gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let created = rebuilt(&self.gpu, self.world_index, &*self.world, size, self.seed);
        let oom = pollster::block_on(self.gpu.device.pop_error_scope());
        let world = created?;
        if let Some(err) = oom {
            drop(world);
            log::error!("out of GPU memory: {err}");
            return Err(anyhow!("not enough GPU memory for {}×{}", size[0], size[1]));
        }
        self.world = world;
        self.world_output_size = size;
        self.reset_measurements(false);
        let percent = self.sim_scale * 100.0;
        log::info!("rebuilt {} at {}x{} ({percent:.0}% resolution)", self.world.name(), size[0], size[1]);
        Ok(())
    }

    /// Store a recipe without resetting or stepping the running simulation.
    fn save_world(&mut self) -> Result<()> {
        let saved = SavedWorld {
            version: 1, name: self.library_name.clone(), seed: self.seed,
            output_size: self.world_output_size, preset: self.world.preset(), modified: self.modified,
            settings: self.world.settings()?, look: self.look, camera: self.camera,
        };
        self.library.save(saved)
    }

    fn load_world(&mut self, index: usize) -> Result<()> {
        let saved = self.library.entries.get(index).context("Save no longer exists")?.saved.clone();
        let (index, world) = saved.instantiate(&self.gpu)?;
        self.world = world;
        self.world_index = index;
        self.world_output_size = saved.output_size;
        self.seed = saved.seed;
        self.look = saved.look;
        self.camera = saved.camera;
        self.modified = saved.modified;
        self.saved_name = Some(saved.name.clone());
        self.library_name = saved.name.clone();
        self.tour_enabled = false;
        self.time = 0.0;
        self.frame = 0;
        self.paused = false;
        self.single_step = false;
        self.reset_measurements(true);
        self.toast(format!("Loaded {} · restarted from its saved seed", saved.name));
        Ok(())
    }

    /// Moves finished measurements into the history and the CSV log.
    fn ingest_samples(&mut self) {
        for sample in self.sampler.collect(&self.gpu) {
            self.history.push(&sample);
            let failed = match &mut self.metrics_log {
                Some(log) => log.write(&sample).err(),
                None => None,
            };
            if let Some(e) = failed {
                log::error!("{e:#}");
                self.stop_metrics_log(Some("the file could not be written"));
            }
        }
    }

    /// The run restarted: forget its traces and its frame-rate average. A world
    /// change also closes the log, because the columns (and the frame counter)
    /// would no longer match.
    fn reset_measurements(&mut self, world_changed: bool) {
        self.history.clear();
        self.sampler.discard();
        self.restart_frame_clock();
        if world_changed {
            self.stop_metrics_log(Some("the world changed"));
        }
    }

    fn start_metrics_log(&mut self) -> Result<()> {
        let metrics = self.world.metrics();
        if metrics.is_empty() {
            return Err(anyhow!("{} has no measurements", self.world.name()));
        }
        std::fs::create_dir_all(&self.output_dir).with_context(|| format!("creating {}", self.output_dir.display()))?;
        let path = unique_path(&self.output_dir, &self.capture_stem(), "csv");
        let log = metrics::CsvLog::create(&path, metrics)?;
        log::info!("logging measurements to {}", path.display());
        self.toast(format!("Logging measurements to {}", path.display()));
        self.metrics_log = Some(log);
        Ok(())
    }

    /// Flushes and closes the measurement log, reporting where it went.
    fn stop_metrics_log(&mut self, reason: Option<&str>) {
        let Some(log) = self.metrics_log.take() else { return };
        let prefix = reason.map(|r| format!("Measurement log stopped ({r}). ")).unwrap_or_default();
        match log.finish() {
            Ok((path, rows)) => {
                log::info!("saved {} ({rows} rows)", path.display());
                self.toast_for(format!("{prefix}Saved {} ({rows} rows)", path.display()), 5.0);
            }
            Err(e) => self.toast_for(format!("{prefix}Measurement log failed: {e:#}"), 6.0),
        }
    }

    /// Runs `f` inside an out-of-memory error scope and returns the error, if any.
    fn catch_oom(&mut self, f: impl FnOnce(&mut Self)) -> Option<wgpu::Error> {
        self.gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        f(self);
        pollster::block_on(self.gpu.device.pop_error_scope())
    }

    fn report_oom(&mut self, err: &wgpu::Error) {
        log::error!("out of GPU memory: {err}");
        self.toast_for(
            "Out of GPU memory. Close other GPU-heavy programs, make the window smaller \
             or lower Tools › Simulation resolution."
                .to_string(),
            6.0,
        );
    }

    fn toast(&mut self, text: String) {
        self.toast_for(text, 3.0);
    }

    fn toast_for(&mut self, text: String, secs: f32) {
        self.toast = Some(Toast { text, shown: Instant::now(), secs });
    }

    fn pointer(&self, view: ViewXform) -> Option<Pointer> {
        let uv = self.cursor_uv()?;
        let dragging = self.input.left || self.input.right;
        if !dragging && self.egui_ctx.is_pointer_over_area() {
            return None;
        }
        let [tw, _] = self.target_size();
        let cells_per_px = view.scale[0] * self.world.size()[0] as f32 / tw.max(1) as f32;
        let radius_px = self.brush_pts * self.egui_ctx.pixels_per_point();
        Some(Pointer {
            pos: self.world.map_position(view, uv),
            primary: self.input.left,
            secondary: self.input.right,
            radius: radius_px * cells_per_px,
        })
    }

    /// Exposure multiplier that fades the image out/in around tour transitions.
    fn tour_fade(&self) -> f32 {
        if !self.tour_enabled {
            return 1.0;
        }
        let fade_in = (self.tour_clock / TOUR_FADE).clamp(0.0, 1.0);
        let fade_out = ((self.tour_secs - self.tour_clock) / TOUR_FADE).clamp(0.0, 1.0);
        let f = fade_in.min(fade_out);
        f * f
    }

    /// Advances the tour clock; moves to the next preset (or world) when due.
    fn tour_tick(&mut self, dt: f32) -> Result<()> {
        if !self.tour_enabled || self.paused {
            return Ok(());
        }
        self.tour_clock += dt;
        if self.tour_clock >= self.tour_secs {
            let next = self.world.preset() + 1;
            if next < self.world.presets().len() {
                self.apply(Action::LoadPreset(next))?;
            } else {
                self.apply(Action::SwitchWorld((self.world_index + 1) % WORLDS.len()))?;
            }
            self.tour_clock = 0.0;
        }
        Ok(())
    }

    fn free_egui_textures(&mut self, ids: &[egui::TextureId]) {
        for id in ids {
            self.egui_renderer.free_texture(id);
        }
    }

    /// True when the GPU has finished the previous frame, so a new frame may
    /// acquire the next swapchain image. Notes (in the log and the title bar)
    /// when frames keep the GPU busy for a long time, and why that seems to be.
    fn gpu_ready(&mut self) -> bool {
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        let now = Instant::now();
        if self.frame_done.load(Ordering::Acquire) {
            if let Some(since) = self.gpu_wait_since.take() {
                let waited = now - since;
                if waited > Duration::from_millis(250) {
                    self.gpu_busy_until = Some(now + Duration::from_secs(5));
                    self.gpu_slow = SlowGpu::classify(waited, self.typical_frame_time());
                    self.warn_gpu_busy(&format!("waited {:.1}s for the GPU to finish a frame", waited.as_secs_f32()));
                }
            }
            return true;
        }
        let since = *self.gpu_wait_since.get_or_insert(now);
        if now - since > Duration::from_millis(1500) && self.gpu_busy_until.is_none_or(|t| t <= now) {
            self.gpu_busy_until = Some(now + Duration::from_secs(5));
            self.gpu_slow = SlowGpu::classify(now - since, self.typical_frame_time());
            self.warn_gpu_busy("the GPU is taking very long to finish frames");
            let title = self.title();
            self.window.set_title(&title);
        }
        false
    }

    /// The average time between frames so far, once there are enough of them
    /// to compare a slow frame with.
    fn typical_frame_time(&self) -> Option<Duration> {
        (self.frames_in_scene >= 10 && self.fps > 0.0).then(|| Duration::from_secs_f32(1.0 / self.fps))
    }

    /// Starts the frame-rate average over after the scene changed, so a
    /// heavier world is not compared with the lighter one before it (and the
    /// time spent building it does not count as a frame).
    fn restart_frame_clock(&mut self) {
        self.fps = 0.0;
        self.frames_in_scene = 0;
        self.last_instant = Instant::now();
    }

    /// Logs a slow-GPU warning, at most once every 10 seconds, with its likely
    /// cause. The first time the simulation itself is too big for the GPU, a
    /// toast suggests a lower resolution.
    fn warn_gpu_busy(&mut self, what: &str) {
        if self.gpu_slow == SlowGpu::Unknown {
            // Often a new world's setup work; the next slow frame will say more.
            log::debug!("{what}");
            return;
        }
        let now = Instant::now();
        if self.gpu_warned_at.is_none_or(|t| now - t > Duration::from_secs(10)) {
            self.gpu_warned_at = Some(now);
            let [w, h] = self.world_output_size;
            match self.gpu_slow {
                SlowGpu::Unknown => {}
                SlowGpu::Contended => log::warn!("{what}; another program may be using the GPU heavily"),
                SlowGpu::Overloaded { ms } => log::warn!(
                    "{what}; this GPU needs about {ms:.0} ms per frame to simulate {w}x{h} \
                     (lower Tools > Simulation resolution, or start with --sim-scale 0.5)"
                ),
            }
        }
        if let SlowGpu::Overloaded { ms } = self.gpu_slow {
            if !self.suggested_lower_resolution && self.sim_scale > theme::SIM_SCALES[0] {
                self.suggested_lower_resolution = true;
                let text = format!(
                    "This GPU needs about {ms:.0} ms per frame here. Tools › Simulation resolution speeds it up."
                );
                self.toast_for(text, 8.0);
            }
        }
    }

    fn title(&self) -> String {
        let paused = if self.paused { " · paused" } else { "" };
        let busy = if self.gpu_busy_until.is_some_and(|t| Instant::now() < t) {
            self.gpu_slow.title_note()
        } else {
            String::new()
        };
        format!("Primordia — {} · {} — {:.0} fps{paused}{busy}", self.world.name(), self.preset_label(), self.fps)
    }

    fn redraw(&mut self) -> Result<()> {
        // The OS can ask for redraws (resize, expose) while the GPU is still busy.
        if self.minimized || !self.gpu_ready() {
            return Ok(());
        }
        self.ingest_samples();
        let now = Instant::now();
        let elapsed = (now - self.last_instant).as_secs_f32();
        // The simulation step is clamped; the fps readout uses the real frame time.
        let dt = elapsed.min(0.1);
        self.last_instant = now;
        if elapsed > 0.0 {
            self.fps = if self.fps == 0.0 { 1.0 / elapsed } else { self.fps * 0.95 + 0.05 / elapsed };
        }
        self.frames_since_start += 1;
        self.frames_in_scene += 1;
        if let Some(n) = self.record_start_in {
            if n == 0 {
                self.record_start_in = None;
                if let Err(e) = self.start_recording() {
                    log::error!("could not start recording: {e:#}");
                    self.toast(format!("Recording failed: {e}"));
                }
            } else {
                self.record_start_in = Some(n - 1);
            }
        }

        // --- UI --------------------------------------------------------------
        let raw_input = self.egui_state.take_egui_input(&self.window);
        let ctx = self.egui_ctx.clone();
        let mut actions = Vec::new();
        let mut full_output = ctx.run(raw_input, |ctx| self.draw_ui(ctx, &mut actions));
        theme::highlight_checked_boxes(&ctx, &mut full_output.shapes);
        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);
        // egui expects every texture delta to be applied, even if this frame is
        // never presented, so upload them before anything can bail out.
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.gpu.device, &self.gpu.queue, *id, delta);
        }
        for action in actions {
            self.apply(action)?;
        }
        self.tour_tick(dt)?;

        // --- acquire the swapchain image before recording any simulation ----
        let surface_texture = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.reconfigure();
                self.free_egui_textures(&full_output.textures_delta.free);
                return Ok(());
            }
            Err(wgpu::SurfaceError::Timeout) => {
                self.free_egui_textures(&full_output.textures_delta.free);
                return Ok(());
            }
            Err(e) => return Err(anyhow!("surface error: {e}")),
        };
        let surface_view = surface_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());

        // --- simulation + scene ---------------------------------------------
        let target = self.target_size();
        let view = self.view();
        let pointer = self.pointer(view);
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let frame =
                Frame { gpu: &self.gpu, time: self.time, dt, frame: self.frame, view, target_size: target, pointer };
            if !self.paused || self.single_step {
                self.world.step(&frame, &mut encoder);
                let mut sink = self.sampler.begin(frame.frame, frame.time);
                self.world.measure(&frame, &mut encoder, &mut sink);
                self.frame += 1;
                self.time += dt;
                self.single_step = false;
            }
            self.world.render(&frame, &mut encoder, self.post.scene_view());
        }
        if let Some(stall) = &self.stall {
            stall.record(&mut encoder);
        }
        let mut look = self.look;
        look.exposure *= self.tour_fade();
        self.post.bloom(&self.gpu, &mut encoder, &look, self.time);
        self.post.composite(&mut encoder, &surface_view);

        // Live recording: a clean (UI-free) copy on a steady 60 fps clock. When the
        // display runs slower, frames are repeated so the video keeps real time.
        let mut record_repeats = 0;
        if let Some(rec) = &self.recorder {
            record_repeats = recording_frames_due(rec.started.elapsed(), self.record_frames);
            if record_repeats > 0 {
                self.record_frames += u64::from(record_repeats);
                self.post.composite(&mut encoder, &rec.view);
                rec.readback.copy_from(&mut encoder, &rec.texture);
            }
        }

        // --- egui on top ------------------------------------------------------
        let paint_jobs = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        let screen =
            egui_wgpu::ScreenDescriptor { size_in_pixels: target, pixels_per_point: full_output.pixels_per_point };
        let extra =
            self.egui_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut encoder, &paint_jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut pass = pass.forget_lifetime();
            self.egui_renderer.render(&mut pass, &paint_jobs, &screen);
        }
        self.gpu.queue.submit(extra.into_iter().chain(std::iter::once(encoder.finish())));
        self.sampler.map();
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        self.gpu.queue.on_submitted_work_done(move || flag.store(true, Ordering::Release));
        self.frame_done = done;
        let suboptimal = surface_texture.suboptimal;
        self.window.pre_present_notify();
        surface_texture.present();
        self.free_egui_textures(&full_output.textures_delta.free);
        // Some setups (e.g. a window on a display driven by another adapter)
        // report every present as suboptimal; only rebuild the swapchain when
        // the window size actually changed.
        if suboptimal && self.window.inner_size() != PhysicalSize::new(self.config.width, self.config.height) {
            self.reconfigure();
        }

        if record_repeats > 0 {
            let sent = match &self.recorder {
                Some(rec) => rec.readback.read(&self.gpu).and_then(|pixels| {
                    rec.sender
                        .as_ref()
                        .context("recorder closed")?
                        .send((pixels, record_repeats))
                        .map_err(|_| anyhow!("ffmpeg stopped accepting frames"))
                }),
                None => Ok(()),
            };
            if let Err(e) = sent {
                log::error!("recording stopped: {e:#}");
                self.stop_recording(Some("the encoder failed"));
            }
        }

        if self.pending_screenshot {
            self.pending_screenshot = false;
            match self.save_screenshot() {
                Ok(path) => self.toast(format!("Saved {}", path.display())),
                Err(e) => {
                    log::error!("screenshot failed: {e:#}");
                    self.toast(format!("Screenshot failed: {e}"));
                }
            }
        }

        if self.last_title.elapsed().as_secs_f32() > 0.5 {
            self.last_title = Instant::now();
            let title = self.title();
            self.window.set_title(&title);
        }
        Ok(())
    }

    /// File name (without extension) for this moment's screenshot, recording or measurement log.
    fn capture_stem(&self) -> String {
        capture_stem(self.world.id(), self.preset_name(), self.modified, self.seed, &utc_timestamp())
    }

    fn save_screenshot(&mut self) -> Result<PathBuf> {
        let size = self.target_size();
        let format = self.config.format;
        let (texture, view) = self.gpu.texture_2d(
            "screenshot",
            size,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let readback = Readback::new(&self.gpu, size, format);
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("screenshot") });
        // Reuses this frame's bloom and settings; only the composite is redone.
        self.post.composite(&mut encoder, &view);
        readback.copy_from(&mut encoder, &texture);
        self.gpu.queue.submit([encoder.finish()]);
        let pixels = readback.read(&self.gpu)?;
        let path = unique_path(&self.output_dir, &self.capture_stem(), "png");
        capture::save_png(&path, size, pixels)?;
        log::info!("saved {}", path.display());
        Ok(path)
    }

    fn start_recording(&mut self) -> Result<()> {
        let source_size = self.target_size();
        let size = [source_size[0].max(2) & !1, source_size[1].max(2) & !1];
        let format = self.config.format;
        // Full surface size (composited 1:1); the readback crops to even dimensions.
        let (texture, view) = self.gpu.texture_2d(
            "recording",
            source_size,
            format,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let readback = Readback::new(&self.gpu, size, format);
        std::fs::create_dir_all(&self.output_dir).with_context(|| format!("creating {}", self.output_dir.display()))?;
        let path = unique_path(&self.output_dir, &self.capture_stem(), "mp4");
        let mut child = headless::spawn_ffmpeg(&path, size, RECORD_FPS as u32, Encode::Realtime)?;
        let mut stdin = child.stdin.take().context("ffmpeg has no stdin")?;
        let (sender, receiver) = mpsc::sync_channel::<(Vec<u8>, u32)>(3);
        let writer = std::thread::Builder::new()
            .name("primordia-recorder".to_string())
            .spawn(move || -> std::io::Result<u64> {
                use std::io::Write as _;
                let mut written = 0;
                for (pixels, repeats) in receiver {
                    for _ in 0..repeats {
                        stdin.write_all(&pixels)?;
                        written += 1;
                    }
                }
                Ok(written)
            })
            .context("starting the recorder thread")?;
        log::info!("recording to {}", path.display());
        self.recorder = Some(Recorder {
            path,
            source_size,
            texture,
            view,
            readback,
            sender: Some(sender),
            writer: Some(writer),
            child,
            started: Instant::now(),
        });
        self.record_frames = 0;
        self.toast("Recording (V to stop)".to_string());
        Ok(())
    }

    /// Finishes the recording (waits for ffmpeg to flush) and reports the result.
    fn stop_recording(&mut self, reason: Option<&str>) {
        let Some(mut rec) = self.recorder.take() else { return };
        drop(rec.sender.take());
        let written = match rec.writer.take().map(JoinHandle::join) {
            Some(Ok(Ok(frames))) => Ok(frames),
            Some(Ok(Err(e))) => Err(format!("writing frames failed: {e}")),
            Some(Err(_)) => Err("the recorder thread panicked".to_string()),
            None => Ok(0),
        };
        let status = rec.child.wait();
        let prefix = reason.map(|r| format!("Recording stopped ({r}). ")).unwrap_or_default();
        let message = match (written, status) {
            (Ok(frames), Ok(s)) if s.success() => {
                log::info!("saved {} ({frames} frames)", rec.path.display());
                format!("{prefix}Saved {} ({:.1}s)", rec.path.display(), frames as f32 / RECORD_FPS)
            }
            (Err(e), _) => format!("{prefix}Recording failed: {e}"),
            (Ok(_), Ok(s)) => format!("{prefix}ffmpeg exited with {s}"),
            (Ok(_), Err(e)) => format!("{prefix}ffmpeg failed: {e}"),
        };
        self.toast_for(message, 5.0);
    }

    fn draw_panel(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        let compact = ui.available_height() < 680.0 || ui.available_width() < 310.0;
        let short = ui.available_height() < 520.0;

        // Reserve capture controls before laying out the scrolling inspector.
        egui::TopBottomPanel::bottom("capture controls")
            .frame(egui::Frame::new().fill(theme::PANEL))
            .show_separator_line(false)
            .show_inside(ui, |ui| {
                ui.add_space(10.0);
                ui.separator();
                ui.add_space(4.0);
                let width = (ui.available_width() - 8.0) / 2.0;
                ui.horizontal(|ui| {
                    if ui
                        .add_sized([width, 34.0], egui::Button::new("Save image"))
                        .on_hover_text(format!("Save a PNG without the UI (F12)\n{}", self.output_dir.display()))
                        .clicked()
                    {
                        actions.push(Action::Screenshot);
                    }
                    let recording = self.recorder.is_some();
                    let text = egui::RichText::new(if recording { "Stop recording" } else { "Record video" })
                        .color(if recording { WARN } else { theme::TEXT });
                    if ui
                        .add_sized([width, 34.0], egui::Button::new(text))
                        .on_hover_text("Start / stop an MP4 recording (V)")
                        .clicked()
                    {
                        actions.push(Action::ToggleRecording);
                    }
                });
                ui.horizontal(|ui| {
                    theme::status_dot(ui, ACCENT);
                    ui.label(egui::RichText::new(format!("{:.0} fps", self.fps)).small().color(ACCENT))
                        .on_hover_text(self.gpu_name.as_str());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new("PNG / MP4 / CSV · clean capture").small().weak())
                            .on_hover_text(format!("Saved to {}", self.output_dir.display()));
                    });
                });
            });

        if short {
            // At the minimum window height, let the whole workspace scroll so
            // every control remains reachable above the fixed capture bar.
            egui::ScrollArea::vertical().id_salt("compact workspace").auto_shrink([false, false]).show(ui, |ui| {
                self.draw_panel_content(ui, actions, compact, false);
            });
        } else {
            self.draw_panel_content(ui, actions, compact, true);
        }
    }

    fn draw_panel_content(
        &mut self,
        ui: &mut egui::Ui,
        actions: &mut Vec<Action>,
        compact: bool,
        scroll_inspector: bool,
    ) {
        ui.horizontal(|ui| {
            theme::mark(ui);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 2.0;
                ui.label(egui::RichText::new("Primordia").size(23.0).strong());
                ui.label(egui::RichText::new("ARTIFICIAL LIFE LAB").size(10.0).color(theme::MUTED));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add_sized([28.0, 28.0], egui::Button::new("‹").frame(false))
                    .on_hover_text("Collapse the control panel (H / Tab)")
                    .clicked()
                {
                    self.show_panel = false;
                }
            });
        });
        ui.add_space(if compact { 4.0 } else { 12.0 });

        theme::eyebrow(ui, "EXPLORE A WORLD");
        if compact {
            egui::ComboBox::from_id_salt("world picker")
                .width(ui.available_width())
                .selected_text(self.world.name())
                .show_ui(ui, |ui| {
                    for (i, entry) in WORLDS.iter().enumerate() {
                        if ui
                            .selectable_label(i == self.world_index, format!("{}   {}", i + 1, entry.name))
                            .on_hover_text(entry.tagline)
                            .clicked()
                            && i != self.world_index
                        {
                            actions.push(Action::SwitchWorld(i));
                        }
                    }
                });
        } else {
            let width = (ui.available_width() - 8.0) / 2.0;
            for row in (0..WORLDS.len()).step_by(2) {
                let width = if row + 1 == WORLDS.len() { ui.available_width() } else { width };
                ui.horizontal(|ui| {
                    for (i, entry) in WORLDS.iter().enumerate().skip(row).take(2) {
                        let detail = match entry.id {
                            "physarum" => "Slime-mould networks",
                            "particle-life" => "Interacting species",
                            "lenia" => "Soft cellular organisms",
                            "reaction-diffusion" => "Living chemistry",
                            "symbiosis" => "Agents + chemistry · coupled worlds",
                            _ => "Artificial life",
                        };
                        if theme::world_card(ui, width, entry.name, detail, i == self.world_index)
                            .on_hover_text(format!("{}\n\nSwitch world: {}", entry.tagline, i + 1))
                            .clicked()
                            && i != self.world_index
                        {
                            actions.push(Action::SwitchWorld(i));
                        }
                    }
                });
            }
        }
        ui.add_space(4.0);

        let presets = self.world.presets();
        let current = self.world.preset();
        let n = presets.len();
        ui.horizontal(|ui| {
            theme::eyebrow(ui, "PRESET");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new(format!("{:02} / {n:02}", (current + 1).min(n))).monospace().weak());
                if self.modified {
                    ui.label(egui::RichText::new("MUTATED").size(10.0).color(ACCENT));
                }
            });
        });
        ui.horizontal(|ui| {
            let picker_width = ui.available_width() - 80.0;
            if ui
                .add_enabled(n > 1, egui::Button::new("‹").min_size(egui::vec2(32.0, 30.0)))
                .on_hover_text("Previous preset ( , )")
                .clicked()
            {
                actions.push(Action::LoadPreset((current + n - 1) % n));
            }
            egui::ComboBox::from_id_salt("preset")
                .selected_text(self.preset_name())
                .width(picker_width)
                .height(320.0)
                .show_ui(ui, |ui| {
                    ui.set_min_width(picker_width);
                    for (i, name) in presets.iter().enumerate() {
                        if ui.selectable_label(i == current, *name).clicked() {
                            actions.push(Action::LoadPreset(i));
                        }
                    }
                });
            if ui
                .add_enabled(n > 1, egui::Button::new("›").min_size(egui::vec2(32.0, 30.0)))
                .on_hover_text("Next preset ( . )")
                .clicked()
            {
                actions.push(Action::LoadPreset((current + 1) % n));
            }
        });

        ui.add_space(2.0);
        let width = ui.available_width();
        ui.horizontal(|ui| {
            let label = if self.paused { "Resume" } else { "Pause" };
            let text = egui::RichText::new(label).strong().color(if self.paused { theme::PANEL } else { ACCENT });
            let button = egui::Button::new(text).fill(if self.paused { ACCENT } else { theme::SELECTED });
            if ui
                .add_sized([width - 88.0, 34.0], button)
                .on_hover_text("Pause / resume the simulation (Space)")
                .clicked()
            {
                self.paused = !self.paused;
            }
            if ui
                .add_sized([80.0, 34.0], egui::Button::new("Step ›"))
                .on_hover_text("Pause and advance one frame (N)")
                .clicked()
            {
                self.paused = true;
                self.single_step = true;
            }
        });
        let action_widths = theme::button_widths(ui, ["Reset", "Mutate", "Save settings"]);
        ui.horizontal(|ui| {
            if ui
                .add_sized([action_widths[0], 30.0], egui::Button::new("Reset"))
                .on_hover_text("Restart with a new seed, keeping these parameters (R)")
                .clicked()
            {
                actions.push(Action::Reset);
            }
            if ui
                .add_sized([action_widths[1], 30.0], egui::Button::new(egui::RichText::new("Mutate").color(ACCENT)))
                .on_hover_text("Discover a new set of parameters (M)")
                .clicked()
            {
                actions.push(Action::Mutate);
            }
            if ui.add_sized([action_widths[2], 30.0], egui::Button::new("Save settings"))
                .on_hover_text("Name and keep this world in your saved library").clicked() {
                self.inspector = Inspector::Library;
                if self.library_name.is_empty() { self.library_name = self.preset_label(); }
            }
        });
        ui.add_space(4.0);
        ui.separator();
        theme::inspector_tabs(ui, &mut self.inspector);
        ui.add_space(4.0);
        if scroll_inspector {
            egui::ScrollArea::vertical()
                .id_salt(("inspector", self.inspector, self.world_index))
                .auto_shrink([false, false])
                .show(ui, |ui| self.draw_inspector(ui, actions));
        } else {
            self.draw_inspector(ui, actions);
        }
    }

    fn draw_inspector(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        // Leave breathing room between controls and the scroll rail.
        ui.set_width(ui.available_width() - 6.0);
        match self.inspector {
            Inspector::World => {
                theme::section(ui, "World parameters", WORLDS[self.world_index].tagline);
                ui.label(egui::RichText::new(self.world.stats()).small().color(theme::MUTED));
                ui.add_space(4.0);
                let names = self.world.comparison_labels();
                let status = theme::MeasurementsStatus {
                    logging: self.metrics_log.as_ref().map(|log| (log.path(), log.rows())),
                    dropped: self.sampler.dropped(),
                };
                if theme::measurements(ui, self.world.metrics(), &self.history, names.as_ref(), status) {
                    actions.push(Action::ToggleMetricsLog);
                }
                ui.add_space(4.0);
                ui.push_id(self.world.id(), |ui| self.world.ui(&self.gpu, ui));
            }
            Inspector::Look => {
                theme::section(ui, "Shape the light", "Finish the image with bloom, colour and film effects.");
                self.look.ui(ui);
                ui.add_space(8.0);
                if ui
                    .button("Restore preset appearance")
                    .on_hover_text("Reset these effects to the active preset's defaults")
                    .clicked()
                {
                    self.look = self.world.post_settings();
                }
                ui.label(
                    egui::RichText::new("World-specific colours and materials are in the World tab.").small().weak(),
                );
            }
            Inspector::Tools => self.draw_tools(ui, actions),
            Inspector::Library => self.draw_library(ui, actions),
        }
        ui.add_space(12.0);
    }

    fn draw_library(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        theme::section(ui, "Saved worlds", "Keep a discovery. Return to it whenever you like.");
        ui.add(egui::TextEdit::singleline(&mut self.library_name).hint_text("Name this world…").char_limit(100).desired_width(f32::INFINITY));
        if ui.add_enabled_ui(!self.library_name.trim().is_empty(), |ui| {
            ui.add_sized([ui.available_width(), 32.0], egui::Button::new(egui::RichText::new("Save current world").color(ACCENT)).fill(theme::SELECTED))
        }).inner.clicked() {
            actions.push(Action::SaveWorld);
        }
        ui.label(egui::RichText::new(format!("Seed {}", self.seed)).monospace().small().weak());
        ui.label(egui::RichText::new("Saves the seed, world settings, appearance and view. Loading restarts the simulation; it does not resume this exact frame.").small().weak());
        ui.add_space(4.0);
        ui.separator();
        ui.horizontal(|ui| {
            theme::eyebrow(ui, &format!("YOUR LIBRARY · {}", self.library.entries.len()));
            if ui.small_button("Refresh").clicked() {
                self.library.refresh(); self.library_edit = None; self.library_delete = None;
            }
        });
        if self.library.entries.is_empty() {
            let command = format!("{} explore -w {} --install", self.program, self.world.id());
            if theme::empty_library(ui, &command) {
                self.toast("Copied".to_string());
            }
        }
        for (index, entry) in self.library.entries.iter().enumerate() {
            ui.push_id(&entry.path, |ui| {
                egui::Frame::new().fill(theme::SURFACE).stroke(egui::Stroke::new(1.0f32, theme::BORDER))
                    .inner_margin(12).corner_radius(8).show(ui, |ui| {
                    ui.set_min_width(ui.available_width());
                    ui.label(egui::RichText::new(&entry.saved.name).strong().color(ACCENT));
                    let world_index = world::find(entry.saved.settings.world_id()).unwrap_or(0);
                    ui.label(egui::RichText::new(WORLDS[world_index].name).small().weak())
                        .on_hover_text(format!("Seed {}", entry.saved.seed));
                    ui.horizontal(|ui| {
                        if ui.add_sized([64.0, 30.0], egui::Button::new(egui::RichText::new("Load").color(ACCENT)).fill(theme::SELECTED)).clicked() {
                            actions.push(Action::LoadWorld(index));
                        }
                        if ui.add(egui::Button::new("Rename").frame(false)).clicked() { self.library_edit = Some((index, entry.saved.name.clone())); self.library_delete = None; }
                        if ui.add(egui::Button::new(egui::RichText::new("Delete").color(theme::MUTED)).frame(false)).clicked() { self.library_delete = Some(index); self.library_edit = None; }
                    });
                    if let Some((editing, name)) = &mut self.library_edit {
                        if *editing == index {
                            ui.add(egui::TextEdit::singleline(name).char_limit(100).desired_width(f32::INFINITY));
                            let mut cancel = false;
                            ui.horizontal(|ui| {
                                if ui.add_enabled(!name.trim().is_empty(), egui::Button::new("Keep name")).clicked() { actions.push(Action::RenameWorld(index, name.clone())); }
                                cancel = ui.button("Cancel").clicked();
                            });
                            if cancel { self.library_edit = None; }
                        }
                    }
                    if self.library_delete == Some(index) {
                        ui.label("Remove this save from your library?");
                        ui.horizontal(|ui| {
                            if ui.button(egui::RichText::new("Remove save").color(WARN)).clicked() { actions.push(Action::DeleteWorld(index)); }
                            if ui.button("Cancel").clicked() { self.library_delete = None; }
                        });
                    }
                });
            });
        }
        for warning in &self.library.warnings { ui.colored_label(WARN, warning); }
        ui.add_space(8.0);
        ui.label(egui::RichText::new(format!("Library folder\n{}", self.library.directory.display())).small().weak());
    }

    fn draw_tools(&mut self, ui: &mut egui::Ui, actions: &mut Vec<Action>) {
        theme::section(ui, "Interact & explore", self.world.controls_hint());
        ui.add(crate::ui::Slider::new(&mut self.brush_pts, 2.0..=400.0).logarithmic(true).text("Brush radius"));
        ui.add(crate::ui::Slider::new(&mut self.camera.zoom, 0.5..=64.0).logarithmic(true).suffix("×").text("Zoom"));
        ui.horizontal(|ui| {
            if ui.button("Reset view").on_hover_text("Centre the camera and reset zoom (0 / Home)").clicked() {
                self.camera = Camera::default();
            }
            let label = if self.window.fullscreen().is_some() { "Leave fullscreen" } else { "Fullscreen" };
            if ui.button(label).on_hover_text("Toggle fullscreen (F / F11)").clicked() {
                self.toggle_fullscreen();
            }
        });
        ui.label(egui::RichText::new("Scroll to zoom · Middle-drag to pan").small().weak());
        ui.add_space(8.0);
        ui.separator();
        theme::section(ui, "Simulation resolution", theme::SIM_SCALE_HINT);
        if let Some(scale) = theme::resolution_picker(ui, self.sim_scale) {
            actions.push(Action::SetSimScale(scale));
        }
        let [w, h] = self.world_output_size;
        let [tw, th] = self.target_size();
        ui.label(
            egui::RichText::new(format!(
                "{:.0}% · simulating {w}×{h} for a {tw}×{th} window · {:.0} fps",
                self.sim_scale * 100.0,
                self.fps
            ))
            .small()
            .weak(),
        );
        ui.add_space(8.0);
        ui.separator();
        theme::section(ui, "Take a tour", "Fade through every preset of every world.");
        if ui.checkbox(&mut self.tour_enabled, "Enable tour").on_hover_text("Toggle tour mode (T)").changed() {
            self.tour_clock = TOUR_FADE;
        }
        ui.add(crate::ui::Slider::new(&mut self.tour_secs, TOUR_RANGE).logarithmic(true).text("Seconds / preset"));
        if self.tour_enabled {
            let left = (self.tour_secs - self.tour_clock).max(0.0);
            ui.add(
                egui::ProgressBar::new((self.tour_clock / self.tour_secs).clamp(0.0, 1.0))
                    .fill(theme::SELECTED)
                    .text(format!("Next preset in {left:.0}s")),
            );
        }
        ui.add_space(8.0);
        egui::CollapsingHeader::new("Keyboard shortcuts").show(ui, |ui| {
            egui::Grid::new("keys").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                for (k, v) in [
                    ("1–5", "Switch world"),
                    (", / .", "Previous / next preset"),
                    ("R / M", "Reset / mutate"),
                    ("Space / N", "Pause / single step"),
                    ("[ / ]", "Brush size"),
                    ("H / Tab", "Hide / show controls"),
                    ("F / F11", "Fullscreen"),
                    ("S / F12", "Save image"),
                    ("V", "Record video"),
                    ("T", "Tour mode"),
                    ("0 / Home", "Reset view"),
                    ("Esc", "Leave fullscreen"),
                    ("Esc Esc", "Quit"),
                ] {
                    ui.label(egui::RichText::new(k).monospace().color(ACCENT));
                    ui.label(v);
                    ui.end_row();
                }
            });
        });
        ui.add_space(8.0);
        ui.label(egui::RichText::new(format!("Capture folder\n{}", self.output_dir.display())).small().weak());
        ui.add_space(8.0);
        ui.separator();
        let commands = command_examples(&self.program, self.world.id(), self.preset_name());
        if theme::about(ui, &self.gpu_detail, &commands) {
            self.toast("Copied".to_string());
        }
        let hint = if self.program.starts_with('.') {
            "Run them in a terminal opened in Primordia's folder; add --help to any of them to see its options."
        } else {
            "Add --help to any of them to see its options."
        };
        ui.label(egui::RichText::new(hint).small().weak());
    }

    fn draw_ui(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let preset_label = self.preset_label();
        if self.show_panel {
            theme::control_panel(ctx).show(ctx, |ui| self.draw_panel(ui, actions));

            // The identity follows the panel edge, leaving the artwork centre clear.
            if ctx.available_rect().width() > 480.0 && self.world.comparison_labels().is_none() {
                egui::Area::new(egui::Id::new("world identity"))
                    .fixed_pos(ctx.available_rect().min + egui::vec2(24.0, 24.0))
                    .interactable(false)
                    .show(ctx, |ui| {
                        theme::overlay().show(ui, |ui| {
                            ui.set_width(236.0);
                            theme::eyebrow(ui, &self.world.name().to_uppercase());
                            ui.label(egui::RichText::new(preset_label.as_str()).size(19.0).color(theme::TEXT));
                        });
                    });
            }
        } else {
            egui::Area::new(egui::Id::new("show controls"))
                .fixed_pos(egui::pos2(16.0, 16.0))
                .show(ctx, |ui| {
                    if ui.add_sized([112.0, 32.0], egui::Button::new(egui::RichText::new("›  Controls").color(ACCENT))
                        .fill(theme::PANEL.gamma_multiply(0.94)).corner_radius(8))
                        .on_hover_text("Open the control panel (H / Tab)").clicked() {
                        self.show_panel = true;
                    }
                });
        }

        if let Some(labels) = self.world.comparison_labels() {
            let screen = ctx.screen_rect();
            let midpoint = screen.center().x;
            for (i, label) in labels.iter().enumerate() {
                let left = if i == 0 { ctx.available_rect().left() } else { midpoint };
                let right = if i == 0 { midpoint } else { screen.right() };
                if right - left < 156.0 { continue; }
                egui::Area::new(egui::Id::new(("comparison label", i)))
                    .fixed_pos(egui::pos2(left + 16.0, 68.0))
                    .interactable(false)
                    .show(ctx, |ui| {
                        theme::overlay().show(ui, |ui| {
                            theme::eyebrow(ui, label);
                        });
                    });
            }
        }

        // Status badges (top right): recording and paused.
        let recording = self.recorder.as_ref().map(|r| r.started.elapsed().as_secs());
        if recording.is_some() || self.paused {
            egui::Area::new(egui::Id::new("badges"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
                .interactable(false)
                .show(ctx, |ui| {
                    theme::overlay().show(ui, |ui| {
                        ui.set_min_width(118.0);
                        if let Some(secs) = recording {
                            ui.horizontal(|ui| {
                                theme::status_dot(ui, WARN);
                                ui.label(
                                    egui::RichText::new(format!("REC {:02}:{:02}", secs / 60, secs % 60))
                                        .monospace()
                                        .color(WARN),
                                );
                            });
                        }
                        if self.paused {
                            ui.label(egui::RichText::new("PAUSED").monospace().strong());
                        }
                    });
                });
        }

        if let Some(toast) = &self.toast {
            let age = toast.shown.elapsed().as_secs_f32();
            if age < toast.secs {
                let alpha = (toast.secs - age).min(1.0);
                egui::Area::new(egui::Id::new("toast"))
                    .anchor(egui::Align2::CENTER_BOTTOM, egui::vec2(0.0, -28.0))
                    .interactable(false)
                    .show(ctx, |ui| {
                        ui.set_opacity(alpha);
                        theme::overlay().show(ui, |ui| {
                            ui.set_max_width((ctx.screen_rect().width() - 60.0).max(200.0));
                            ui.label(egui::RichText::new(toast.text.as_str()).size(15.0));
                        });
                    });
            }
        }

        // Brush outline under the cursor (not shown in screenshots or videos).
        if let Some(c) = self.input.cursor {
            let dragging = self.input.left || self.input.right;
            if dragging || !ctx.is_pointer_over_area() {
                let ppp = ctx.pixels_per_point();
                let color = if self.input.left {
                    ACCENT
                } else if self.input.right {
                    WARN
                } else {
                    egui::Color32::from_white_alpha(70)
                };
                ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("brush"))).circle_stroke(
                    egui::pos2(c[0] / ppp, c[1] / ppp),
                    self.brush_pts,
                    egui::Stroke::new(1.0f32, color),
                );
            }
        }

        self.popup_was_open = egui::Popup::is_any_open(ctx);
    }
}

/// Use wall time, including stalls and skipped redraws, to keep videos at real speed.
fn recording_frames_due(elapsed: Duration, written: u64) -> u32 {
    let total = (elapsed.as_secs_f64() * f64::from(RECORD_FPS)).floor() as u64 + 1;
    total.saturating_sub(written).min(u64::from(u32::MAX)) as u32
}

/// `<world>_<preset>[-mutated]_s<seed>_<utc timestamp>`, e.g.
/// `lenia_pearl-reef_s1234_2026-09-22_10-15-00`. With the seed, `primordia
/// render --world lenia --preset "Pearl Reef" --seed 1234` starts from the same
/// initial state (at the same simulation size).
fn capture_stem(world: &str, preset: &str, mutated: bool, seed: u64, timestamp: &str) -> String {
    let mutated = if mutated { "-mutated" } else { "" };
    format!("{world}_{}{mutated}_s{seed}_{timestamp}", headless::slug(preset))
}

/// A fresh copy of `world` (`WORLDS[index]`) for an output of `size`, with the
/// same preset and parameters, started from `seed`. Worlds without a
/// saved-settings recipe come back as their preset.
fn rebuilt(gpu: &Gpu, index: usize, world: &dyn World, size: [u32; 2], seed: u64) -> Result<Box<dyn World>> {
    let settings = world.settings().ok();
    let mut copy = world::create_at(gpu, index, size, Some(world.preset()), seed)?;
    if let Some(settings) = &settings {
        copy.restore_settings(gpu, settings, seed)?;
    }
    Ok(copy)
}

/// Simulation scale when `--sim-scale` is not given: integrated, virtual and
/// software GPUs start at [`WEAK_GPU_SIM_SCALE`].
fn default_sim_scale(device: wgpu::DeviceType) -> f32 {
    match device {
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::VirtualGpu | wgpu::DeviceType::Cpu => WEAK_GPU_SIM_SCALE,
        wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::Other => 1.0,
    }
}

/// "an integrated GPU", "a discrete GPU", ... for messages.
fn adapter_kind(device: wgpu::DeviceType) -> &'static str {
    match device {
        wgpu::DeviceType::IntegratedGpu => "an integrated GPU",
        wgpu::DeviceType::DiscreteGpu => "a discrete GPU",
        wgpu::DeviceType::VirtualGpu => "a virtual GPU",
        wgpu::DeviceType::Cpu => "a software renderer",
        wgpu::DeviceType::Other => "a GPU of unknown type",
    }
}

/// "NVIDIA GeForce RTX 4090 · Vulkan · discrete GPU".
fn describe_adapter(info: &wgpu::AdapterInfo) -> String {
    let backend = match info.backend {
        wgpu::Backend::Vulkan => "Vulkan",
        wgpu::Backend::Metal => "Metal",
        wgpu::Backend::Dx12 => "Direct3D 12",
        wgpu::Backend::Gl => "OpenGL",
        other => other.to_str(),
    };
    let kind = adapter_kind(info.device_type);
    let kind = kind.split_once(' ').map_or(kind, |(_article, rest)| rest);
    format!("{} · {backend} · {kind}", info.name)
}

/// Commands to try in a terminal, for the world on screen. `program` is how
/// the exe is called there (its file name without `.exe`).
fn command_examples(program: &str, world: &str, preset: &str) -> String {
    let plain = |c: char| c.is_ascii_alphanumeric() || c == '-' || c == '_';
    let preset = if preset.chars().all(plain) { preset.to_string() } else { format!("\"{preset}\"") };
    format!("{program} list\n{program} render -w {world} -p {preset}\n{program} explore -w {world} --install")
}

/// How to run this exe from a terminal: its name when its folder is on the
/// PATH (e.g. after `cargo install`), else `.\name` (`./name`) for a terminal
/// opened in that folder. The name is the file's own, as downloaded or renamed,
/// unless it would need quoting; then it is `primordia`.
fn program_name() -> String {
    let exe = std::env::current_exe().ok();
    let plain = |name: &String| name.chars().all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
    let name = exe
        .as_deref()
        .and_then(Path::file_stem)
        .map(|stem| stem.to_string_lossy().into_owned())
        .filter(plain)
        .unwrap_or_else(|| "primordia".to_string());
    let folder = exe.as_deref().and_then(Path::parent);
    let on_path = std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| Some(dir.as_path()) == folder));
    match (on_path, cfg!(windows)) {
        (true, _) => name,
        (false, true) => format!(".\\{name}"),
        (false, false) => format!("./{name}"),
    }
}

/// Title-bar and taskbar icons for a window on a display with this scale factor.
fn with_icons(mut attrs: WindowAttributes, scale_factor: f64) -> WindowAttributes {
    let icon = |points: f64| {
        let size = (points * scale_factor).round().clamp(16.0, 256.0) as u32;
        theme::icon_rgba(theme::ICON, size)
            .and_then(|rgba| Ok(Icon::from_rgba(rgba, size, size)?))
            .map_err(|e| log::warn!("window icon: {e:#}"))
            .ok()
    };
    // Windows shows this one in the title bar; X11 window managers scale one
    // larger icon as they need; macOS and Wayland ignore it.
    attrs = attrs.with_window_icon(icon(if cfg!(windows) { 16.0 } else { 64.0 }));
    // The big one goes in the taskbar and Alt+Tab.
    #[cfg(windows)]
    {
        use winit::platform::windows::WindowAttributesExtWindows as _;
        attrs = attrs.with_taskbar_icon(icon(32.0));
    }
    attrs
}

fn scaled(size: [u32; 2], scale: f32) -> [u32; 2] {
    let s = scale.clamp(0.1, 4.0);
    [((size[0] as f32 * s) as u32).max(16), ((size[1] as f32 * s) as u32).max(16)]
}

/// `dir/stem.ext`, or `dir/stem-2.ext`, `-3`, ... if that already exists.
fn unique_path(dir: &Path, stem: &str, ext: &str) -> PathBuf {
    let mut path = dir.join(format!("{stem}.{ext}"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{stem}-{n}.{ext}"));
        n += 1;
    }
    path
}

/// Current UTC time as `YYYY-MM-DD_hh-mm-ss` (no date crate needed).
fn utc_timestamp() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil-from-days algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}_{:02}-{:02}-{:02}", rem / 3600, rem / 60 % 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_names_carry_the_seed() {
        let stem = capture_stem("lenia", "Pearl Reef", false, 1234, "2026-09-22_10-15-00");
        assert_eq!(stem, "lenia_pearl-reef_s1234_2026-09-22_10-15-00");
        let stem = capture_stem("particle-life", "Predator & Prey", true, u64::MAX, "2026-09-22_10-15-00");
        assert_eq!(stem, format!("particle-life_predator-prey-mutated_s{}_2026-09-22_10-15-00", u64::MAX));
        // The stamp keeps its own format and stays a valid file name everywhere.
        let stem = capture_stem("physarum", "Dendrites", false, 7, &utc_timestamp());
        assert!(stem.starts_with("physarum_dendrites_s7_20"), "{stem}");
        assert!(stem.chars().all(|c| c.is_ascii_alphanumeric() || "_-".contains(c)), "{stem}");
    }

    #[test]
    fn a_resolution_change_keeps_every_worlds_preset_and_parameters() {
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let mut shrunk = 0;
        for (index, entry) in WORLDS.iter().enumerate() {
            let mut world = world::create_at(&gpu, index, [320, 180], Some(1), 7).unwrap();
            world.mutate(&gpu, 8);
            let copy = rebuilt(&gpu, index, &*world, [160, 90], 9).unwrap();
            // (Lenia's mutation may start from another preset.)
            assert_eq!(copy.preset(), world.preset(), "{}", entry.id);
            let json = |w: &dyn World| w.settings().map(|s| serde_json::to_value(s).unwrap()).ok();
            assert_eq!(json(&*copy), json(&*world), "{} lost its parameters", entry.id);
            // Particle Life's domain does not follow the output size; the grid worlds' does.
            assert!(copy.size()[0] <= world.size()[0], "{} grew", entry.id);
            shrunk += usize::from(copy.size()[0] < world.size()[0]);
        }
        assert!(shrunk >= 3, "only {shrunk} worlds simulate fewer cells at a lower resolution");
    }

    #[test]
    fn weak_gpus_start_at_half_resolution() {
        use wgpu::DeviceType as D;
        assert_eq!(default_sim_scale(D::DiscreteGpu), 1.0);
        assert_eq!(default_sim_scale(D::Other), 1.0);
        for weak in [D::IntegratedGpu, D::VirtualGpu, D::Cpu] {
            assert_eq!(default_sim_scale(weak), WEAK_GPU_SIM_SCALE);
        }
        // Every picker option is a scale `scaled` accepts unchanged.
        for scale in theme::SIM_SCALES {
            assert_eq!(scaled([1600, 900], scale), [(1600.0 * scale) as u32, (900.0 * scale) as u32]);
        }
    }

    #[test]
    fn slow_frames_blame_another_program_only_when_they_are_sudden() {
        let ms = Duration::from_millis;
        // Nothing to compare with yet.
        assert_eq!(SlowGpu::classify(ms(900), None), SlowGpu::Unknown);
        // 16 ms frames, then one takes 1.5 s: something else wants the GPU.
        assert_eq!(SlowGpu::classify(ms(1500), Some(ms(16))), SlowGpu::Contended);
        // Frames have taken about a third of a second all along: the simulation is too big.
        assert_eq!(SlowGpu::classify(ms(320), Some(ms(330))), SlowGpu::Overloaded { ms: 330.0 });
        assert_eq!(SlowGpu::classify(ms(600), Some(ms(330))), SlowGpu::Overloaded { ms: 600.0 });
        assert!(SlowGpu::Contended.title_note().contains("another program"));
        let note = SlowGpu::Overloaded { ms: 330.4 }.title_note();
        assert!(note.contains("330 ms per frame") && !note.contains("another program"), "{note}");
        assert!(!SlowGpu::Unknown.title_note().contains("another program"));
    }

    #[test]
    fn about_describes_the_gpu_and_suggests_commands_for_the_world_on_screen() {
        let info = wgpu::AdapterInfo {
            name: "AMD Radeon(TM) Graphics".to_string(),
            vendor: 0x1002,
            device: 0x164e,
            device_type: wgpu::DeviceType::IntegratedGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };
        assert_eq!(describe_adapter(&info), "AMD Radeon(TM) Graphics · Vulkan · integrated GPU");
        let info = wgpu::AdapterInfo { device_type: wgpu::DeviceType::Cpu, backend: wgpu::Backend::Dx12, ..info };
        assert_eq!(describe_adapter(&info), "AMD Radeon(TM) Graphics · Direct3D 12 · software renderer");

        let commands = command_examples(".\\primordia", "lenia", "Pearl Reef");
        assert_eq!(
            commands,
            ".\\primordia list\n.\\primordia render -w lenia -p \"Pearl Reef\"\n.\\primordia explore -w lenia --install"
        );
        assert!(command_examples("primordia", "physarum", "Dendrites").contains("render -w physarum -p Dendrites\n"));
        // Every world id and preset name the commands can show must parse back.
        use clap::Parser as _;
        #[derive(clap::Parser)]
        struct Render {
            #[arg(short, long)]
            world: String,
            #[arg(short, long)]
            preset: String,
        }
        let examples = [("lenia", "Pearl Reef"), ("particle-life", "Predator & Prey"), ("physarum", "Dendrites")];
        for (world, preset) in examples {
            let line = command_examples("primordia", world, preset).lines().nth(1).unwrap().to_string();
            // A shell's word splitting, for the double quotes the commands use.
            let mut words = vec![String::new()];
            let mut quoted = false;
            for c in line.chars() {
                match c {
                    '"' => quoted = !quoted,
                    ' ' if !quoted => words.push(String::new()),
                    c => words.last_mut().unwrap().push(c),
                }
            }
            let parsed = Render::try_parse_from(&words[1..]).unwrap();
            assert_eq!((parsed.world.as_str(), parsed.preset.as_str()), (world, preset));
        }
        // The test binary's own name, bare or relative depending on whether cargo
        // put its folder on the PATH (it does on Windows, to find DLLs).
        let program = program_name();
        let stem = std::env::current_exe().unwrap().file_stem().unwrap().to_string_lossy().into_owned();
        assert!([format!(".\\{stem}"), format!("./{stem}"), stem].contains(&program), "{program}");
    }

    #[test]
    fn recording_catches_up_after_slow_or_skipped_frames() {
        assert_eq!(recording_frames_due(Duration::ZERO, 0), 1);
        assert_eq!(recording_frames_due(Duration::from_secs(1), 1), 60);
        assert_eq!(recording_frames_due(Duration::from_secs(3), 61), 120);
        assert_eq!(recording_frames_due(Duration::from_secs(3), 181), 0);
    }
}
