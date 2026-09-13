//! Interactive window: winit event loop, wgpu surface, egui control panel.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Context as _, Result};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{Fullscreen, Window, WindowId};

use crate::capture::{self, Readback};
use crate::gpu::Gpu;
use crate::headless::{self, Encode};
use crate::post::{Post, PostSettings};
use crate::rng;
use crate::world::{self, Camera, Frame, Pointer, ViewXform, World, WORLDS};

const ACCENT: egui::Color32 = egui::Color32::from_rgb(94, 234, 212);
const WARN: egui::Color32 = egui::Color32::from_rgb(255, 90, 90);
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

pub struct AppOptions {
    pub world: String,
    pub preset: Option<String>,
    pub seed: Option<u64>,
    /// Logical window size; `None` picks a default that fits the screen.
    pub window_size: Option<[u32; 2]>,
    pub fullscreen: bool,
    pub vsync: bool,
    pub sim_scale: f32,
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
            Shortcut::Step | Shortcut::PrevPreset | Shortcut::NextPreset | Shortcut::BrushSmaller | Shortcut::BrushBigger
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

    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    popup_was_open: bool,

    input: Input,
    show_panel: bool,
    paused: bool,
    single_step: bool,
    /// Brush radius in logical points, so it looks the same at any DPI.
    brush_pts: f32,

    time: f32,
    frame: u64,
    last_instant: Instant,
    started: Instant,
    frames_since_start: u64,
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
    /// Until when the title shows "GPU busy" (extended by every slow frame).
    gpu_busy_until: Option<Instant>,
    /// Last time a slow-GPU warning was logged (they are rate-limited).
    gpu_warned_at: Option<Instant>,
    /// Development aid: extra GPU work per frame (`PRIMORDIA_DEBUG_GPU_STALL_MS`).
    stall: Option<crate::stall::Stall>,
}

impl Drop for State {
    fn drop(&mut self) {
        // Close ffmpeg's input so an in-progress recording is finalised on exit.
        self.stop_recording(None);
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
        let window = Arc::new(event_loop.create_window(attrs).context("creating window")?);

        let instance = Gpu::create_instance();
        let surface = instance.create_surface(window.clone()).context("creating surface")?;
        let gpu = pollster::block_on(Gpu::new(instance, Some(&surface)))?;
        let gpu_name = gpu.adapter_name();

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
        let sim_size = scaled([w, h], opts.sim_scale);
        let (world_index, world) = world::create(&gpu, &opts.world, sim_size, opts.preset.as_deref(), seed)?;
        let look = world.post_settings();

        let egui_ctx = egui::Context::default();
        let mut visuals = egui::Visuals::dark();
        visuals.selection.bg_fill = egui::Color32::from_rgb(22, 120, 110);
        visuals.hyperlink_color = ACCENT;
        // Pin the theme: following the OS into light mode would put dark text on
        // our dark translucent panel.
        egui_ctx.set_theme(egui::Theme::Dark);
        egui_ctx.set_visuals_of(egui::Theme::Dark, visuals);
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

        log::info!(
            "opened {} at {}x{} (world domain {}x{})",
            world.name(),
            w,
            h,
            world.size()[0],
            world.size()[1]
        );
        let now = Instant::now();
        let stall = crate::stall::Stall::from_env(&gpu);
        let mut state = Self {
            window,
            gpu,
            gpu_name,
            surface,
            config,
            minimized: false,
            post,
            world,
            world_index,
            look,
            camera: Camera::default(),
            sim_scale: opts.sim_scale,
            modified: false,
            egui_ctx,
            egui_state,
            egui_renderer,
            popup_was_open: false,
            input: Input::default(),
            show_panel: !opts.hide_ui,
            paused: false,
            single_step: false,
            brush_pts: 30.0,
            time: 0.0,
            frame: 0,
            last_instant: now,
            started: now,
            frames_since_start: 0,
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
            gpu_warned_at: None,
            stall,
        };
        state.toast_for(
            "Drag to interact · wheel zooms · middle-drag pans · H hides the panel · Space pauses".to_string(),
            6.0,
        );
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
                        self.camera.center = [self.camera.center[0].rem_euclid(1.0), self.camera.center[1].rem_euclid(1.0)];
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
                if event.state == ElementState::Pressed && !response.consumed && !self.egui_ctx.wants_keyboard_input() =>
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
        let anchor = self.view().apply(cursor_uv);
        self.camera.zoom = (self.camera.zoom * factor).clamp(0.5, 64.0);
        let after = self.view();
        self.camera.center = [
            (anchor[0] - (cursor_uv[0] - 0.5) * after.scale[0]).rem_euclid(1.0),
            (anchor[1] - (cursor_uv[1] - 0.5) * after.scale[1]).rem_euclid(1.0),
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
                self.toast(self.preset_name().to_string());
            }
            Action::Reset => {
                if let Some(err) = self.catch_oom(|s| s.world.reset(&s.gpu, seed)) {
                    self.report_oom(&err);
                }
            }
            Action::Mutate => {
                if let Some(err) = self.catch_oom(|s| s.world.mutate(&s.gpu, seed)) {
                    self.report_oom(&err);
                    return Ok(());
                }
                self.look = self.world.post_settings();
                self.modified = true;
                self.toast("Mutated".to_string());
            }
            Action::Screenshot => self.pending_screenshot = true,
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

    /// Runs `f` inside an out-of-memory error scope and returns the error, if any.
    fn catch_oom(&mut self, f: impl FnOnce(&mut Self)) -> Option<wgpu::Error> {
        self.gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        f(self);
        pollster::block_on(self.gpu.device.pop_error_scope())
    }

    fn report_oom(&mut self, err: &wgpu::Error) {
        log::error!("out of GPU memory: {err}");
        self.toast_for(
            "Out of GPU memory. Close other GPU-heavy programs, or make the window smaller.".to_string(),
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
            pos: view.apply(uv),
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
    /// when another program keeps the GPU busy for a long time.
    fn gpu_ready(&mut self) -> bool {
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
        let now = Instant::now();
        if self.frame_done.load(Ordering::Acquire) {
            if let Some(since) = self.gpu_wait_since.take() {
                let waited = now - since;
                if waited > Duration::from_millis(250) {
                    self.gpu_busy_until = Some(now + Duration::from_secs(5));
                    self.warn_gpu_busy(&format!("waited {:.1}s for the GPU to finish a frame", waited.as_secs_f32()));
                }
            }
            return true;
        }
        let since = *self.gpu_wait_since.get_or_insert(now);
        if now - since > Duration::from_millis(1500) && self.gpu_busy_until.is_none_or(|t| t <= now) {
            self.gpu_busy_until = Some(now + Duration::from_secs(5));
            self.warn_gpu_busy("the GPU is taking very long to finish frames");
            let title = self.title();
            self.window.set_title(&title);
        }
        false
    }

    /// Logs a slow-GPU warning, at most once every 10 seconds.
    fn warn_gpu_busy(&mut self, what: &str) {
        let now = Instant::now();
        if self.gpu_warned_at.is_none_or(|t| now - t > Duration::from_secs(10)) {
            self.gpu_warned_at = Some(now);
            log::warn!("{what}; another program may be using the GPU heavily");
        }
    }

    fn title(&self) -> String {
        let paused = if self.paused { " · paused" } else { "" };
        let busy = if self.gpu_busy_until.is_some_and(|t| Instant::now() < t) {
            " · GPU busy (another program may be using it heavily)"
        } else {
            ""
        };
        format!("Primordia — {} · {} — {:.0} fps{paused}{busy}", self.world.name(), self.preset_label(), self.fps)
    }

    fn redraw(&mut self) -> Result<()> {
        // The OS can ask for redraws (resize, expose) while the GPU is still busy.
        if self.minimized || !self.gpu_ready() {
            return Ok(());
        }
        let now = Instant::now();
        let elapsed = (now - self.last_instant).as_secs_f32();
        // The simulation step is clamped; the fps readout uses the real frame time.
        let dt = elapsed.min(0.1);
        self.last_instant = now;
        if elapsed > 0.0 {
            self.fps = if self.fps == 0.0 { 1.0 / elapsed } else { self.fps * 0.95 + 0.05 / elapsed };
        }
        self.frames_since_start += 1;
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
        let full_output = ctx.run(raw_input, |ctx| self.draw_ui(ctx, &mut actions));
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
            let frame = Frame { gpu: &self.gpu, time: self.time, dt, frame: self.frame, view, target_size: target, pointer };
            if !self.paused || self.single_step {
                self.world.step(&frame, &mut encoder);
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
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: target, pixels_per_point: full_output.pixels_per_point };
        let extra = self.egui_renderer.update_buffers(&self.gpu.device, &self.gpu.queue, &mut encoder, &paint_jobs, &screen);
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

    /// `<world>_<preset>_<utc timestamp>` for screenshot / recording file names.
    fn capture_stem(&self) -> String {
        let mutated = if self.modified { "-mutated" } else { "" };
        format!("{}_{}{mutated}_{}", self.world.id(), headless::slug(self.preset_name()), utc_timestamp())
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
        let mut encoder = self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("screenshot") });
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

    fn draw_ui(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let preset_label = self.preset_label();
        if self.show_panel {
            let frame = egui::Frame::side_top_panel(&ctx.style())
                .fill(egui::Color32::from_rgba_unmultiplied(10, 12, 18, 225))
                .inner_margin(egui::Margin::same(12));
            egui::SidePanel::left("controls").resizable(false).exact_width(310.0).frame(frame).show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    ui.label(egui::RichText::new("PRIMORDIA").size(24.0).strong().color(ACCENT));
                    ui.label(egui::RichText::new("artificial life on the GPU").weak().italics());
                    ui.label(
                        egui::RichText::new(format!("{:.0} fps · {}", self.fps, self.gpu_name)).small().weak(),
                    );
                    ui.separator();

                    for (i, entry) in WORLDS.iter().enumerate() {
                        let selected = i == self.world_index;
                        let text = egui::RichText::new(format!("{}  {}", i + 1, entry.name)).size(15.0);
                        if ui.selectable_label(selected, text).on_hover_text(entry.tagline).clicked() && !selected {
                            actions.push(Action::SwitchWorld(i));
                        }
                    }
                    ui.label(egui::RichText::new(WORLDS[self.world_index].tagline).weak().italics());
                    ui.label(egui::RichText::new(self.world.controls_hint()).small().color(ACCENT));
                    ui.separator();

                    let presets = self.world.presets();
                    let current = self.world.preset();
                    let n = presets.len();
                    ui.horizontal(|ui| {
                        if ui.add_enabled(n > 1, egui::Button::new("◀")).on_hover_text("Previous preset ( , )").clicked() {
                            actions.push(Action::LoadPreset((current + n - 1) % n));
                        }
                        egui::ComboBox::from_id_salt("preset").selected_text(preset_label.as_str()).width(200.0).show_ui(
                            ui,
                            |ui| {
                                for (i, name) in presets.iter().enumerate() {
                                    if ui.selectable_label(i == current, *name).clicked() {
                                        actions.push(Action::LoadPreset(i));
                                    }
                                }
                            },
                        );
                        if ui.add_enabled(n > 1, egui::Button::new("▶")).on_hover_text("Next preset ( . )").clicked() {
                            actions.push(Action::LoadPreset((current + 1) % n));
                        }
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Reset").on_hover_text("Restart from a new seed (R)").clicked() {
                            actions.push(Action::Reset);
                        }
                        if ui.button("Mutate").on_hover_text("Random new parameters (M)").clicked() {
                            actions.push(Action::Mutate);
                        }
                        let label = if self.paused { "Play" } else { "Pause" };
                        if ui.button(label).on_hover_text("Space").clicked() {
                            self.paused = !self.paused;
                        }
                        if ui.button("Step").on_hover_text("Advance one frame (N)").clicked() {
                            self.paused = true;
                            self.single_step = true;
                        }
                    });
                    let stats = self.world.stats();
                    if !stats.is_empty() {
                        ui.label(egui::RichText::new(stats).monospace().small());
                    }
                    ui.separator();

                    egui::CollapsingHeader::new("Simulation").default_open(true).show(ui, |ui| {
                        self.world.ui(&self.gpu, ui);
                    });
                    egui::CollapsingHeader::new("Look").show(ui, |ui| {
                        self.look.ui(ui);
                        if ui.button("Restore preset look").clicked() {
                            self.look = self.world.post_settings();
                        }
                    });
                    egui::CollapsingHeader::new("Brush & camera").show(ui, |ui| {
                        ui.label(egui::RichText::new(self.world.controls_hint()).weak());
                        ui.add(egui::Slider::new(&mut self.brush_pts, 2.0..=400.0).logarithmic(true).text("Brush radius"));
                        ui.add(egui::Slider::new(&mut self.camera.zoom, 0.5..=64.0).logarithmic(true).text("Zoom"));
                        if ui.button("Reset view (0)").clicked() {
                            self.camera = Camera::default();
                        }
                    });
                    egui::CollapsingHeader::new("Tour").show(ui, |ui| {
                        ui.label(egui::RichText::new("Screensaver: fade through every preset of every world.").weak());
                        if ui.checkbox(&mut self.tour_enabled, "Tour mode (T)").changed() {
                            self.tour_clock = TOUR_FADE;
                        }
                        ui.add(egui::Slider::new(&mut self.tour_secs, TOUR_RANGE).logarithmic(true).text("Seconds / preset"));
                        if self.tour_enabled {
                            let left = (self.tour_secs - self.tour_clock).max(0.0);
                            ui.label(egui::RichText::new(format!("next in {left:.0}s")).small().weak());
                        }
                    });
                    egui::CollapsingHeader::new("Keys").show(ui, |ui| {
                        egui::Grid::new("keys").num_columns(2).spacing([12.0, 2.0]).show(ui, |ui| {
                            for (k, v) in [
                                ("1-4", "switch world"),
                                (", / .", "previous / next preset"),
                                ("R / M", "reset / mutate"),
                                ("Space / N", "pause / single step"),
                                ("Wheel", "zoom at cursor"),
                                ("Middle drag", "pan"),
                                ("[ / ]", "brush size"),
                                ("H / Tab", "hide panel"),
                                ("F / F11", "fullscreen"),
                                ("S / F12", "screenshot"),
                                ("V", "record video (MP4)"),
                                ("T", "tour mode"),
                                ("0 / Home", "reset view"),
                                ("Esc", "leave fullscreen · Esc Esc quits"),
                            ] {
                                ui.label(egui::RichText::new(k).monospace().color(ACCENT));
                                ui.label(v);
                                ui.end_row();
                            }
                        });
                    });
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        if ui.button("Screenshot (F12)").clicked() {
                            actions.push(Action::Screenshot);
                        }
                        let label = if self.recorder.is_some() { "Stop recording (V)" } else { "Record video (V)" };
                        if ui.button(label).clicked() {
                            actions.push(Action::ToggleRecording);
                        }
                    });
                    ui.label(egui::RichText::new(format!("Saved to {}", self.output_dir.display())).small().weak());
                });
            });
        }

        // Status badges (top right): recording and paused.
        let recording = self.recorder.as_ref().map(|r| r.started.elapsed().as_secs());
        if recording.is_some() || self.paused {
            egui::Area::new(egui::Id::new("badges"))
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-16.0, 16.0))
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        if let Some(secs) = recording {
                            ui.label(
                                egui::RichText::new(format!("● REC {:02}:{:02}", secs / 60, secs % 60))
                                    .monospace()
                                    .color(WARN)
                                    .strong(),
                            );
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
                        egui::Frame::popup(ui.style()).show(ui, |ui| {
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
                    egui::Stroke::new(1.0, color),
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
    fn recording_catches_up_after_slow_or_skipped_frames() {
        assert_eq!(recording_frames_due(Duration::ZERO, 0), 1);
        assert_eq!(recording_frames_due(Duration::from_secs(1), 1), 60);
        assert_eq!(recording_frames_due(Duration::from_secs(3), 61), 120);
        assert_eq!(recording_frames_due(Duration::from_secs(3), 181), 0);
    }
}
