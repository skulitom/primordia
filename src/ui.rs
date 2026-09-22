//! Shared visual language for the interactive laboratory.

use egui::{Color32, FontId, RichText, Stroke, TextStyle};

#[cfg(test)]
#[path = "ui_preview.rs"]
mod preview;

pub const ACCENT: Color32 = Color32::from_rgb(123, 224, 196);
pub const TEXT: Color32 = Color32::from_rgb(228, 235, 234);
pub const MUTED: Color32 = Color32::from_rgb(146, 164, 166);
pub const PANEL: Color32 = Color32::from_rgb(16, 23, 27);
pub const SURFACE: Color32 = Color32::from_rgb(24, 34, 39);
pub const BORDER: Color32 = Color32::from_rgb(43, 58, 63);
pub const SELECTED: Color32 = Color32::from_rgb(30, 65, 59);
pub const WARN: Color32 = Color32::from_rgb(255, 128, 130);
pub const PANEL_WIDTH: f32 = 364.0;
/// Corner radius of buttons, pickers, text fields and checkboxes. egui draws a
/// checkbox's 14 px box with it too: much above 3 turns the box into a disc
/// that reads as a radio button or a status dot.
const WIDGET_RADIUS: u8 = 3;

/// The app icon: a disc cut from a Physarum "Galaxy" render, as PNG images
/// from 16 to 256 pixels. build.rs embeds the same file in the Windows exe.
pub const ICON: &[u8] = include_bytes!("../assets/primordia.ico");

/// Where to find Primordia on the web (from Cargo.toml).
pub const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

pub fn control_panel(ctx: &egui::Context) -> egui::SidePanel {
    let frame = egui::Frame::side_top_panel(&ctx.style())
        .fill(PANEL)
        .stroke(Stroke::new(1.0f32, BORDER))
        .inner_margin(egui::Margin::same(18));
    egui::SidePanel::left("controls")
        .resizable(false)
        // An overflowing child expands the response beyond the clipped fill.
        // egui draws its automatic separator at that response edge, over the
        // artwork. Use only the frame border, which respects the panel clip.
        .show_separator_line(false)
        .exact_width(PANEL_WIDTH.min(ctx.screen_rect().width()))
        .frame(frame)
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Inspector {
    #[default]
    World,
    Look,
    Tools,
    Library,
}

pub fn configure(ctx: &egui::Context) {
    // Always use the dark theme: the simulation is the background of the UI.
    ctx.set_theme(egui::Theme::Dark);
    let mut style = (*ctx.style()).clone();
    style.text_styles = [
        (TextStyle::Heading, FontId::proportional(21.0)),
        (TextStyle::Body, FontId::proportional(13.0)),
        (TextStyle::Button, FontId::proportional(13.0)),
        (TextStyle::Small, FontId::proportional(11.5)),
        (TextStyle::Monospace, FontId::monospace(11.5)),
    ]
    .into();
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);
    style.spacing.interact_size = egui::vec2(32.0, 26.0);
    style.spacing.slider_width = 100.0;
    style.spacing.combo_width = 140.0;
    style.spacing.indent = 12.0;
    style.spacing.scroll = egui::style::ScrollStyle::solid();
    style.spacing.scroll.bar_width = 4.0;
    style.wrap_mode = Some(egui::TextWrapMode::Wrap);
    style.animation_time = 0.15;

    let v = &mut style.visuals;
    v.panel_fill = PANEL;
    v.window_fill = SURFACE;
    v.extreme_bg_color = PANEL;
    v.faint_bg_color = SURFACE;
    v.code_bg_color = SURFACE;
    v.weak_text_color = Some(MUTED);
    v.selection.bg_fill = SELECTED;
    v.selection.stroke = Stroke::new(1.0f32, ACCENT);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = Color32::from_rgb(235, 194, 130);
    v.error_fg_color = WARN;
    v.window_stroke = Stroke::new(1.0f32, BORDER);
    v.window_corner_radius = 10.into();
    v.menu_corner_radius = 8.into();
    v.slider_trailing_fill = true;
    v.collapsing_header_frame = true;
    for widget in [
        &mut v.widgets.noninteractive,
        &mut v.widgets.inactive,
        &mut v.widgets.hovered,
        &mut v.widgets.active,
        &mut v.widgets.open,
    ] {
        widget.corner_radius = WIDGET_RADIUS.into();
        widget.bg_stroke = Stroke::new(1.0f32, BORDER);
        widget.fg_stroke = Stroke::new(1.0f32, TEXT);
        widget.expansion = 0.0;
    }
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.weak_bg_fill = SURFACE;
    v.widgets.inactive.bg_fill = Color32::from_rgb(36, 49, 55);
    v.widgets.inactive.weak_bg_fill = SURFACE;
    v.widgets.hovered.bg_fill = Color32::from_rgb(47, 67, 70);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(35, 49, 54);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0f32, MUTED);
    v.widgets.active.bg_fill = SELECTED;
    v.widgets.active.weak_bg_fill = SELECTED;
    v.widgets.active.bg_stroke = Stroke::new(1.0f32, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0f32, ACCENT);
    v.widgets.open.bg_fill = SELECTED;
    v.widgets.open.weak_bg_fill = SELECTED;
    ctx.set_style(style);
}

pub fn eyebrow(ui: &mut egui::Ui, title: &str) {
    ui.label(RichText::new(title).size(10.5).strong().color(MUTED));
}

pub fn section(ui: &mut egui::Ui, title: &str, description: &str) {
    ui.label(RichText::new(title).size(16.0).strong().color(TEXT));
    ui.label(RichText::new(description).small().color(MUTED));
    ui.add_space(4.0);
}

/// A label above a full-width picker. Long selected names stay on one line;
/// the tooltip and popup keep the full text available in narrow inspectors.
pub fn dropdown<R>(
    ui: &mut egui::Ui,
    label: &str,
    selected: &str,
    contents: impl FnOnce(&mut egui::Ui) -> R,
) -> egui::InnerResponse<Option<R>> {
    ui.vertical(|ui| {
        ui.spacing_mut().item_spacing.y = 4.0;
        let label_response = ui.label(RichText::new(label).color(MUTED));
        let width = ui.available_width();
        let mut response = egui::ComboBox::from_id_salt(label)
            .width(width)
            .height(300.0)
            .wrap_mode(egui::TextWrapMode::Truncate)
            .selected_text(selected)
            .show_ui(ui, |ui| {
                ui.set_min_width((width - 16.0).max(0.0));
                contents(ui)
            });
        response.response = response.response.labelled_by(label_response.id).on_hover_text(selected);
        response
    })
    .inner
}

pub fn overlay() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL.gamma_multiply(0.96))
        .stroke(Stroke::new(1.0f32, BORDER))
        .corner_radius(10)
        .inner_margin(egui::Margin::symmetric(14, 10))
}

/// Small procedural mark: a nucleus and three orbiting cells.
pub fn mark(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    p.circle_stroke(c, 14.0, Stroke::new(1.0f32, ACCENT.gamma_multiply(0.5)));
    p.circle_stroke(c, 8.0, Stroke::new(1.0f32, ACCENT));
    p.circle_filled(c, 3.0, ACCENT);
    for offset in [egui::vec2(0.0, -14.0), egui::vec2(12.1, 7.0), egui::vec2(-12.1, 7.0)] {
        p.circle_filled(c + offset, 2.5, ACCENT);
    }
}

pub fn status_dot(ui: &mut egui::Ui, color: Color32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), 3.0, color);
}

pub fn world_card(ui: &mut egui::Ui, width: f32, name: &str, detail: &str, selected: bool) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 59.0), egui::Sense::click());
    response
        .widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, ui.is_enabled(), selected, name));
    let focused = response.hovered() || response.has_focus();
    let fill = if selected {
        SELECTED
    } else if focused {
        Color32::from_rgb(33, 46, 51)
    } else {
        SURFACE
    };
    let border = if selected {
        ACCENT.gamma_multiply(0.65)
    } else if focused {
        MUTED
    } else {
        BORDER
    };
    ui.painter().rect(rect, 8.0, fill, Stroke::new(1.0f32, border), egui::StrokeKind::Inside);
    let title_size = if name.len() > 16 { 12.5 } else { 14.0 };
    ui.painter().text(
        rect.min + egui::vec2(12.0, 17.0),
        egui::Align2::LEFT_CENTER,
        name,
        FontId::proportional(title_size),
        if selected { ACCENT } else { TEXT },
    );
    ui.painter().text(
        rect.min + egui::vec2(12.0, 39.0),
        egui::Align2::LEFT_CENTER,
        detail,
        FontId::proportional(11.0),
        MUTED,
    );
    response
}

/// Share spare space after reserving each label's measured width, so a long
/// label can borrow room that shorter neighbours do not need.
pub fn button_widths<const N: usize>(ui: &egui::Ui, labels: [&str; N]) -> [f32; N] {
    let font = TextStyle::Button.resolve(ui.style());
    let text_widths =
        labels.map(|label| ui.painter().layout_no_wrap(label.to_owned(), font.clone(), TEXT).size().x.ceil());
    let gaps = ui.spacing().item_spacing.x * N.saturating_sub(1) as f32;
    let spare = ((ui.available_width() - gaps - text_widths.iter().sum::<f32>()) / N as f32).max(0.0);
    text_widths.map(|width| width + spare)
}

pub fn inspector_tabs(ui: &mut egui::Ui, selected: &mut Inspector) {
    let tabs = [
        (Inspector::World, "World", "Simulation parameters, colours and materials"),
        (Inspector::Look, "Appearance", "Bloom, exposure and colour grading"),
        (Inspector::Tools, "Tools", "Brush, camera, resolution, tour, shortcuts and about"),
        (Inspector::Library, "Library", "Save and reload your favourite worlds"),
    ];
    let frame = egui::Frame::new().fill(SURFACE).stroke(Stroke::new(1.0f32, BORDER)).corner_radius(8).inner_margin(3);
    frame.show(ui, |ui| {
        ui.spacing_mut().item_spacing.x = 2.0;
        ui.spacing_mut().button_padding = egui::vec2(8.0, 6.0);
        let widths = button_widths(ui, tabs.map(|(_, title, _)| title));
        ui.horizontal(|ui| {
            for ((tab, title, hint), width) in tabs.into_iter().zip(widths) {
                let active = *selected == tab;
                let text = RichText::new(title).color(if active { ACCENT } else { MUTED });
                let button = egui::Button::new(text)
                    .selected(active)
                    .stroke(Stroke::NONE)
                    .corner_radius(5)
                    .wrap_mode(egui::TextWrapMode::Extend);
                if ui.add_sized([width, 32.0], button).on_hover_text(hint).clicked() {
                    *selected = tab;
                }
            }
        });
    });
}

/// A labelled native slider with room for long parameter names. Keep egui's
/// numeric editing, clamping, logarithmic mapping and change tracking intact.
pub struct Slider<'a> {
    inner: egui::Slider<'a>,
    label: egui::WidgetText,
}

impl<'a> Slider<'a> {
    pub fn new<Num: egui::emath::Numeric>(value: &'a mut Num, range: std::ops::RangeInclusive<Num>) -> Self {
        Self { inner: egui::Slider::new(value, range), label: Default::default() }
    }

    pub fn text(mut self, label: impl Into<egui::WidgetText>) -> Self {
        self.label = label.into();
        self
    }

    pub fn logarithmic(mut self, enabled: bool) -> Self {
        self.inner = self.inner.logarithmic(enabled);
        self
    }

    pub fn suffix(mut self, suffix: impl ToString) -> Self {
        self.inner = self.inner.suffix(suffix);
        self
    }

    pub fn fixed_decimals(mut self, decimals: usize) -> Self {
        self.inner = self.inner.fixed_decimals(decimals);
        self
    }
}

impl egui::Widget for Slider<'_> {
    fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 3.0;
            // Reserve a consistent value column even for precise chemistry
            // values. Do not round or change the underlying slider behaviour.
            ui.spacing_mut().slider_width = (ui.available_width() - 88.0).max(40.0);
            ui.spacing_mut().interact_size = egui::vec2(80.0, 24.0);
            let label = ui.add(egui::Label::new(self.label).wrap());
            ui.add(self.inner).labelled_by(label.id) | label
        })
        .inner
    }
}

/// Colours of the first and second measurement series: a habitat and, while
/// comparing, its reference.
pub const SERIES_COLORS: [Color32; 2] = [ACCENT, Color32::from_rgb(235, 194, 130)];

/// A compact time-series card: `label` top-left, `value` top-right and one
/// polyline per series (oldest to newest) on a shared y-range. Non-finite
/// points are skipped, a single point becomes a dot and a flat trace sits in
/// the middle of the plot. `tooltip` is shown on hover.
pub fn sparkline(
    ui: &mut egui::Ui,
    width: f32,
    label: &str,
    value: &str,
    series: &[(&[f32], Color32)],
    tooltip: &str,
) -> egui::Response {
    const HEIGHT: f32 = 52.0;
    const PAD: f32 = 10.0;
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, HEIGHT), egui::Sense::hover());
    let painter = ui.painter();
    painter.rect(rect, 8.0, SURFACE, Stroke::new(1.0f32, BORDER), egui::StrokeKind::Inside);
    painter.text(rect.min + egui::vec2(PAD, 12.0), egui::Align2::LEFT_CENTER, label, FontId::proportional(11.0), MUTED);
    painter.text(
        egui::pos2(rect.max.x - PAD, rect.min.y + 12.0),
        egui::Align2::RIGHT_CENTER,
        value,
        FontId::monospace(11.5),
        TEXT,
    );
    let plot = egui::Rect::from_min_max(
        egui::pos2(rect.min.x + PAD, rect.min.y + 22.0),
        egui::pos2(rect.max.x - PAD, rect.max.y - 7.0),
    );
    let (lo, hi) = series
        .iter()
        .flat_map(|(values, _)| values.iter().copied())
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), v| (lo.min(v), hi.max(v)));
    if lo <= hi {
        let (lo, hi) = if hi > lo {
            (lo, hi)
        } else {
            let pad = lo.abs().max(1e-6) * 0.5;
            (lo - pad, hi + pad)
        };
        let painter = ui.painter_at(plot);
        for (values, colour) in series {
            let n = values.len();
            let points: Vec<egui::Pos2> = values
                .iter()
                .enumerate()
                .filter(|(_, v)| v.is_finite())
                .map(|(i, v)| {
                    let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.0 };
                    egui::pos2(plot.min.x + t * plot.width(), plot.max.y - (v - lo) / (hi - lo) * plot.height())
                })
                .collect();
            match points.len() {
                0 => {}
                1 => {
                    painter.circle_filled(points[0], 2.0, *colour);
                }
                _ => {
                    painter.add(egui::Shape::line(points, Stroke::new(1.25f32, *colour)));
                }
            }
        }
    }
    if tooltip.is_empty() { response } else { response.on_hover_text(tooltip) }
}

/// What the measurements section reports besides the traces.
pub struct MeasurementsStatus<'a> {
    /// The open CSV log: its path and the rows written so far.
    pub logging: Option<(&'a std::path::Path, u64)>,
    /// Frames whose measurements were skipped because the readback ring was full.
    pub dropped: u64,
}

/// The World tab's "Measurements" section: a collapsible header carrying the
/// CSV log toggle, a legend while two habitats are compared, one sparkline
/// per metric and a status line. Returns true when the log toggle was
/// clicked. Draws nothing for a world without measurements.
pub fn measurements(
    ui: &mut egui::Ui,
    metrics: &[crate::metrics::MetricDesc],
    history: &crate::metrics::History,
    series_names: Option<&[String; 2]>,
    status: MeasurementsStatus<'_>,
) -> bool {
    use std::fmt::Write as _;
    if metrics.is_empty() {
        return false;
    }
    let mut toggle_log = false;
    let id = ui.make_persistent_id("measurements");
    egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, true)
        .show_header(ui, |ui| {
            ui.label(RichText::new("Measurements").strong().color(TEXT));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (text, colour, hint) = match status.logging {
                    Some((path, rows)) => {
                        ("Stop log", WARN, format!("Logging every frame to {}\n{rows} rows so far", path.display()))
                    }
                    None => ("Log CSV", TEXT, "Write every frame's measurements to a CSV file in the capture folder".to_string()),
                };
                if ui.add(egui::Button::new(RichText::new(text).small().color(colour))).on_hover_text(hint).clicked() {
                    toggle_log = true;
                }
            });
        })
        .body_unindented(|ui| {
            ui.spacing_mut().item_spacing.y = 6.0;
            let series = history.series();
            if let Some(names) = series_names.filter(|_| series > 1) {
                ui.horizontal_wrapped(|ui| {
                    for (name, colour) in names.iter().zip(SERIES_COLORS) {
                        status_dot(ui, colour);
                        ui.label(RichText::new(name).small().color(MUTED));
                    }
                });
            }
            let width = ui.available_width();
            for (i, metric) in metrics.iter().enumerate() {
                let traces: Vec<Vec<f32>> = (0..series).map(|s| history.trace(s, i)).collect();
                let lines: Vec<(&[f32], Color32)> =
                    traces.iter().zip(SERIES_COLORS).map(|(trace, colour)| (trace.as_slice(), colour)).collect();
                let value = history.latest(0, i).map_or_else(|| "—".to_string(), |v| metric.unit.format(v));
                let mut tooltip = format!("{}\n{}", metric.label, metric.hint);
                for s in 0..series {
                    if let (Some(latest), Some((lo, hi))) = (history.latest(s, i), history.range(s, i)) {
                        let name = series_names.filter(|_| series > 1).map_or(String::new(), |n| format!("{}: ", n[s]));
                        let _ = write!(
                            tooltip,
                            "\n{name}{} (range {} to {})",
                            metric.unit.format(latest),
                            metric.unit.format(lo),
                            metric.unit.format(hi)
                        );
                    }
                }
                if let Some((first, last)) = history.frame_span() {
                    let _ = write!(tooltip, "\nframes {first} to {last}, one point every {} frame(s)", history.stride());
                }
                sparkline(ui, width, metric.label, &value, &lines, &tooltip);
            }
            let mut line = match history.latest_frame() {
                Some(frame) => format!("frame {frame}"),
                None => "waiting for the first frame".to_string(),
            };
            if history.stride() > 1 {
                let _ = write!(line, " · one point every {} frames", history.stride());
            }
            if status.dropped > 0 {
                let _ = write!(line, " · {} frame(s) skipped", status.dropped);
            }
            if let Some((_, rows)) = status.logging {
                let _ = write!(line, " · logging, {rows} rows");
            }
            ui.label(RichText::new(line).small().weak());
        });
    toggle_log
}

/// Simulation resolutions offered in the Tools tab, as fractions of the
/// window's pixel size.
pub const SIM_SCALES: [f32; 4] = [0.25, 0.5, 0.75, 1.0];

/// Description of the Tools tab's "Simulation resolution" section.
pub const SIM_SCALE_HINT: &str =
    "Lower is faster on laptops and integrated GPUs. Changing it restarts this world from its seed.";

/// One button per entry of [`SIM_SCALES`], the one equal to `current`
/// highlighted (none, after `--sim-scale` chose another value). Returns the
/// scale clicked.
pub fn resolution_picker(ui: &mut egui::Ui, current: f32) -> Option<f32> {
    let labels = SIM_SCALES.map(|scale| format!("{:.0}%", scale * 100.0));
    let widths = button_widths(ui, labels.each_ref().map(String::as_str));
    let mut picked = None;
    ui.horizontal(|ui| {
        for ((scale, label), width) in SIM_SCALES.into_iter().zip(&labels).zip(widths) {
            let active = (scale - current).abs() < 1e-3;
            let text = RichText::new(label.as_str()).color(if active { ACCENT } else { TEXT });
            if ui
                .add_sized([width, 30.0], egui::Button::new(text).selected(active))
                .on_hover_text(format!("Simulate at {label} of the window's pixel size"))
                .clicked()
                && !active
            {
                picked = Some(scale);
            }
        }
    });
    picked
}

/// A few terminal commands under an eyebrow `title`, in a monospace box whose
/// text can be selected, with a Copy button beside the title. Returns true when
/// the button put them on the clipboard.
pub fn command_box(ui: &mut egui::Ui, title: &str, commands: &str) -> bool {
    let mut copied = false;
    ui.horizontal(|ui| {
        eyebrow(ui, title);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.small_button("Copy").on_hover_text("Copy these commands to the clipboard").clicked() {
                ui.ctx().copy_text(commands.to_string());
                copied = true;
            }
        });
    });
    egui::Frame::new()
        .fill(SURFACE)
        .stroke(Stroke::new(1.0f32, BORDER))
        .corner_radius(8)
        .inner_margin(egui::Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.spacing_mut().item_spacing = egui::vec2(6.0, 4.0);
            // A prompt before each command, so a long one that wraps is still visibly one command.
            for command in commands.lines() {
                ui.horizontal_top(|ui| {
                    ui.label(RichText::new("›").monospace().color(MUTED));
                    ui.add(egui::Label::new(RichText::new(command).monospace().color(TEXT)).selectable(true).wrap());
                });
            }
        });
    copied
}

/// What the Library tab shows before anything is saved: the two ways to fill
/// it, with `command` (an `explore --install` run) ready to copy. Returns true
/// when the command was copied.
pub fn empty_library(ui: &mut egui::Ui, command: &str) -> bool {
    ui.label("Nothing saved yet.");
    ui.label(
        RichText::new(
            "Name the world on screen above and press Save current world. Or let Primordia search this world's \
             mutations for the most novel ones: run the command below in a terminal, then press Refresh to find \
             its discoveries here.",
        )
        .small()
        .weak(),
    );
    command_box(ui, "FILL YOUR LIBRARY", command)
}

/// The Tools tab's "About" section: version, GPU, links to the project and
/// `commands` to try in a terminal. Returns true when the commands were copied.
pub fn about(ui: &mut egui::Ui, gpu: &str, commands: &str) -> bool {
    section(ui, "About", "Primordia is open source: read the guide, share a discovery or report a problem.");
    ui.label(RichText::new(format!("Primordia {}", env!("CARGO_PKG_VERSION"))).strong().color(TEXT));
    ui.label(RichText::new(format!("GPU: {gpu}")).small().color(MUTED));
    ui.horizontal_wrapped(|ui| {
        ui.hyperlink_to("Repository", REPOSITORY);
        ui.hyperlink_to("README", format!("{REPOSITORY}#readme"));
        ui.hyperlink_to("Report an issue", format!("{REPOSITORY}/issues"));
    });
    ui.add_space(4.0);
    command_box(ui, "FROM THE COMMAND LINE", commands)
}

/// Repaints the checked checkboxes of a finished frame as accent-filled boxes
/// with a bold dark tick. egui paints every checkbox, the worlds' ones included,
/// as a box in the colours of the widget's state followed, when checked, by a
/// thin tick in the text colour, so on its own "on" differs from "off" only by
/// that line. Call it on each frame's shapes before tessellating them.
pub fn highlight_checked_boxes(ctx: &egui::Context, shapes: &mut [egui::epaint::ClippedShape]) {
    use egui::epaint::{PathStroke, Shape};
    let style = ctx.style();
    let side = style.spacing.icon_width;
    let widgets = &style.visuals.widgets;
    for i in 1..shapes.len() {
        let (before, after) = shapes.split_at_mut(i);
        let (Shape::Rect(frame), Shape::Path(tick)) = (&mut before[i - 1].shape, &mut after[0].shape) else {
            continue;
        };
        // The box is an icon-sized square and the tick an open three-point line inside it.
        let square = (frame.rect.width() - side).abs() < 0.5 && (frame.rect.height() - side).abs() < 0.5;
        if !square || tick.closed || tick.points.len() != 3 || !tick.points.iter().all(|p| frame.rect.contains(*p)) {
            continue;
        }
        // Hovered and pressed boxes keep a hint of their state, a disabled one its fade.
        let fill = if frame.fill == widgets.hovered.bg_fill {
            ACCENT.lerp_to_gamma(Color32::WHITE, 0.3)
        } else if frame.fill == widgets.active.bg_fill {
            ACCENT.lerp_to_gamma(PANEL, 0.25)
        } else {
            ACCENT
        };
        let opacity = f32::from(frame.fill.a()) / 255.0;
        frame.fill = fill.gamma_multiply(opacity);
        frame.stroke = Stroke::new(1.0f32, frame.fill);
        tick.stroke = PathStroke::new(2.0f32, PANEL.gamma_multiply(opacity));
    }
}

/// RGBA pixels of a `size` x `size` icon taken from an `.ico` file whose images
/// are PNG-compressed, as all of [`ICON`]'s are: the smallest image at least
/// that large (else the largest), resized if it does not match exactly.
pub fn icon_rgba(ico: &[u8], size: u32) -> anyhow::Result<Vec<u8>> {
    use anyhow::{Context as _, ensure};
    ensure!((1..=1024).contains(&size), "unsupported icon size {size}");
    let u16_at = |at: usize| ico.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |at: usize| ico.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize);
    ensure!(u16_at(0) == Some(0) && u16_at(2) == Some(1), "not an .ico file");
    let mut images = Vec::new();
    // Each 16-byte directory entry: width (0 = 256), height, colours, reserved,
    // planes, bits per pixel, image length and image offset.
    for entry in (0..usize::from(u16_at(4).unwrap_or(0))).map(|i| 6 + 16 * i) {
        let width = *ico.get(entry).context("the .ico directory is truncated")?;
        let (len, offset) = u32_at(entry + 8).zip(u32_at(entry + 12)).context("the .ico directory is truncated")?;
        let data = ico.get(offset..offset.saturating_add(len)).context("an .ico image is truncated")?;
        images.push((if width == 0 { 256 } else { u32::from(width) }, data));
    }
    let (_, data) = images
        .iter()
        .filter(|(width, _)| *width >= size)
        .min_by_key(|(width, _)| *width)
        .or_else(|| images.iter().max_by_key(|(width, _)| *width))
        .context("the .ico file holds no images")?;
    let image = image::load_from_memory_with_format(data, image::ImageFormat::Png)
        .context("decoding an .ico image (only PNG-compressed ones are supported)")?
        .into_rgba8();
    let image = if image.dimensions() == (size, size) {
        image
    } else {
        image::imageops::resize(&image, size, size, image::imageops::FilterType::Lanczos3)
    };
    Ok(image.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inspector_navigation_fits_the_minimum_window_width() {
        let ctx = egui::Context::default();
        configure(&ctx);
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(320.0, 600.0))),
                ..Default::default()
            },
            |ctx| {
                control_panel(ctx).show(ctx, |ui| {
                    inspector_tabs(ui, &mut Inspector::World);
                    let titles = ["Reset", "Mutate", "Save settings"];
                    let widths = button_widths(ui, titles);
                    ui.horizontal(|ui| {
                        for (title, width) in titles.into_iter().zip(widths) {
                            ui.add_sized([width, 30.0], egui::Button::new(title));
                        }
                    });
                });
            },
        );
        let labels: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|clipped| if let egui::epaint::Shape::Text(text) = &clipped.shape { Some(text) } else { None })
            .collect();
        assert_eq!(labels.len(), 7);
        for text in labels {
            assert_eq!(text.galley.rows.len(), 1);
            assert!(text.pos.x + text.galley.size().x < 320.0, "{} extends outside the window", text.galley.job.text);
        }
    }

    #[test]
    fn inspector_labels_fit_on_one_line_at_narrow_widths_and_display_scales() {
        for width in [264.0, 282.0, 308.0, 326.0] {
            for scale in [1.0, 1.25, 1.5, 2.0] {
                let ctx = egui::Context::default();
                configure(&ctx);
                let mut input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 100.0))),
                    ..Default::default()
                };
                input.viewports.get_mut(&egui::ViewportId::ROOT).unwrap().native_pixels_per_point = Some(scale);
                let output = ctx.run(input, |ctx| {
                    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
                        inspector_tabs(ui, &mut Inspector::World);
                    });
                });
                let labels: Vec<_> =
                    output
                        .shapes
                        .iter()
                        .filter_map(|clipped| {
                            if let egui::epaint::Shape::Text(text) = &clipped.shape { Some(text) } else { None }
                        })
                        .collect();
                assert_eq!(labels.len(), 4);
                for (text, expected) in labels.iter().zip(["World", "Appearance", "Tools", "Library"]) {
                    assert_eq!(text.galley.job.text, expected);
                    assert_eq!(text.galley.rows.len(), 1, "{expected} wrapped at {width}px / {scale}x");
                    assert!(!text.galley.elided);
                    assert!(
                        text.pos.x >= 0.0 && text.pos.x + text.galley.size().x <= width,
                        "{expected}: x={} w={} at {width}px / {scale}x",
                        text.pos.x,
                        text.galley.size().x
                    );
                    assert!((text.pos.y - labels[0].pos.y).abs() < 1.0, "Tab labels lost their common baseline");
                }
            }
        }
    }

    #[test]
    fn overflowing_controls_do_not_paint_a_line_over_the_artwork() {
        let ctx = egui::Context::default();
        configure(&ctx);
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 800.0))),
                ..Default::default()
            },
            |ctx| {
                control_panel(ctx).show(ctx, |ui| {
                    // Reproduce the oversized child that puts the automatic
                    // separator 36 pixels beyond the visible panel.
                    ui.set_min_width(PANEL_WIDTH);
                    ui.label("World parameters");
                });
            },
        );
        for clipped in output.shapes {
            if let egui::epaint::Shape::LineSegment { points, stroke } = clipped.shape {
                let is_vertical = points[0].x == points[1].x;
                let over_art = points[0].x > PANEL_WIDTH && clipped.clip_rect.max.x > PANEL_WIDTH;
                assert!(!(is_vertical && over_art && stroke.width > 0.0 && stroke.color != egui::Color32::TRANSPARENT));
            }
        }
    }

    /// Runs `draw` once in a 300 px wide panel and returns the painted shapes.
    fn paint(draw: impl FnOnce(&mut egui::Ui)) -> Vec<egui::epaint::ClippedShape> {
        let ctx = egui::Context::default();
        configure(&ctx);
        let mut draw = Some(draw);
        ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 900.0))),
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| {
                    if let Some(draw) = draw.take() {
                        draw(ui);
                    }
                });
            },
        )
        .shapes
    }

    /// The trace polylines: paths stroked in a series colour (the collapsing
    /// header's arrow is a path too, in the text colour).
    fn paths(shapes: &[egui::epaint::ClippedShape]) -> Vec<&egui::epaint::PathShape> {
        shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::epaint::Shape::Path(path)
                    if SERIES_COLORS.iter().any(|c| path.stroke.color == egui::epaint::ColorMode::Solid(*c)) =>
                {
                    Some(path)
                }
                _ => None,
            })
            .collect()
    }

    fn texts(shapes: &[egui::epaint::ClippedShape]) -> Vec<String> {
        shapes
            .iter()
            .filter_map(|clipped| {
                if let egui::epaint::Shape::Text(text) = &clipped.shape {
                    assert_eq!(text.galley.rows.len(), 1, "{} wrapped", text.galley.job.text);
                    Some(text.galley.job.text.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn sparkline_draws_each_series_inside_its_card() {
        let rising: Vec<f32> = (0..50).map(|i| i as f32 * 0.01).collect();
        let falling: Vec<f32> = (0..50).map(|i| 1.0 - i as f32 * 0.02).collect();
        let shapes = paint(|ui| {
            sparkline(ui, 280.0, "Growth cover", "12.3%", &[(&rising, ACCENT), (&falling, SERIES_COLORS[1])], "hint");
        });
        let lines = paths(&shapes);
        assert_eq!(lines.len(), 2);
        let card = shapes
            .iter()
            .find_map(|c| match &c.shape {
                egui::epaint::Shape::Rect(rect) if rect.fill == SURFACE => Some(rect.rect),
                _ => None,
            })
            .expect("the card background");
        assert_eq!(card.size(), egui::vec2(280.0, 52.0));
        for (line, colour) in lines.iter().zip(SERIES_COLORS) {
            assert_eq!(line.points.len(), 50);
            assert!(!line.closed);
            assert_eq!(line.stroke.color, egui::epaint::ColorMode::Solid(colour));
            assert!(line.points.iter().all(|p| card.contains(*p)), "points leave the card");
        }
        // The rising trace ends at the top of the shared range, the falling one at the bottom.
        assert!(lines[0].points[49].y < lines[1].points[49].y);
        assert!(lines[0].points[0].y > lines[1].points[0].y);
        assert_eq!(texts(&shapes), ["Growth cover", "12.3%"]);
    }

    #[test]
    fn sparkline_survives_empty_constant_and_non_finite_series() {
        let shapes = paint(|ui| {
            sparkline(ui, 280.0, "Nothing yet", "—", &[(&[], ACCENT)], "");
        });
        assert!(paths(&shapes).is_empty());
        assert!(!shapes.iter().any(|c| matches!(c.shape, egui::epaint::Shape::Circle(_))));

        let shapes = paint(|ui| {
            sparkline(ui, 280.0, "One point", "1.00", &[(&[1.0], ACCENT)], "");
        });
        assert!(paths(&shapes).is_empty());
        assert!(shapes.iter().any(|c| matches!(c.shape, egui::epaint::Shape::Circle(_))));

        let flat = [0.25; 20];
        let mixed = [0.0, f32::NAN, 1.0, f32::INFINITY, 0.5];
        let shapes = paint(|ui| {
            sparkline(ui, 280.0, "Flat", "0.250", &[(&flat, ACCENT)], "");
            sparkline(ui, 280.0, "Mixed", "0.500", &[(&mixed, ACCENT)], "");
        });
        let lines = paths(&shapes);
        assert_eq!(lines.len(), 2);
        let ys: Vec<f32> = lines[0].points.iter().map(|p| p.y).collect();
        assert!(ys.iter().all(|y| (y - ys[0]).abs() < 1e-3), "a flat trace stays level");
        assert_eq!(lines[1].points.len(), 3, "non-finite points are skipped");
        assert!(lines[1].points.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    #[test]
    fn checked_boxes_are_accent_squares_with_a_bold_tick() {
        use egui::epaint::{ColorMode, Shape};
        let ctx = egui::Context::default();
        configure(&ctx);
        let mut output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 900.0))),
                ..Default::default()
            },
            |ctx| {
                egui::CentralPanel::default().show(ctx, |ui| {
                    ui.checkbox(&mut true, "Compare habitats");
                    ui.checkbox(&mut false, "Revive / respawn life");
                    ui.add_enabled(false, egui::Checkbox::new(&mut true, "Disabled"));
                    // Not a checkbox: a collapsing header's arrow is a filled triangle.
                    egui::CollapsingHeader::new("Fertility cycle").default_open(true).show(ui, |ui| ui.label("inside"));
                });
            },
        );
        highlight_checked_boxes(&ctx, &mut output.shapes);
        let side = ctx.style().spacing.icon_width;
        let boxes: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|c| match &c.shape {
                Shape::Rect(rect) if rect.rect.width() == side && rect.rect.height() == side => Some(rect),
                _ => None,
            })
            .collect();
        assert_eq!(boxes.len(), 3, "one box per checkbox");
        for b in &boxes {
            // At the box's 14 px a radius near 7 would make it a disc.
            assert!(b.corner_radius.nw <= 3 && b.corner_radius.se <= 3, "{:?}", b.corner_radius);
        }
        assert_eq!(boxes[0].fill, ACCENT, "checked");
        assert_ne!(boxes[1].fill, ACCENT, "unchecked");
        let faded = boxes[2].fill;
        assert!(faded.a() < 255 && faded.g() > faded.r(), "disabled and checked: a faded accent, not {faded:?}");
        let paths: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|c| if let Shape::Path(path) = &c.shape { Some(path) } else { None })
            .collect();
        let ticks: Vec<_> = paths.iter().filter(|p| p.points.len() == 3 && !p.closed).collect();
        assert_eq!(ticks.len(), 2, "only checked boxes have a tick");
        assert!(ticks.iter().all(|t| t.stroke.width >= 2.0));
        assert_eq!(ticks[0].stroke.color, ColorMode::Solid(PANEL));
        // The collapsing header's arrow keeps its colour.
        let arrow = paths.iter().find(|p| p.closed).expect("the header's arrow");
        assert_ne!(arrow.fill, PANEL);
        assert_ne!(arrow.fill, ACCENT);
    }

    #[test]
    fn the_app_icon_decodes_at_every_size_the_window_asks_for() {
        for size in [16, 20, 24, 28, 32, 40, 48, 56, 64, 128, 256, 300] {
            let rgba = icon_rgba(ICON, size).unwrap();
            assert_eq!(rgba.len(), (size * size * 4) as usize, "{size} px");
            let alpha = |x: u32, y: u32| rgba[((y * size + x) * 4 + 3) as usize];
            // A disc: clear corners, an opaque and visible middle.
            assert_eq!(alpha(0, 0), 0, "{size} px corner");
            assert_eq!(alpha(size / 2, size / 2), 255, "{size} px centre");
            let lit = rgba.chunks_exact(4).filter(|px| px[3] > 0 && px[..3].iter().any(|&c| c > 96)).count();
            assert!(lit > (size * size / 20) as usize, "{size} px icon is too dark");
        }
    }

    #[test]
    fn icon_decoding_rejects_damaged_files() {
        assert!(icon_rgba(b"", 32).is_err());
        assert!(icon_rgba(b"\x89PNG\r\n\x1a\n", 32).is_err());
        assert!(icon_rgba(&ICON[..40], 32).is_err(), "directory entries pointing past the end");
        let mut empty = ICON[..6].to_vec();
        empty[4..6].copy_from_slice(&0u16.to_le_bytes());
        assert!(icon_rgba(&empty, 32).is_err(), "no images");
        let mut garbled = ICON.to_vec();
        let first = u32::from_le_bytes(garbled[18..22].try_into().unwrap()) as usize;
        garbled[first..first + 8].fill(0);
        assert!(icon_rgba(&garbled, 16).is_err(), "not a PNG");
        assert!(icon_rgba(ICON, 0).is_err());
    }

    #[test]
    fn tools_and_library_sections_fit_the_narrowest_panel() {
        let commands = ".\\primordia list\n.\\primordia render -w reaction-diffusion -p \"Crescent Gliders\"\n\
                        .\\primordia explore -w reaction-diffusion --install";
        let ctx = egui::Context::default();
        configure(&ctx);
        let mut picked = None;
        let output = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(320.0, 900.0))),
                ..Default::default()
            },
            |ctx| {
                control_panel(ctx).show(ctx, |ui| {
                    picked = resolution_picker(ui, 0.5);
                    assert!(!about(ui, "NVIDIA GeForce RTX 4090 · Vulkan · discrete GPU", commands));
                    assert!(!empty_library(ui, ".\\primordia explore -w reaction-diffusion --install"));
                });
            },
        );
        assert_eq!(picked, None);
        let texts: Vec<_> = output
            .shapes
            .iter()
            .filter_map(|clipped| if let egui::epaint::Shape::Text(text) = &clipped.shape { Some(text) } else { None })
            .collect();
        for label in ["25%", "50%", "75%", "100%", "Repository", "README", "Report an issue", "Copy"] {
            let text = texts.iter().find(|t| t.galley.job.text == label).unwrap_or_else(|| panic!("{label} missing"));
            assert_eq!(text.galley.rows.len(), 1, "{label} wrapped");
        }
        let version = format!("Primordia {}", env!("CARGO_PKG_VERSION"));
        assert!(texts.iter().any(|t| t.galley.job.text == version));
        let selected = texts.iter().find(|t| t.galley.job.text == "50%").unwrap();
        assert_eq!(selected.galley.job.sections[0].format.color, ACCENT, "the current scale is highlighted");
        for text in &texts {
            let right = text.pos.x + text.galley.size().x;
            assert!(right <= 320.0, "{} extends outside the panel", text.galley.job.text);
        }
        // Each command is a label of its own behind a prompt: a long one wraps
        // under itself, clear of the prompt, rather than under the next one.
        for command in commands.lines().chain([".\\primordia explore -w reaction-diffusion --install"]) {
            let text = texts.iter().find(|t| t.galley.job.text == command).unwrap_or_else(|| panic!("{command} missing"));
            let prompt = texts
                .iter()
                .find(|t| t.galley.job.text == "›" && (t.pos.y - text.pos.y).abs() < 1.0)
                .unwrap_or_else(|| panic!("no prompt before {command}"));
            assert!(prompt.pos.x + prompt.galley.size().x < text.pos.x);
        }
        let render = texts.iter().find(|t| t.galley.job.text.contains("render")).unwrap();
        assert!(render.galley.rows.len() > 1, "at 320 px the render command wraps");
    }

    #[test]
    fn measurements_section_shows_one_card_per_metric_and_nothing_without_metrics() {
        use crate::metrics::{History, MetricDesc, Sample, Unit, MAX_METRICS, MAX_SERIES};
        let metrics = [
            MetricDesc { id: "cover", label: "Cover", unit: Unit::Fraction, hint: "Covered cells." },
            MetricDesc { id: "mean", label: "Mean", unit: Unit::Scalar, hint: "Mean value." },
            MetricDesc { id: "active", label: "Active", unit: Unit::Fraction, hint: "Changing cells." },
        ];
        let mut history = History::default();
        for frame in 0..40 {
            let mut values = [[0.0; MAX_METRICS]; MAX_SERIES];
            for (s, lanes) in values.iter_mut().enumerate() {
                for (k, v) in lanes.iter_mut().enumerate() {
                    *v = (frame as f32 * 0.1 + k as f32 + s as f32 * 0.5).sin() * 0.5 + 0.5;
                }
            }
            history.push(&Sample { frame, time: frame as f32 / 60.0, series: 2, values });
        }
        let names = ["COUPLING 0.70".to_string(), "COUPLING OFF".to_string()];
        let dir = std::path::Path::new("screenshots/run.csv");
        let shapes = paint(|ui| {
            let status = MeasurementsStatus { logging: Some((dir, 80)), dropped: 2 };
            assert!(!measurements(ui, &metrics, &history, Some(&names), status));
        });
        assert_eq!(paths(&shapes).len(), metrics.len() * 2, "two traces per metric");
        let labels = texts(&shapes);
        for metric in &metrics {
            assert!(labels.iter().any(|t| t == metric.label), "{} card missing", metric.label);
        }
        assert!(labels.iter().any(|t| t == "Stop log"));
        assert!(labels.iter().any(|t| t == "COUPLING OFF"), "legend for the second series");
        assert!(labels.iter().any(|t| t.starts_with("frame 39") && t.contains("2 frame(s) skipped") && t.contains("80 rows")));

        let shapes = paint(|ui| {
            let status = MeasurementsStatus { logging: None, dropped: 0 };
            assert!(!measurements(ui, &[], &history, None, status));
        });
        assert!(texts(&shapes).is_empty() && paths(&shapes).is_empty());
    }
}
