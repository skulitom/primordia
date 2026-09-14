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

pub fn control_panel(ctx: &egui::Context) -> egui::SidePanel {
    let frame = egui::Frame::side_top_panel(&ctx.style())
        .fill(PANEL)
        .stroke(Stroke::new(1.0, BORDER))
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
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.hyperlink_color = ACCENT;
    v.warn_fg_color = Color32::from_rgb(235, 194, 130);
    v.error_fg_color = WARN;
    v.window_stroke = Stroke::new(1.0, BORDER);
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
        widget.corner_radius = 6.into();
        widget.bg_stroke = Stroke::new(1.0, BORDER);
        widget.fg_stroke = Stroke::new(1.0, TEXT);
        widget.expansion = 0.0;
    }
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.weak_bg_fill = SURFACE;
    v.widgets.inactive.bg_fill = Color32::from_rgb(36, 49, 55);
    v.widgets.inactive.weak_bg_fill = SURFACE;
    v.widgets.hovered.bg_fill = Color32::from_rgb(47, 67, 70);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(35, 49, 54);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, MUTED);
    v.widgets.active.bg_fill = SELECTED;
    v.widgets.active.weak_bg_fill = SELECTED;
    v.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = Stroke::new(1.0, ACCENT);
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
        .stroke(Stroke::new(1.0, BORDER))
        .corner_radius(10)
        .inner_margin(egui::Margin::symmetric(14, 10))
}

/// Small procedural mark: a nucleus and three orbiting cells.
pub fn mark(ui: &mut egui::Ui) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::hover());
    let p = ui.painter();
    let c = rect.center();
    p.circle_stroke(c, 14.0, Stroke::new(1.0, ACCENT.gamma_multiply(0.5)));
    p.circle_stroke(c, 8.0, Stroke::new(1.0, ACCENT));
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
    ui.painter().rect(rect, 8.0, fill, Stroke::new(1.0, border), egui::StrokeKind::Inside);
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
        (Inspector::Tools, "Tools", "Brush, camera, tour and shortcuts"),
        (Inspector::Library, "Library", "Save and reload your favourite worlds"),
    ];
    egui::Frame::new().fill(SURFACE).stroke(Stroke::new(1.0, BORDER)).corner_radius(8).inner_margin(3).show(ui, |ui| {
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
}
