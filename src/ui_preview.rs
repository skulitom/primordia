//! Manual, offscreen visual QA of the real inspector widgets. No application
//! window is opened or controlled; previews go to target/ui-preview.

use super::*;
use crate::{capture, palette, post::PostSettings, world};

#[test]
#[ignore = "writes offscreen UI previews to target/ui-preview"]
fn render_inspector_previews() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    for width in [320, 364] {
        for tab in ["world", "appearance", "symbiosis", "comparison", "fertility", "measurements"] {
            let appearance = tab == "appearance";
            let measurements = tab == "measurements";
            let size = [width, 900];
            let ctx = egui::Context::default();
            configure(&ctx);
            let (world_index, mut specimen) = if tab == "symbiosis" || tab == "comparison" || tab == "fertility" || measurements {
                world::create(&gpu, "symbiosis", [800, 600], None, 42).unwrap()
            } else {
                world::create(&gpu, "lenia", [800, 600], Some("Necklaces"), 42).unwrap()
            };
            if tab == "fertility" {
                specimen.load_preset(&gpu, 5, 42);
            }
            // A synthetic two-habitat run: the traces are what a comparison looks like.
            let mut history = crate::metrics::History::default();
            if measurements {
                use crate::metrics::{MAX_METRICS, MAX_SERIES, Sample};
                for frame in 0..1500u64 {
                    let t = frame as f32 / 60.0;
                    let mut values = [[0.0; MAX_METRICS]; MAX_SERIES];
                    for (s, lanes) in values.iter_mut().enumerate() {
                        for (k, v) in lanes.iter_mut().enumerate() {
                            let phase = k as f32 * 0.9 + s as f32 * 0.6;
                            *v = 0.5 + 0.35 * (t * (0.3 + 0.1 * k as f32) + phase).sin() * (-t * 0.02).exp();
                        }
                    }
                    history.push(&Sample { frame, time: t, series: 2, values });
                }
            }
            if tab == "comparison" || tab == "fertility" || measurements {
                let mut recipe = specimen.settings().unwrap();
                if let crate::library::WorldSettings::Symbiosis { params, .. } = &mut recipe {
                    params.compare = true;
                    if tab == "fertility" {
                        params.layer = world::symbiosis::Layer::Fertility;
                    }
                }
                specimen.restore_settings(&gpu, &recipe, 42).unwrap();
            }
            let mut post = PostSettings::default();
            let mut selected_palette = 2;
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width as f32, 900.0))),
                    ..Default::default()
                },
                |ctx| {
                    control_panel(ctx).show(ctx, |ui| {
                        ui.horizontal(|ui| {
                            mark(ui);
                            ui.vertical(|ui| {
                                ui.label(RichText::new("Primordia").size(23.0).color(ACCENT));
                                ui.label(RichText::new("ARTIFICIAL LIFE LAB").size(10.0).color(MUTED));
                            });
                        });
                        ui.add_space(12.0);
                        let titles = ["Reset", "Mutate", "Save settings"];
                        let widths = button_widths(ui, titles);
                        ui.horizontal(|ui| {
                            for (title, width) in titles.into_iter().zip(widths) {
                                ui.add_sized([width, 30.0], egui::Button::new(title));
                            }
                        });
                        inspector_tabs(ui, &mut if appearance { Inspector::Look } else { Inspector::World });
                        ui.add_space(8.0);
                        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                            ui.set_width(ui.available_width() - 6.0);
                            if appearance {
                                section(ui, "Shape the light", "Finish the image with bloom, colour and film effects.");
                                post.ui(ui);
                                ui.separator();
                                palette::combo(ui, "preview palette", &mut selected_palette);
                                dropdown(ui, "Material", "Nacre (thin-film iridescence)", |ui| {
                                    ui.label("Nacre (thin-film iridescence)");
                                });
                            } else {
                                section(
                                    ui,
                                    "World parameters",
                                    world::WORLDS[world_index].tagline,
                                );
                                ui.label(RichText::new(specimen.stats()).small().color(MUTED));
                                ui.add_space(4.0);
                                if measurements {
                                    let names = specimen.comparison_labels();
                                    let path = std::path::Path::new("screenshots/symbiosis_living-reef.csv");
                                    let status = MeasurementsStatus { logging: Some((path, 3000)), dropped: 0 };
                                    super::measurements(ui, specimen.metrics(), &history, names.as_ref(), status);
                                    ui.add_space(4.0);
                                }
                                specimen.ui(&gpu, ui);
                            }
                        });
                    });
                },
            );
            let format = wgpu::TextureFormat::Rgba8Unorm;
            let (texture, view) = gpu.texture_2d(
                "UI preview",
                size,
                format,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let mut renderer = egui_wgpu::Renderer::new(&gpu.device, format, None, 1, false);
            for (id, delta) in &output.textures_delta.set {
                renderer.update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
            let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
            let screen =
                egui_wgpu::ScreenDescriptor { size_in_pixels: size, pixels_per_point: output.pixels_per_point };
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            let extra = renderer.update_buffers(&gpu.device, &gpu.queue, &mut encoder, &jobs, &screen);
            {
                let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
                renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
            }
            let readback = capture::Readback::new(&gpu, size, format);
            readback.copy_from(&mut encoder, &texture);
            gpu.queue.submit(extra.into_iter().chain([encoder.finish()]));
            let path = format!("target/ui-preview/{tab}-{width}.png");
            capture::save_png(std::path::Path::new(&path), size, readback.read(&gpu).unwrap()).unwrap();
            println!("Saved {path}");
        }
    }
}
