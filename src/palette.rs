//! Colour palettes: hand-picked gradient stops interpolated in OKLab and baked
//! into a 256x1 lookup texture that shaders sample via `palette_lookup`.

use crate::gpu::Gpu;

pub struct Palette {
    pub name: &'static str,
    /// (position 0..1, sRGB hex colour)
    pub stops: &'static [(f32, u32)],
}

pub const PALETTES: &[Palette] = &[
    Palette {
        name: "Bioluminescence",
        stops: &[(0.0, 0x000308), (0.2, 0x04213a), (0.42, 0x0a5c6e), (0.62, 0x1fb5a8), (0.82, 0x8ff2d0), (1.0, 0xf4fff9)],
    },
    Palette {
        name: "Ember",
        stops: &[
            (0.0, 0x000004),
            (0.15, 0x280b54),
            (0.3, 0x65156e),
            (0.45, 0x9f2a63),
            (0.6, 0xd44842),
            (0.75, 0xf57d15),
            (0.88, 0xfac127),
            (1.0, 0xfcffa4),
        ],
    },
    Palette {
        name: "Aurora",
        stops: &[
            (0.0, 0x02010a),
            (0.2, 0x1a0b3d),
            (0.38, 0x3b1f8f),
            (0.55, 0x1f7bbf),
            (0.72, 0x21d4a7),
            (0.88, 0xb8f56a),
            (1.0, 0xf7ffd6),
        ],
    },
    Palette {
        name: "Nacre",
        stops: &[
            (0.0, 0x05030c),
            (0.2, 0x2b1b4a),
            (0.4, 0x7b3f8c),
            (0.6, 0xe0799b),
            (0.78, 0xffc9a8),
            (0.92, 0xd7f6ff),
            (1.0, 0xffffff),
        ],
    },
    Palette {
        name: "Lagoon",
        stops: &[(0.0, 0x0d0221), (0.3, 0x3b2c85), (0.55, 0x21918c), (0.78, 0x5ec962), (1.0, 0xfde725)],
    },
    Palette {
        name: "Moss",
        stops: &[(0.0, 0x010401), (0.25, 0x0b2410), (0.5, 0x2e6b1f), (0.75, 0x9bc53d), (1.0, 0xf2f5c8)],
    },
    Palette {
        name: "Solar",
        stops: &[(0.0, 0x030100), (0.2, 0x2b0f00), (0.4, 0x7a3100), (0.62, 0xd47a06), (0.82, 0xffd35c), (1.0, 0xfffbe8)],
    },
    Palette {
        name: "Glacier",
        stops: &[(0.0, 0x01030a), (0.25, 0x0e1f45), (0.5, 0x2e5fa8), (0.72, 0x7fb8e6), (0.9, 0xdff3ff), (1.0, 0xffffff)],
    },
    Palette {
        name: "Coral",
        stops: &[(0.0, 0x020612), (0.25, 0x0b2c4d), (0.45, 0x5a4f8f), (0.65, 0xe0707a), (0.85, 0xffc39b), (1.0, 0xfff4e6)],
    },
    Palette {
        name: "Neon",
        stops: &[(0.0, 0x000000), (0.2, 0x2a0033), (0.4, 0xa1007a), (0.55, 0xff2e88), (0.8, 0x00e1ff), (1.0, 0xe8ffff)],
    },
    Palette {
        name: "Blood",
        stops: &[(0.0, 0x000000), (0.3, 0x2a0000), (0.6, 0x8a0303), (0.85, 0xff4d2e), (1.0, 0xffe0c7)],
    },
    Palette {
        name: "Ink",
        stops: &[(0.0, 0xf3ede2), (0.35, 0xb9ad9c), (0.65, 0x3b342e), (1.0, 0x0a0908)],
    },
];

/// Index of the palette whose name matches `name` (case-insensitive).
pub fn find(name: &str) -> Option<usize> {
    PALETTES.iter().position(|p| p.name.eq_ignore_ascii_case(name))
}

impl Palette {
    /// Linear-RGB colour at `t` in [0, 1] (OKLab interpolation between stops).
    pub fn sample(&self, t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);
        let stops = self.stops;
        let mut i = 0;
        while i + 1 < stops.len() - 1 && t > stops[i + 1].0 {
            i += 1;
        }
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[(i + 1).min(stops.len() - 1)];
        let f = if t1 > t0 { ((t - t0) / (t1 - t0)).clamp(0.0, 1.0) } else { 0.0 };
        let a = linear_to_oklab(hex_to_linear(c0));
        let b = linear_to_oklab(hex_to_linear(c1));
        let lab = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
        let rgb = oklab_to_linear(lab);
        [rgb[0].clamp(0.0, 1.0), rgb[1].clamp(0.0, 1.0), rgb[2].clamp(0.0, 1.0)]
    }

    /// 8-bit sRGB colour at `t`, e.g. for UI swatches.
    pub fn sample_srgb8(&self, t: f32) -> [u8; 3] {
        let c = self.sample(t);
        [to_u8(linear_to_srgb(c[0])), to_u8(linear_to_srgb(c[1])), to_u8(linear_to_srgb(c[2]))]
    }

    /// 256 RGBA8 sRGB texels.
    pub fn lut_rgba8(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256 * 4);
        for i in 0..256 {
            let [r, g, b] = self.sample_srgb8(i as f32 / 255.0);
            out.extend_from_slice(&[r, g, b, 255]);
        }
        out
    }
}

/// A palette baked into a 256x1 `Rgba8UnormSrgb` texture plus a linear clamp sampler.
/// Bind `view` as `texture_2d<f32>` and `sampler` as a filtering sampler, then call
/// `palette_lookup(lut, samp, t)` from WGSL.
pub struct PaletteLut {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    index: usize,
}

impl PaletteLut {
    pub fn new(gpu: &Gpu, index: usize) -> Self {
        let (texture, view) = gpu.texture_2d(
            "palette lut",
            [256, 1],
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        let sampler = gpu.sampler(wgpu::FilterMode::Linear, wgpu::AddressMode::ClampToEdge);
        let lut = Self { texture, view, sampler, index: usize::MAX };
        let mut lut = lut;
        lut.set(gpu, index);
        lut
    }

    pub fn index(&self) -> usize {
        self.index
    }

    pub fn set(&mut self, gpu: &Gpu, index: usize) {
        let index = index.min(PALETTES.len() - 1);
        if index == self.index {
            return;
        }
        self.index = index;
        gpu.queue.write_texture(
            self.texture.as_image_copy(),
            &PALETTES[index].lut_rgba8(),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256 * 4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 256, height: 1, depth_or_array_layers: 1 },
        );
    }

    /// Palette picker; updates the texture and returns true when changed.
    pub fn ui(&mut self, gpu: &Gpu, ui: &mut egui::Ui) -> bool {
        let mut index = self.index;
        if combo(ui, "palette", &mut index) {
            self.set(gpu, index);
            true
        } else {
            false
        }
    }
}

/// Palette combo box with gradient previews. Returns true when the selection changed.
pub fn combo(ui: &mut egui::Ui, id_salt: &str, index: &mut usize) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label("Palette");
        egui::ComboBox::from_id_salt(id_salt)
            .selected_text(PALETTES[(*index).min(PALETTES.len() - 1)].name)
            .width(140.0)
            .show_ui(ui, |ui| {
                for (i, p) in PALETTES.iter().enumerate() {
                    ui.horizontal(|ui| {
                        swatch(ui, p, egui::vec2(44.0, 12.0));
                        if ui.selectable_label(*index == i, p.name).clicked() {
                            *index = i;
                            changed = true;
                        }
                    });
                }
            });
        swatch(ui, &PALETTES[(*index).min(PALETTES.len() - 1)], egui::vec2(56.0, 14.0));
    });
    changed
}

/// Paints a horizontal gradient preview of `palette`.
pub fn swatch(ui: &mut egui::Ui, palette: &Palette, size: egui::Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
    let painter = ui.painter_at(rect);
    const STEPS: usize = 32;
    for k in 0..STEPS {
        let t0 = k as f32 / STEPS as f32;
        let t1 = (k + 1) as f32 / STEPS as f32;
        let x0 = egui::lerp(rect.left()..=rect.right(), t0);
        let x1 = egui::lerp(rect.left()..=rect.right(), t1);
        let [r, g, b] = palette.sample_srgb8((t0 + t1) * 0.5);
        painter.rect_filled(
            egui::Rect::from_min_max(egui::pos2(x0, rect.top()), egui::pos2(x1 + 0.5, rect.bottom())),
            0.0,
            egui::Color32::from_rgb(r, g, b),
        );
    }
}

// --- colour science --------------------------------------------------------

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
}

pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.003_130_8 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
}

/// sRGB hex (0xRRGGBB) to linear RGB.
pub fn hex_to_linear(hex: u32) -> [f32; 3] {
    let r = ((hex >> 16) & 0xff) as f32 / 255.0;
    let g = ((hex >> 8) & 0xff) as f32 / 255.0;
    let b = (hex & 0xff) as f32 / 255.0;
    [srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)]
}

fn linear_to_oklab([r, g, b]: [f32; 3]) -> [f32; 3] {
    let l = 0.412_221_46 * r + 0.536_332_55 * g + 0.051_445_995 * b;
    let m = 0.211_903_5 * r + 0.680_699_5 * g + 0.107_396_96 * b;
    let s = 0.088_302_46 * r + 0.281_718_85 * g + 0.629_978_7 * b;
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    [
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    ]
}

fn oklab_to_linear([ll, a, b]: [f32; 3]) -> [f32; 3] {
    let l = ll + 0.396_337_78 * a + 0.215_803_76 * b;
    let m = ll - 0.105_561_346 * a - 0.063_854_17 * b;
    let s = ll - 0.089_484_18 * a - 1.291_485_5 * b;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}
