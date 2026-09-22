//! GPU texture readback (screenshots, headless renders, video frames) and the
//! PNG writer, which records where each image came from and its recipe.

use std::path::Path;

use anyhow::{anyhow, ensure, Context as _, Result};

use crate::gpu::Gpu;
use crate::library::SavedWorld;

/// A reusable staging buffer that copies an RGBA8/BGRA8 texture back to the CPU.
pub struct Readback {
    buffer: wgpu::Buffer,
    size: [u32; 2],
    padded_row: u32,
    bgra: bool,
}

impl Readback {
    pub fn new(gpu: &Gpu, size: [u32; 2], format: wgpu::TextureFormat) -> Self {
        use wgpu::TextureFormat as F;
        let bgra = matches!(format, F::Bgra8Unorm | F::Bgra8UnormSrgb);
        assert!(
            bgra || matches!(format, F::Rgba8Unorm | F::Rgba8UnormSrgb),
            "readback only supports 8-bit RGBA/BGRA formats, got {format:?}"
        );
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_row = (size[0] * 4).div_ceil(align) * align;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: padded_row as u64 * size[1] as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self { buffer, size, padded_row, bgra }
    }

    /// Records a copy of `texture` (which must match the readback size) into the staging buffer.
    pub fn copy_from(&self, encoder: &mut wgpu::CommandEncoder, texture: &wgpu::Texture) {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.size[1]),
                },
            },
            wgpu::Extent3d { width: self.size[0], height: self.size[1], depth_or_array_layers: 1 },
        );
    }

    /// Waits for the GPU, then returns tightly packed RGBA8 pixels (alpha forced to 255).
    /// Call after submitting the encoder that recorded [`Readback::copy_from`].
    pub fn read(&self, gpu: &Gpu) -> Result<Vec<u8>> {
        let slice = self.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device.poll(wgpu::PollType::Wait).map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
        rx.recv().context("readback callback dropped")?.context("buffer mapping failed")?;

        let [w, h] = self.size;
        let row = (w * 4) as usize;
        let mut out = Vec::with_capacity(row * h as usize);
        {
            let data = slice.get_mapped_range();
            for y in 0..h as usize {
                let start = y * self.padded_row as usize;
                out.extend_from_slice(&data[start..start + row]);
            }
        }
        self.buffer.unmap();

        for px in out.chunks_exact_mut(4) {
            if self.bgra {
                px.swap(0, 2);
            }
            px[3] = 255;
        }
        Ok(out)
    }
}

/// The first eight bytes of every PNG file.
pub const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Keyword of the iTXt chunk that carries a PNG's recipe: the JSON of a
/// library save ([`SavedWorld`]), which `render --recipe` reads back.
pub const RECIPE_KEYWORD: &str = "primordia:recipe";

/// What a PNG written by Primordia says about where it came from. Every PNG
/// gets tEXt `Software` ("Primordia <version>") and `Source` (the repository);
/// the rest is written when known.
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    /// `Title`: "<World> · <Preset>".
    pub title: Option<String>,
    /// `Comment`: a command that renders the image again.
    pub command: Option<String>,
    /// `primordia:gpu`: the GPU that rendered it (a run replays exactly only on
    /// the same GPU, driver and backend).
    pub gpu: Option<String>,
    /// iTXt [`RECIPE_KEYWORD`]: the recipe of the world shown.
    pub recipe: Option<SavedWorld>,
}

impl Provenance {
    /// An image of `recipe` rendered on `gpu`, titled "<World> · <Preset>".
    pub fn of(recipe: &SavedWorld, gpu: &Gpu) -> Self {
        Self { title: Some(title(recipe)), command: None, gpu: Some(gpu_name(gpu)), recipe: Some(recipe.clone()) }
    }

    /// The same, with `command` as the `Comment`.
    pub fn with_command(self, command: String) -> Self {
        Self { command: Some(command), ..self }
    }
}

/// "<World> · <Preset>" for a recipe.
pub fn title(recipe: &SavedWorld) -> String {
    let world = crate::world::find(recipe.settings.world_id()).map(|index| &crate::world::WORLDS[index]);
    let name = world.map_or("Primordia", |w| w.name);
    let preset = world.and_then(|w| (w.presets)().get(recipe.preset).copied()).unwrap_or("custom");
    format!("{name} · {preset}")
}

/// The adapter's name and backend, e.g. "NVIDIA GeForce RTX 4090 (Vulkan)".
pub fn gpu_name(gpu: &Gpu) -> String {
    let info = gpu.adapter.get_info();
    format!("{} ({:?})", info.name, info.backend)
}

/// Writes an RGBA8 image as a PNG with only the `Software` and `Source` chunks.
pub fn save_png(path: &Path, size: [u32; 2], rgba: Vec<u8>) -> Result<()> {
    write_png(path, size, &rgba, &Provenance::default())
}

/// Writes an RGBA8 image as a PNG that carries `provenance` in text chunks:
/// tEXt where the text is Latin-1 (as the PNG specification asks), iTXt otherwise.
pub fn write_png(path: &Path, size: [u32; 2], rgba: &[u8], provenance: &Provenance) -> Result<()> {
    use std::io::Write as _;
    ensure!(rgba.len() as u64 == u64::from(size[0]) * u64::from(size[1]) * 4, "image buffer has the wrong size");
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let mut text = vec![
        ("Software", format!("Primordia {}", env!("CARGO_PKG_VERSION"))),
        ("Source", env!("CARGO_PKG_REPOSITORY").to_string()),
    ];
    text.extend(provenance.title.clone().map(|t| ("Title", t)));
    text.extend(provenance.command.clone().map(|c| ("Comment", c)));
    text.extend(provenance.gpu.clone().map(|g| ("primordia:gpu", g)));
    let recipe = provenance.recipe.as_ref().map(serde_json::to_string).transpose().context("serialising the recipe")?;

    let written = (|| -> Result<()> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let mut encoder = png::Encoder::new(&mut file, size[0], size[1]);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        // The image crate's defaults, which wrote Primordia's PNGs before.
        encoder.set_compression(png::Compression::Fast);
        encoder.set_filter(png::Filter::Adaptive);
        for (keyword, value) in text {
            if value.chars().all(|c| u32::from(c) <= 0xff) {
                encoder.add_text_chunk(keyword.to_string(), value)?;
            } else {
                encoder.add_itxt_chunk(keyword.to_string(), value)?;
            }
        }
        if let Some(recipe) = recipe {
            encoder.add_itxt_chunk(RECIPE_KEYWORD.to_string(), recipe)?;
        }
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgba)?;
        writer.finish()?;
        file.flush()?;
        Ok(())
    })();
    written.with_context(|| format!("writing {}", path.display()))
}

/// The JSON text of the recipe a PNG carries ([`RECIPE_KEYWORD`]), or `None`.
/// Only the chunks before the image data are read, which is where
/// [`write_png`] puts them.
pub fn read_recipe(path: &Path) -> Result<Option<String>> {
    let file = std::fs::File::open(path)?;
    let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
    // Text chunks count against this limit: a recipe needs a few kilobytes.
    decoder.set_limits(png::Limits { bytes: 4 * crate::library::MAX_SAVE_BYTES as usize });
    let reader = decoder.read_info().context("not a readable PNG")?;
    let chunk = reader.info().utf8_text.iter().find(|chunk| chunk.keyword == RECIPE_KEYWORD);
    chunk.map(|chunk| chunk.get_text().context("reading the recipe chunk")).transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> SavedWorld {
        serde_json::from_str(include_str!("../tests/fixtures/reaction-diffusion.json")).unwrap()
    }

    /// Every text chunk of a PNG: (keyword, text, whether it was iTXt).
    fn text_chunks(path: &Path) -> Vec<(String, String, bool)> {
        let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path).unwrap()));
        let reader = decoder.read_info().unwrap();
        let info = reader.info();
        let latin1 = info.uncompressed_latin1_text.iter().map(|c| (c.keyword.clone(), c.text.clone(), false));
        let utf8 = info.utf8_text.iter().map(|c| (c.keyword.clone(), c.get_text().unwrap(), true));
        latin1.chain(utf8).collect()
    }

    fn pixels(size: [u32; 2]) -> Vec<u8> {
        (0..size[0] * size[1] * 4).map(|i| (i * 37 % 251) as u8).collect()
    }

    #[test]
    fn pngs_carry_provenance_and_their_recipe() {
        let dir = tempfile::tempdir().unwrap();
        let size = [7, 5];
        let path = dir.path().join("nested").join("reef.png");
        let provenance = Provenance {
            title: Some(title(&fixture())),
            command: Some("primordia render --recipe reef.png --frames 600".into()),
            gpu: Some("Test GPU (Vulkan)".into()),
            recipe: Some(fixture()),
        };
        write_png(&path, size, &pixels(size), &provenance).unwrap();

        let image = image::open(&path).unwrap().to_rgba8();
        assert_eq!((image.width(), image.height()), (7, 5));
        assert_eq!(image.into_raw(), pixels(size), "the pixels are stored exactly");
        let chunks = text_chunks(&path);
        let get = |key: &str| chunks.iter().find(|c| c.0 == key).map(|c| (c.1.as_str(), c.2));
        assert_eq!(get("Software"), Some((concat!("Primordia ", env!("CARGO_PKG_VERSION")), false)));
        assert_eq!(get("Source"), Some(("https://github.com/skulitom/primordia", false)));
        assert_eq!(get("Title"), Some(("Reaction-Diffusion · Coral Reef", false)), "Latin-1 text stays tEXt");
        assert_eq!(get("Comment"), Some(("primordia render --recipe reef.png --frames 600", false)));
        assert_eq!(get("primordia:gpu"), Some(("Test GPU (Vulkan)", false)));
        let (recipe, itxt) = get(RECIPE_KEYWORD).unwrap();
        assert!(itxt, "the recipe is an iTXt chunk");
        let back: SavedWorld = serde_json::from_str(recipe).unwrap();
        assert_eq!(serde_json::to_value(back).unwrap(), serde_json::to_value(fixture()).unwrap());
        assert_eq!(read_recipe(&path).unwrap().as_deref(), Some(recipe));

        // Text outside Latin-1 goes into iTXt; a plain save has only the two fixed chunks.
        let plain = dir.path().join("plain.png");
        let command = "primordia render --recipe рецепт.png".to_string();
        let unicode = Provenance { command: Some(command.clone()), ..Default::default() };
        write_png(&plain, size, &pixels(size), &unicode).unwrap();
        let chunks = text_chunks(&plain);
        assert!(chunks.contains(&("Comment".into(), command, true)), "{chunks:?}");
        save_png(&plain, size, pixels(size)).unwrap();
        let keys: Vec<String> = text_chunks(&plain).into_iter().map(|c| c.0).collect();
        assert_eq!(keys, ["Software", "Source"]);
        assert_eq!(read_recipe(&plain).unwrap(), None);
        assert!(write_png(&plain, [8, 8], &pixels(size), &Provenance::default()).is_err(), "wrong buffer size");
    }

    #[test]
    fn recipes_load_from_pngs_and_pngs_without_one_are_invalid_input() {
        let dir = tempfile::tempdir().unwrap();
        let size = [4, 4];
        let with = dir.path().join("with.png");
        let provenance = Provenance { recipe: Some(fixture()), ..Default::default() };
        write_png(&with, size, &pixels(size), &provenance).unwrap();
        let loaded = crate::library::load(&with).unwrap();
        assert_eq!(serde_json::to_value(loaded).unwrap(), serde_json::to_value(fixture()).unwrap());

        let without = dir.path().join("without.png");
        save_png(&without, size, pixels(size)).unwrap();
        let broken = dir.path().join("broken.png");
        let mut bytes = PNG_SIGNATURE.to_vec();
        bytes.extend_from_slice(b"not really");
        std::fs::write(&broken, bytes).unwrap();
        let mut bad_recipe = fixture();
        bad_recipe.version = 7;
        let future = dir.path().join("future.png");
        let provenance = Provenance { recipe: Some(bad_recipe), ..Default::default() };
        write_png(&future, size, &pixels(size), &provenance).unwrap();
        for (path, expected) in [
            (&without, "this PNG carries no recipe"),
            (&broken, "not a readable PNG"),
            (&future, "Unsupported save version"),
            (&dir.path().join("missing.json"), "cannot use the recipe"),
        ] {
            let error = crate::library::load(path).unwrap_err();
            assert!(format!("{error:#}").contains(expected), "{}: {error:#}", path.display());
            assert!(error.to_string().starts_with("cannot use the recipe "), "{error}");
            assert_eq!(crate::failure::exit_code(&error), 2, "{error:#}");
        }
    }
}
