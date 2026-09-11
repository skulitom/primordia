//! GPU texture readback (screenshots, headless renders, video frames).

use std::path::Path;

use anyhow::{anyhow, Context as _, Result};

use crate::gpu::Gpu;

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

pub fn save_png(path: &Path, size: [u32; 2], rgba: Vec<u8>) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let image = image::RgbaImage::from_raw(size[0], size[1], rgba).context("image buffer has the wrong size")?;
    image.save_with_format(path, image::ImageFormat::Png).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}
