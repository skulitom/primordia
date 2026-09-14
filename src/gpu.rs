//! GPU context plus small helpers that keep the world implementations terse.

use std::borrow::Cow;
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context as _, Result};
use wgpu::util::DeviceExt as _;

/// WGSL prelude prepended to every shader created through [`Gpu::shader`].
pub const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");

/// Format of the HDR scene texture every world renders into.
pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Hold this before creating an instance and until its GPU is dropped. The
/// Windows Vulkan loader can crash during concurrent instance creation/drop
/// across otherwise independent tests. CPU-only tests can still run in parallel.
#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Shared GPU handles. Cloning is cheap: every wgpu handle is reference counted.
#[derive(Clone)]
pub struct Gpu {
    /// Held so the instance outlives everything created from it (e.g. surfaces).
    #[allow(dead_code)]
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// First unrecoverable GPU problem (device lost, out of memory, validation).
    fatal: Arc<Mutex<Option<String>>>,
}

impl Gpu {
    /// Creates a wgpu instance honouring the usual `WGPU_BACKEND` style env vars.
    pub fn create_instance() -> wgpu::Instance {
        wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default())
    }

    /// Picks the high-performance adapter (compatible with `surface`, if given)
    /// and opens a device with every limit the adapter supports.
    pub async fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::from_env()
                    .unwrap_or(wgpu::PowerPreference::HighPerformance),
                force_fallback_adapter: false,
                compatible_surface: surface,
            })
            .await
            .context("no suitable GPU adapter found")?;

        let info = adapter.get_info();
        log::info!("GPU: {} ({:?}, {:?})", info.name, info.backend, info.device_type);

        // Optional niceties; only requested when the adapter has them.
        let wanted = wgpu::Features::FLOAT32_FILTERABLE;
        let required_features = adapter.features() & wanted;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("primordia device"),
                required_features,
                // Desktop GPUs allow far bigger buffers than the WebGPU defaults;
                // ask for everything so worlds can hold millions of agents.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to open GPU device")?;

        // Record problems instead of panicking inside wgpu's callbacks, so the
        // app and headless renders can stop cleanly with a readable message.
        let fatal = Arc::new(Mutex::new(None::<String>));
        let record = |fatal: &Arc<Mutex<Option<String>>>, text: String| {
            log::error!("{text}");
            fatal.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert(text);
        };
        {
            let fatal = fatal.clone();
            device.set_device_lost_callback(move |reason, message| {
                record(&fatal, format!("the GPU device was lost ({reason:?}): {message}"));
            });
        }
        {
            let fatal = fatal.clone();
            device.on_uncaptured_error(Box::new(move |error| record(&fatal, format!("wgpu error: {error}"))));
        }

        Ok(Self { instance, adapter, device, queue, fatal })
    }

    /// The first unrecoverable problem wgpu reported (device lost, out of memory,
    /// a validation error), if any. Callers should stop using the GPU when set.
    pub fn fatal_error(&self) -> Option<String> {
        self.fatal.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    pub fn adapter_name(&self) -> String {
        self.adapter.get_info().name
    }

    /// Compiles a WGSL module with [`COMMON_WGSL`] prepended.
    pub fn shader(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        let full = format!("{COMMON_WGSL}\n{source}");
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(Cow::Owned(full)),
        })
    }

    pub fn uniform_buffer<T: bytemuck::Pod>(&self, label: &str, value: &T) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(value),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Zero-initialised storage buffer (`STORAGE | COPY_DST | COPY_SRC` + `extra`).
    pub fn storage_buffer(&self, label: &str, size: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(16),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | extra,
            mapped_at_creation: false,
        })
    }

    /// Storage buffer initialised with `contents` (`STORAGE | COPY_DST | COPY_SRC` + `extra`).
    pub fn storage_buffer_init(&self, label: &str, contents: &[u8], extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | extra,
        })
    }

    /// Uploads a plain-old-data value to the start of `buffer`.
    ///
    /// Note: `queue.write_buffer` lands before the *whole* next submission, so
    /// every dispatch recorded in that submission sees the last value written.
    pub fn write<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer, value: &T) {
        self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(value));
    }

    pub fn texture_2d(
        &self,
        label: &str,
        size: [u32; 2],
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    pub fn sampler(&self, filter: wgpu::FilterMode, address: wgpu::AddressMode) -> wgpu::Sampler {
        self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("primordia sampler"),
            address_mode_u: address,
            address_mode_v: address,
            address_mode_w: address,
            mag_filter: filter,
            min_filter: filter,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        })
    }

    pub fn bind_group_layout(&self, label: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
    }

    /// Bind group whose entries are bound at bindings `0..resources.len()`.
    pub fn bind_group(
        &self,
        label: &str,
        layout: &wgpu::BindGroupLayout,
        resources: &[wgpu::BindingResource<'_>],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
            .iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
    }

    pub fn pipeline_layout(&self, label: &str, layouts: &[&wgpu::BindGroupLayout]) -> wgpu::PipelineLayout {
        self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: layouts,
            push_constant_ranges: &[],
        })
    }

    pub fn compute_pipeline(
        &self,
        label: &str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    /// Render pipeline drawing the prelude's fullscreen triangle (`vs_fullscreen`)
    /// with fragment entry point `fs_entry`.
    pub fn fullscreen_pipeline(
        &self,
        label: &str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        fs_entry: &str,
        format: wgpu::TextureFormat,
        blend: Option<wgpu::BlendState>,
    ) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some(fs_entry),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview: None,
            cache: None,
        })
    }

    /// Blocks until all submitted GPU work has finished.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait);
    }

    /// Blocking readback of a whole buffer, for tests that cross-check GPU
    /// results against a CPU computation.
    #[cfg(test)]
    pub fn read_buffer<T: bytemuck::Pod>(&self, source: &wgpu::Buffer) -> Vec<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test readback"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
        self.wait_idle();
        rx.recv().unwrap().unwrap();
        let values = bytemuck::pod_collect_to_vec(&staging.slice(..).get_mapped_range());
        staging.unmap();
        values
    }
}

/// Records a render pass that draws the fullscreen triangle once into `target`.
/// `clear = None` keeps (loads) the existing contents, e.g. for additive blending.
pub fn fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    clear: Option<wgpu::Color>,
    pipeline: &wgpu::RenderPipeline,
    bind_groups: &[&wgpu::BindGroup],
) {
    let load = match clear {
        Some(color) => wgpu::LoadOp::Clear(color),
        None => wgpu::LoadOp::Load,
    };
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    pass.set_pipeline(pipeline);
    for (i, group) in bind_groups.iter().enumerate() {
        pass.set_bind_group(i as u32, Some(*group), &[]);
    }
    pass.draw(0..3, 0..1);
}

/// Additive blending (`dst += src`), handy for splatting particles / glow.
pub const BLEND_ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

/// Shorthand constructors for bind group layout entries.
pub mod layout {
    use wgpu::{BindGroupLayoutEntry as Entry, BindingType, BufferBindingType, ShaderStages};

    pub fn uniform(binding: u32, visibility: ShaderStages) -> Entry {
        Entry {
            binding,
            visibility,
            ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }
    }

    pub fn storage(binding: u32, visibility: ShaderStages, read_only: bool) -> Entry {
        Entry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    /// Sampled 2D float texture (`texture_2d<f32>`).
    pub fn texture(binding: u32, visibility: ShaderStages, filterable: bool) -> Entry {
        Entry {
            binding,
            visibility,
            ty: BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        }
    }

    /// Storage texture (`texture_storage_2d<format, access>`).
    pub fn storage_texture(
        binding: u32,
        visibility: ShaderStages,
        format: wgpu::TextureFormat,
        access: wgpu::StorageTextureAccess,
    ) -> Entry {
        Entry {
            binding,
            visibility,
            ty: BindingType::StorageTexture { access, format, view_dimension: wgpu::TextureViewDimension::D2 },
            count: None,
        }
    }

    pub fn sampler(binding: u32, visibility: ShaderStages, filtering: bool) -> Entry {
        let ty = if filtering { wgpu::SamplerBindingType::Filtering } else { wgpu::SamplerBindingType::NonFiltering };
        Entry { binding, visibility, ty: BindingType::Sampler(ty), count: None }
    }
}

/// Splits a 1D dispatch of `count` invocations with workgroup size `wg` into
/// (x, y) workgroup counts that respect the 65535-per-dimension limit. Shaders
/// reconstruct the linear index as
/// `gid.x + gid.y * num_workgroups.x * WG` (with `@builtin(num_workgroups)`).
pub fn dispatch_linear(count: u32, wg: u32) -> (u32, u32) {
    let groups = count.div_ceil(wg).max(1);
    const MAX: u32 = 65535;
    if groups <= MAX { (groups, 1) } else { (MAX, groups.div_ceil(MAX)) }
}
