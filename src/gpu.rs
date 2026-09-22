//! GPU context plus small helpers that keep the world implementations terse.

use std::borrow::Cow;
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{anyhow, bail, Context as _, Result};
use wgpu::util::DeviceExt as _;

/// WGSL prelude prepended to every shader created through [`Gpu::shader`].
pub const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");

/// Format of the HDR scene texture every world renders into.
pub const SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Restricts wgpu to these graphics backends (comma-separated, read by wgpu).
pub const BACKEND_ENV: &str = "WGPU_BACKEND";
/// Picks the adapter whose name contains this text, ignoring case (read by [`Gpu::new`]).
pub const ADAPTER_NAME_ENV: &str = "WGPU_ADAPTER_NAME";
/// The values of [`BACKEND_ENV`] worth suggesting.
const BACKEND_VALUES: &str = "vulkan, dx12, metal, gl";

/// What wgpu needs on this platform, for the no-GPU message.
const GRAPHICS_APIS: &str = if cfg!(target_os = "macos") {
    "Metal"
} else if cfg!(windows) {
    "Vulkan or DirectX 12"
} else {
    "Vulkan"
};

/// Hold this before creating an instance and until its GPU is dropped. The
/// Windows Vulkan loader can crash during concurrent instance creation/drop
/// across otherwise independent tests. CPU-only tests can still run in parallel.
/// Tests normally take it through [`test_gpu`].
#[cfg(test)]
pub fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A GPU for a test, with [`test_lock`] held while the guard lives. Bind it as
/// `let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };` (`_`
/// would release the lock at once; `_guard` comes first so it outlives the GPU).
/// With `PRIMORDIA_GPU_TESTS=skip` this prints a note and returns `None`, so the
/// test passes without running; otherwise a missing adapter fails the test.
#[cfg(test)]
pub fn test_gpu() -> Option<(std::sync::MutexGuard<'static, ()>, Gpu)> {
    if std::env::var("PRIMORDIA_GPU_TESTS").is_ok_and(|v| v == "skip") {
        eprintln!("skipped: PRIMORDIA_GPU_TESTS=skip");
        return None;
    }
    let guard = test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None))
        .expect("GPU tests need a GPU adapter (set PRIMORDIA_GPU_TESTS=skip to skip them)");
    Some((guard, gpu))
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
    /// Creates a wgpu instance honouring the usual `WGPU_BACKEND` style env vars,
    /// warning about backend names wgpu does not know (it ignores them silently).
    pub fn create_instance() -> wgpu::Instance {
        if let Some(warning) = std::env::var(BACKEND_ENV).ok().and_then(|value| backend_env_warning(&value)) {
            log::warn!("{warning}");
        }
        wgpu::Instance::new(&wgpu::InstanceDescriptor::from_env_or_default())
    }

    /// Picks an adapter and opens a device with every limit the adapter
    /// supports. `WGPU_ADAPTER_NAME` names the adapter (any part of its name,
    /// ignoring case); otherwise wgpu picks the high-performance one
    /// (`WGPU_POWER_PREF=low` asks for the other). Either way it must be able to
    /// draw to `surface`, if given.
    pub async fn new(instance: wgpu::Instance, surface: Option<&wgpu::Surface<'_>>) -> Result<Self> {
        let adapter = match env_value(ADAPTER_NAME_ENV) {
            Some(wanted) => named_adapter(&instance, surface, &wanted)?,
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::from_env()
                        .unwrap_or(wgpu::PowerPreference::HighPerformance),
                    force_fallback_adapter: false,
                    compatible_surface: surface,
                })
                .await
                .map_err(|e| {
                    log::debug!("wgpu found no adapter: {e}");
                    anyhow!(no_gpu_message(std::env::var(BACKEND_ENV).ok().as_deref()))
                })?,
        };

        let info = adapter.get_info();
        log::info!("GPU: {}", describe_adapter(&info));

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

/// A variable's value, or `None` when it is unset or blank.
fn env_value(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// "NVIDIA GeForce RTX 4090 (Vulkan, DiscreteGpu)".
pub fn describe_adapter(info: &wgpu::AdapterInfo) -> String {
    format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
}

/// The first adapter whose name contains `wanted` (ignoring case) and that can
/// draw to `surface`, if given. The error lists every adapter wgpu found.
fn named_adapter(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
    wanted: &str,
) -> Result<wgpu::Adapter> {
    let mut adapters = instance.enumerate_adapters(wgpu::Backends::all());
    if adapters.is_empty() {
        bail!(no_gpu_message(std::env::var(BACKEND_ENV).ok().as_deref()));
    }
    let infos: Vec<wgpu::AdapterInfo> = adapters.iter().map(wgpu::Adapter::get_info).collect();
    let usable: Vec<bool> = adapters.iter().map(|a| surface.is_none_or(|s| a.is_surface_supported(s))).collect();
    match find_adapter(&infos, &usable, wanted) {
        Some(index) => Ok(adapters.swap_remove(index)),
        None => bail!(no_adapter_match_message(wanted, &infos, &usable)),
    }
}

/// Index of the first usable adapter whose name contains `wanted`, ignoring case.
fn find_adapter(infos: &[wgpu::AdapterInfo], usable: &[bool], wanted: &str) -> Option<usize> {
    let wanted = wanted.trim().to_lowercase();
    infos.iter().zip(usable).position(|(info, &ok)| ok && info.name.to_lowercase().contains(&wanted))
}

/// Why no adapter matched [`ADAPTER_NAME_ENV`], listing every adapter once.
fn no_adapter_match_message(wanted: &str, infos: &[wgpu::AdapterInfo], usable: &[bool]) -> String {
    let mut lines: Vec<String> = Vec::new();
    for (info, &ok) in infos.iter().zip(usable) {
        let line = format!("  {}{}", describe_adapter(info), if ok { "" } else { " - cannot draw to this window" });
        if !lines.contains(&line) {
            lines.push(line);
        }
    }
    format!(
        "{ADAPTER_NAME_ENV}='{wanted}' matches no GPU adapter. Available adapters:\n{}\n\
         Set {ADAPTER_NAME_ENV} to part of one of these names, or unset it to choose automatically.",
        lines.join("\n")
    )
}

/// The error when wgpu finds no adapter at all. `backend_env` is [`BACKEND_ENV`]'s value, if set.
fn no_gpu_message(backend_env: Option<&str>) -> String {
    let mut message = format!(
        "Primordia needs a GPU with {GRAPHICS_APIS} support and could not find one. \
         Update your graphics driver, then run `primordia selftest` to check it."
    );
    if let Some(value) = backend_env {
        message += &format!(
            "\n{BACKEND_ENV}='{value}' limits the search to those backends (valid values: {BACKEND_VALUES}); \
             unset it to try them all."
        );
    }
    message
}

/// A warning about entries of [`BACKEND_ENV`] (`value`) that name no backend;
/// wgpu skips them without a word, and with none left it finds no GPU.
fn backend_env_warning(value: &str) -> Option<String> {
    let unknown: Vec<&str> =
        value.split(',').map(str::trim).filter(|b| wgpu::Backends::from_comma_list(b).is_empty()).collect();
    if wgpu::Backends::from_comma_list(value).is_empty() {
        Some(format!(
            "{BACKEND_ENV}='{value}' names no graphics backend (valid values: {BACKEND_VALUES}), so no GPU can be found"
        ))
    } else if !unknown.is_empty() {
        let unknown = unknown.join("', '");
        Some(format!("{BACKEND_ENV}: ignoring unknown backend '{unknown}' (valid values: {BACKEND_VALUES})"))
    } else {
        None
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

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::{Backend, DeviceType};

    fn info(name: &str, backend: Backend, device_type: DeviceType) -> wgpu::AdapterInfo {
        let (driver, driver_info) = (String::new(), String::new());
        wgpu::AdapterInfo { name: name.to_string(), vendor: 0, device: 0, device_type, driver, driver_info, backend }
    }

    /// The adapters of the development machine, in wgpu's order.
    fn machine() -> Vec<wgpu::AdapterInfo> {
        vec![
            info("NVIDIA GeForce RTX 4090", Backend::Vulkan, DeviceType::DiscreteGpu),
            info("AMD Radeon(TM) Graphics", Backend::Vulkan, DeviceType::IntegratedGpu),
            info("NVIDIA GeForce RTX 4090", Backend::Dx12, DeviceType::DiscreteGpu),
            info("AMD Radeon(TM) Graphics", Backend::Dx12, DeviceType::IntegratedGpu),
            info("NVIDIA GeForce RTX 4090", Backend::Dx12, DeviceType::DiscreteGpu),
            info("Microsoft Basic Render Driver", Backend::Dx12, DeviceType::Cpu),
            info("NVIDIA GeForce RTX 4090/PCIe/SSE2", Backend::Gl, DeviceType::Other),
        ]
    }

    #[test]
    fn adapters_are_described_by_name_backend_and_type() {
        let text = describe_adapter(&info("AMD Radeon(TM) Graphics", Backend::Vulkan, DeviceType::IntegratedGpu));
        assert_eq!(text, "AMD Radeon(TM) Graphics (Vulkan, IntegratedGpu)");
    }

    #[test]
    fn adapter_names_match_any_part_ignoring_case() {
        let infos = machine();
        let all = vec![true; infos.len()];
        assert_eq!(find_adapter(&infos, &all, "radeon"), Some(1), "the first match wins");
        assert_eq!(find_adapter(&infos, &all, "  RTX 4090 "), Some(0));
        assert_eq!(find_adapter(&infos, &all, "basic render"), Some(5));
        assert_eq!(find_adapter(&infos, &all, "sse2"), Some(6));
        assert_eq!(find_adapter(&infos, &all, "nonexistent"), None);
        // Adapters that cannot draw to the window are passed over.
        let usable = [false, false, true, true, true, true, true];
        assert_eq!(find_adapter(&infos, &usable, "radeon"), Some(3));
        assert_eq!(find_adapter(&infos, &[false; 7], "radeon"), None);
    }

    #[test]
    fn a_missing_adapter_name_lists_every_adapter_once() {
        let infos = machine();
        let mut usable = vec![true; infos.len()];
        usable[6] = false;
        let message = no_adapter_match_message("nonexistent", &infos, &usable);
        let lines: Vec<&str> = message.lines().collect();
        assert_eq!(lines[0], "WGPU_ADAPTER_NAME='nonexistent' matches no GPU adapter. Available adapters:");
        assert_eq!(
            &lines[1..7],
            [
                "  NVIDIA GeForce RTX 4090 (Vulkan, DiscreteGpu)",
                "  AMD Radeon(TM) Graphics (Vulkan, IntegratedGpu)",
                "  NVIDIA GeForce RTX 4090 (Dx12, DiscreteGpu)",
                "  AMD Radeon(TM) Graphics (Dx12, IntegratedGpu)",
                "  Microsoft Basic Render Driver (Dx12, Cpu)",
                "  NVIDIA GeForce RTX 4090/PCIe/SSE2 (Gl, Other) - cannot draw to this window",
            ]
        );
        assert!(lines[7].starts_with("Set WGPU_ADAPTER_NAME to part of one of these names"), "{message}");
    }

    #[test]
    fn the_no_gpu_message_says_what_to_do_and_names_wgpu_backend() {
        let plain = no_gpu_message(None);
        assert!(plain.starts_with(&format!("Primordia needs a GPU with {GRAPHICS_APIS} support")), "{plain}");
        assert!(plain.contains("Update your graphics driver") && plain.contains("`primordia selftest`"), "{plain}");
        assert!(!plain.contains("WGPU_BACKEND"), "{plain}");
        let limited = no_gpu_message(Some("bogus"));
        assert!(limited.starts_with(&plain), "{limited}");
        assert!(limited.contains("WGPU_BACKEND='bogus' limits the search"), "{limited}");
        assert!(limited.contains("vulkan, dx12, metal, gl"), "{limited}");
    }

    #[test]
    fn unknown_wgpu_backend_values_are_reported() {
        for fine in ["vulkan", "DX12", "vulkan, dx12", "gl", "metal"] {
            assert_eq!(backend_env_warning(fine), None, "{fine}");
        }
        let none = backend_env_warning("bogus").unwrap();
        assert!(none.starts_with("WGPU_BACKEND='bogus' names no graphics backend"), "{none}");
        assert!(none.ends_with("so no GPU can be found"), "{none}");
        assert!(backend_env_warning("").is_some(), "an empty value disables every backend");
        let partly = backend_env_warning("vulkan,bogus,dx13").unwrap();
        assert!(partly.starts_with("WGPU_BACKEND: ignoring unknown backend 'bogus', 'dx13'"), "{partly}");
    }
}
