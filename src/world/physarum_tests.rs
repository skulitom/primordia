use super::*;
use crate::world::Camera;

/// Mirrors `Agent` in physarum.wgsl (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Agent {
    pos: [f32; 2],
    heading: f32,
    state: u32,
}

fn advance(gpu: &Gpu, world: &mut Physarum, frames: u32) {
    for n in 0..frames {
        let frame = Frame {
            gpu,
            time: n as f32 / 60.0,
            dt: 1.0 / 60.0,
            frame: n as u64,
            view: ViewXform::fit(world.size, world.size, &Camera::default()),
            target_size: world.size,
            pointer: None,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.step(&frame, &mut encoder);
        gpu.queue.submit([encoder.finish()]);
        if n % 32 == 31 {
            gpu.wait_idle();
        }
    }
    gpu.wait_idle();
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

/// Runs the measurement kernels on the latest state and reads their totals.
fn measure(gpu: &Gpu, world: &Physarum) -> Vec<f32> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.record_measure(gpu, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let totals: Vec<f32> = gpu.read_buffer(world.reduction.totals());
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    totals
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mut mantissa = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if mantissa == 0 => sign,
        0 => {
            // Subnormal: renormalise.
            let mut e = 127 - 15 + 1;
            while mantissa & 0x400 == 0 {
                mantissa <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((mantissa & 0x3ff) << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

/// Blocking readback of a whole rgba32float / rgba16float texture.
fn read_texture(gpu: &Gpu, texture: &wgpu::Texture, size: [u32; 2], format: wgpu::TextureFormat) -> Vec<[f32; 4]> {
    let bytes_per_pixel = match format {
        wgpu::TextureFormat::Rgba32Float => 16,
        wgpu::TextureFormat::Rgba16Float => 8,
        other => panic!("unexpected field format {other:?}"),
    };
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_row = (size[0] * bytes_per_pixel).div_ceil(align) * align;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("physarum test readback"),
        size: u64::from(padded_row) * u64::from(size[1]),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row), rows_per_image: None },
        },
        wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
    );
    gpu.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    gpu.wait_idle();
    rx.recv().unwrap().unwrap();
    let bytes = staging.slice(..).get_mapped_range();
    let mut texels = Vec::with_capacity((size[0] * size[1]) as usize);
    for y in 0..size[1] as usize {
        let row = &bytes[y * padded_row as usize..][..(size[0] * bytes_per_pixel) as usize];
        for x in 0..size[0] as usize {
            let px = &row[x * bytes_per_pixel as usize..][..bytes_per_pixel as usize];
            texels.push(std::array::from_fn(|c| match format {
                wgpu::TextureFormat::Rgba32Float => f32::from_le_bytes(px[c * 4..c * 4 + 4].try_into().unwrap()),
                _ => f16_to_f32(u16::from_le_bytes([px[c * 2], px[c * 2 + 1]])),
            }));
        }
    }
    drop(bytes);
    staging.unmap();
    texels
}

/// The metrics of `physarum_measure.wgsl`, recomputed on the CPU: per-cell
/// and per-agent terms in f32 as the kernels evaluate them, sums in f64.
fn cpu_metrics(gpu: &Gpu, world: &Physarum) -> Vec<f64> {
    let format = trail_format(gpu);
    let latest = read_texture(gpu, &world.sim._trail_textures[world.current], world.size, format);
    let previous = read_texture(gpu, &world.sim._trail_textures[1 - world.current], world.size, format);
    let traffic = read_texture(gpu, &world.sim._traffic_textures[world.current], world.size, TRAFFIC_FORMAT);
    let agents: Vec<Agent> = gpu.read_buffer(&world.agents);
    let mu = world.measure_uniform();
    let [marked, vein, travelled, _] = mu.thresholds;
    let density = |t: &[f32; 4]| -> f32 { (0..4).map(|s| t[s].max(0.0) * mu.inv_level[s]).sum() };
    let mut sums = [0.0_f64; 8];
    for i in 0..latest.len() {
        let x = density(&latest[i]);
        let x0 = density(&previous[i]);
        let g: f32 = (0..world.population.species).map(|s| traffic[i][s].max(0.0)).sum();
        let cell = [
            f32::from(x > marked),
            f32::from(x > vein),
            x.clamp(2f32.powi(-24), 16.0).log2(),
            f32::from(g > travelled),
            x.min(10_000.0),
            f32::from(x > x0),
        ];
        for (sum, term) in sums.iter_mut().zip(cell) {
            *sum += f64::from(term) * f64::from(mu.inv_cells);
        }
    }
    let [w, h] = world.size.map(|n| n as i64);
    for agent in agents.iter().take(mu.agent_count as usize) {
        let s = (agent.state & 3) as usize;
        let cx = (agent.pos[0].floor() as i64).rem_euclid(w);
        let cy = (agent.pos[1].floor() as i64).rem_euclid(h);
        let t = latest[(cy * w + cx) as usize][s].max(0.0);
        let xs = (t * mu.inv_level[s]).clamp(0.0, 10_000.0);
        sums[6] += f64::from(u8::from(xs > vein)) * f64::from(mu.inv_agents);
        sums[7] += f64::from(xs) * f64::from(mu.inv_agents);
    }
    sums.to_vec()
}

fn check_against_cpu(gpu: &Gpu, world: &Physarum) -> Vec<f32> {
    let totals = measure(gpu, world);
    let expected = cpu_metrics(gpu, world);
    assert_eq!(expected.len(), METRICS.len());
    let population = f64::from((world.size[0] * world.size[1]).min(world.population.agents.max(1)));
    for ((metric, gpu_value), cpu_value) in METRICS.iter().zip(&totals).zip(&expected) {
        // A dot product decides the thresholds, so a handful of cells may flip.
        let tolerance = 2e-4 * cpu_value.abs().max(1.0) + 16.0 / population;
        assert!(
            (f64::from(*gpu_value) - cpu_value).abs() <= tolerance,
            "{}: GPU {gpu_value} vs CPU {cpu_value}",
            metric.id
        );
    }
    assert!(totals[METRICS.len()..].iter().all(|v| *v == 0.0), "unused lanes stay zero");
    totals
}

#[test]
fn gpu_measurements_match_a_cpu_reduction_and_a_cleared_run_reads_empty() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = Physarum::new(&gpu, [256, 144], 42);
    let rivals = world.presets().iter().position(|p| *p == "Rival Colonies").unwrap();
    world.load_preset(&gpu, rivals, 42);
    assert!(world.population.species >= 2, "a multi-species preset exercises the per-species levels");
    advance(&gpu, &mut world, 120);
    let totals = check_against_cpu(&gpu, &world);
    let [ground, veins, concentration, travelled, trail_mass, reinforced, on_vein, agent_trail] = totals[..8] else {
        unreachable!()
    };
    assert!(ground > 0.0 && veins <= ground && travelled > 0.0, "{totals:?}");
    // Jensen: the mean log density never exceeds the log of the mean density.
    assert!(concentration <= trail_mass.log2() + 1e-3 && concentration >= -24.0, "{totals:?}");
    assert!((0.05..50.0).contains(&trail_mass), "the trail settles within an order of its mean level: {totals:?}");
    assert!((0.0..=1.0).contains(&reinforced) && (0.0..=1.0).contains(&on_vein) && agent_trail > 0.0, "{totals:?}");

    // Right after a reset the fields are cleared: nothing is marked, and the
    // log density sits at its floor.
    world.reset(&gpu, 7);
    let totals = measure(&gpu, &world);
    assert_eq!(&totals[..2], &[0.0, 0.0]);
    assert!((totals[2] + 24.0).abs() < 1e-3, "{totals:?}");
    assert_eq!(&totals[3..8], &[0.0; 5]);

    // Tripling the population reallocates the agent buffer and the bind groups.
    world.pending.agents = (world.population.agents * 3).min(world.max_agents);
    world.reset(&gpu, 9);
    assert_eq!(world.agent_capacity, world.population.agents);
    advance(&gpu, &mut world, 30);
    let totals = check_against_cpu(&gpu, &world);
    assert!(totals.iter().all(|v| v.is_finite()) && totals[0] > 0.0, "{totals:?}");
}

#[test]
fn gpu_measurements_are_finite_and_bounded_for_every_preset() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = Physarum::new(&gpu, [256, 144], 42);
    for (index, name) in world.presets().iter().copied().enumerate() {
        world.load_preset(&gpu, index, 314159);
        advance(&gpu, &mut world, 60);
        let totals = measure(&gpu, &world);
        for (metric, value) in METRICS.iter().zip(&totals) {
            assert!(value.is_finite(), "{name}: {} is {value}", metric.id);
            if metric.unit == Unit::Fraction {
                assert!((0.0..=1.0).contains(value), "{name}: {} = {value}", metric.id);
            }
        }
        assert!((-24.0..=4.0).contains(&totals[2]), "{name}: concentration {}", totals[2]);
        assert!(totals[0] > 0.0 && totals[4] > 0.0, "{name}: agents must leave trails: {totals:?}");
    }
}
