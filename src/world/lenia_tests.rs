use super::*;
use crate::world::Camera;

fn frame_at<'a>(gpu: &'a Gpu, world: &Lenia, n: u64) -> Frame<'a> {
    Frame {
        gpu,
        time: n as f32 / 60.0,
        dt: 1.0 / 60.0,
        frame: n,
        view: ViewXform::fit(world.dom.size, world.output, &Camera::default()),
        target_size: world.output,
        pointer: None,
    }
}

fn scene(gpu: &Gpu, world: &Lenia) -> wgpu::TextureView {
    gpu.texture_2d("lenia test scene", world.output, SCENE_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT).1
}

/// Steps and renders like the app does: the compose pass measures the mass
/// that drives revival, so a dying soup is reseeded.
fn advance(gpu: &Gpu, world: &mut Lenia, scene: &wgpu::TextureView, frames: u32) {
    for n in 0..frames {
        let frame = frame_at(gpu, world, u64::from(n));
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.step(&frame, &mut encoder);
        world.render(&frame, &mut encoder, scene);
        gpu.queue.submit([encoder.finish()]);
        if n % 32 == 31 {
            gpu.wait_idle();
        }
    }
    gpu.wait_idle();
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

/// Runs the measurement kernel on the latest state and reads its totals.
fn measure(gpu: &Gpu, world: &Lenia) -> Vec<f32> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.record_measure(gpu, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let totals: Vec<f32> = gpu.read_buffer(world.dom.reduction.totals());
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    totals
}

/// The metrics of `lenia_measure.wgsl`, recomputed on the CPU: per-cell
/// terms in f32 as the kernel evaluates them, sums in f64.
fn cpu_metrics(gpu: &Gpu, world: &Lenia) -> Vec<f64> {
    let sim = &world.dom.sim;
    let latest: Vec<f32> = gpu.read_buffer(&sim.state[sim.current]);
    let previous: Vec<f32> = gpu.read_buffer(&sim.state[1 - sim.current]);
    let growth: Vec<[f32; 4]> = gpu.read_buffer(&sim.growth);
    let mu = world.measure_uniform();
    let n = (mu.size[0] * mu.size[1]) as usize;
    let on = |c: usize| if (c as u32) < mu.channels { 1.0 } else { 0.0 };
    let mut sums = [0.0_f64; 8];
    for i in 0..n {
        let a: [f32; 3] = std::array::from_fn(|c| latest[c * n + i].clamp(0.0, 1.0) * on(c));
        let b: [f32; 3] = std::array::from_fn(|c| previous[c * n + i].clamp(0.0, 1.0) * on(c));
        let g: [f32; 3] = std::array::from_fn(|c| growth[i][c].clamp(-1.0, 1.0) * on(c));
        let total = a[0] + a[1] + a[2];
        let change = (a[0] - b[0]).abs() + (a[1] - b[1]).abs() + (a[2] - b[2]).abs();
        let cell = [
            total,
            a[0],
            a[1],
            a[2],
            f32::from(total > mu.occupied),
            f32::from(change > mu.active),
            g[0] + g[1] + g[2],
            f32::from(total > mu.dense),
        ];
        for (sum, term) in sums.iter_mut().zip(cell) {
            *sum += f64::from(term);
        }
    }
    sums.into_iter().map(|sum| sum * f64::from(mu.inv_cells)).collect()
}

#[test]
fn gpu_measurements_match_a_cpu_reduction_and_the_compose_pass_mass() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = Lenia::new(&gpu, [640, 360], 42);
    let scene = scene(&gpu, &world);
    let soup = world.presets().iter().position(|p| *p == "Hydrogeminium").unwrap();
    world.load_preset(&gpu, soup, 42);
    advance(&gpu, &mut world, &scene, 120);
    let cells = f64::from(world.dom.cells());

    let totals = measure(&gpu, &world);
    let expected = cpu_metrics(&gpu, &world);
    assert_eq!(expected.len(), METRICS.len());
    for ((metric, gpu_value), cpu_value) in METRICS.iter().zip(&totals).zip(&expected) {
        let tolerance = 1e-4 * cpu_value.abs().max(1.0) + 8.0 / cells;
        assert!(
            (f64::from(*gpu_value) - cpu_value).abs() <= tolerance,
            "{}: GPU {gpu_value} vs CPU {cpu_value}",
            metric.id
        );
    }
    assert!(totals[METRICS.len()..].iter().all(|v| *v == 0.0), "unused lanes stay zero");
    let [mass, m1, m2, m3, occupied, active, _, dense] = totals[..8] else { unreachable!() };
    assert!((mass - (m1 + m2 + m3)).abs() < 1e-5, "{totals:?}");
    assert!(mass > 0.0 && occupied > 0.0 && active > 0.0 && dense <= occupied, "{totals:?}");

    // The compose pass counts the same mass in fixed point (truncated per cell and channel).
    let frame = frame_at(&gpu, &world, 120);
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.render(&frame, &mut encoder, &scene);
    gpu.queue.submit([encoder.finish()]);
    let fixed: Vec<u32> = gpu.read_buffer(&world.dom.mass);
    let composed = f64::from(fixed[0]) / f64::from(MASS_SCALE);
    let measured = f64::from(mass) * cells;
    let channels = f64::from(world.params.active_channels() as u32);
    assert!((composed - measured).abs() <= cells * channels / f64::from(MASS_SCALE) + 1.0, "{composed} vs {measured}");
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

#[test]
fn gpu_measurements_are_finite_and_bounded_for_every_preset() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = Lenia::new(&gpu, [640, 360], 42);
    let scene = scene(&gpu, &world);
    for (index, name) in world.presets().iter().copied().enumerate() {
        world.load_preset(&gpu, index, 314159);
        advance(&gpu, &mut world, &scene, 60);
        let totals = measure(&gpu, &world);
        for (metric, value) in METRICS.iter().zip(&totals) {
            assert!(value.is_finite(), "{name}: {} is {value}", metric.id);
            if metric.unit == Unit::Fraction {
                assert!((0.0..=1.0).contains(value), "{name}: {} = {value}", metric.id);
            }
        }
        assert!((0.0..=3.0).contains(&totals[0]) && (-3.0..=3.0).contains(&totals[6]), "{name}: {totals:?}");
        assert!(totals[0] > 0.0, "{name}: the world must hold some mass after a second");
        for c in world.params.active_channels()..CHANNELS {
            assert_eq!(totals[1 + c], 0.0, "{name}: disabled channel {} reads zero", c + 1);
        }
    }
}
