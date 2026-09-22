use super::*;
use crate::world::Camera;

fn advance(gpu: &Gpu, world: &mut ParticleLife, frames: u32) {
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

/// Runs the measurement kernel on the latest state and reads its totals.
fn measure(gpu: &Gpu, world: &ParticleLife) -> Vec<f32> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.record_measure(gpu, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let totals: Vec<f32> = gpu.read_buffer(world.buffers.reduction.totals());
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    totals
}

/// The metrics of `particle_life_measure.wgsl`, recomputed on the CPU:
/// per-particle terms in f32 as the kernel evaluates them, sums in f64.
fn cpu_metrics(gpu: &Gpu, world: &ParticleLife) -> Vec<f64> {
    let vel: Vec<[f32; 2]> = gpu.read_buffer(&world.buffers.vel);
    let sorted: Vec<[f32; 4]> = gpu.read_buffer(&world.buffers._scratch[0]);
    let starts: Vec<u32> = gpu.read_buffer(&world.buffers._scratch[2]);
    let mu = world.measure_uniform();
    let cell_index = |p: [f32; 2]| -> usize {
        let cx = ((p[0] / mu.cell[0]).floor() as i64).clamp(0, i64::from(mu.grid[0]) - 1);
        let cy = ((p[1] / mu.cell[1]).floor() as i64).clamp(0, i64::from(mu.grid[1]) - 1);
        (cy * i64::from(mu.grid[0]) + cx) as usize
    };
    let mut sums = [0.0_f64; 7];
    for i in 0..mu.count as usize {
        let speed = (vel[i][0] * vel[i][0] + vel[i][1] * vel[i][1]).sqrt().min(1e6);
        let q = sorted[i];
        let c = cell_index([q[0], q[1]]);
        let first = starts[c];
        let last = starts[c + 1].max(first);
        let n = (last - first).max(1) as f32;
        let same = (first..last).filter(|&j| sorted[j as usize][2] == q[2]).count() as f32;
        let segregation = if n > 1.0 { (((same - 1.0) / (n - 1.0) - mu.mix) * mu.inv_unmix).clamp(-1.0, 1.0) } else { 0.0 };
        let particle = [
            speed * mu.inv_r_max,
            f32::from(speed > mu.hot),
            f32::from(speed < mu.slow),
            n * mu.inv_crowd,
            f32::from(n > mu.dense),
            segregation,
        ];
        for (sum, term) in sums.iter_mut().zip(particle) {
            *sum += f64::from(term) * f64::from(mu.inv_count);
        }
    }
    for c in 0..mu.cells as usize {
        sums[6] += f64::from(u8::from(starts[c + 1] == starts[c])) * f64::from(mu.inv_cells);
    }
    sums.to_vec()
}

fn check_against_cpu(gpu: &Gpu, world: &ParticleLife) -> Vec<f32> {
    let totals = measure(gpu, world);
    let expected = cpu_metrics(gpu, world);
    assert_eq!(expected.len(), METRICS.len());
    for ((metric, gpu_value), cpu_value) in METRICS.iter().zip(&totals).zip(&expected) {
        let tolerance = 1e-4 * cpu_value.abs().max(1.0) + 8.0 / f64::from(world.count.max(1));
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
fn gpu_measurements_match_a_cpu_reduction_before_and_after_reallocation() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = ParticleLife::new(&gpu, [256, 144], 42);
    world.params.count = MIN_COUNT;
    world.reset(&gpu, 42);
    assert_eq!(world.count, MIN_COUNT);
    advance(&gpu, &mut world, 120);
    let totals = check_against_cpu(&gpu, &world);
    let [speed, hot, slow, crowding, dense, _, void] = totals[..7] else { unreachable!() };
    assert!(speed > 0.0 && hot + slow <= 1.0 && crowding > 0.0 && dense <= 1.0, "{totals:?}");
    assert!((0.0..1.0).contains(&void), "{totals:?}");

    // A larger population reallocates the buffers, the bind group and the reduction.
    world.params.count = 65_536;
    world.reset(&gpu, 7);
    assert_eq!(world.buffers.particles, 65_536);
    advance(&gpu, &mut world, 30);
    let totals = check_against_cpu(&gpu, &world);
    assert!(totals.iter().all(|v| v.is_finite()) && totals[0] > 0.0, "{totals:?}");
}

#[test]
fn gpu_measurements_are_finite_and_bounded_for_every_preset() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = ParticleLife::new(&gpu, [256, 144], 42);
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
        assert!(totals[0] >= 0.0 && totals[3] > 0.0 && (-1.0..=1.0).contains(&totals[5]), "{name}: {totals:?}");
        assert!(totals[1] + totals[2] <= 1.0 + 1e-6, "{name}: hot {} + slow {}", totals[1], totals[2]);
    }
}
