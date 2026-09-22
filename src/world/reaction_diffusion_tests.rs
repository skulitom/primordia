use super::*;
use crate::world::Camera;

fn advance(gpu: &Gpu, world: &mut ReactionDiffusion, frames: u32) {
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

/// Runs the measurement kernel on the latest field and reads its totals.
fn measure(gpu: &Gpu, world: &ReactionDiffusion) -> Vec<f32> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.record_measure(gpu, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let totals: Vec<f32> = gpu.read_buffer(world.reduction.totals());
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    totals
}

/// The metrics of `reaction_diffusion_measure.wgsl`, recomputed on the CPU:
/// per-cell terms in f32 as the kernel evaluates them, sums in f64.
fn cpu_metrics(gpu: &Gpu, world: &ReactionDiffusion) -> Vec<f64> {
    let latest: Vec<[f32; 2]> = gpu.read_buffer(&world.buffers[world.current]);
    let previous: Vec<[f32; 2]> = gpu.read_buffer(&world.buffers[1 - world.current]);
    let mu = world.measure_uniform();
    let [alive_v, body_v, filled_v, active_dv] = mu.thresholds;
    let [w, h] = world.size.map(|n| n as i64);
    let alive_at = |x: i64, y: i64| latest[(y.rem_euclid(h) * w + x.rem_euclid(w)) as usize][1] > alive_v;
    let mut sums = [0.0_f64; 8];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) as usize;
            let c = latest[i].map(|v| v.clamp(0.0, 1.0));
            let b = previous[i].map(|v| v.clamp(0.0, 1.0));
            let dv = c[1] - b[1];
            let alive = c[1] > alive_v;
            let edge = alive
                && !(alive_at(x + 1, y) && alive_at(x - 1, y) && alive_at(x, y + 1) && alive_at(x, y - 1));
            let cell = [
                f32::from(alive),
                f32::from(c[1] > body_v),
                c[1],
                c[0],
                f32::from(dv.abs() > active_dv),
                dv,
                f32::from(edge),
                f32::from(c[1] > filled_v),
            ];
            for (sum, term) in sums.iter_mut().zip(cell) {
                *sum += f64::from(term);
            }
        }
    }
    sums.into_iter().map(|sum| sum * f64::from(mu.inv_cells)).collect()
}

#[test]
fn gpu_measurements_match_a_cpu_reduction_of_the_field() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = ReactionDiffusion::new(&gpu, [256, 144], 42);
    world.params.steps_per_frame = 5;
    world.reset(&gpu, 42);
    advance(&gpu, &mut world, 120);
    let cells = (world.size[0] * world.size[1]) as f64;

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
    let [alive, body, _, _, active, _, edge, filled] = totals[..8] else { unreachable!() };
    assert!(alive >= body && body >= filled, "{totals:?}");
    assert!(edge <= alive && edge > 0.0, "a coral pattern has a perimeter: {totals:?}");
    assert!(alive > 0.01 && active > 0.0, "{totals:?}");

    // Right after a reset both buffers hold the seed: nothing has changed yet.
    world.reset(&gpu, 7);
    let totals = measure(&gpu, &world);
    assert_eq!((totals[4], totals[5]), (0.0, 0.0));
    assert!(totals[0] > 0.0, "the seed patches are alive");
}

#[test]
fn gpu_measurements_are_finite_and_bounded_for_every_preset() {
    let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
    let mut world = ReactionDiffusion::new(&gpu, [256, 144], 42);
    for (index, name) in preset_names().iter().enumerate() {
        world.load_preset(&gpu, index, 314159);
        advance(&gpu, &mut world, 60);
        let totals = measure(&gpu, &world);
        for (metric, value) in METRICS.iter().zip(&totals) {
            assert!(value.is_finite(), "{name}: {} is {value}", metric.id);
            if metric.unit == Unit::Fraction {
                assert!((0.0..=1.0).contains(value), "{name}: {} = {value}", metric.id);
            }
        }
        assert!((0.0..=1.0).contains(&totals[2]) && (0.0..=1.0).contains(&totals[3]), "{name}: {totals:?}");
        assert!((-1.0..=1.0).contains(&totals[5]), "{name}: drift {}", totals[5]);
        // Fragile presets (solitons) can die out at some seeds and sizes; the
        // coral reef always survives.
        assert!(index > 0 || totals[0] > 0.0, "{name}: the pattern must be alive after a second");
    }
}
