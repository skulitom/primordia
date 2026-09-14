use super::*;
use crate::world::Camera;

fn read<T: Pod>(gpu: &Gpu, source: &wgpu::Buffer) -> Vec<T> {
    gpu.read_buffer(source)
}

/// Runs the measurement kernel on the habitat's latest state and reads its totals.
fn measure(gpu: &Gpu, world: &Symbiosis) -> Vec<f32> {
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.record_measure(gpu, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let totals = read::<f32>(gpu, world.reduction.totals());
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    totals
}

/// The metrics of `symbiosis_measure.wgsl`, recomputed on the CPU: the
/// per-cell terms in f32 exactly as the kernel evaluates them, the sums in f64.
fn cpu_metrics(gpu: &Gpu, world: &Symbiosis) -> Vec<f64> {
    let latest = read::<[f32; 4]>(gpu, &world.fields[world.current]);
    let previous = read::<[f32; 4]>(gpu, &world.fields[1 - world.current]);
    let soil = read::<f32>(gpu, &world.fertility[world.current]);
    let deposits = read::<u32>(gpu, &world.deposits);
    let mu = world.measure_uniform();
    let [growth, active, exhausted, route] = mu.thresholds;
    let mut sums = [0.0_f64; 8];
    for i in 0..latest.len() {
        let v = latest[i][1].clamp(0.0, 1.0);
        let dv = v - previous[i][1].clamp(0.0, 1.0);
        let f = soil[i].clamp(0.0, 1.0);
        let d = deposits[i].min(mu.count) as f32;
        let growing = f32::from(v > growth);
        let cell = [
            growing,
            v,
            f32::from(dv.abs() > active),
            dv,
            f32::from(latest[i][2] > route),
            f,
            f32::from(f < exhausted),
            d * growing,
        ];
        for (sum, term) in sums.iter_mut().zip(cell) {
            *sum += f64::from(term);
        }
    }
    let weights = [f64::from(mu.inv_cells); 7].into_iter().chain([f64::from(mu.inv_count)]);
    sums.into_iter().zip(weights).map(|(sum, weight)| sum * weight).collect()
}

fn advance(gpu: &Gpu, world: &mut Symbiosis, frames: u32) {
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

fn chemicals(field: &[[f32; 4]]) -> Vec<[f32; 2]> {
    field.iter().map(|f| [f[0], f[1]]).collect()
}

#[test]
fn gpu_feedback_works_in_both_directions_and_zero_decouples() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    // Odd sub-step counts also exercise alternating frame-start bind groups.
    world.params.steps = 7;
    world.params.depletion = 0.8;
    let count = world.count;
    for (relationship, coupling) in Relationship::ALL.into_iter().flat_map(|r| [0.0, 1.0].map(|c| (r, c))) {
        world.params.relationship = relationship;
        world.params.coupling = coupling;
        world.count = count;
        world.reset(&gpu, 42);
        advance(&gpu, &mut world, 40);
        let original = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
        let original_agents = read::<u8>(&gpu, &world.agents);

        world.reset(&gpu, 42);
        advance(&gpu, &mut world, 40);
        assert_eq!(original, read::<[f32; 4]>(&gpu, &world.fields[world.current]), "reset must reproduce the habitat");
        assert_eq!(original_agents, read::<u8>(&gpu, &world.agents), "reset must reproduce the agents");

        // Remove the agents, retaining identical chemical seeds and parameters.
        world.reset(&gpu, 42);
        world.count = 0;
        advance(&gpu, &mut world, 40);
        let no_agents = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
        if coupling == 0.0 {
            assert_eq!(chemicals(&original), chemicals(&no_agents));
        } else {
            let delta: f32 = original.iter().zip(&no_agents).map(|(a, b)| (a[1] - b[1]).abs()).sum();
            assert!(delta > 0.1, "agent trails must change the chemistry: {delta}");
        }

        // Remove the chemical seeds, retaining identical agents.
        world.count = count;
        world.reset(&gpu, 42);
        let empty = vec![[1.0_f32, 0.0, 0.0, 0.0]; (world.size[0] * world.size[1]) as usize];
        for buffer in &world.fields {
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&empty));
        }
        advance(&gpu, &mut world, 40);
        let no_chemistry = read::<u8>(&gpu, &world.agents);
        if coupling == 0.0 {
            assert_eq!(original_agents, no_chemistry);
        } else {
            assert_ne!(original_agents, no_chemistry, "chemistry must steer the agents");
        }
    }
}

#[test]
fn gpu_presets_keep_a_finite_living_habitat_at_full_coupling() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [160, 120], 42);
    let mut coverage = Vec::new();
    for (index, name, seed) in PRESETS.iter().enumerate().flat_map(|(i, name)| [42, 314159].map(|seed| (i, name, seed)))
    {
        world.load_preset(&gpu, index, seed);
        world.params.coupling = 1.0;
        advance(&gpu, &mut world, 3600);
        let field = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
        assert!(read::<f32>(&gpu, &world.fertility[world.current]).iter().all(|f| (0.0..=1.0).contains(f)));
        for f in &field {
            assert!(f.iter().all(|v| v.is_finite()), "{name}: {f:?}");
            assert!((0.0..=1.0).contains(&f[0]) && (0.0..=1.0).contains(&f[1]), "{name}: {f:?}");
            assert!((0.0..=32.0).contains(&f[2]), "{name}: {f:?}");
        }
        let occupied = field.iter().filter(|f| f[1] > 0.1).count() as f32 / field.len() as f32;
        let trail: f32 = field.iter().map(|f| f[2]).sum::<f32>() / field.len() as f32;
        println!("{name}, seed {seed}: occupied {occupied:.3}, mean trail {trail:.3}");
        coverage.push(occupied);
        assert!((0.01..0.95).contains(&occupied), "{name} should retain growth and open ground");
        assert!(trail > 0.01, "{name} should retain agent trails");
        for agent in read::<Agent>(&gpu, &world.agents) {
            assert!((0.0..world.size[0] as f32).contains(&agent.pos[0]));
            assert!((0.0..world.size[1] as f32).contains(&agent.pos[1]));
            assert!(agent.heading.is_finite());
        }
    }
    let min = coverage.iter().copied().fold(f32::INFINITY, f32::min);
    let max = coverage.iter().copied().fold(0.0, f32::max);
    assert!(max - min > 0.3, "presets must retain distinct growth densities after a minute");
}

#[test]
fn relationships_change_growth_in_opposite_directions_and_weave_germinates_bare_routes() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    world.params.steps = 1;
    world.params.scale = 1.0;
    world.params.terrain = 0.0;
    world.count = 0;
    let mut growth = Vec::new();
    for relationship in Relationship::ALL {
        world.params.relationship = relationship;
        for activator in [0.0, 0.2] {
            for coupling in [0.0, 1.0] {
                world.params.coupling = coupling;
                let cells = vec![[0.8_f32, activator, 8.0, 0.0]; 96 * 80];
                for buffer in &world.fields {
                    gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&cells));
                }
                advance(&gpu, &mut world, 1);
                growth.push(read::<[f32; 4]>(&gpu, &world.fields[world.current])[0][1]);
            }
        }
    }
    assert!(growth[3] > growth[2], "cultivators nourish existing growth");
    assert!(growth[7] < growth[6], "grazers consume existing growth");
    assert_eq!(growth[0], 0.0);
    assert_eq!(growth[1], 0.0);
    assert_eq!(growth[4], 0.0);
    assert_eq!(growth[5], 0.0);
    assert_eq!(growth[8], 0.0);
    assert!(growth[9] > 0.001, "weavers germinate new growth along busy routes");
}

#[test]
fn gpu_brush_wraps_at_the_edges_and_display_layers_leave_simulation_unchanged() {
    let _guard = crate::gpu::test_lock();
    use crate::capture::{self, Readback};
    use crate::post::Post;
    use crate::world::Pointer;

    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    let blank = vec![[1.0_f32, 0.0, 0.0, 0.0]; 96 * 80];
    for buffer in &world.fields {
        gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&blank));
    }
    let mut frame = Frame {
        gpu: &gpu,
        time: 0.0,
        dt: 1.0 / 60.0,
        frame: 0,
        view: ViewXform::fit(world.size, world.size, &Camera::default()),
        target_size: world.size,
        pointer: Some(Pointer { pos: [-0.005, 1.005], primary: true, secondary: false, radius: 8.0 }),
    };
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.step(&frame, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let painted = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
    let corners = [0, 95, 96 * 79, 96 * 80 - 1];
    assert!(corners.iter().all(|&i| painted[i][1] > 0.1), "brush must paint across both wrapped edges");
    assert!(painted[96 * 40 + 48][1] < 0.001, "brush must stay local");
    frame.pointer = frame.pointer.map(|p| Pointer { primary: false, secondary: true, ..p });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.step(&frame, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let erased = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
    assert!(corners.iter().all(|&i| erased[i][1] < painted[i][1] * 0.5));

    world.reset(&gpu, 42);
    advance(&gpu, &mut world, 120);
    let before = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
    let size = [384, 320];
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let post = Post::new(&gpu, size, format);
    let (texture, target) = gpu.texture_2d(
        "symbiosis layer test",
        size,
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let readback = Readback::new(&gpu, size, format);
    let mut views = Vec::new();
    frame.pointer = None;
    frame.target_size = size;
    frame.view = ViewXform::fit(world.size, size, &Camera { zoom: 0.5, ..Camera::default() });
    let soil_before = read::<f32>(&gpu, &world.fertility[world.current]);
    for layer in [Layer::Together, Layer::Chemistry, Layer::Trails, Layer::Fertility] {
        world.params.layer = layer;
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.render(&frame, &mut encoder, post.scene_view());
        post.run(&gpu, &mut encoder, &world.post, 0.0, &target);
        readback.copy_from(&mut encoder, &texture);
        gpu.queue.submit([encoder.finish()]);
        let pixels = readback.read(&gpu).unwrap();
        let path = format!("target/symbiosis-preview/layer-{}.png", layer as u32);
        capture::save_png(std::path::Path::new(&path), size, pixels.clone()).unwrap();
        views.push(pixels);
    }
    assert_ne!(views[0], views[1]);
    assert_ne!(views[0], views[2]);
    assert_ne!(views[1], views[2]);
    assert!(views[..3].iter().all(|view| *view != views[3]));
    assert_eq!(before, read::<[f32; 4]>(&gpu, &world.fields[world.current]));
    assert_eq!(soil_before, read::<f32>(&gpu, &world.fertility[world.current]));
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

#[test]
#[ignore = "renders the Symbiosis presets to target/symbiosis-preview"]
fn render_symbiosis_previews() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut images = Vec::new();
    for (name, seed) in PRESETS.iter().flat_map(|name| [42, 314159].map(|seed| (name, seed))) {
        let mut job = crate::headless::RenderJob::new("symbiosis");
        job.preset = Some((*name).to_owned());
        job.seed = seed;
        job.size = [1280, 800];
        job.frames = 3600;
        job.max_fps = 0.0;
        job.out = Some(format!("target/symbiosis-preview/{seed}/{}.png", name.to_lowercase().replace(' ', "-")).into());
        crate::headless::render_with(&gpu, &job).unwrap();
        images.push((if seed == 42 { "42" } else { "314159" }, job.out.unwrap()));
    }
    // Keep each seed in a row, with presets in their menu order.
    images.sort_by_key(|(seed, path)| {
        let index =
            PRESETS.iter().position(|name| path.file_stem().unwrap() == crate::headless::slug(name).as_str()).unwrap();
        (*seed != "42", index)
    });
    crate::headless::contact_sheet(&images, std::path::Path::new("target/symbiosis-preview/presets.png"), 5).unwrap();
}

#[test]
fn mutations_explore_relationships_geography_and_seeding_with_valid_recipes() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    let mut relationships = std::collections::BTreeSet::new();
    let mut seedings = std::collections::BTreeSet::new();
    let mut scales = Vec::new();
    for seed in 0..32 {
        world.mutate(&gpu, seed);
        world.params.validate().unwrap();
        relationships.insert(world.params.relationship as u32);
        seedings.insert(world.params.seeding as u32);
        scales.push(world.params.scale);
        let recipe = world.settings().unwrap();
        let decoded = serde_json::from_str(&serde_json::to_string(&recipe).unwrap()).unwrap();
        world.restore_settings(&gpu, &decoded, seed).unwrap();
        assert_eq!(serde_json::to_value(world.settings().unwrap()).unwrap(), serde_json::to_value(recipe).unwrap());
    }
    assert_eq!(relationships.len(), Relationship::ALL.len());
    assert_eq!(seedings.len(), Seeding::ALL.len());
    let span = scales.iter().copied().fold(0.0, f32::max) - scales.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(span > 0.8, "mutation should explore visibly different growth scales");
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

#[test]
fn comparison_replays_the_seed_stays_independent_and_restores_saved_settings() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    let initial = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
    advance(&gpu, &mut world, 25);
    world.set_comparison(&gpu, true);
    assert_eq!(initial, read::<[f32; 4]>(&gpu, &world.fields[world.current]), "enabling starts from the seed");
    let reference = world.reference.as_ref().unwrap();
    assert!(reference.reference.is_none());
    assert_eq!(initial, read::<[f32; 4]>(&gpu, &reference.fields[reference.current]));
    assert_eq!(read::<u8>(&gpu, &world.agents), read::<u8>(&gpu, &reference.agents));

    world.params.steps = 7;
    world.params.speed = 2.1;
    world.params.coupling = 0.0;
    advance(&gpu, &mut world, 75);
    let reference = world.reference.as_ref().unwrap();
    assert_eq!(reference.params.speed, world.params.speed);
    assert_eq!(
        read::<[f32; 4]>(&gpu, &world.fields[world.current]),
        read::<[f32; 4]>(&gpu, &reference.fields[reference.current]),
        "both sides must match when coupling is zero"
    );

    world.params.coupling = 0.9;
    world.reset(&gpu, world.seed);
    let recipe = world.settings().unwrap();
    advance(&gpu, &mut world, 75);
    let coupled = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
    let reference = world.reference.as_ref().unwrap();
    let uncoupled = read::<[f32; 4]>(&gpu, &reference.fields[reference.current]);
    assert_ne!(coupled, uncoupled, "feedback should produce a different habitat");
    assert_eq!(reference.params.coupling, 0.0);
    world.set_comparison(&gpu, false);
    assert!(world.reference.is_none());
    assert_eq!(
        coupled,
        read::<[f32; 4]>(&gpu, &world.fields[world.current]),
        "leaving comparison keeps the current state"
    );

    let decoded: WorldSettings = serde_json::from_str(&serde_json::to_string(&recipe).unwrap()).unwrap();
    world.restore_settings(&gpu, &decoded, 42).unwrap();
    advance(&gpu, &mut world, 75);
    assert_eq!(coupled, read::<[f32; 4]>(&gpu, &world.fields[world.current]));
    let reference = world.reference.as_ref().unwrap();
    assert_eq!(uncoupled, read::<[f32; 4]>(&gpu, &reference.fields[reference.current]));

    world.load_preset(&gpu, 1, 73);
    assert!(world.params.compare);
    assert_eq!(world.reference.as_ref().unwrap().seed, 73);
    world.mutate(&gpu, 91);
    assert!(world.params.compare);
    assert_eq!(world.reference.as_ref().unwrap().seed, 91);

    // Recipes written before comparison and ecological controls keep their
    // original relationship, seeding, scale and uniform habitat.
    let mut old_recipe = serde_json::to_value(recipe).unwrap();
    old_recipe["params"].as_object_mut().unwrap().remove("compare");
    for field in ["relationship", "seeding", "terrain", "scale", "trail_light", "depletion", "recovery", "reference"] {
        old_recipe["params"].as_object_mut().unwrap().remove(field);
    }
    let legacy: WorldSettings = serde_json::from_value(old_recipe).unwrap();
    world.restore_settings(&gpu, &legacy, 42).unwrap();
    assert!(!world.params.compare && world.reference.is_none());
    assert_eq!(world.params.relationship, Relationship::Cultivate);
    assert_eq!(world.params.seeding, Seeding::Islands);
    assert_eq!(world.params.terrain, 0.0);
    assert_eq!(world.params.scale, 1.0);
    assert_eq!(world.params.trail_light, 1.0);
    assert_eq!(world.params.depletion, 0.0);
    assert_eq!(world.params.recovery, recovery_default());
    assert_eq!(world.params.reference, Reference::CouplingOff);
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

#[test]
fn fertility_remembers_local_traffic_and_recovers_after_it_leaves() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    world.count = 0;
    world.params.depletion = 1.0;
    world.params.coupling = 1.0;
    world.params.retention = 0.99;
    world.params.steps = 7;
    let mut cells = vec![[1.0_f32, 0.0, 0.0, 0.0]; 96 * 80];
    for y in 25..55 {
        for x in 30..66 {
            cells[y * 96 + x][2] = 8.0;
        }
    }
    for buffer in &world.fields {
        gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&cells));
    }
    advance(&gpu, &mut world, 150);
    let exhausted = read::<f32>(&gpu, &world.fertility[world.current]);
    let centre = 40 * 96 + 48;
    assert!(exhausted[centre] < 0.5, "busy ground must exhaust: {}", exhausted[centre]);
    assert!(exhausted[0] > 0.999, "untouched ground must remain fertile");
    // Remove the traffic. Fertility must retain its past, then recover slowly.
    let blank = vec![[1.0_f32, 0.0, 0.0, 0.0]; 96 * 80];
    for buffer in &world.fields {
        gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&blank));
    }
    advance(&gpu, &mut world, 1);
    let resting = read::<f32>(&gpu, &world.fertility[world.current]);
    assert!((resting[centre] - exhausted[centre]).abs() < 0.001, "clearing trails must not erase history");
    advance(&gpu, &mut world, 600);
    let recovered = read::<f32>(&gpu, &world.fertility[world.current]);
    assert!(recovered[centre] > resting[centre] + 0.1);
    assert!(recovered.iter().all(|f| (0.0..=1.0).contains(f)));
    world.reset(&gpu, 42);
    assert!(read::<f32>(&gpu, &world.fertility[world.current]).iter().all(|f| *f == 1.0));
}

#[test]
fn fertility_recovery_uses_frames_not_chemical_substeps_and_depletion_inhibits_growth() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [64, 64], 42);
    world.count = 0;
    world.params.depletion = 1.0;
    world.params.coupling = 1.0;
    world.params.recovery = 5.0;
    for steps in [1, 7, 20] {
        world.params.steps = steps;
        world.params.scale = if steps == 7 { 2.0 } else { 0.6 };
        for buffer in &world.fields {
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vec![[1.0_f32, 0.0, 0.0, 0.0]; 64 * 64]));
        }
        for buffer in &world.fertility {
            gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vec![0.25_f32; 64 * 64]));
        }
        advance(&gpu, &mut world, 300);
        let recovered = read::<f32>(&gpu, &world.fertility[world.current]);
        let expected = 1.0 - 0.75 * (-1.0_f32).exp();
        assert!((recovered[0] - expected).abs() < 0.0005, "{steps} steps: {} != {expected}", recovered[0]);
    }
    world.params.steps = 1;
    world.params.scale = 1.0;
    let mut growth = Vec::new();
    for depletion in [0.0, 1.0] {
        world.params.depletion = depletion;
        for reserve in [0.0_f32, 1.0] {
            for buffer in &world.fields {
                gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vec![[0.8_f32, 0.2, 8.0, 0.0]; 64 * 64]));
            }
            for buffer in &world.fertility {
                gpu.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vec![reserve; 64 * 64]));
            }
            advance(&gpu, &mut world, 1);
            growth.push(read::<[f32; 4]>(&gpu, &world.fields[world.current])[0][1]);
        }
    }
    assert_eq!(growth[0], growth[1], "switching off depletion must bypass even an exhausted habitat's history");
    assert!(growth[2] < growth[3], "exhausted ground must inhibit growth");
}

#[test]
fn fertility_comparison_isolates_the_cycle_and_replays_saved_recipes() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [128, 96], 42);
    world.load_preset(&gpu, 5, 42);
    world.set_comparison(&gpu, true);
    let recipe = world.settings().unwrap();
    advance(&gpu, &mut world, 600);
    let habitat = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
    let soil = read::<f32>(&gpu, &world.fertility[world.current]);
    let reference = world.reference.as_ref().unwrap();
    let control = read::<[f32; 4]>(&gpu, &reference.fields[reference.current]);
    assert_eq!(reference.params.coupling, world.params.coupling);
    assert_eq!(reference.params.depletion, 0.0);
    assert_ne!(habitat, control);
    assert!(soil.iter().any(|f| *f < 0.8));
    assert!(read::<f32>(&gpu, &reference.fertility[reference.current]).iter().all(|f| *f == 1.0));
    let decoded = serde_json::from_str(&serde_json::to_string(&recipe).unwrap()).unwrap();
    world.restore_settings(&gpu, &decoded, 42).unwrap();
    advance(&gpu, &mut world, 600);
    assert_eq!(habitat, read::<[f32; 4]>(&gpu, &world.fields[world.current]));
    assert_eq!(soil, read::<f32>(&gpu, &world.fertility[world.current]));
    world.set_comparison(&gpu, false);
    assert_eq!(soil, read::<f32>(&gpu, &world.fertility[world.current]));
    world.params.depletion = 0.0;
    world.set_comparison(&gpu, true);
    advance(&gpu, &mut world, 80);
    let reference = world.reference.as_ref().unwrap();
    assert_eq!(
        read::<[f32; 4]>(&gpu, &world.fields[world.current]),
        read::<[f32; 4]>(&gpu, &reference.fields[reference.current])
    );
}

#[test]
fn fertility_settings_reject_unsafe_values() {
    for value in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
        let mut params = Params::preset(0).0;
        params.depletion = value;
        assert!(params.validate().is_err());
    }
    for value in [0.0, 4.9, 120.1, f32::NAN, f32::INFINITY] {
        let mut params = Params::preset(0).0;
        params.recovery = value;
        assert!(params.validate().is_err());
    }
}

#[test]
fn comparison_maps_brushes_and_zoom_to_the_same_place_in_both_panes() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    world.set_comparison(&gpu, true);
    for target in [[1280, 800], [801, 600], [320, 900]] {
        let camera = Camera { center: [-0.2, 1.3], zoom: 2.4 };
        let view = ViewXform::fit(world.size, target, &camera);
        for left in [[0.125, 0.3], [0.25, 0.5], [0.375, 0.7]] {
            let right = [left[0] + 0.5, left[1]];
            let a = world.map_position(view, left);
            let b = world.map_position(view, right);
            assert_eq!(a, b, "corresponding points must paint the same cells");
            // Match the cropped viewport used by draw_region.
            let pane = ViewXform {
                scale: [view.scale[0] * 0.5, view.scale[1]],
                offset: [view.offset[0] + view.scale[0] * 0.25, view.offset[1]],
            };
            let rendered = pane.apply([left[0] * 2.0, left[1]]);
            assert!(a.iter().zip(rendered).all(|(a, b)| (a - b).abs() < 1e-6));
            for uv in [left, right] {
                let anchor = world.map_position(view, uv);
                let mut zoomed = Camera { zoom: camera.zoom * 1.5, ..camera };
                let after = world.map_position(ViewXform::fit(world.size, target, &zoomed), uv);
                zoomed.center = [zoomed.center[0] + anchor[0] - after[0], zoomed.center[1] + anchor[1] - after[1]];
                let actual = world.map_position(ViewXform::fit(world.size, target, &zoomed), uv);
                assert!(anchor.iter().zip(actual).all(|(a, b)| (a - b).abs() < 1e-6), "zoom must stay anchored");
            }
        }
    }
    // The mapped brush is sent to both simulations, including while uncoupled.
    world.params.coupling = 0.0;
    let view = ViewXform::fit(world.size, [1280, 800], &Camera::default());
    let frame = Frame {
        gpu: &gpu,
        time: 0.0,
        dt: 1.0 / 60.0,
        frame: 0,
        view,
        target_size: [1280, 800],
        pointer: Some(crate::world::Pointer {
            pos: world.map_position(view, [0.7, 0.4]),
            primary: true,
            secondary: false,
            radius: 7.0,
        }),
    };
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.step(&frame, &mut encoder);
    gpu.queue.submit([encoder.finish()]);
    let reference = world.reference.as_ref().unwrap();
    assert_eq!(
        read::<[f32; 4]>(&gpu, &world.fields[world.current]),
        read::<[f32; 4]>(&gpu, &reference.fields[reference.current])
    );
    world.set_comparison(&gpu, false);
    assert_eq!(world.map_position(view, [0.7, 0.4]), view.apply([0.7, 0.4]));
}

#[test]
#[ignore = "renders paired habitats to target/symbiosis-preview/comparison.png"]
fn render_comparison_preview() {
    let _guard = crate::gpu::test_lock();
    use crate::capture::{self, Readback};
    use crate::post::Post;
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [640, 400], 42);
    world.set_comparison(&gpu, true);
    advance(&gpu, &mut world, 720);
    let size = [1280, 800];
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let post = Post::new(&gpu, size, format);
    let (texture, view) = gpu.texture_2d(
        "comparison preview",
        size,
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let frame = Frame {
        gpu: &gpu,
        time: 12.0,
        dt: 1.0 / 60.0,
        frame: 720,
        view: ViewXform::fit(world.size, size, &Camera::default()),
        target_size: size,
        pointer: None,
    };
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    world.render(&frame, &mut encoder, post.scene_view());
    post.run(&gpu, &mut encoder, &world.post, 12.0, &view);
    let readback = Readback::new(&gpu, size, format);
    readback.copy_from(&mut encoder, &texture);
    gpu.queue.submit([encoder.finish()]);
    capture::save_png(
        std::path::Path::new("target/symbiosis-preview/comparison.png"),
        size,
        readback.read(&gpu).unwrap(),
    )
    .unwrap();
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

#[test]
#[ignore = "renders a two-minute fertility experiment and timelapse frames to target/fertility-preview"]
fn render_fertility_cycle_preview() {
    let _guard = crate::gpu::test_lock();
    use crate::capture::{self, Readback};
    use crate::post::Post;
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [640, 360], 42);
    world.load_preset(&gpu, 5, 42);
    world.set_comparison(&gpu, true);
    let size = [1280, 720];
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let post = Post::new(&gpu, size, format);
    let (texture, target) = gpu.texture_2d(
        "fertility preview",
        size,
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    );
    let readback = Readback::new(&gpu, size, format);
    for n in 0..=600 {
        if n > 0 {
            advance(&gpu, &mut world, 12);
        }
        let frame = Frame {
            gpu: &gpu,
            time: n as f32 / 5.0,
            dt: 1.0 / 60.0,
            frame: n as u64 * 12,
            view: ViewXform::fit(world.size, size, &Camera::default()),
            target_size: size,
            pointer: None,
        };
        for (layer, name) in [(Layer::Together, "living"), (Layer::Fertility, "fertility")] {
            world.params.layer = layer;
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            world.render(&frame, &mut encoder, post.scene_view());
            post.run(&gpu, &mut encoder, &world.post, frame.time, &target);
            readback.copy_from(&mut encoder, &texture);
            gpu.queue.submit([encoder.finish()]);
            let path = format!("target/fertility-preview/{name}/{n:04}.png");
            capture::save_png(std::path::Path::new(&path), size, readback.read(&gpu).unwrap()).unwrap();
        }
        if n % 50 == 0 {
            let soil = read::<f32>(&gpu, &world.fertility[world.current]);
            let field = read::<[f32; 4]>(&gpu, &world.fields[world.current]);
            let coverage = field.iter().filter(|f| f[1] > 0.1).count() as f32 / field.len() as f32;
            let mean = soil.iter().sum::<f32>() / soil.len() as f32;
            println!("frame {}: occupied {coverage:.3}, mean fertility {mean:.3}", n * 12);
        }
    }
    assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
}

#[test]
fn gpu_measurements_match_a_cpu_reduction_of_the_habitat() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [160, 120], 42);
    world.params.steps = 7;
    world.params.depletion = 0.8;
    world.reset(&gpu, 42);
    advance(&gpu, &mut world, 200);
    let cells = (world.size[0] * world.size[1]) as f64;
    assert_eq!(read::<u32>(&gpu, &world.deposits).iter().sum::<u32>(), world.count, "one deposit per agent");

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
    // A busy habitat has growth, moving chemistry, busy routes and some worn ground.
    assert!(totals[0] > 0.01 && totals[2] > 0.0 && totals[4] > 0.01 && totals[4] < 1.0 && totals[6] > 0.0, "{totals:?}");

    // Without agents the agent share is exactly zero, not NaN.
    world.count = 0;
    advance(&gpu, &mut world, 1);
    let totals = measure(&gpu, &world);
    assert_eq!(totals[7], 0.0);
    assert!(totals.iter().all(|v| v.is_finite()));
}

#[test]
fn gpu_comparison_measures_both_panes_in_label_order() {
    use crate::metrics::Sampler;
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [96, 80], 42);
    world.params.coupling = 0.0;
    world.params.reference = Reference::CouplingOff;
    world.set_comparison(&gpu, true);
    let mut sampler = Sampler::new(&gpu, 2);
    let mut sample_after = |world: &mut Symbiosis, frames: u32| {
        advance(&gpu, world, frames);
        let frame = Frame {
            gpu: &gpu,
            time: 0.0,
            dt: 1.0 / 60.0,
            frame: u64::from(frames),
            view: ViewXform::fit(world.size, world.size, &Camera::default()),
            target_size: world.size,
            pointer: None,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        world.step(&frame, &mut encoder);
        let mut sink = sampler.begin(frame.frame, frame.time);
        world.measure(&frame, &mut encoder, &mut sink);
        gpu.queue.submit([encoder.finish()]);
        sampler.map();
        let samples = sampler.flush(&gpu).unwrap();
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
        assert_eq!(samples.len(), 1);
        samples[0]
    };

    // At zero coupling the reference is an identical habitat: both series agree bit for bit.
    let sample = sample_after(&mut world, 30);
    assert_eq!((sample.frame, sample.series), (30, 2));
    assert_eq!(sample.values[0], sample.values[1]);
    assert!(sample.values[0][0] > 0.0, "the seeded habitat has growth");
    assert_eq!(world.comparison_labels().unwrap()[1], "COUPLING OFF");

    // Coupling only changes the left pane, which is series 0.
    let reference_before = sample.values[1];
    world.params.coupling = 0.9;
    let sample = sample_after(&mut world, 60);
    assert_eq!(sample.series, 2);
    assert_ne!(sample.values[0], sample.values[1], "coupling must change the measured habitat");
    let reference = read::<[f32; 4]>(&gpu, &world.reference.as_ref().unwrap().fields[0]);
    assert!(reference.iter().all(|f| f.iter().all(|v| v.is_finite())));
    assert!(reference_before[0] > 0.0);

    // Leaving comparison mode drops the second series.
    world.set_comparison(&gpu, false);
    let sample = sample_after(&mut world, 1);
    assert_eq!(sample.series, 1);
}

#[test]
fn gpu_measurements_are_finite_and_bounded_for_every_preset() {
    let _guard = crate::gpu::test_lock();
    let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
    let mut world = Symbiosis::new(&gpu, [256, 144], 42);
    for (index, name) in PRESETS.iter().enumerate() {
        world.load_preset(&gpu, index, 314159);
        advance(&gpu, &mut world, 60);
        let totals = measure(&gpu, &world);
        for (metric, value) in METRICS.iter().zip(&totals) {
            assert!(value.is_finite(), "{name}: {} is {value}", metric.id);
            if metric.unit == Unit::Fraction {
                assert!((0.0..=1.0).contains(value), "{name}: {} = {value}", metric.id);
            }
        }
        assert!((0.0..=1.0).contains(&totals[1]), "{name}: mean growth {}", totals[1]);
        assert!((0.0..=1.0).contains(&totals[5]), "{name}: mean fertility {}", totals[5]);
        assert!(totals[0] > 0.0 && totals[4] > 0.0, "{name}: growth {} routes {}", totals[0], totals[4]);
    }
}
