// A shared habitat: x = nutrient U, y = activator V, z = slime trail.
// Agents read yesterday's habitat; integer deposits are resolved in a separate
// pass. Every chemical/trail sub-step then reads a complete, immutable field.
struct Sim {
    size: vec2<u32>, count: u32, steps: u32,
    chemistry: vec4<f32>, // feed, kill, coupling, dt
    motion: vec4<f32>, // sensor distance, sensor angle, turn angle, speed
    trail: vec4<f32>, // retention, deposit, wander, diffusion (per sub-step)
    pointer: vec2<f32>, radius: f32, pointer_mode: u32,
};
struct Agent { pos: vec2<f32>, heading: f32, state: u32 };
@group(0) @binding(0) var<uniform> sim: Sim;
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(4) var<storage, read_write> deposits: array<atomic<u32>>;

fn cell(p: vec2<i32>) -> vec4<f32> { return src[wrap_index(p, sim.size)]; }
fn sample_field(p: vec2<f32>) -> vec4<f32> {
    let q = vec2<i32>(floor(p));
    let t = fract(p);
    return mix(mix(cell(q), cell(q + vec2<i32>(1, 0)), t.x),
               mix(cell(q + vec2<i32>(0, 1)), cell(q + vec2<i32>(1, 1)), t.x), t.y);
}
fn sense(p: vec2<f32>, a: f32) -> f32 {
    let f = sample_field(p + vec2<f32>(cos(a), sin(a)) * sim.motion.x);
    let trail = f.z / (0.35 + f.z);
    // Prefer the growing margin to either empty ground or a saturated centre.
    let margin = exp(-100.0 * (f.y - 0.14) * (f.y - 0.14));
    return trail + sim.chemistry.z * 2.5 * margin;
}

@compute @workgroup_size(256)
fn cs_agents(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= sim.count { return; }
    var a = agents[i];
    a.state += 0x9e3779b9u;
    let noise = rand1(a.state) * 2.0 - 1.0;
    let forward = sense(a.pos, a.heading);
    let left = sense(a.pos, a.heading - sim.motion.y);
    let right = sense(a.pos, a.heading + sim.motion.y);
    if forward < left && forward < right {
        a.heading += select(-sim.motion.z, sim.motion.z, noise > 0.0);
    } else if left > right && left > forward {
        a.heading -= sim.motion.z;
    } else if right > left && right > forward {
        a.heading += sim.motion.z;
    }
    a.heading += noise * sim.trail.z;
    a.heading = a.heading - TAU * floor(a.heading / TAU);
    a.pos += vec2<f32>(cos(a.heading), sin(a.heading)) * sim.motion.w;
    let size = vec2<f32>(sim.size);
    a.pos = fract(a.pos / size) * size;
    agents[i] = a;
    atomicAdd(&deposits[wrap_index(vec2<i32>(floor(a.pos)), sim.size)], 1u);
}

@compute @workgroup_size(16, 16)
fn cs_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid.xy >= sim.size) { return; }
    let p = vec2<i32>(gid.xy);
    let c = cell(p);
    let cardinal = cell(p + vec2<i32>(1, 0)) + cell(p + vec2<i32>(-1, 0))
                 + cell(p + vec2<i32>(0, 1)) + cell(p + vec2<i32>(0, -1));
    let diagonal = cell(p + vec2<i32>(1, 1)) + cell(p + vec2<i32>(-1, 1))
                 + cell(p + vec2<i32>(1, -1)) + cell(p + vec2<i32>(-1, -1));
    let lap = cardinal * 0.2 + diagonal * 0.05 - c;
    let support = sim.chemistry.z * c.z / (1.0 + c.z);
    let feed = sim.chemistry.x;
    let kill = sim.chemistry.y - 0.004 * support;
    // Trails reduce local loss and catalyse the existing growth front.
    let reaction = c.x * c.y * c.y + 0.0005 * support * c.x * clamp(c.y * 8.0, 0.0, 1.0);
    let du = lap.x - reaction + feed * (1.0 - c.x);
    let dv = lap.y * 0.5 + reaction - (feed + kill) * c.y;
    let i = gid.y * sim.size.x + gid.x;
    let deposit = f32(atomicLoad(&deposits[i])) * sim.trail.y;
    var next = vec4<f32>(clamp(c.xy + vec2<f32>(du, dv) * sim.chemistry.w, vec2<f32>(0.0), vec2<f32>(1.0)),
                        clamp((c.z + lap.z * sim.trail.w) * sim.trail.x + deposit, 0.0, 32.0), abs(dv));
    if sim.pointer_mode != 0u {
        let d = length(torus_delta(vec2<f32>(gid.xy), sim.pointer, vec2<f32>(sim.size)));
        let brush = (1.0 - smoothstep(sim.radius * 0.5, sim.radius, d)) * 0.3;
        if sim.pointer_mode == 1u {
            next.x = mix(next.x, 0.5, brush);
            next.y = mix(next.y, 0.28, brush);
        } else {
            next = mix(next, vec4<f32>(1.0, 0.0, 0.0, 0.0), brush);
        }
    }
    dst[i] = next;
}
