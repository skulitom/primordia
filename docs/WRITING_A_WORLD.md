# Writing a world

A *world* is a self-contained GPU simulation. The engine owns the window, the camera,
post-processing, the UI shell, screenshots, recording and headless rendering; a world
only has to simulate and paint itself.

`src/world/placeholder.rs` is the smallest possible world and a good file to copy
when starting. `src/world/reaction_diffusion.rs` (+ `src/shaders/reaction_diffusion.wgsl`)
is a complete, full-featured reference that uses every convention below.

## Frame flow

```
            ┌───────────── once per displayed frame ─────────────┐
 input ──►  world.step(frame, encoder)      compute: advance the simulation
            world.render(frame, encoder, scene)   paint into the HDR scene
            post.run(...)                   bloom → exposure → tonemap → vignette/dither
            egui                            control panel (interactive app only)
            └────────────────────────────────────────────────────┘
```

* `render` runs for every displayed frame, so display-only state such as motion
  trails can accumulate there. `step` is skipped while paused.
* The scene texture is `SCENE_FORMAT` (`Rgba16Float`), linear HDR, at the *output*
  size (`frame.target_size`). Values above about 0.6 start to bloom (soft knee),
  and values above 1.0 glow clearly.
* Headless renders (`primordia render` / `gallery`) call exactly the same methods
  at a fixed 60 fps timestep, so they are a faithful preview of the live app.

## Checklist

1. **Module.** Add `src/world/my_world.rs` exposing
   `pub fn create(gpu: &Gpu, output_size: [u32; 2], seed: u64) -> Box<dyn World>`
   and register it in `WORLDS` in `src/world/mod.rs` (id, name, aliases, tagline).
   `create` chooses the simulation domain size, usually proportional to the output.
2. **Shaders.** Put WGSL in `src/shaders/my_world.wgsl` and compile it with
   `gpu.shader(label, include_str!(...))`. The prelude (`src/shaders/common.wgsl`) is
   prepended automatically and gives you:
   * `vs_fullscreen` / `FullscreenOut`, a fullscreen triangle for display passes
     (`gpu.fullscreen_pipeline` + `gpu::fullscreen_pass`);
   * `ViewXform` + `view_apply(view, screen_uv)`, which map screen uv to world uv;
   * `wrap_i`, `wrap_index`, `torus_delta` for torus arithmetic;
   * `pcg_hash`, `rand1/2/3`, `luminance`, sRGB helpers, `palette_lookup`, `cosine_palette`.
3. **Parameters.** Keep a plain Rust `Params` struct and mirror what the GPU needs into
   `#[repr(C)]` `bytemuck::Pod` uniform structs. Upload them with `gpu.write` at the
   start of `step` / `render`.
4. **Presets.** Store presets as a `const` table; return the names from `presets()`
   (a `OnceLock<Vec<&str>>` works well). `load_preset` sets params, palette and look,
   then reseeds. `reset` reseeds with the current params. `mutate` jumps to a random but
   *reliably interesting* parameter set.
5. **Rendering.** Clear the target and draw a fullscreen pass. Map `in.uv` through
   `frame.view` and **wrap** the result so zoom, pan and zoomed-out tiling work.
   Colour through a `PaletteLut` (`palette_lookup`) or your own scheme. Suggest a look
   per preset via `post_settings()`.
6. **Interaction.** `frame.pointer` (interactive only) gives the brush position in
   world uv, the button state and the radius in domain cells. Document the buttons in
   `controls_hint()`.
7. **UI.** `ui()` is drawn inside the **World** tab of the side panel. Use the shared
   `crate::ui::Slider`, `crate::ui::dropdown` and `crate::palette::combo` controls to keep labels readable
   at narrow panel widths. Every
   slider range must be safe: nothing the user can do should produce NaN, a black
   screen that never recovers, or a crash. Settings that reallocate (e.g. agent counts)
   should apply on reset or behind an explicit button.
8. **Saved settings.** Derive serde serialization for parameters, add a variant to
   `library::WorldSettings`, and implement `settings()` / `restore_settings()`.
   Include all editable state and use stable palette names. Validate allocation
   sizes, indices and parameter ranges before rebuilding GPU state, then reseed
   with the supplied seed. Recipes restart a simulation; they do not contain its
   evolving GPU buffers. Use serde defaults when adding optional fields to keep
   existing saves readable. The library tests exercise every registered preset.

For a custom viewport such as Symbiosis's comparison panes, override
`map_position()` to match the display transform, so painting and cursor-anchored
zoom stay aligned. `comparison_labels()` can supply labels for the app overlay.

## Gotchas

* **Wrap coordinates with `wrap_i` / `wrap_index`, never with your own `%` or
  `floor` division.** GPU float division is not correctly rounded, so naive wraps
  put visible seams on the torus at some domain sizes. `primordia selftest`
  checks the prelude's wrap on your GPU.
* `queue.write_buffer` lands before the *whole* submission, so every dispatch recorded
  in one `step` sees the last value written. For per-sub-step data, use a GPU-side
  counter, a storage buffer of per-step values, or dynamic offsets.
* Uniform layout: avoid `vec3` members, give arrays a 16-byte stride
  (`array<vec4<f32>, N>`), and keep struct sizes a multiple of 16. Big tables are
  easier as storage buffers.
* A dispatch can use at most 65 535 workgroups per dimension. For large 1D dispatches,
  use `gpu::dispatch_linear` and rebuild the index from `@builtin(num_workgroups)`.
* Race-free accumulation (particles depositing into a grid, for example) needs atomics:
  use fixed-point `atomic<u32>` and resolve into floats in a separate pass.

## Testing without a window

```bash
cargo run -- render -w my-world -p 1 --frames 600 --width 1280 --height 720 --out renders/a.png
cargo run -- render -w my-world -p 1 --zoom 0.5 --out renders/tiling.png   # torus seams
cargo run -- render -w my-world -p 1 --zoom 4 --center 0.3,0.6              # detail
cargo run -- render -w my-world --frames 240 --every 60 --frames-dir renders/seq
cargo run -- gallery -w my-world --out-dir renders/gallery                  # every preset
```

Headless renders print frames per second, which makes a quick performance check
(try `--width 1920 --height 1080`).

Run `cargo test --locked -- --include-ignored` for the GPU tests and visual
previews, then inspect the PNGs under `target/`. Acquire `gpu::test_lock()` before
creating a GPU in a test, and hold the guard until the GPU is dropped. This keeps
concurrent tests from racing through native driver initialization.
