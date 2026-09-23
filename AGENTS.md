# Agent guide

Primordia is a GPU artificial-life laboratory: one Rust binary, `primordia`, with five simulations ("worlds") that run
as WGSL compute shaders through wgpu, an egui control panel, and headless commands that render, measure and explore
them. This file is for coding agents working on the repository. To use the command line instead, read the
[primordia skill](.claude/skills/primordia/SKILL.md) and the generated [command-line reference](docs/CLI.md).

## Layout

| Path | What is there |
|---|---|
| `src/main.rs` | The command line (clap) with its help texts, and the tests that generate the docs |
| `src/app.rs`, `src/ui.rs` | The interactive window: winit, egui, capture and the library tab |
| `src/headless.rs`, `src/explore.rs` | `render` and `gallery`; `explore`, the novelty search |
| `src/recipe.rs`, `src/library.rs`, `src/capture.rs` | Recipes and `--set`, saved worlds, PNGs with their provenance |
| `src/metrics.rs` | GPU measurements: the reduction, the asynchronous readback ring, CSV |
| `src/gpu.rs`, `src/post.rs`, `src/palette.rs` | The wgpu layer, bloom and tonemapping, OKLab palettes |
| `src/world/` | One file per world with its `<world>_tests.rs`; `mod.rs` holds the `World` trait, the world conventions and the registry of ids, aliases, presets, measurements and palettes |
| `src/shaders/` | WGSL; `common.wgsl` is the prelude every module gets |
| `docs/` | [Writing a world](docs/WRITING_A_WORLD.md), the CLI reference, one page per world, images |

## Checks

```bash
cargo test --locked                                  # about half the tests need a GPU adapter, never a window
PRIMORDIA_GPU_TESTS=skip cargo test --locked         # without a GPU: the GPU tests pass as skipped
cargo clippy --all-targets --locked -- -D warnings   # clean on Rust 1.86 (the MSRV) and on current stable
```

- The first build compiles wgpu, naga and egui with optimisations, even for tests, and takes several minutes.
- GPU tests hold a lock and run one at a time. Name a slow test that loops over every preset or world with
  `every_preset`, `every_world` or `full_coupling`: CI's quick set skips those and a weekly run includes them.
- `docs/CLI.md` and `.claude/skills/primordia/references/measurements.md` are generated. After changing any help text
  or measurement, run `PRIMORDIA_BLESS=1 cargo test --locked generated_` and commit the result. Another test checks
  that the skill names only registered worlds, presets and measurements.
- `primordia list` needs no GPU and the headless commands need no window. A bare `primordia` opens a window and blocks
  until it is closed, so never run it unattended without `--exit-after SECS`.

## Rules

- Do not run `cargo fmt`. The code is formatted by hand to about 120 columns; match the surrounding layout, naming and
  comment density.
- British English in prose, UI strings and comments: colour, behaviour, normalise, mould.
- Every dependency must build on Rust 1.86; that is why wgpu, winit and egui are pinned.
- Saved recipes must keep loading. Never rename or remove a serialised field, give new ones a serde default, and keep
  `SavedWorld.version` at 1.
- World code follows the conventions at the top of `src/world/mod.rs` and the checklist in
  [docs/WRITING_A_WORLD.md](docs/WRITING_A_WORLD.md). Among them: wrap coordinates with `wrap_i`, never let a sampler
  share its binding with another resource in the same module, normalise measurements in the kernel and cross-check
  each one against a CPU recomputation, and never block a frame on a readback.
- Commit messages have an imperative subject in sentence case with no type prefix, and a body of short paragraphs
  saying what changed and why. User-visible changes also get a line in [CHANGELOG.md](CHANGELOG.md).

[CONTRIBUTING.md](CONTRIBUTING.md) has the details, including how to report a GPU problem.
