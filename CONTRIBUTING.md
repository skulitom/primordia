# Contributing to Primordia

Thanks for your interest. Bug reports, shared worlds, fixes, new presets and whole new worlds are all welcome. This
page covers building and testing, the house style, commit messages and how to report a GPU problem well.

## Building

You need **Rust 1.86 or newer** (1.86 is the minimum supported version, the MSRV) and a GPU with Vulkan, DirectX 12
or Metal. [ffmpeg](https://ffmpeg.org) is optional and only used for video.

```bash
git clone https://github.com/skulitom/primordia
cd primordia
cargo run --release
```

`wgpu`, `winit` and `egui` are pinned to the newest releases that still build on Rust 1.86. A new dependency must
build on 1.86 too; prefer crates that are already in `Cargo.lock`.

## Checks

Run these before you open a pull request. CI runs the tests and clippy (see below), but only your machine runs the
GPU tests on a real GPU.

```bash
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo +1.86 clippy --all-targets --locked -- -D warnings   # if your default toolchain is newer
cargo build --release --locked
cargo run --release --locked -- selftest
```

Clippy has to be clean both on 1.86 and on current stable, since newer compilers add lints;
`rustup toolchain install 1.86` gets the older one.

`docs/CLI.md` and `.claude/skills/primordia/references/measurements.md` are generated from the help texts and the
world registry, and tests fail when they are out of date. After changing either, run
`PRIMORDIA_BLESS=1 cargo test --locked generated_` and commit the files it rewrites.

### GPU tests

About half of the tests need a GPU adapter, though never a window. They hold a lock for as long as they have a
device, so they run one at a time: some Windows Vulkan drivers crash when instances are created or dropped
concurrently. A GPU test gets its GPU, and the lock, with
`let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };`.

Without a usable adapter those tests fail, so a broken driver cannot pass unnoticed. On a machine without a GPU, set
`PRIMORDIA_GPU_TESTS=skip`: the GPU tests then print a note and pass without running, and the CPU-side tests run as
usual.

```bash
PRIMORDIA_GPU_TESTS=skip cargo test --locked          # bash
$env:PRIMORDIA_GPU_TESTS = "skip"; cargo test --locked  # PowerShell
```

`cargo test --locked -- --include-ignored` also renders PNG previews into `target/` for inspection by eye.

CI (`.github/workflows/ci.yml`) runs clippy on 1.86 and on stable, and runs the tests on Linux on lavapipe, Mesa's
software Vulkan driver, with `WGPU_BACKEND=vulkan`. Pushes and pull requests skip the slow tests that loop over every
preset or every world (`--skip every_preset --skip every_world --skip full_coupling`); a weekly run includes them.
Name exhaustive GPU tests with `every_preset`, `every_world` or `full_coupling` so that the quick set recognises
them. This includes tests that replay exploration recipes or check invalid saves across all worlds: a small output
image can still trigger a large Lenia nursery. Both Linux suites run one test at a time and stream output, so the
logs identify the active test instead of reporting other tests waiting for the GPU lock as slow.
Windows and macOS runners have no usable GPU, so they build everything and run the tests with
`PRIMORDIA_GPU_TESTS=skip`.

## Style

- **Do not run `cargo fmt`.** The code is formatted by hand to about 120 columns, and reformatting would bury your
  change in a large diff. Match the surrounding code: its layout, naming, idioms and how much it comments.
- British English in prose, UI strings and comments: colour, behaviour, normalise, mould.
- Saved recipes (Library saves, explore recipes and `tests/fixtures/reaction-diffusion.json`) must keep loading.
  Never rename or remove a serialised field; give new ones a serde default. `SavedWorld.version` stays 1.
- Add tests for new CPU-side logic (parsing, name resolution, JSON shapes, file handling) in the style of the existing
  `#[cfg(test)]` modules.

## Writing a world

Copy `src/world/placeholder.rs`, the smallest possible world, register it in `src/world/mod.rs` and follow
[docs/WRITING_A_WORLD.md](docs/WRITING_A_WORLD.md). It explains the frame flow, the checklist every world goes
through, the usual WGSL gotchas and how to test a world without a window.

## Commit messages

- A subject in the imperative and in sentence case, at most about 70 characters, with no type prefix such as
  `feat:`. For example: `Add a recovering fertility cycle to Symbiosis`.
- A body of short prose paragraphs that say what changed and why, when the subject alone does not.
- Split separable changes into separate commits.

## Reporting a GPU problem

Most problems depend on the GPU, driver and backend, so please use the **Bug report** form and include:

1. `primordia --version` (or the commit, if you built it yourself) and the exact command you ran.
2. The line Primordia logs when it starts, such as `[INFO ] GPU: NVIDIA GeForce RTX 4090 (Vulkan, DiscreteGpu)`.
3. The full output of `primordia selftest`, which checks on your GPU the shader maths every world relies on.
4. Your operating system and its version.

It also helps to know whether another backend behaves differently. Set `WGPU_BACKEND` to `vulkan`, `dx12`, `metal` or
`gl` before starting Primordia (`$env:WGPU_BACKEND = "dx12"` in PowerShell). On a laptop with two GPUs,
`WGPU_POWER_PREF=low` or `high` picks the integrated or the discrete one. `RUST_LOG=warn,primordia=debug` makes the
log more detailed while keeping wgpu's warnings.

## Sharing a world

Found something worth keeping? Open a **Share a world** issue with a screenshot and the recipe JSON, from the Library
folder or from `primordia explore`, so others can load it too.

## Releases

Maintainers describe the version in `CHANGELOG.md`, set it in `Cargo.toml` and `CITATION.cff` (with the date), commit,
and push a tag such as `v0.2.0`. `.github/workflows/release.yml` then builds Windows, Linux and macOS archives and
attaches them, with `SHA256SUMS.txt`, to a draft release to review and publish by hand. The workflow stops if the tag
and `Cargo.toml` disagree.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
