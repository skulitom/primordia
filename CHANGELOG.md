# Changelog

Notable changes to Primordia, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Before 1.0, a minor version may change the command line and the files Primordia writes; recipes saved by an earlier
version keep loading.

## [0.2.0] - 2026-09-23

Downloads for Windows, Linux and macOS, a command line that scripts and AI agents can drive, recipes that render and
edit without the app, and fixes for the v0.1.0 Windows exe.

### Added

- Release archives for Windows (x86-64), Linux (x86-64, glibc 2.35 or newer) and macOS (universal), with
  `SHA256SUMS.txt`.
- `primordia list [--world W] [--json]` describes every world without opening a GPU: ids, aliases, presets,
  measurements with their units, ranges and meanings, and palettes.
- A global `--json` flag: stdout carries one result object, `{"ok":true,...}` or `{"ok":false,"error":{...}}`, with
  seeds as strings. Logs and progress go to stderr; `-q` keeps warnings and errors, `-v` adds debug lines.
- Exit status by kind of failure: 1 another failure, 2 invalid input, 3 no usable GPU or a GPU failure, 4 ffmpeg
  missing or failed. Names, sizes and output paths are checked before the GPU starts.
- "Did you mean" suggestions for world, preset and measurement names; an ambiguous prefix lists its candidates.
- `render --recipe FILE` renders a library or explore recipe, or a PNG that Primordia wrote, at its own size, seed,
  look and camera; the options override any of them.
- `--set KEY=VALUE` changes any setting of a preset or recipe in `render`, `explore` and the new `recipe` command,
  which prints a complete recipe. `render --save-recipe` writes the recipe it rendered.
- Every PNG from `render`, `gallery` and `explore` carries its recipe, how many frames the world ran and at what
  rate, the GPU, and a command that renders it again. `primordia render --recipe picture.png` reproduces it with no
  other options.
- `explore --recipe` and `--set` start a search from a saved recipe.
- Captions on every tile of the gallery and explore contact sheets.
- `WGPU_ADAPTER_NAME` chooses the GPU by name; a name that matches nothing lists every adapter.
- **Tools → Simulation resolution** (25 to 100%). Integrated, virtual and software GPUs start at half resolution.
- An **About** section in the Tools tab with the version, the GPU, links and command lines ready to copy. An empty
  Library explains how to fill it.
- The window and the Windows exe carry the Primordia icon, and the exe its version information.
- A short report for internal failures (exit status 3) that names the backend and the adapter and says where to
  report it.
- For contributors: `PRIMORDIA_GPU_TESTS=skip` runs the tests without a GPU; CI runs clippy on Rust 1.86 and stable
  and the GPU tests on lavapipe; CONTRIBUTING.md, issue forms, a pull request template and CITATION.cff.

### Changed

- Each explore run writes into a folder of its own, `explore/<world>-<preset>-s<seed>` by default, with a `run.json`
  that records the command, version, GPU, settings, the meaning of every CSV column and the files written. Running
  again into the same folder replaces exactly those files; `--overwrite` also removes an older build's outputs.
- Explore's refinement changes behaviour only: colours, palettes and the rest of the look are no longer perturbed.
  `--select max:<id>` and `min:<id>` skip inert candidates unless `--keep-inert` is given.
- `candidates.csv` numbers presets from 1, as `--preset` does, and adds a `preset_name` column.
- Seeds that Primordia draws stay below 2^53, so JSON tools keep them exact, and recipes also accept seeds written as
  strings. The same `explore --seed` therefore evaluates different candidates than in 0.1.0.
- Screenshots, recordings and measurement logs have the seed in their names: `<world>_<preset>_s<seed>_<time>`.
- Sizes outside 16 to 16384 pixels, zero frames or counts, and zooms outside 0.05 to 256 are refused instead of
  clamped.
- A double-clicked Windows exe closes its console window and shows errors in a message box.
- On DirectX 12, Primordia says when it is compiling shaders, which can take minutes on the first start.
- Checkboxes are squares, filled with the accent colour when checked.
- Lenia mutations log their kernel tables only at debug level.

### Fixed

- The Windows exe runs without the Visual C++ redistributable. The v0.1.0 exe stopped with "VCRUNTIME140.dll was
  not found" on PCs that lack it.
- On DirectX 12, Reaction-Diffusion no longer crashes within a second of starting, nor Lenia after minutes of
  shader compilation.
- A `WGPU_BACKEND` value that wgpu does not know is reported, instead of looking like a missing GPU.

## [0.1.0] - 2026-09-20

The first release: five worlds (Physarum, Particle Life, Lenia, Reaction-Diffusion and Symbiosis), 42 presets and
Mutate, live measurements, novelty search with `primordia explore`, a library of saved worlds, tour mode,
screenshots, MP4 recording and headless rendering. It shipped as a Windows exe.

[0.2.0]: https://github.com/skulitom/primordia/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/skulitom/primordia/releases/tag/v0.1.0
