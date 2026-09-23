---
name: primordia
description: Render, measure and explore Primordia's GPU artificial-life simulations (Physarum slime mould or slime mold, Particle Life, Lenia, Gray-Scott reaction-diffusion and Symbiosis) from the command line. Use when asked for stills, frame sequences, videos or preset galleries of these worlds, for per-frame measurements as CSV, for a novelty search over parameter mutations (primordia explore) that returns images, a contact sheet and loadable JSON recipes, or to render, edit or inspect a recipe or a PNG that Primordia wrote.
compatibility: Needs the primordia binary (built with Rust 1.86 or newer) and a GPU with Vulkan, DirectX 12 or Metal; ffmpeg only for videos.
---

# Primordia on the command line

Primordia is one binary holding five GPU simulations, called worlds. With a command it runs headless and exits:
`list` (needs no GPU), `render`, `gallery`, `explore`, `recipe` and `selftest`. Without a command it opens the
interactive window. Every option and default is in [docs/CLI.md](../../../docs/CLI.md), which is generated from
`primordia <command> --help`.

## Getting the binary

- Check what is installed: `primordia --version`, then `primordia list --json`. If `list --json` is rejected, the
  binary predates this skill (the v0.1.0 download does); build one from the source.
- In a clone, `cargo build --release --locked` (a few minutes the first time) gives `target/release/primordia`, or
  `target\release\primordia.exe` on Windows. `cargo install --path . --locked` puts it on the PATH, and
  `cargo run --release --locked -- <args>` builds and runs in one step. Without a clone:
  `cargo install --git https://github.com/skulitom/primordia --locked`.
- From v0.2.0 on, the releases (https://github.com/skulitom/primordia/releases) have archives for Windows, Linux and
  macOS; check a downloaded binary the same way.
- `primordia selftest` checks the GPU. When one opens, its first line is `PASS: ...` or `FAIL: ...` and it ends with
  the adapters it can see, `*` marking the one in use. Without a GPU, stdout stays empty and stderr says why. Either
  failure exits with 3.
- The commands below say `primordia`; use the path you have, such as `./target/release/primordia`. No display is
  needed, only a GPU; on Linux without one, Mesa's lavapipe (software Vulkan) works, slowly.

## Ground rules

- **Never run `primordia` without a command** unless the user wants the window: it opens one and blocks until someone
  closes it. To show the user a world, name it (`primordia --world lenia --preset necklaces`); in a script or a
  smoke test always add `--exit-after 5`.
- **Start small, then scale.** The defaults are 1920x1080, 600 frames and a 240 fps cap. Probe with
  `--width 320 --height 180 --frames 120 --max-fps 0`, about a second per command on a desktop GPU (most of it
  opening the GPU), and render the deliverable at full size last.
- **Determinism is per GPU.** The same world and preset (or recipe), seed, size, `--frames` and `--fps` give a
  bit-identical image again on the same GPU, driver and backend. On another GPU the chaotic worlds (Particle Life,
  Physarum and Symbiosis) diverge within a few hundred frames: the same kind of pattern, not the same pixels. Always
  report the seed. Compare runs from different machines statistically, through
  their measurements over several seeds, never through image hashes.
- **stdout is the result, stderr the log.** Without `--json`, stdout lists every file written, one path per line,
  as given (backslashes on Windows). With the global `--json` (before or after the command), stdout is exactly one
  JSON object: `{"ok": true, "command": ..., ...}` on success (`list --json` prints the catalogue itself instead),
  `{"ok": false, "command": ..., "error": {"kind": "usage|gpu|ffmpeg|io|other", "message": ..., "exit_code": ...}}`
  on failure. A name that did not resolve adds `what`, `query`, `available` and a `suggestion` or the ambiguous
  `candidates` to `error`. Seeds in JSON are strings. Progress goes to stderr; `-q` keeps only warnings and errors,
  `-v` adds debug lines. `selftest` and the window take no `--json`.
- **Exit status:** 0 success; 1 another failure, such as reading or writing a file; 2 invalid input: a bad option,
  an unknown world, preset, measurement or setting, a recipe that cannot be read or that the world rejects, a size out
  of range or an output path that cannot be written; 3 no usable GPU, or the GPU failed; 4 ffmpeg is missing or
  failed. Names, sizes, paths and ffmpeg are checked before the GPU opens (in about 0.1 s), and so are `--set` edits
  of a recipe; `--set` edits of a preset are checked once its world exists.
- **Names.** A world is an id, a number 1-5, an alias or a unique prefix of an id (`rd`, `5`, `lenia`). A preset is a
  name (case and punctuation are ignored), a unique prefix or a number from 1 (`-p "Pearl Reef"`, `-p pearl`,
  `-p 4`). Measurement ids must be exact. A typo fails at once with exit 2 and a "did you mean".
- **Where files go.** Paths are relative to the working directory, and the defaults (`renders/`, `gallery/`,
  `explore/`, `frames/`) are created there, so work in a scratch folder. `explore --install` writes into the user's
  real library: only do that when asked, or set `PRIMORDIA_LIBRARY_DIR` to a scratch folder first.

## Worlds and presets

| # | World | Aliases | Presets (numbered from 1) |
|---|---|---|---|
| 1 | `physarum` | slime, slime-mold, slime-mould, mold, mould | Dendrites, Neural Lace, Mycelium, Rival Colonies, Symbiosis, Honeycomb, Synapses, Currents, Chasing Waves, Galaxy |
| 2 | `particle-life` | particles, particlelife, pl, life | Tidepool, Living Cells, Serpents, Rotating Suns, Lace, Marbling, Predator & Prey, Plankton, Necklaces |
| 3 | `lenia` | smoothlife, continuous-ca | Orbium, Leviathans, Menagerie, Pearl Reef, Necklaces, Hydrogeminium, Tessellatium |
| 4 | `reaction-diffusion` | rd, gray-scott, grayscott, turing, coral | Coral Reef, Mitosis, Fingerprints, Worms, Solitons, Crescent Gliders, Spiral Waves, Bubbles, Spots & Stripes, Morphology Atlas |
| 5 | `symbiosis` | coupled, hybrid, ecosystem | Living Reef, Wandering Veins, Coral Maze, Spore Tide, Root Atlas, Fallow Gardens |

Each world's first preset is its default. "Symbiosis" is also Physarum's preset 5, while `--world symbiosis` is the
fifth world. The alias `coral` means Reaction-Diffusion; Coral Maze is a Symbiosis preset. Prefixes match ids only:
`-w l` is Lenia, and `-w p` is ambiguous.

## Discovery: `primordia list --json`

`list` needs no GPU and answers in milliseconds, so start there rather than guessing names. `list --world rd` prints
the same catalogue for people.

```text
{"schema": 1, "version": "0.2.0", "recipe_version": 1, "library_dir": "...", "tonemaps": ["agx", ...],
 "worlds": [{"index": 1, "id": "physarum", "name": "Physarum", "aliases": [...], "tagline": "...",
             "presets": [{"index": 1, "name": "Dendrites", "slug": "dendrites"}, ...],
             "metrics": [{"id": "ground", "label": "...", "unit": "fraction", "range": [0.0, 1.0],
                          "hint": "...", "vital": true}, ...],
             "palettes": ["Ember Gold", ...], "palette_setting": "palette"}, ...]}
```

`palette_setting` names the recipe setting that picks a palette: `palette` (one name), Lenia's `palettes` (one
name per channel) or Particle Life's `params.colors` (a colour scheme by 0-based index).

## Render

```bash
primordia render -w lenia -p "Pearl Reef" --seed 7 --width 1280 --height 720 --frames 600 -o lenia.png
primordia render -w rd -p mitosis --width 320 --height 180 --frames 120 --max-fps 0 --json
```

- `-f/--frames N` frames are simulated at a time step of 1/`--fps`, so N/fps simulated seconds. The defaults are 600
  frames at 60 fps, or what a PNG given to `--recipe` records.
- `-o/--out FILE.png`; the default is `renders/<world>-<preset slug>-s<seed>.png`.
- `--video clip.mp4` encodes every frame (also .mov, .mkv, .webm, .gif; needs ffmpeg; odd sizes round down to
  even). With `--video` and no `-o`, no PNG is written.
- `--every N --frames-dir DIR` saves `DIR/<world>_NNNNN.png` every N frames (DIR defaults to `frames`).
- Camera: `--zoom Z` from 0.05 to 256 (below 1 shows the torus tiling, above 1 is a close-up) and `--center=X,Y` in
  world uv (0-1; write the `=` so negative values parse).
- Look: `--exposure`, `--bloom` (0 turns it off), `--bloom-threshold`, `--tonemap agx|aces|reinhard|linear`.
- Scripted mouse: `--brush primary|secondary [--brush-at=X,Y] [--brush-radius R]` holds a button all run long.
- `--max-fps 0` lifts the 240 fps cap; leave the cap on for long unattended batches.
- `--json` result: `world`, `preset` `{index, name, slug}`, `seed`, `size`, `frames`, `fps`, `source`, `set`,
  `modified`, `files` `{png, video, frames, metrics, recipe}`, `metrics` (see below, or null) and `secs`.

Every PNG that `render` (the final image and `--every` frames), `gallery` and `explore` write carries text chunks:
`Software`, `Source`, `Title`, `Comment` (a command that renders the image again, run from the PNG's folder),
`primordia:gpu` (the adapter and backend), `primordia:frames` and `primordia:fps` (how long the world ran, and at
what rate) and `primordia:recipe` (the recipe, which `--recipe` reads back). Contact sheets and the app's screenshots
carry only `Software` and `Source`, so `--recipe` refuses them with exit 2. Read the chunks with Pillow:
`python -c "from PIL import Image; print(Image.open('lenia.png').text)"`.

## Gallery

```bash
primordia gallery -w lenia --width 640 --height 360 --frames 300 --max-fps 0 -o gallery
```

It renders the final frame of every preset (of every world without `-w`, 42 images) to
`<out-dir>/<world>-NN-<preset slug>.png`, plus `contact-sheet.png` with each tile captioned (skip it with
`--no-sheet`). Defaults: `gallery/`, 1280x720, 600 frames, seed 1 for every preset. The `--json` result lists
`images` (`{world, preset, path}`) and `sheet`.

## Measurements

Every world computes a few numbers about its state on the GPU each frame. `render --metrics run.csv` logs them:

```bash
primordia render -w rd -p mitosis --width 640 --height 360 --frames 600 --max-fps 0 --metrics mitosis.csv --json
```

- The CSV header is `frame,time,series,` then the world's measurement ids in the order below. There is one row
  per frame and series; `frame` counts from 0, `time` is simulated seconds (frame/fps), values have six decimals.
- `series` 0 is the habitat. A Symbiosis comparison (`--set params.compare=true`) adds series 1, the reference
  habitat, and the `--json` result names both in `metrics.series` (such as `["CYCLE 0.80", "CYCLE OFF"]`).
- A fraction is a share of cells or agents from 0 to 1; a scalar is a plain number (a mean, a ratio or a signed
  rate). Vital measurements (`*`) are the ones that fall to zero when a world is dead, empty or frozen.
- With `--json`, `metrics` is `{rows, series, last}`, and `last` holds the final value of every measurement, one
  object per series, so a probe needs no CSV parsing.
- Measurements cost nothing unless asked for: `render` computes them only with `--metrics`, `explore` always.

| World | Measurements, in CSV column order (* = vital) |
|---|---|
| `physarum` | `ground`*, `veins`, `concentration`, `travelled`, `trail_mass`, `reinforced`, `on_vein`, `agent_trail` |
| `particle-life` | `speed`*, `hot`, `slow`, `crowding`, `dense`, `segregation`, `void` |
| `lenia` | `mass`*, `mass_1`, `mass_2`, `mass_3`, `occupied`, `active`*, `growth`, `dense` |
| `reaction-diffusion` | `alive`*, `body`, `v_mean`, `u_mean`, `active`*, `v_drift`, `edge`, `filled` |
| `symbiosis` | `growth_cover`*, `growth_mean`, `growth_active`*, `growth_drift`, `routes`, `fertility_mean`, `exhausted`, `agents_on_growth` |

The unit, label and definition of each are in [references/measurements.md](references/measurements.md), generated
from the same registry that `primordia list` prints.

## Explore

`explore` turns mutation into a search. It evaluates the base (a preset, or `--recipe`) and `--runs` mutations,
then `--refine` rounds of `--children` perturbed copies of the kept recipes. It measures every candidate and keeps
`--keep` of them: the most mutually different (`--select novelty`, the default), or the largest or smallest late
mean of one measurement (`--select max:<id>` or `min:<id>`).

```bash
primordia explore -w symbiosis --select max:growth_cover --runs 8 --refine 0 --keep 4 --frames 300 --max-fps 0
primordia explore -w rd -p mitosis --select min:edge --runs 24 --keep 6 --max-fps 0 --json
```

- Defaults: `--runs 48 --refine 1 --children 24 --strength 0.15 --keep 12 --frames 600`, 640x360, `--seed 1`:
  73 candidates, a minute or two with `--max-fps 0` on a fast GPU, several at the cap, and longer for Lenia.
- Quick probe: `--runs 8 --refine 0 --keep 4 --frames 300 --max-fps 0` evaluates 9 candidates in a few seconds
  (about 20 s for Lenia, whose mutations hatch creatures first). Use it before committing to a full search.
- Explore always steps at 1/60 s. A candidate is inert when any vital measurement's mean over the last 40% of its
  frames is below 0.001; inert and failed candidates are never kept unless `--keep-inert` (failed ones never).
- Refinement multiplies floats, and integers above 32, by log-normal noise (`--strength`, each with probability one
  half) and moves smaller integers by one. Integers inside lists, switches and text stay as they are, and so do
  colours, palettes and the rest of the look, so children keep their parent's look. Children reuse their parent's
  seed.
- `--seed` is the master seed: every candidate seed follows from it, so the same command repeats the same search on
  the same GPU (v0.1.0 drew other candidates from the same seed). The base runs at `--seed`, or at a recipe's own seed.
- `--recipe FILE` and `--set KEY=VALUE` start the search from a recipe or an edited preset instead. Every candidate,
  the recipe included, runs at `--width` x `--height` (640x360 by default), not at the recipe's own size.
- When every candidate is inert or failed, explore exits with 1 and says that `--keep-inert` admits dead and frozen
  ones.

Each run writes into its own folder, `explore/<world>-<preset slug>-s<seed>` (or `-<recipe file stem>-` with
`--recipe`; `-o/--out-dir` names another):

- `NN-seedS.png`: the kept candidate of rank NN (the rank, not the seed, identifies it).
- `recipes/NN-seedS.json`: its recipe, with a `provenance` object saying how explore found it.
- `candidates.csv`: one row per candidate: `index, round, origin` (preset, recipe, mutation or child), `parent,
  seed, preset` (from 1), `preset_name, rank` (empty unless kept), `novelty, status` (ok, inert or failed), `secs`,
  then `<id>_mean` (last 40% of the frames), `<id>_std` and `<id>_drift` (that mean minus the first 20%'s) per
  measurement.
- `contact-sheet.png` captioned "#NN · seed S" (not with `--no-sheet`), and `all/NNN-rR-seedS.png` with `--all`.
- `run.json`, written last: `tool, schema, version, complete, error, argv, started, secs, gpu, world, base,
  settings`, `csv` (the meaning of every column), `files` (everything written) and, when complete, `evaluated,
  inert, failed, kept`.

A second run into the same folder first deletes exactly the files its `run.json` lists and leaves everything else.
Explore-looking files that no `run.json` lists make it refuse the folder with exit 2 unless `--overwrite`. The
`--json` result has `kept` (`rank, candidate, round, origin, parent, seed, preset, novelty, image, recipe,
installed`), `files` (`out_dir, csv, sheet, manifest, all`), `evaluated`, `inert` and `library`.

Kept recipes remember the size they were explored at (their name says it, such as "(seed S, 640x360)"). Rendering
one larger with `--width`/`--height` runs the same rules on a larger domain, which is a different run; explore at
the final size when the exact behaviour matters. `--install` also copies the kept recipes into the app's library.

## Recipes

A recipe is the JSON that the app's library, explore, `render --save-recipe` and `primordia recipe` write, and that
every PNG from `render`, `gallery` and `explore` embeds: `{"version": 1, "name", "seed", "output_size": [w, h], "preset"` (from 0), `"modified",
"settings": {"world": <id>, "params": {...}, ..., "post": {...}}, "look": {...}, "camera": {"center": [x, y],
"zoom"}}`. Explore's recipes add `provenance`.

```bash
primordia recipe -w rd -p mitosis
primordia recipe -w rd -p mitosis --set params.feed=0.031 -o mito.json
primordia render --recipe mito.json --frames 600 --video mito.mp4
primordia render --recipe explore/symbiosis-living-reef-s1/recipes/01-seed1.json --width 1920 --height 1080
primordia render --recipe renders/lenia-pearl-reef-s7.png -o again.png
primordia render -w rd -p mitosis --set params.kill=0.06 --set post.exposure=1.3 --save-recipe mine.json
primordia explore --recipe mine.json --runs 8 --refine 1 --children 8 --max-fps 0
```

- `primordia recipe` prints a preset's (or `--recipe` file's) complete recipe after any `--set`: every setting
  there is to change. It needs a GPU because the world validates the values. `-o FILE.json` writes it instead, and
  `--json` wraps it in a result object.
- `render --recipe FILE` takes a `.json` recipe or a PNG Primordia wrote, and reproduces it at its own seed, size,
  look and camera; options override them. Given a PNG, it also runs the frame count and rate the PNG records, so
  `render --recipe picture.png` alone renders the same image again. A `.json` recipe records neither and runs 600
  frames at 60 fps: to reproduce an explore image, pass explore's `--frames`, or use the kept `NN-seedS.png` as the
  recipe. It conflicts with `-w` and `-p`. The default output is
  `renders/<recipe file stem>.png` (plus `-s<seed>` with a different `--seed`); when that would overwrite the PNG
  it came from, it exits with 2: name the image with `-o`.
- `--set KEY=VALUE` (render, explore and recipe; repeatable, applied in order). KEY is a dotted path into
  `settings`, with list indices as numbers: `params.feed`, `palette`, `post.exposure`, `params.kernels.0.mu`,
  `palettes.1`. VALUE is JSON, or text when it is not JSON (`palette=Frost`, one of Physarum's palettes); it must
  keep the JSON kind the setting has, then the world checks it. Keys under `post` also change the look. Palette names
  are checked against the world's own list in `list`; Particle Life picks its colours with `params.colors.Scheme=3`. A Symbiosis comparison is
  `params.compare=true`. An unknown key exits with 2, suggesting the closest key and listing its siblings.
- `render --save-recipe FILE.json` writes the recipe that was rendered, after `--set` and the look and camera
  options, for the next `--recipe`.
- Seeds: `--seed` takes any whole number from 0 to 2^64-1. Recipes may write the seed as a number or as a string of
  digits, and Primordia draws new seeds below 2^53 so that JavaScript and other float-based JSON tools keep them
  exact. Older recipes can hold larger ones: edit those with a tool that keeps big integers (Python's `json`), or
  write the seed as a string. Every seed in `--json` output is a string.
- Recipes over 4 MB, with a version other than 1, or with numbers outside the 32-bit float range are refused (exit 2).
- To open a recipe in the app, copy it into the library folder (`library_dir` in `list --json`) and press Refresh in
  the Library tab, or use `explore --install`.

## Environment

| Variable | Effect |
|---|---|
| `PRIMORDIA_LIBRARY_DIR` | Library folder (default: `%APPDATA%\Primordia\library`, `~/Library/Application Support/Primordia/library` or `$XDG_DATA_HOME/primordia/library`, which defaults to `~/.local/share/primordia/library`) |
| `PRIMORDIA_FFMPEG` | ffmpeg executable for `--video` (default: `ffmpeg` on the PATH) |
| `RUST_LOG` | Log filter (default `primordia=info,wgpu_core=error,wgpu_hal=error,naga=warn`); `-q` and `-v` still apply |
| `WGPU_BACKEND` | Backends to try: `vulkan`, `dx12`, `metal`, `gl` (comma-separated) |
| `WGPU_POWER_PREF` | `high` (default) or `low`, which prefers an integrated GPU |
| `WGPU_ADAPTER_NAME` | Use the GPU whose name contains this text (`selftest` lists the names) |

## Cookbook

A request such as "find the liveliest Lenia creatures, render the best one large and say how it behaves" takes four
steps, one command each:

1. **Discover** (no GPU): pick the world, a preset and the measurement that fits the request, here Menagerie and
   `active` (see [references/measurements.md](references/measurements.md)).
2. **Probe**: a small render with measurements. Read `metrics.last` and look at the PNG (a black image means the
   world died) before spending minutes on a search.
3. **Search**: a quick explore, then a full one if the probe looks promising. Look at `contact-sheet.png`, and pick
   ranks from `kept` with the numbers in `candidates.csv`.
4. **Deliver**: render the chosen recipe (its path is in `kept[].recipe`) at full size and for longer, with
   measurements and a video if wanted. A larger world is a new run of the same rules, so check that the behaviour
   held. Report the files, the seed, the recipe and the GPU (`gpu` in `run.json`).

```bash
primordia list --world lenia --json
primordia render -w lenia -p menagerie --width 320 --height 180 --frames 300 --max-fps 0 --metrics probe.csv --json
primordia explore -w lenia -p menagerie --select max:active --runs 8 --refine 0 --keep 4 --frames 300 --max-fps 0 --json
primordia render --recipe explore/lenia-menagerie-s1/recipes/01-seed<S>.json --width 1920 --height 1080 --frames 1200 --metrics final.csv -o final.png
```

## When something fails

- Exit 3, "needs a GPU": run `primordia selftest`. `WGPU_ADAPTER_NAME` or `WGPU_BACKEND` choose another adapter.
- On Windows, DirectX 12 compiles shaders slowly (a minute or more per world, several for Lenia; the log says
  "compiling its shaders for DirectX 12"). `WGPU_BACKEND=vulkan` avoids it where Vulkan exists.
- Exit 4: install ffmpeg or set `PRIMORDIA_FFMPEG`; PNG output needs no ffmpeg.
- Exit 2 from explore about "explore outputs that no run.json lists": choose another `-o`, or add `--overwrite`.
- A black or empty image usually means the world died: check the vital measurements, try another seed or preset.
