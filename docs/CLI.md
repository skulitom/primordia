# Command-line reference

Every option of `primordia` and its commands, as `primordia --help` and `primordia <command> --help` print them.
`<library folder>` stands for the folder of saved recipes on the machine running them (`primordia list` names it).
The [README](../README.md) has examples and background; coding agents start with the [primordia skill](../.claude/skills/primordia/SKILL.md).

This file is generated from `src/main.rs` by the test `generated_cli_reference_is_current`. After changing any help text, run
`PRIMORDIA_BLESS=1 cargo test --locked generated_` and commit the result.

Contents: [primordia](#primordia) · [primordia list](#primordia-list) · [primordia render](#primordia-render) · [primordia gallery](#primordia-gallery) · [primordia recipe](#primordia-recipe) · [primordia explore](#primordia-explore) · [primordia selftest](#primordia-selftest)

## primordia

```text
GPU artificial-life laboratory: slime moulds, particle life, Lenia, reaction-diffusion and Symbiosis.

Without a command, primordia opens the interactive window and blocks until it is closed; add --exit-after SECS to quit on its own (smoke tests, scripts). The commands list, render, gallery, explore, recipe and selftest run headless and exit when they are done.

Usage: primordia [OPTIONS]
       primordia [--json] [-q | -v] <COMMAND> [ARGS]

Commands:
  list      List worlds with their presets, measurements and palettes (no GPU needed)
  render    Render a world offscreen to a PNG (and optionally a video via ffmpeg)
  gallery   Render the final frame of every preset into a folder
  recipe    Print the complete recipe of a preset or recipe file, after --set: every setting there is (needs a GPU)
  explore   Search a world's mutations for the most novel behaviour and keep them as recipes
  selftest  Verify on this GPU the shader maths the simulations rely on
  help      Print this message or the help of the given subcommand(s)

Options:
  -w, --world <WORLD>
          World to use: an id, a number or an alias, ignoring case and punctuation; a prefix works when only one id starts with it ("re" = reaction-diffusion).
            1  physarum            slime, slime-mold, slime-mould, mold, mould
            2  particle-life       particles, particlelife, pl, life
            3  lenia               smoothlife, continuous-ca
            4  reaction-diffusion  rd, gray-scott, grayscott, turing, coral
            5  symbiosis           coupled, hybrid, ecosystem

          [default: physarum]

  -p, --preset <PRESET>
          Preset: a name, a unique prefix or a number from 1 (see `primordia list --world W`)

      --seed <SEED>
          Random seed (default: time based)

      --width <WIDTH>
          Window width in logical pixels [default: 1600, shrunk to fit the screen]

      --height <HEIGHT>
          Window height in logical pixels [default: 900, shrunk to fit the screen]

      --fullscreen
          Start in borderless fullscreen

      --no-vsync
          Disable vsync (uncapped frame rate)

      --sim-scale <SIM_SCALE>
          Simulation resolution relative to the window's pixel size [default: 1; 0.5 on integrated and software GPUs]

      --hide-ui
          Start with the control panel hidden (H or Tab shows it)

      --exit-after <SECS>
          Quit automatically after this many seconds (useful for smoke tests)

      --screenshot-dir <SCREENSHOT_DIR>
          Folder for screenshots and recordings

          [default: screenshots]

      --tour <SECS>
          Screensaver mode: fade to the next preset (then world) every SECS seconds

      --record
          Start recording an MP4 (into --screenshot-dir) immediately; V toggles it (needs ffmpeg)

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version

Global options:
      --json
          Print one JSON object with the result (or the error) on stdout

  -q, --quiet
          Only print warnings and errors on stderr

  -v, --verbose...
          Also print debug messages on stderr (-vv: trace)

Examples:
  primordia --world lenia --preset necklaces           open a world (blocks until the window is closed)
  primordia list --world rd                            presets, measurements and palettes of one world
  primordia render -w rd -p mitosis -o mitosis.png     render a still
  primordia render -w physarum --video network.mp4     render a video (needs ffmpeg)
  primordia gallery -o gallery                         every preset of every world, plus a contact sheet
  primordia explore -w symbiosis --runs 24 --json      search for novel behaviour and keep recipes
  primordia render --recipe explore/symbiosis-living-reef-s1/recipes/01-seed1.json --width 3840 --height 2160
                                                       render a kept recipe (or a PNG Primordia wrote) again
  primordia recipe -w rd -p mitosis --set params.feed=0.031 -o mito.json
                                                       a preset's complete recipe, with one setting changed

Output:
  Logs and progress go to stderr (-q: warnings and errors only, -v: debug). stdout carries results only: the
  path of every file written, one per line, or with --json one JSON object at the end: {"ok":true,
  "command":"render",...} on success, {"ok":false,"error":{"kind":"usage|gpu|ffmpeg|io|other",
  "message":...}} on failure. Seeds are JSON strings. `primordia list --json` describes every world.

Environment:
  RUST_LOG               Log filter [default: primordia=info,wgpu_core=error,wgpu_hal=error,naga=warn]
  WGPU_BACKEND           GPU backends to try: vulkan, dx12, metal or gl (comma-separated)
  WGPU_POWER_PREF        Which GPU to prefer when there are several: high (default) or low
  WGPU_ADAPTER_NAME      Use the GPU whose name contains this text
  PRIMORDIA_FFMPEG       ffmpeg executable for --video, --record and V [default: ffmpeg on the PATH]
  PRIMORDIA_LIBRARY_DIR  Folder of saved recipes [now: <library folder>]

Exit status:
  0  success
  1  another failure (for example reading or writing a file)
  2  invalid input: bad arguments, an unknown or ambiguous world, preset, measurement or setting, a
     recipe that cannot be read or that the world rejects, a size out of range, an output path that
     cannot be written or has the wrong extension
  3  no usable GPU, or the GPU failed during the run
  4  ffmpeg is missing or failed
```

## primordia list

```text
List worlds with their presets, measurements and palettes (no GPU needed)

Usage: primordia list [OPTIONS]

Options:
  -w, --world <WORLD>
          Only this world (an id, 1-5, an alias or a unique prefix)

  -h, --help
          Print help (see a summary with '-h')

Global options:
      --json
          Print one JSON object with the result (or the error) on stdout

  -q, --quiet
          Only print warnings and errors on stderr

  -v, --verbose...
          Also print debug messages on stderr (-vv: trace)

Output, environment variables and exit status: see `primordia --help`.
```

## primordia render

```text
Render a world offscreen to a PNG (and optionally a video via ffmpeg)

Usage: primordia render [OPTIONS]

Options:
  -w, --world <WORLD>
          World to use: an id, a number or an alias, ignoring case and punctuation; a prefix works when only one id starts with it ("re" = reaction-diffusion).
            1  physarum            slime, slime-mold, slime-mould, mold, mould
            2  particle-life       particles, particlelife, pl, life
            3  lenia               smoothlife, continuous-ca
            4  reaction-diffusion  rd, gray-scott, grayscott, turing, coral
            5  symbiosis           coupled, hybrid, ecosystem

          [default: physarum]

  -p, --preset <PRESET>
          Preset: a name, a unique prefix or a number from 1 [default: the world's first preset]

      --recipe <FILE>
          Start from this recipe instead of a preset: a library or explore .json file, or a PNG written by Primordia (it carries its recipe)

      --set <KEY=VALUE>
          Change one setting of the preset or recipe; repeat it for several, applied in order. KEY is a path into the recipe's settings, with dots between keys and list indices: params.feed, palette, post.exposure, params.kernels.0.mu. VALUE is JSON, or text when it is not JSON (palette=Frost). Keys under post also change the look. `primordia recipe -w W -p P` prints every setting a preset has.

      --seed <SEED>
          Random seed (the same seed always gives the same image on the same GPU) [default: 1, or the recipe's]

      --width <WIDTH>
          Output width in pixels (16-16384; a video rounds it down to an even number) [default: 1920, or the recipe's]

      --height <HEIGHT>
          Output height in pixels (16-16384; a video rounds it down to an even number) [default: 1080, or the recipe's]

  -f, --frames <FRAMES>
          Frames to simulate before the final image [default: 600, or the frame count of a PNG given to --recipe]

      --fps <FPS>
          Frames per simulated second: sets the time step (1/fps) and the video frame rate [default: 60, or the rate of a PNG given to --recipe]

  -o, --out <OUT>
          Output PNG of the final frame [default: renders/<world>-<preset>-s<seed>.png, or renders/<recipe file name>.png; skipped when only --video is given]

      --save-recipe <PATH>
          Also write the recipe that was rendered (after --set and the look and camera options) to this .json file

      --video <VIDEO>
          Encode every frame into a video with ffmpeg: .mp4, .mov, .mkv (H.264), .webm or .gif

      --every <N>
          Save a PNG every N frames into --frames-dir (0 = off)

          [default: 0]

      --frames-dir <FRAMES_DIR>
          Folder for the --every frame PNGs

          [default: frames]

      --exposure <EXPOSURE>
          Override the preset's (or recipe's) exposure

      --bloom <BLOOM>
          Override the preset's (or recipe's) bloom strength (0 disables bloom)

      --bloom-threshold <BLOOM_THRESHOLD>
          Override the preset's (or recipe's) bloom threshold

      --tonemap <TONEMAP>
          Override the preset's (or recipe's) tonemapper

          [possible values: agx, aces, reinhard, linear]

      --zoom <ZOOM>
          Camera zoom, 0.05-256 (1 = whole world, >1 = close-up, <1 = show the torus tiling) [default: 1, or the recipe's]

      --center <X,Y>
          Camera centre in world uv, as X,Y [default: 0.5,0.5, or the recipe's]

      --brush <BRUSH>
          Hold a scripted mouse button for the whole render (tests interaction)

          Possible values:
          - primary:   Left button: the world's create / attract / paint action
          - secondary: Right button: the world's destroy / repel / erase action

      --brush-at <X,Y>
          Where to hold the brush, in world uv (e.g. 0.3,0.6) [default: orbit the centre]

      --brush-radius <BRUSH_RADIUS>
          Brush radius in world cells

          [default: 40]

      --max-fps <MAX_FPS>
          Frame-rate ceiling that keeps the GPU from running flat out (0 = unlimited)

          [default: 240]

      --metrics <PATH>
          Write every frame's measurements to a CSV file: frame, time, series, then one column per measurement (ids, units and meanings: `primordia list --world W`)

  -h, --help
          Print help (see a summary with '-h')

Global options:
      --json
          Print one JSON object with the result (or the error) on stdout

  -q, --quiet
          Only print warnings and errors on stderr

  -v, --verbose...
          Also print debug messages on stderr (-vv: trace)

Examples:
  primordia render -w rd -p mitosis -o mitosis.png
  primordia render -w lenia -p 5 --frames 1200 --video necklaces.mp4 --metrics necklaces.csv
  primordia render -w physarum --zoom 3 --center 0.25,0.5 --json
  primordia render --recipe explore/symbiosis-living-reef-s1/recipes/01-seed1.json --width 3840 --height 2160
  primordia render -w rd -p mitosis --set params.feed=0.031 --save-recipe mito.json
  primordia render --recipe mito.png -o mito-again.png

--recipe reproduces a recipe exactly at its own size, seed, look and camera; --width and --height run the same rules on a larger or smaller world, and the other options override the recipe's. Every PNG written carries its recipe, how many frames it ran and at what rate, the command that renders it again and the GPU that rendered it (runs repeat exactly only on the same GPU, driver and backend). Given a PNG, --recipe also takes its frame count and rate as the defaults of --frames and --fps, so `render --recipe image.png` renders that image again; a .json recipe runs 600 frames at 60 fps unless they say otherwise.

Output, environment variables and exit status: see `primordia --help`.
```

## primordia gallery

```text
Render the final frame of every preset into a folder

Usage: primordia gallery [OPTIONS]

Options:
  -o, --out-dir <OUT_DIR>
          Folder to write the images into

          [default: gallery]

  -w, --world <WORLD>
          Only this world (an id, 1-5, an alias or a unique prefix) [default: every world]

      --width <WIDTH>
          Image width in pixels (16-16384)

          [default: 1280]

      --height <HEIGHT>
          Image height in pixels (16-16384)

          [default: 720]

  -f, --frames <FRAMES>
          Frames to simulate per preset

          [default: 600]

      --seed <SEED>
          Random seed used for every preset

          [default: 1]

      --no-sheet
          Don't write contact-sheet.png (all images tiled into one overview)

      --max-fps <MAX_FPS>
          Frame-rate ceiling per render, keeps the GPU from running flat out (0 = unlimited)

          [default: 240]

  -h, --help
          Print help (see a summary with '-h')

Global options:
      --json
          Print one JSON object with the result (or the error) on stdout

  -q, --quiet
          Only print warnings and errors on stderr

  -v, --verbose...
          Also print debug messages on stderr (-vv: trace)

Output, environment variables and exit status: see `primordia --help`.
```

## primordia recipe

```text
Print the complete recipe of a preset or recipe file, after --set: every setting there is (needs a GPU)

Usage: primordia recipe [OPTIONS]

Options:
  -w, --world <WORLD>
          World to use: an id, a number or an alias, ignoring case and punctuation; a prefix works when only one id starts with it ("re" = reaction-diffusion).
            1  physarum            slime, slime-mold, slime-mould, mold, mould
            2  particle-life       particles, particlelife, pl, life
            3  lenia               smoothlife, continuous-ca
            4  reaction-diffusion  rd, gray-scott, grayscott, turing, coral
            5  symbiosis           coupled, hybrid, ecosystem

          [default: physarum]

  -p, --preset <PRESET>
          Preset: a name, a unique prefix or a number from 1 [default: the world's first preset]

      --recipe <FILE>
          Start from this recipe instead of a preset: a library or explore .json file, or a PNG written by Primordia (it carries its recipe)

      --set <KEY=VALUE>
          Change one setting of the preset or recipe; repeat it for several, applied in order. KEY is a path into the recipe's settings, with dots between keys and list indices: params.feed, palette, post.exposure, params.kernels.0.mu. VALUE is JSON, or text when it is not JSON (palette=Frost). Keys under post also change the look. `primordia recipe -w W -p P` prints every setting a preset has.

      --seed <SEED>
          Seed the recipe runs from [default: 1, or the recipe's]

      --width <WIDTH>
          Output width the recipe is for, in pixels (16-16384) [default: 1920, or the recipe's]

      --height <HEIGHT>
          Output height the recipe is for, in pixels (16-16384) [default: 1080, or the recipe's]

  -o, --out <PATH>
          Write the recipe to this .json file instead of printing it

  -h, --help
          Print help (see a summary with '-h')

Global options:
      --json
          Print one JSON object with the result (or the error) on stdout

  -q, --quiet
          Only print warnings and errors on stderr

  -v, --verbose...
          Also print debug messages on stderr (-vv: trace)

Examples:
  primordia recipe -w rd -p mitosis
  primordia recipe -w rd -p mitosis --set params.feed=0.031 -o mito.json
  primordia render --recipe mito.json --video mito.mp4

A recipe is the JSON that the app's library and explore write: version (1), name, seed (a number, or a string of digits), output_size, preset (numbered from 0), modified, settings (the world's parameters, which --set changes), look and camera. A PNG written by Primordia carries the recipe of its image in an iTXt chunk named primordia:recipe, and --recipe reads it back.

Without -o the recipe is printed on stdout; with --json it is the "recipe" field of the result, with its seed as a string.

Output, environment variables and exit status: see `primordia --help`.
```

## primordia explore

```text
Search a world's mutations for the most novel behaviour and keep them as recipes

Usage: primordia explore [OPTIONS]

Options:
  -w, --world <WORLD>
          World to use: an id, a number or an alias, ignoring case and punctuation; a prefix works when only one id starts with it ("re" = reaction-diffusion).
            1  physarum            slime, slime-mold, slime-mould, mold, mould
            2  particle-life       particles, particlelife, pl, life
            3  lenia               smoothlife, continuous-ca
            4  reaction-diffusion  rd, gray-scott, grayscott, turing, coral
            5  symbiosis           coupled, hybrid, ecosystem

          [default: physarum]

  -p, --preset <PRESET>
          Preset to start from, a name or a number from 1 (mutations of some worlds keep parts of it)

      --recipe <FILE>
          Start from this recipe instead of a preset (a library or explore .json file, or a PNG written by Primordia); it is evaluated first, from its own seed, at --width x --height

      --set <KEY=VALUE>
          Change one setting of the preset or recipe; repeat it for several, applied in order. KEY is a path into the recipe's settings, with dots between keys and list indices: params.feed, palette, post.exposure, params.kernels.0.mu. VALUE is JSON, or text when it is not JSON (palette=Frost). Keys under post also change the look. `primordia recipe -w W -p P` prints every setting a preset has.

      --seed <SEED>
          Master seed: every candidate seed and perturbation follows from it (and the preset runs from it)

          [default: 1]

      --runs <RUNS>
          Mutations to evaluate in the first round (the preset itself is evaluated too)

          [default: 48]

      --refine <REFINE>
          Refinement rounds, each perturbing recipes of the current archive

          [default: 1]

      --children <CHILDREN>
          Children to evaluate per refinement round

          [default: 24]

      --strength <STRENGTH>
          Relative size of a perturbation (log-normal noise on the recipe's numbers; colours, palettes and the rest of the look are never perturbed)

          [default: 0.15]

      --keep <KEEP>
          Candidates to keep

          [default: 12]

  -f, --frames <FRAMES>
          Frames to simulate per candidate (at 60 frames per simulated second)

          [default: 600]

      --width <WIDTH>
          Image width in pixels (16-16384)

          [default: 640]

      --height <HEIGHT>
          Image height in pixels (16-16384)

          [default: 360]

      --max-fps <MAX_FPS>
          Frame-rate ceiling per candidate, keeps the GPU from running flat out (0 = unlimited)

          [default: 240]

  -o, --out-dir <OUT_DIR>
          Folder for the images, recipes/, candidates.csv, contact-sheet.png and run.json [default: explore/<world>-<preset>-s<seed>, or explore/<world>-<recipe file name>-s<seed> with --recipe]

      --overwrite
          Also replace explore outputs in --out-dir that no run.json lists (from an older build, another tool or a killed run); without it such a folder is refused. The files an earlier run.json lists are always replaced

      --select <SELECT>
          What to keep: "novelty" (mutually most different), "max:<metric>" or "min:<metric>" (metric ids: `primordia list --world W`)

          [default: novelty]

      --install
          Also save the kept recipes into the app's library (<library folder>; PRIMORDIA_LIBRARY_DIR overrides)

      --no-sheet
          Don't write contact-sheet.png

      --all
          Also write every evaluated candidate's final frame under all/

      --keep-inert
          Let dead, empty or frozen candidates into the archive

  -h, --help
          Print help (see a summary with '-h')

Global options:
      --json
          Print one JSON object with the result (or the error) on stdout

  -q, --quiet
          Only print warnings and errors on stderr

  -v, --verbose...
          Also print debug messages on stderr (-vv: trace)

Each run writes into a folder of its own, explore/<world>-<preset>-s<seed> unless --out-dir names one: NN-seedS.png for the kept candidate of rank NN, recipes/NN-seedS.json (loadable by the app's library, render --recipe and explore --recipe; its "provenance" says how explore found it), candidates.csv, contact-sheet.png (captioned with rank and seed), all/NNN-rR-seedS.png with --all, and run.json: the command line, version, GPU, settings, what every candidates.csv column holds, the kept candidates and every file the run wrote. A later run into the same folder first removes exactly the files its run.json lists; a folder holding explore outputs that no run.json lists is refused unless --overwrite.

candidates.csv has one row per candidate: index, round, origin (preset, recipe, mutation or child), parent, seed, preset (from 1, like --preset), preset_name, rank, novelty, status (ok, inert or failed), secs, then three columns per measurement: <id>_mean (mean of the last 40% of the frames), <id>_std (its standard deviation) and <id>_drift (that mean minus the mean of the first 20%). Children reuse their parent's seed.

Output, environment variables and exit status: see `primordia --help`.
```

## primordia selftest

```text
Verify on this GPU the shader maths the simulations rely on

Usage: primordia selftest [OPTIONS]

Options:
  -h, --help
          Print help (see a summary with '-h')

Global options:
      --json
          Print one JSON object with the result (or the error) on stdout

  -q, --quiet
          Only print warnings and errors on stderr

  -v, --verbose...
          Also print debug messages on stderr (-vv: trace)

Output, environment variables and exit status: see `primordia --help`.
```
