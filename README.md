# Primordia

**A GPU artificial-life laboratory in Rust + wgpu.** Millions of slime-mould agents weave living transport networks. Particle species with asymmetric attractions self-assemble into cells and serpents. Continuous cellular automata hatch soft, gliding organisms. Reaction-diffusion chemistry paints coral, fingerprints and spiral waves. Everything simulates in real time on compute shaders and is rendered as living light, and you can poke at all of it with the mouse.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Rust 1.86+](https://img.shields.io/badge/rust-1.86%2B-orange.svg)
![wgpu 25](https://img.shields.io/badge/wgpu-25-6f42c1.svg)
![Platforms](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

<p align="center">
  <picture>
    <source srcset="docs/images/hero.webp" type="image/webp">
    <img src="docs/images/physarum.png" alt="Primordia in motion: a Physarum galaxy, particle-life serpents, a Lenia pearl reef and Gray-Scott spiral waves" width="800">
  </picture>
</p>

Primordia is a playground for **emergence** and **self-organisation**. It has four classic artificial-life (ALife) systems, **36 curated presets** and a *Mutate* button that jumps to a random but reliably interesting region of parameter space. A cinematic HDR pipeline is shared by all four worlds. Use it as a screensaver, a generative-art tool, a teaching aid for agent-based models and cellular automata, or as a starting point for your own GPU simulations.

- [The four worlds](#the-four-worlds)
- [Features](#features)
- [Quick start](#quick-start)
- [Controls](#controls)
- [Rendering without a window](#rendering-without-a-window)
- [How it works](#how-it-works)
- [Inspiration and references](#inspiration-and-references)

## The four worlds

### 1 · Physarum (slime mould)

![Physarum slime-mould transport network, Dendrites preset](docs/images/physarum.png)

A multi-species version of Jeff Jones' *Physarum polycephalum* agent model, in the spirit of Sage Jenson's work. About **3-6 million agents** (at 1080p) sense the chemical trail ahead of them, steer toward it and deposit more. Out of that loop grow self-optimising transport networks: white-hot arteries fed by fine capillaries that keep remodelling. Up to four species attract or repel each other through an interaction matrix and fight over territory. Left-click drops food; right-click repels.

**Presets:** Dendrites · Neural Lace · Mycelium · Rival Colonies · Symbiosis · Honeycomb · Synapses · Currents · Chasing Waves · Galaxy

![All Physarum presets](docs/images/physarum-presets.png)

### 2 · Particle Life

![Particle Life serpents: chains of particles self-assembled from asymmetric attraction rules](docs/images/particle-life.png)

Particle life after Jeffrey Ventrella's *Clusters* and Tom Mohr: up to 8 species, each with its own asymmetric attraction or repulsion toward every other species. Those simple pairwise rules produce membranes, cells, serpents, rotating suns and predator-prey chases. Neighbour search is a GPU uniform grid rebuilt every step, and forces are summed in fixed point, so every run replays exactly from its seed. Left-click attracts; right-click scatters.

**Presets:** Tidepool · Living Cells · Serpents · Rotating Suns · Lace · Marbling · Predator & Prey · Plankton · Necklaces

![All Particle Life presets](docs/images/particle-life-presets.png)

### 3 · Lenia

![Lenia continuous cellular automaton: Orbium gliders among pearl colonies](docs/images/lenia.png)

Bert Chan's Lenia, a continuous cellular automaton, in its multi-channel, multi-kernel "expanded" form: up to 3 channels and 16 kernels. The classic **Orbium** gliders are hatched from the published pattern and released at random headings. Around them live giants and minnows, pearl reefs that grow into rings, the worm colonies of *Hydrogeminium* and the gyrating *Tessellatium*. Left-click paints life; right-click erases it.

**Presets:** Orbium · Leviathans · Menagerie · Pearl Reef · Necklaces · Hydrogeminium · Tessellatium

![All Lenia presets](docs/images/lenia-presets.png)

### 4 · Reaction-Diffusion (Gray-Scott)

![Gray-Scott reaction-diffusion: pink coral colonies in relief on deep navy](docs/images/reaction-diffusion.png)

Two virtual chemicals, U and V, react (`U + 2V → 3V`) and diffuse. Depending on the feed and kill rates, that gives coral, dividing cells, fingerprints, worms, solitons, crescent gliders, spiral waves or soap-film foam. Slow, periodic "weather" keeps every pattern alive and evolving. The *Morphology Atlas* preset sweeps feed and kill across space, so the whole Pearson zoo lives on one seamless torus. The field is lit as a height map in one of five materials, including thin-film nacre and ink on paper. Left-click seeds chemistry; right-click erases.

**Presets:** Coral Reef · Mitosis · Fingerprints · Worms · Solitons · Crescent Gliders · Spiral Waves · Bubbles · Spots & Stripes · Morphology Atlas

![All reaction-diffusion presets](docs/images/reaction-diffusion-presets.png)

## Features

- **Real-time GPU simulation** in WGSL compute shaders. Every world runs at hundreds of fps at 1080p on an RTX 4090 (see [Performance](#performance)).
- **36 curated presets** and **Mutate** (`M`), which lands on a new, random but interesting parameter set.
- **Interactive.** Paint, feed, attract, repel and erase with the mouse. Zoom into any detail, and pan across the seamless toroidal world.
- **Cinematic look.** HDR bloom (the Jimenez 13-tap chain with level falloff), AgX, ACES or Reinhard tonemapping, perceptual OKLab palettes, vignette, film grain and dithering.
- **Live control panel** (egui) exposing every parameter of every world.
- **Tour mode** (`T`): a screensaver that fades through every preset of every world.
- **Screenshots** (`F12`) and **MP4 recording** (`V`) straight from the app.
- **Headless rendering.** PNGs, frame sequences, MP4s, galleries and contact sheets from the command line, deterministic from a seed. A scripted mouse lets you test interactions too.
- **Extensible.** A small `World` trait; see [docs/WRITING_A_WORLD.md](docs/WRITING_A_WORLD.md).

## Quick start

You need **Rust 1.86 or newer** and a GPU with Vulkan, DirectX 12 or Metal. [ffmpeg](https://ffmpeg.org) is optional and is only used for video.

```bash
git clone https://github.com/skulitom/primordia
cd primordia
cargo run --release
```

Jump straight to a world or preset, or start the screensaver:

```bash
cargo run --release -- --world lenia --preset "Pearl Reef"
cargo run --release -- --world particle-life --preset 3
cargo run --release -- --tour 20 --fullscreen
```

## Controls

| Input | Action |
|---|---|
| **Left drag** | The world's primary action (feed / attract / paint life / seed chemistry) |
| **Right drag** | The world's secondary action (repel / erase) |
| **Wheel** / **middle drag** | Zoom at the cursor / pan |
| `1`–`4` | Switch world |
| `,` `.` | Previous / next preset |
| `R` / `M` | Reset with a new seed / mutate the parameters |
| `Space` / `N` | Pause / single step |
| `[` `]` | Brush size |
| `H` or `Tab` | Hide the control panel |
| `F` or `F11` | Fullscreen |
| `S` or `F12` | Screenshot (PNG) |
| `V` | Start / stop recording an MP4 |
| `T` | Tour mode |
| `0` or `Home` | Reset the view |
| `Esc` | Leave fullscreen; press it twice to quit |

Shortcuts use physical key positions, so they work on any keyboard layout.

## Rendering without a window

```bash
# A 1080p still of one preset after 600 frames
primordia render --world physarum --preset "Rival Colonies" --frames 600 --out physarum.png

# A 4K close-up, and a zoomed-out view showing the torus tiles seamlessly
primordia render --world reaction-diffusion --width 3840 --height 2160 --zoom 3 --center 0.3,0.6
primordia render --world lenia --zoom 0.5

# A 10-second MP4 (needs ffmpeg), or a PNG every 30 frames
primordia render --world particle-life --frames 600 --video clip.mp4
primordia render --world lenia --frames 900 --every 30 --frames-dir frames/

# Every preset of every world, plus contact sheets
primordia gallery --out-dir gallery --width 1280 --height 720

# Scripted mouse input, to test interaction headlessly
primordia render --world physarum --brush primary --brush-at 0.5,0.5

primordia list       # worlds, presets and palettes
primordia selftest   # verify this GPU's torus-wrapping maths
```

Headless renders are capped at 240 fps by default, so long batch jobs don't run your GPU flat out; `--max-fps 0` removes the cap. `primordia help render` lists every option.

## How it works

```
 each frame:  world.step()    compute passes advance the simulation (usually several sub-steps)
              world.render()  a fullscreen pass paints the state into an HDR (Rgba16Float) scene
              post            bloom -> exposure -> tonemap -> vignette / grain / dither
              egui            control panel (interactive app only)
```

- **Engine** (`src/`): a thin wgpu layer (`gpu.rs`), post-processing (`post.rs`, `shaders/post.wgsl`), OKLab palettes (`palette.rs`), the winit + egui app (`app.rs`), and headless rendering with ffmpeg video (`headless.rs`).
- **Worlds** (`src/world/`): each one owns its GPU state and implements the `World` trait. That covers `step`, `render`, presets, `mutate`, a parameter UI, and the post look it suggests.
- **Shared WGSL prelude** (`src/shaders/common.wgsl`): a fullscreen triangle, the camera transform, exact torus wrapping, PCG hashing and palette lookup.

Under the hood, per world:

- **Physarum**
  - Deposits are race-free: `u32` atomics in fixed point, split along each step so fast agents still draw continuous lines.
  - A diffuse pass blurs, decays and softly caps crowding in the trail.
  - A drifting, exactly periodic "terrain" field varies how fertile the ground is, so dense meshes and sparse ranges shift over time.
  - The display runs the trail through a logarithmic tone curve. That keeps single-agent filigree visible while arteries glow into HDR, and a GPU histogram sets an adaptive black point.
- **Particle Life**
  - The neighbour grid is rebuilt each step with a GPU counting sort.
  - Every step computes the pair forces between neighbouring cells in a fixed-point pass.
  - Motion-stretched Gaussian glows accumulate in an HDR trail, with crisp particle heads drawn on top.
  - A copy is instanced per visible torus tile, so zooming out tiles the world seamlessly.
- **Lenia**
  - Convolution is direct. The kernel weights are normalised and precomputed, and four kernels are packed per `vec4` in shared-memory tiles.
  - Orbium-family creatures are hatched in isolated "nursery" tori, then released into the world.
  - Glass-like bodies, glowing nuclei and soft wakes come from a separate compose pass.
- **Reaction-Diffusion**
  - It uses Karl Sims' 3×3 Laplacian on ping-ponged buffers.
  - Periodic modulations keep the chemistry alive: kill-rate "weather", drifting lagoons, pattern-size variation, and a ridge flow that closes stripes into whorls.
  - Auto-contrast comes from a GPU histogram.
  - Five lighting materials are available for the height field.

One lesson learned along the way: on some GPU/driver combinations, WGSL's signed integer `%` is wrong for negative operands, and wrapping with float division is off by one at some sizes. Both show up as seams on a torus. The prelude's `wrap_i` corrects for both, and `primordia selftest` checks it against the CPU on your hardware.

### Performance

These are headless numbers on an RTX 4090 at 1920×1080, uncapped, for each world's default preset. They include bloom, tonemapping and the final frame's readback.

| World | fps | GPU time per frame |
|---|---|---|
| Physarum (Dendrites, ~3 M agents) | 339 | ~2.9 ms |
| Particle Life (Tidepool) | 859 | ~1.2 ms |
| Lenia (Orbium) | 888 | ~1.1 ms |
| Reaction-Diffusion (Coral Reef) | 903 | ~1.1 ms |

## Write your own world

Copy `src/world/placeholder.rs`, the smallest possible world, register it in `src/world/mod.rs`, and follow [docs/WRITING_A_WORLD.md](docs/WRITING_A_WORLD.md). The engine gives you the window, camera, bloom, tonemapping, UI shell, screenshots, recording, headless rendering and contact sheets for free.

## Inspiration and references

Primordia stands on the shoulders of these:

- **Physarum**: Jeff Jones, *Characteristics of pattern formation and evolution in approximations of Physarum transport networks*, Artificial Life 16(2), 2010. [Sage Jenson's *physarum*](https://cargocollective.com/sagejenson/physarum) artwork.
- **Particle Life**: Jeffrey Ventrella's [*Clusters*](https://www.ventrella.com/Clusters/); Tom Mohr's [particle-life](https://github.com/tom-mohr/particle-life).
- **Lenia**: Bert Chan, [*Lenia: Biology of Artificial Life*](https://arxiv.org/abs/1812.05433) (2019) and [*Lenia and Expanded Universe*](https://arxiv.org/abs/2005.03742) (2020); [Chakazul/Lenia](https://github.com/Chakazul/Lenia).
- **Reaction-diffusion**: John E. Pearson, *Complex Patterns in a Simple System*, Science 261, 1993; Karl Sims' [reaction-diffusion tutorial](https://www.karlsims.com/rd.html); Robert Munafo's [xmorphia](https://www.mrob.com/pub/comp/xmorphia/).
- **Rendering**: Jorge Jimenez, *Next Generation Post Processing in Call of Duty: Advanced Warfare* (SIGGRAPH 2014), for the bloom chain; Troy Sobotka's [AgX](https://github.com/sobotka/AgX) and Benjamin Wrensch's [minimal AgX fit](https://iolite-engine.com/blog_posts/minimal_agx_implementation); Björn Ottosson's [OKLab](https://bottosson.github.io/posts/oklab/).
- Built with [wgpu](https://wgpu.rs), [winit](https://github.com/rust-windowing/winit) and [egui](https://github.com/emilk/egui).

If you like this, you might also enjoy exploring cellular automata, agent-based models, swarm simulation, evolutionary art and other creative-coding experiments in the ALife community.

## License

[MIT](LICENSE)
