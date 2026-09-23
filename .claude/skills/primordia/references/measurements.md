# Measurements

Every measurement of every world, in the order of the columns a `primordia render --metrics` CSV file has after
`frame,time,series`. The ids are also what `primordia explore --select max:<id>` and `min:<id>` take, and
`<id>_mean`, `<id>_std` and `<id>_drift` are explore's candidates.csv columns. `primordia list --world <id>` prints
the same table, and `primordia list --json` the same data.

A fraction is a share of cells or agents from 0 to 1; a scalar is a plain number (a mean, a ratio or a signed
rate). Explore counts a candidate as inert, and keeps it only with `--keep-inert`, when the mean of any of its vital
measurements over the last 40% of its frames is below 0.001.

This file is generated from the world registry (`src/world/mod.rs`) by the test
`generated_measurement_reference_is_current` in `src/main.rs`. After changing a measurement, run
`PRIMORDIA_BLESS=1 cargo test --locked generated_` and commit the result.

## physarum

Physarum. Vital: `ground`.

| Id | Unit | Vital | Label | Meaning |
|---|---|---|---|---|
| `ground` | fraction | yes | Marked ground | Cells holding at least a tenth of the trail an even spread of the agents would leave. |
| `veins` | fraction |  | Vein cells | Cells holding more than four times the trail of an even spread: the network's arteries. |
| `concentration` | scalar |  | Concentration | Mean log2 of the relative trail density: an even spread at the nominal level gives 0, and the value drops the more the trail gathers into a network. |
| `travelled` | fraction |  | Travelled ground | Cells whose long-exposure traffic is at least a quarter of an even spread. |
| `trail_mass` | scalar |  | Trail mass | Mean relative trail density; about 1, shifted by terrain and crowding. |
| `reinforced` | fraction |  | Reinforced cells | Cells whose trail grew during the last step, against those decaying. |
| `on_vein` | fraction |  | Agents on veins | Agents standing on a cell where their own species' trail exceeds four times an even spread. |
| `agent_trail` | scalar |  | Trail under agents | Mean relative trail density of the agents' own species under them: about 1 when scattered, far more on a network. |

## particle-life

Particle Life. Vital: `speed`.

| Id | Unit | Vital | Label | Meaning |
|---|---|---|---|---|
| `speed` | scalar | yes | Mean speed | Mean particle speed in interaction radii per second. |
| `hot` | fraction |  | Fast particles | Particles moving faster than 1.5 radii per second: chases and boiling. |
| `slow` | fraction |  | Settled particles | Particles moving slower than 0.1 radii per second: frozen lattices and resting cells. |
| `crowding` | scalar |  | Crowding | Mean particles per grid cell around each particle, relative to a uniform spread (1 = random, higher = clustered). |
| `dense` | fraction |  | Clustered particles | Particles whose grid cell holds more than three times the uniform expectation. |
| `segregation` | scalar |  | Species segregation | How much cell-mates share a particle's species beyond chance: 0 = mixed, 1 = pure species clusters, negative = alternating. |
| `void` | fraction |  | Empty cells | Grid cells without any particle: the open space between structures. |

## lenia

Lenia. Vital: `mass`, `active`.

| Id | Unit | Vital | Label | Meaning |
|---|---|---|---|---|
| `mass` | scalar | yes | Mean density | Mean cell value summed over the channels: the creatures' total mass per cell. |
| `mass_1` | scalar |  | Channel 1 density | Mean value of channel 1. |
| `mass_2` | scalar |  | Channel 2 density | Mean value of channel 2 (zero when the preset uses fewer channels). |
| `mass_3` | scalar |  | Channel 3 density | Mean value of channel 3 (zero when the preset uses fewer channels). |
| `occupied` | fraction |  | Occupied cells | Cells whose summed value exceeds 0.1: the creatures' footprint. |
| `active` | fraction | yes | Changing cells | Cells whose summed value is changing by more than 0.001 per frame, judged from the last step. |
| `growth` | scalar |  | Net growth | Mean growth rate of the last step summed over the channels: positive while creatures grow, negative while they fade. |
| `dense` | fraction |  | Dense cores | Cells whose summed value exceeds 0.5: the solid bodies of the creatures. |

## reaction-diffusion

Reaction-Diffusion. Vital: `alive`, `active`.

| Id | Unit | Vital | Label | Meaning |
|---|---|---|---|---|
| `alive` | fraction | yes | Alive cells | Cells whose V exceeds 0.03: the footprint of the pattern. |
| `body` | fraction |  | Body cells | Cells whose V exceeds 0.25: the solid interior of spots, stripes and worms. |
| `v_mean` | scalar |  | Mean V | Mean concentration of the activator V. |
| `u_mean` | scalar |  | Mean U | Mean concentration of the substrate U. |
| `active` | fraction | yes | Changing cells | Cells whose V is changing by more than 0.001 per frame, judged from the last step: zero once a pattern has frozen. |
| `v_drift` | scalar |  | V drift | Mean change of V per step: positive while the pattern spreads, negative while it dies back. |
| `edge` | fraction |  | Boundary cells | Alive cells with a dead neighbour: the pattern's perimeter, high for fine labyrinths and low for blobs. |
| `filled` | fraction |  | Filled cells | Cells whose V exceeds 0.4: flooded ground, the state the revive logic watches for. |

## symbiosis

Symbiosis. Vital: `growth_cover`, `growth_active`.

| Id | Unit | Vital | Label | Meaning |
|---|---|---|---|---|
| `growth_cover` | fraction | yes | Growth cover | Cells whose activator V exceeds 0.1: the extent of the chemical colonies. |
| `growth_mean` | scalar |  | Mean growth | Mean activator concentration V over the habitat. |
| `growth_active` | fraction | yes | Changing cells | Cells whose V is changing by more than 0.001 per frame, judged from the last chemistry step: how much of the pattern is still evolving. |
| `growth_drift` | scalar |  | Growth drift | Mean change of V per chemistry step: positive while colonies grow, negative while they die back. |
| `routes` | fraction |  | Busy routes | Cells whose trail exceeds 0.6, the level from which traffic wears the ground: a tight network covers few, a diffuse one many. |
| `fertility_mean` | scalar |  | Mean fertility | Mean ground fertility; 1 is fully rested ground. |
| `exhausted` | fraction |  | Exhausted ground | Cells whose fertility has fallen below 0.5. |
| `agents_on_growth` | fraction |  | Agents on growth | Share of agents standing on a cell with growth (V above 0.1). |
