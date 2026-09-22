//! Live measurements.
//!
//! Every world reduces a few scalar quantities of its state on the GPU each
//! simulation frame (a coverage fraction, a mean concentration, ...). The
//! engine copies each frame's totals into a small ring of staging buffers,
//! maps them asynchronously and hands the finished samples to the control
//! panel's sparklines and to CSV logs. Nothing here waits for the GPU on the
//! interactive path; headless renders and tests flush at the end.
//!
//! ```text
//!  world cs_measure   per-workgroup partial sums (16 floats, normalised) -> Reduction::partials
//!  cs_reduce          one workgroup sums the partials                    -> Reduction::totals (64 B)
//!  Sink::push         copies the totals into this frame's staging slot   -> Sampler ring
//!  Sampler::map       map_async once the frame is submitted
//!  Sampler::collect   completed slots -> Sample (a frame or two later, without blocking)
//! ```

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use anyhow::{anyhow, bail, Context, Result};
use bytemuck::Zeroable;
use wgpu::ShaderStages;

use crate::gpu::{layout, Gpu};

/// Lanes in a totals record: four `vec4<f32>`.
pub const MAX_METRICS: usize = 16;
/// Series per frame: one habitat, or the two panes of a comparison.
pub const MAX_SERIES: usize = 2;
/// Bytes of one series' totals.
pub const SERIES_BYTES: u64 = (MAX_METRICS * 4) as u64;
/// Points a history keeps per trace before it halves its resolution.
pub const HISTORY_POINTS: usize = 720;

/// Shared WGSL for the worlds' measure kernels (`metric_reduce`). Compile a
/// measure module as `gpu.shader(label, &format!("{WGSL}\n{source}"))`.
pub const WGSL: &str = include_str!("shaders/metrics.wgsl");
const REDUCE_WGSL: &str = include_str!("shaders/metrics_reduce.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// A share of cells or agents in `0..=1`, shown as a percentage.
    Fraction,
    /// A plain number: a mean concentration, a relative speed, a signed rate.
    Scalar,
}

impl Unit {
    /// Compact text for the panel; non-finite values show as a dash.
    pub fn format(self, value: f32) -> String {
        if !value.is_finite() {
            return "—".to_string();
        }
        match self {
            Unit::Fraction => {
                let pct = value * 100.0;
                if pct.abs() < 10.0 { format!("{pct:.2}%") } else { format!("{pct:.1}%") }
            }
            Unit::Scalar => {
                let a = value.abs();
                if a == 0.0 {
                    "0".to_string()
                } else if a >= 100.0 {
                    format!("{value:.0}")
                } else if a >= 10.0 {
                    format!("{value:.1}")
                } else if a >= 1.0 {
                    format!("{value:.2}")
                } else if a >= 0.001 {
                    format!("{value:.3}")
                } else {
                    format!("{value:.2e}")
                }
            }
        }
    }
}

/// One measurement a world publishes; the order of a world's table is the
/// lane order of its totals record and the column order of its CSV.
#[derive(Clone, Copy, Debug)]
pub struct MetricDesc {
    /// CSV column name: lowercase ASCII letters, digits and underscores, unique within the world.
    pub id: &'static str,
    pub label: &'static str,
    pub unit: Unit,
    /// One sentence for the tooltip: what the number means and how it is measured.
    pub hint: &'static str,
}

// --- GPU reduction -----------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    count: u32,
    _pad: [u32; 3],
}

/// The second half of a world's measurement: sums the partial records its
/// `cs_measure` kernel wrote (one 64-byte record per workgroup, see
/// `shaders/metrics.wgsl`) into a 64-byte totals record. Worlds embed one per
/// habitat, next to their pipelines.
pub struct Reduction {
    partials: wgpu::Buffer,
    totals: wgpu::Buffer,
    params: wgpu::Buffer,
    group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    capacity: u32,
}

impl Reduction {
    /// `max_workgroups`: the most partial records the measure kernel(s) can write per frame.
    pub fn new(gpu: &Gpu, label: &str, max_workgroups: u32) -> Self {
        let capacity = max_workgroups.max(1);
        let partials = gpu.storage_buffer(
            &format!("{label} partials"),
            u64::from(capacity) * SERIES_BYTES,
            wgpu::BufferUsages::empty(),
        );
        let totals = gpu.storage_buffer(&format!("{label} totals"), SERIES_BYTES, wgpu::BufferUsages::empty());
        let params = gpu.uniform_buffer(&format!("{label} reduce"), &ReduceParams::zeroed());
        let bgl = gpu.bind_group_layout(
            label,
            &[
                layout::uniform(0, ShaderStages::COMPUTE),
                layout::storage(1, ShaderStages::COMPUTE, true),
                layout::storage(2, ShaderStages::COMPUTE, false),
            ],
        );
        let group = gpu.bind_group(
            label,
            &bgl,
            &[params.as_entire_binding(), partials.as_entire_binding(), totals.as_entire_binding()],
        );
        let module = gpu.shader("metrics reduce", &format!("{WGSL}\n{REDUCE_WGSL}"));
        let pl = gpu.pipeline_layout(label, &[&bgl]);
        let pipeline = gpu.compute_pipeline(label, &pl, &module, "cs_reduce");
        Self { partials, totals, params, group, pipeline, capacity }
    }

    /// Bind as `var<storage, read_write> partials: array<vec4<f32>>`: workgroup
    /// `g` writes `partials[4g .. 4g + 4]`, all four vec4s every frame (zero the
    /// unused lanes), already normalised.
    pub fn partials(&self) -> &wgpu::Buffer {
        &self.partials
    }

    /// Metric `i` at float `i`; what a world hands to [`Sink::push`].
    pub fn totals(&self) -> &wgpu::Buffer {
        &self.totals
    }

    /// Sums the first `workgroups` partial records into `totals`. Record it
    /// once per frame, after the measure kernel(s).
    pub fn record(&self, gpu: &Gpu, encoder: &mut wgpu::CommandEncoder, workgroups: u32) {
        debug_assert!(workgroups <= self.capacity, "{workgroups} partial records exceed the capacity {}", self.capacity);
        gpu.write(&self.params, &ReduceParams { count: workgroups.min(self.capacity), _pad: [0; 3] });
        let mut pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("metrics reduce"), timestamp_writes: None });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

// --- asynchronous readback ---------------------------------------------------

/// One frame's measurements: `values[s]` for each of the `series` recorded.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// `Frame::frame` of the step that produced the state measured.
    pub frame: u64,
    pub time: f32,
    /// Series recorded this frame: 1, or 2 while a world compares two habitats.
    pub series: usize,
    pub values: [[f32; MAX_METRICS]; MAX_SERIES],
}

enum SlotState {
    Free,
    /// Between `begin` and `map`; `series` totals records copied so far.
    Recording { series: usize },
    /// Mapping requested; the callback reports through `rx`.
    Pending { series: usize, rx: mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> },
}

struct Slot {
    staging: wgpu::Buffer,
    state: SlotState,
    frame: u64,
    time: f32,
    generation: u32,
}

/// A ring of staging buffers that carries totals records from the GPU to the
/// CPU without ever blocking the interactive frame loop.
pub struct Sampler {
    slots: Vec<Slot>,
    /// Slot opened by `begin` for the frame being recorded.
    current: Option<usize>,
    /// Slots with a mapping in flight, oldest first.
    pending: VecDeque<usize>,
    /// Bumped by `discard`; samples from older generations are dropped on arrival.
    generation: u32,
    /// Frames whose measurements were skipped because every slot was in flight.
    dropped: u64,
}

impl Sampler {
    /// Enough for the interactive loop, which polls every frame.
    pub const APP_SLOTS: usize = 4;
    /// Headless renders only wait for the GPU every fourth frame.
    pub const HEADLESS_SLOTS: usize = 6;

    pub fn new(gpu: &Gpu, slots: usize) -> Self {
        let slots = (0..slots.max(1))
            .map(|_| Slot {
                staging: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("metrics staging"),
                    size: SERIES_BYTES * MAX_SERIES as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: SlotState::Free,
                frame: 0,
                time: 0.0,
                generation: 0,
            })
            .collect();
        Self { slots, current: None, pending: VecDeque::new(), generation: 0, dropped: 0 }
    }

    /// Opens this frame's slot for the world's `measure` call. When every slot
    /// is still in flight the sink is dead (pushes are ignored) and the frame
    /// is counted in [`Sampler::dropped`].
    pub fn begin(&mut self, frame: u64, time: f32) -> Sink<'_> {
        // A frame abandoned before its submit leaves its slot recording; reclaim it.
        if let Some(i) = self.current.take() {
            if matches!(self.slots[i].state, SlotState::Recording { .. }) {
                self.slots[i].state = SlotState::Free;
            }
        }
        let Some(i) = self.slots.iter().position(|s| matches!(s.state, SlotState::Free)) else {
            self.dropped += 1;
            return Sink { slot: None };
        };
        self.current = Some(i);
        let slot = &mut self.slots[i];
        slot.state = SlotState::Recording { series: 0 };
        slot.frame = frame;
        slot.time = time;
        slot.generation = self.generation;
        Sink { slot: Some(slot) }
    }

    /// Call right after the frame's `queue.submit`: requests the mapping of the
    /// slot opened by `begin` (or frees it when nothing was pushed).
    pub fn map(&mut self) {
        let Some(i) = self.current.take() else { return };
        let slot = &mut self.slots[i];
        match slot.state {
            SlotState::Recording { series: 0 } => slot.state = SlotState::Free,
            SlotState::Recording { series } => {
                let (tx, rx) = mpsc::channel();
                slot.staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
                slot.state = SlotState::Pending { series, rx };
                self.pending.push_back(i);
            }
            _ => {}
        }
    }

    /// Non-blocking: polls the device and returns the samples whose mappings
    /// have completed, oldest first.
    pub fn collect(&mut self, gpu: &Gpu) -> Vec<Sample> {
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        self.drain()
    }

    /// Blocking: waits for every mapping in flight (headless renders, tests).
    pub fn flush(&mut self, gpu: &Gpu) -> Result<Vec<Sample>> {
        let mut out = Vec::new();
        while !self.pending.is_empty() {
            gpu.device.poll(wgpu::PollType::Wait).map_err(|e| anyhow!("GPU poll failed: {e:?}"))?;
            let before = self.pending.len();
            out.extend(self.drain());
            if self.pending.len() == before {
                bail!("the GPU did not complete a measurement readback");
            }
        }
        Ok(out)
    }

    /// Forgets the samples still in flight (the world was reset or replaced).
    /// Their slots are recycled when the mappings complete.
    pub fn discard(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// True when `begin` would return a dead sink.
    pub fn is_full(&self) -> bool {
        self.slots.iter().all(|s| !matches!(s.state, SlotState::Free))
    }

    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    fn drain(&mut self) -> Vec<Sample> {
        let mut out = Vec::new();
        while let Some(&i) = self.pending.front() {
            let slot = &mut self.slots[i];
            let (series, result) = match &slot.state {
                SlotState::Pending { series, rx } => (*series, rx.try_recv()),
                _ => {
                    self.pending.pop_front();
                    continue;
                }
            };
            match result {
                // Mappings complete in submission order; a later one waits for the next poll.
                Err(mpsc::TryRecvError::Empty) => break,
                Ok(Ok(())) => {
                    if slot.generation == self.generation {
                        let mut values = [[0.0; MAX_METRICS]; MAX_SERIES];
                        let bytes = slot.staging.slice(..).get_mapped_range();
                        for (s, lanes) in values.iter_mut().enumerate().take(series) {
                            for (k, value) in lanes.iter_mut().enumerate() {
                                let o = s * SERIES_BYTES as usize + k * 4;
                                *value = f32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]);
                            }
                        }
                        drop(bytes);
                        out.push(Sample { frame: slot.frame, time: slot.time, series, values });
                    }
                    slot.staging.unmap();
                    slot.state = SlotState::Free;
                }
                // A failed mapping never mapped the buffer, so it must not be unmapped.
                Ok(Err(e)) => {
                    log::warn!("measurement readback failed: {e}");
                    slot.state = SlotState::Free;
                }
                Err(mpsc::TryRecvError::Disconnected) => slot.state = SlotState::Free,
            }
            self.pending.pop_front();
        }
        out
    }
}

/// The per-frame handle a world pushes its totals into; the n-th push is series n.
pub struct Sink<'a> {
    slot: Option<&'a mut Slot>,
}

impl Sink<'_> {
    /// False when the ring is exhausted: the world may skip its measure passes.
    pub fn is_live(&self) -> bool {
        self.slot.is_some()
    }

    /// Copies one 64-byte totals record into this frame's slot.
    pub fn push(&mut self, encoder: &mut wgpu::CommandEncoder, totals: &wgpu::Buffer) {
        let Some(slot) = self.slot.as_deref_mut() else { return };
        let SlotState::Recording { series } = &mut slot.state else { return };
        if *series >= MAX_SERIES {
            debug_assert!(false, "a world pushed more than {MAX_SERIES} series");
            return;
        }
        encoder.copy_buffer_to_buffer(totals, 0, &slot.staging, *series as u64 * SERIES_BYTES, SERIES_BYTES);
        *series += 1;
    }
}

// --- history -----------------------------------------------------------------

type Point = (u64, [f32; MAX_METRICS]);

/// The run since the last reset, for the sparklines. Points are kept at full
/// resolution until a trace holds [`HISTORY_POINTS`]; then every other point is
/// dropped and the stride doubles, so a long experiment stays visible whole.
pub struct History {
    traces: [VecDeque<Point>; MAX_SERIES],
    capacity: usize,
    stride: u64,
    since_last: u64,
    series: usize,
    latest: Option<Sample>,
}

impl Default for History {
    fn default() -> Self {
        Self::new(HISTORY_POINTS)
    }
}

impl History {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(2);
        Self {
            traces: std::array::from_fn(|_| VecDeque::with_capacity(capacity + 1)),
            capacity,
            stride: 1,
            since_last: 1,
            series: 0,
            latest: None,
        }
    }

    pub fn clear(&mut self) {
        for trace in &mut self.traces {
            trace.clear();
        }
        self.stride = 1;
        self.since_last = 1;
        self.series = 0;
        self.latest = None;
    }

    pub fn push(&mut self, sample: &Sample) {
        let series = sample.series.min(MAX_SERIES);
        if series > self.series && self.series > 0 {
            // A comparison starting restarts both habitats from the seed (inside
            // the world's own controls, unseen by the app): the old run is over.
            self.clear();
        }
        // A series that stopped reporting (comparison switched off) is retired.
        for trace in &mut self.traces[series..] {
            trace.clear();
        }
        self.series = series;
        self.latest = Some(*sample);
        if self.since_last < self.stride {
            self.since_last += 1;
            return;
        }
        self.since_last = 1;
        for (s, trace) in self.traces.iter_mut().enumerate().take(series) {
            trace.push_back((sample.frame, sample.values[s]));
        }
        if self.traces[0].len() > self.capacity {
            for trace in &mut self.traces {
                let mut keep = false;
                trace.retain(|_| {
                    keep = !keep;
                    keep
                });
            }
            self.stride *= 2;
        }
    }

    /// Series currently reporting.
    pub fn series(&self) -> usize {
        self.series
    }

    /// Simulation frames between kept points.
    pub fn stride(&self) -> u64 {
        self.stride
    }

    /// Newest value received, regardless of decimation.
    pub fn latest(&self, series: usize, metric: usize) -> Option<f32> {
        let sample = self.latest.as_ref()?;
        (series < sample.series && metric < MAX_METRICS).then(|| sample.values[series][metric])
    }

    pub fn latest_frame(&self) -> Option<u64> {
        self.latest.as_ref().map(|s| s.frame)
    }

    /// Oldest and newest frame with a kept point.
    pub fn frame_span(&self) -> Option<(u64, u64)> {
        Some((self.traces[0].front()?.0, self.traces[0].back()?.0))
    }

    /// Kept values of one metric, oldest first.
    pub fn trace(&self, series: usize, metric: usize) -> Vec<f32> {
        match self.traces.get(series) {
            Some(trace) if metric < MAX_METRICS => trace.iter().map(|(_, v)| v[metric]).collect(),
            _ => Vec::new(),
        }
    }

    /// Finite minimum and maximum of the kept values.
    pub fn range(&self, series: usize, metric: usize) -> Option<(f32, f32)> {
        self.trace(series, metric)
            .into_iter()
            .filter(|v| v.is_finite())
            .fold(None, |acc, v| Some(acc.map_or((v, v), |(lo, hi): (f32, f32)| (lo.min(v), hi.max(v)))))
    }
}

// --- CSV ---------------------------------------------------------------------

/// `frame,time,series,<metric ids>`.
pub fn csv_header(metrics: &[MetricDesc]) -> String {
    let mut line = String::from("frame,time,series");
    for metric in metrics {
        line.push(',');
        line.push_str(metric.id);
    }
    line.push('\n');
    line
}

/// One line per series recorded in `sample`, with the first `metric_count` lanes.
pub fn csv_rows(sample: &Sample, metric_count: usize) -> String {
    let mut lines = String::new();
    for series in 0..sample.series.min(MAX_SERIES) {
        let _ = write!(lines, "{},{:.6},{series}", sample.frame, sample.time);
        for value in &sample.values[series][..metric_count.min(MAX_METRICS)] {
            let _ = write!(lines, ",{value:.6}");
        }
        lines.push('\n');
    }
    lines
}

/// A CSV file receiving every sample of a run.
pub struct CsvLog {
    path: PathBuf,
    writer: BufWriter<File>,
    metric_count: usize,
    rows: u64,
}

impl CsvLog {
    /// Creates the file (and its folder) and writes the header.
    pub fn create(path: &Path, metrics: &[MetricDesc]) -> Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writer.write_all(csv_header(metrics).as_bytes()).with_context(|| format!("writing {}", path.display()))?;
        Ok(Self { path: path.to_path_buf(), writer, metric_count: metrics.len(), rows: 0 })
    }

    pub fn write(&mut self, sample: &Sample) -> Result<()> {
        self.writer
            .write_all(csv_rows(sample, self.metric_count).as_bytes())
            .with_context(|| format!("writing {}", self.path.display()))?;
        self.rows += sample.series.min(MAX_SERIES) as u64;
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Data rows written so far (one per series per frame).
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Flushes and closes the file.
    pub fn finish(mut self) -> Result<(PathBuf, u64)> {
        self.writer.flush().with_context(|| format!("writing {}", self.path.display()))?;
        Ok((self.path, self.rows))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(frame: u64, series: usize, base: f32) -> Sample {
        let mut values = [[0.0; MAX_METRICS]; MAX_SERIES];
        for (s, lanes) in values.iter_mut().enumerate() {
            for (k, v) in lanes.iter_mut().enumerate() {
                *v = base + s as f32 * 100.0 + k as f32;
            }
        }
        Sample { frame, time: frame as f32 / 60.0, series, values }
    }

    #[test]
    fn history_keeps_every_point_until_full_then_halves_its_resolution() {
        let mut history = History::new(8);
        assert!(history.latest_frame().is_none());
        for frame in 0..8 {
            history.push(&sample(frame, 1, frame as f32));
        }
        assert_eq!(history.stride(), 1);
        assert_eq!(history.trace(0, 0), (0..8).map(|f| f as f32).collect::<Vec<_>>());
        assert_eq!(history.trace(0, 3), (0..8).map(|f| f as f32 + 3.0).collect::<Vec<_>>());
        assert_eq!(history.frame_span(), Some((0, 7)));

        // The ninth point overflows: every other point is kept and the stride doubles.
        history.push(&sample(8, 1, 8.0));
        assert_eq!(history.stride(), 2);
        assert_eq!(history.trace(0, 0), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        // Only every second frame is recorded from now on, but the latest value is always current.
        history.push(&sample(9, 1, 9.0));
        assert_eq!(history.trace(0, 0), vec![0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(history.latest(0, 0), Some(9.0));
        assert_eq!(history.latest_frame(), Some(9));
        history.push(&sample(10, 1, 10.0));
        assert_eq!(history.trace(0, 0), vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
        assert_eq!(history.range(0, 0), Some((0.0, 10.0)));
        assert!(history.trace(1, 0).is_empty());
        assert_eq!(history.latest(1, 0), None);

        history.clear();
        assert!(history.latest_frame().is_none());
        assert_eq!(history.stride(), 1);
        assert!(history.trace(0, 0).is_empty());
        assert_eq!(history.range(0, 0), None);
    }

    #[test]
    fn history_retires_a_series_that_stops_reporting_and_ignores_non_finite_ranges() {
        let mut history = History::new(16);
        history.push(&sample(0, 2, 1.0));
        history.push(&sample(1, 2, 2.0));
        assert_eq!(history.series(), 2);
        assert_eq!(history.trace(1, 0), vec![101.0, 102.0]);
        history.push(&sample(2, 1, 3.0));
        assert_eq!(history.series(), 1);
        assert!(history.trace(1, 0).is_empty());
        assert_eq!(history.trace(0, 0), vec![1.0, 2.0, 3.0]);
        // A comparison starting means both habitats restarted: the run begins anew.
        history.push(&sample(3, 2, 4.0));
        assert_eq!(history.series(), 2);
        assert_eq!(history.trace(0, 0), vec![4.0]);
        assert_eq!(history.trace(1, 0), vec![104.0]);
        history.push(&sample(4, 1, 5.0));

        let mut nan = sample(5, 1, 6.0);
        nan.values[0][0] = f32::NAN;
        history.push(&nan);
        assert_eq!(history.range(0, 0), Some((4.0, 5.0)));
        assert!(history.latest(0, 0).unwrap().is_nan());
    }

    #[test]
    fn csv_uses_metric_ids_and_six_decimals_and_one_row_per_series() {
        let metrics = [
            MetricDesc { id: "cover", label: "Cover", unit: Unit::Fraction, hint: "" },
            MetricDesc { id: "mean_v", label: "Mean V", unit: Unit::Scalar, hint: "" },
        ];
        assert_eq!(csv_header(&metrics), "frame,time,series,cover,mean_v\n");
        let mut s = sample(12, 2, 0.5);
        s.values[1][1] = f32::NAN;
        assert_eq!(
            csv_rows(&s, metrics.len()),
            "12,0.200000,0,0.500000,1.500000\n12,0.200000,1,100.500000,NaN\n"
        );
        assert_eq!(csv_rows(&sample(3, 1, 0.0), 0), "3,0.050000,0\n");

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("run.csv");
        let mut log = CsvLog::create(&path, &metrics).unwrap();
        log.write(&sample(0, 1, 0.25)).unwrap();
        log.write(&sample(1, 2, 0.75)).unwrap();
        assert_eq!(log.rows(), 3);
        let (written, rows) = log.finish().unwrap();
        assert_eq!((written.as_path(), rows), (path.as_path(), 3));
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "frame,time,series,cover,mean_v");
        assert_eq!(lines[1], "0,0.000000,0,0.250000,1.250000");
        assert_eq!(lines[3], "1,0.016667,1,100.750000,101.750000");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn units_format_compactly_and_survive_non_finite_values() {
        assert_eq!(Unit::Fraction.format(0.01234), "1.23%");
        assert_eq!(Unit::Fraction.format(0.5), "50.0%");
        assert_eq!(Unit::Fraction.format(1.0), "100.0%");
        assert_eq!(Unit::Scalar.format(0.0), "0");
        assert_eq!(Unit::Scalar.format(1234.6), "1235");
        assert_eq!(Unit::Scalar.format(12.345), "12.3");
        assert_eq!(Unit::Scalar.format(-1.2345), "-1.23");
        assert_eq!(Unit::Scalar.format(0.01234), "0.012");
        assert_eq!(Unit::Scalar.format(0.00001234), "1.23e-5");
        assert_eq!(Unit::Scalar.format(f32::NAN), "—");
        assert_eq!(Unit::Fraction.format(f32::INFINITY), "—");
    }

    /// Partial record `g`, lane `k`: multiples of 1/8 whose sums stay exactly
    /// representable, so the GPU total must match bit for bit.
    fn partial(g: u32, k: usize) -> f32 {
        ((g % 13) as f32 + k as f32) * 0.125
    }

    #[test]
    fn gpu_reduce_sums_partials_exactly_and_the_ring_returns_frames_in_order() {
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let reduction = Reduction::new(&gpu, "test", 1000);
        assert_eq!(reduction.capacity, 1000);
        let partials: Vec<f32> = (0..1000).flat_map(|g| (0..MAX_METRICS).map(move |k| partial(g, k))).collect();
        gpu.queue.write_buffer(reduction.partials(), 0, bytemuck::cast_slice(&partials));
        let expected = |count: u32| -> [f32; MAX_METRICS] {
            std::array::from_fn(|k| (0..count).map(|g| partial(g, k) as f64).sum::<f64>() as f32)
        };

        let mut sampler = Sampler::new(&gpu, 2);
        for (frame, count) in [(7, 1000), (8, 1)] {
            let mut encoder = gpu.device.create_command_encoder(&Default::default());
            reduction.record(&gpu, &mut encoder, count);
            let mut sink = sampler.begin(frame, frame as f32 * 0.5);
            assert!(sink.is_live());
            sink.push(&mut encoder, reduction.totals());
            gpu.queue.submit([encoder.finish()]);
            sampler.map();
        }
        // Both slots are in flight: the next frame is dropped, not blocked on.
        assert!(sampler.is_full());
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        let mut dead = sampler.begin(9, 4.5);
        assert!(!dead.is_live());
        dead.push(&mut encoder, reduction.totals());
        let _ = &dead;
        gpu.queue.submit([encoder.finish()]);
        sampler.map();
        assert_eq!(sampler.dropped(), 1);

        let samples = sampler.flush(&gpu).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!((samples[0].frame, samples[0].time, samples[0].series), (7, 3.5, 1));
        assert_eq!(samples[0].values[0], expected(1000));
        assert_eq!((samples[1].frame, samples[1].series), (8, 1));
        assert_eq!(samples[1].values[0], expected(1));
        assert!(!sampler.is_full());

        // Two series per frame land in two records; a zero count reads all zeros;
        // discarded samples never come back but free their slots.
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        reduction.record(&gpu, &mut encoder, 0);
        let mut sink = sampler.begin(10, 5.0);
        sink.push(&mut encoder, reduction.totals());
        sink.push(&mut encoder, reduction.totals());
        gpu.queue.submit([encoder.finish()]);
        sampler.map();
        let samples = sampler.flush(&gpu).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].series, 2);
        assert_eq!(samples[0].values, [[0.0; MAX_METRICS]; MAX_SERIES]);

        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        reduction.record(&gpu, &mut encoder, 2);
        let mut sink = sampler.begin(11, 5.5);
        sink.push(&mut encoder, reduction.totals());
        gpu.queue.submit([encoder.finish()]);
        sampler.map();
        sampler.discard();
        assert!(sampler.flush(&gpu).unwrap().is_empty());
        assert!(!sampler.is_full());
        // A frame that pushes nothing releases its slot without a mapping.
        sampler.begin(12, 6.0);
        sampler.map();
        assert!(sampler.flush(&gpu).unwrap().is_empty());
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    }

    #[test]
    fn every_world_publishes_a_valid_metric_table_and_finite_measurements() {
        use crate::world::{Frame, ViewXform, WORLDS};
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let mut sampler = Sampler::new(&gpu, 2);
        for entry in WORLDS {
            let size = [96, 64];
            let (_, mut world) = crate::world::create(&gpu, entry.id, size, None, 1).unwrap();
            let metrics = world.metrics();
            assert!(metrics.len() <= MAX_METRICS, "{}: too many metrics", entry.id);
            for (i, metric) in metrics.iter().enumerate() {
                assert!(
                    !metric.id.is_empty()
                        && metric.id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'),
                    "{}: metric id '{}' is not a CSV-friendly identifier",
                    entry.id,
                    metric.id
                );
                assert!(metrics[..i].iter().all(|m| m.id != metric.id), "{}: duplicate id '{}'", entry.id, metric.id);
                assert!(!metric.label.is_empty() && !metric.hint.is_empty(), "{}: {} lacks a label or hint", entry.id, metric.id);
            }
            let view = ViewXform::fit(world.size(), size, &crate::world::Camera::default());
            for frame in 0..3u64 {
                let frame = Frame {
                    gpu: &gpu,
                    time: frame as f32 / 60.0,
                    dt: 1.0 / 60.0,
                    frame,
                    view,
                    target_size: size,
                    pointer: None,
                };
                let mut encoder = gpu.device.create_command_encoder(&Default::default());
                world.step(&frame, &mut encoder);
                let mut sink = sampler.begin(frame.frame, frame.time);
                world.measure(&frame, &mut encoder, &mut sink);
                gpu.queue.submit([encoder.finish()]);
                sampler.map();
                let samples = sampler.flush(&gpu).unwrap();
                assert!(gpu.fatal_error().is_none(), "{}: {:?}", entry.id, gpu.fatal_error());
                if metrics.is_empty() {
                    assert!(samples.is_empty(), "{}: measurements without a metric table", entry.id);
                    continue;
                }
                assert_eq!(samples.len(), 1, "{}: one sample per frame", entry.id);
                let sample = &samples[0];
                assert_eq!(sample.frame, frame.frame);
                assert!((1..=MAX_SERIES).contains(&sample.series));
                for series in 0..sample.series {
                    for (metric, value) in metrics.iter().zip(sample.values[series]) {
                        assert!(value.is_finite(), "{}: {} is {value}", entry.id, metric.id);
                        if metric.unit == Unit::Fraction {
                            assert!((0.0..=1.0).contains(&value), "{}: {} = {value} is not a fraction", entry.id, metric.id);
                        }
                    }
                }
            }
        }
    }
}
