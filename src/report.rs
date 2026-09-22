//! What the CLI prints on stdout: the `primordia list` catalogue and the result
//! of each headless command, as text or as one JSON object (`--json`); and the
//! JSON explore leaves in its folder (`run.json`, the recipes' `provenance`).
//!
//! Conventions of every JSON object: seeds are strings (a `u64` does not fit a
//! JSON number exactly), including the seed of a recipe object; presets are
//! `{"index", "name", "slug"}` with the 1-based index `--preset` takes
//! (recipes store it 0-based), paths are printed as given, and `"ok"` says
//! whether the command succeeded (failures come from [`crate::failure::json`]).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::explore::{self, Candidate, Origin};
use crate::headless::{self, GallerySummary, RenderJob, RenderSummary};
use crate::library::{self, SavedWorld};
use crate::metrics::MetricDesc;
use crate::recipe::{RecipeJob, RecipeSummary, Setting};
use crate::world::WORLDS;

/// Version of the `list --json` layout; bumped only when fields change meaning or go away.
pub const LIST_SCHEMA: u32 = 1;

fn path(path: &Path) -> Value {
    Value::String(path.display().to_string())
}

fn optional_path(p: Option<&Path>) -> Value {
    p.map_or(Value::Null, path)
}

/// An `f32` with the digits it actually has (0.1589202, not 0.15892019867897034);
/// `null` when it is not finite.
fn number(value: f32) -> Value {
    value.to_string().parse::<f64>().ok().and_then(serde_json::Number::from_f64).map_or(Value::Null, Value::Number)
}

/// `{"index" (1-based), "name", "slug"}` of preset `index` (0-based) of world `world`.
pub fn preset(world: usize, index: usize) -> Value {
    let name = (WORLDS[world].presets)().get(index).copied().unwrap_or("custom");
    json!({ "index": index + 1, "name": name, "slug": headless::slug(name) })
}

/// A measurement's description: id (the CSV column), label, unit, range, hint
/// and whether it is vital (explore counts a candidate whose vital
/// measurements stay near zero as inert).
pub fn metric(metric: &MetricDesc, vital: bool) -> Value {
    json!({
        "id": metric.id,
        "label": metric.label,
        "unit": metric.unit.name(),
        "range": metric.unit.range(),
        "hint": metric.hint,
        "vital": vital,
    })
}

/// Everything `list` knows about world `index`.
pub fn world(index: usize) -> Value {
    let entry = &WORLDS[index];
    let presets = (entry.presets)();
    json!({
        "index": index + 1,
        "id": entry.id,
        "name": entry.name,
        "aliases": entry.aliases,
        "tagline": entry.tagline,
        "presets": (0..presets.len()).map(|i| preset(index, i)).collect::<Vec<_>>(),
        "metrics": entry.metrics.iter().map(|m| metric(m, entry.vital.contains(&m.id))).collect::<Vec<_>>(),
        "palettes": (entry.palettes)(),
        "palette_setting": entry.palette_setting,
    })
}

/// `list --json`: every world (or only `only`), the tonemappers and where the library lives.
pub fn list_json(only: Option<usize>, tonemaps: &[&str]) -> Value {
    let worlds: Vec<Value> = (0..WORLDS.len()).filter(|&i| only.is_none_or(|o| o == i)).map(world).collect();
    json!({
        "schema": LIST_SCHEMA,
        "version": env!("CARGO_PKG_VERSION"),
        "recipe_version": library::RECIPE_VERSION,
        "library_dir": path(&library::default_directory()),
        "tonemaps": tonemaps,
        "worlds": worlds,
    })
}

/// What a world's palette list is and where a recipe chooses from it.
fn palette_heading(setting: &str) -> String {
    match setting {
        "palettes" => "Palettes (recipe field `palettes`, one per channel)".to_string(),
        "params.colors" => "Colour schemes (recipe field `params.colors`, by 0-based index)".to_string(),
        other => format!("Palettes (recipe field `{other}`)"),
    }
}

/// `list` for people: every world (or only `only`) with its presets,
/// measurements and palettes.
pub fn list_text(only: Option<usize>, tonemaps: &[&str]) -> String {
    let mut text = String::from("Worlds (open one with `primordia --world <id> --preset <name or number>`):\n");
    for (index, entry) in WORLDS.iter().enumerate() {
        if only.is_some_and(|o| o != index) {
            continue;
        }
        let _ = writeln!(text, "\n{}. {} ({})\n   {}", index + 1, entry.name, entry.id, entry.tagline);
        let _ = writeln!(text, "   Aliases: {}", entry.aliases.join(", "));
        let _ = writeln!(text, "   Presets:");
        for (i, name) in (entry.presets)().iter().enumerate() {
            let _ = writeln!(text, "     {:>2}. {name}", i + 1);
        }
        let _ = writeln!(text, "   Measurements (CSV columns; * = vital):");
        let width = entry.metrics.iter().map(|m| m.id.len() + 1).max().unwrap_or(0);
        for m in entry.metrics {
            let id = if entry.vital.contains(&m.id) { format!("{}*", m.id) } else { m.id.to_string() };
            let _ = writeln!(text, "     {id:<width$}  {:<8}  {}", m.unit.name(), m.hint);
        }
        let _ = writeln!(text, "   {}:\n     {}", palette_heading(entry.palette_setting), (entry.palettes)().join(", "));
    }
    let _ = writeln!(
        text,
        "\nA fraction is a share of cells or agents in 0-1; a scalar is unbounded. Explore counts a candidate as \
         inert when a\nvital (*) measurement stays near zero. Presets are numbered from 1 here and in --preset; \
         recipes store them from 0."
    );
    let _ = writeln!(text, "\nTonemaps: {}", tonemaps.join(", "));
    let _ = writeln!(text, "Library: {} (PRIMORDIA_LIBRARY_DIR overrides)", library::default_directory().display());
    text
}

/// The `--set` arguments as given.
fn settings(sets: &[Setting]) -> Value {
    Value::Array(sets.iter().map(|s| Value::String(s.to_string())).collect())
}

/// A recipe as JSON, with its seed as a string like every other seed here
/// (a recipe file keeps it a number; both load).
pub fn recipe_json(saved: &SavedWorld) -> Value {
    let mut value = serde_json::to_value(saved).unwrap_or(Value::Null);
    value["seed"] = Value::String(saved.seed.to_string());
    value
}

/// `recipe --json`: the complete recipe and the file it was written to.
pub fn recipe(summary: &RecipeSummary, job: &RecipeJob) -> Value {
    let saved = &summary.saved;
    json!({
        "ok": true,
        "command": "recipe",
        "world": WORLDS[summary.world].id,
        "preset": preset(summary.world, saved.preset),
        "source": optional_path(job.recipe.as_ref().map(|r| r.path.as_path())),
        "set": settings(&job.sets),
        "modified": saved.modified,
        "seed": saved.seed.to_string(),
        "size": saved.output_size,
        "file": optional_path(summary.file.as_deref()),
        "recipe": recipe_json(saved),
    })
}

/// `render --json`.
pub fn render(summary: &RenderSummary, job: &RenderJob) -> Value {
    let world = summary.world;
    let metrics = summary.metrics.as_ref().map(|log| {
        let ids: Vec<&str> = WORLDS[world].metrics.iter().map(|m| m.id).collect();
        let last: Vec<Value> = log
            .last
            .iter()
            .flat_map(|sample| (0..sample.series).map(move |s| sample.values[s]))
            .map(|values| {
                let fields = ids.iter().zip(values).map(|(id, v)| ((*id).to_string(), number(v)));
                Value::Object(fields.collect())
            })
            .collect();
        json!({ "rows": log.rows, "series": log.series, "last": last })
    });
    json!({
        "ok": true,
        "command": "render",
        "world": WORLDS[world].id,
        "preset": preset(world, summary.preset),
        "source": optional_path(job.recipe.as_ref().map(|r| r.path.as_path())),
        "set": settings(&job.sets),
        "modified": summary.modified,
        "seed": job.seed.to_string(),
        "size": summary.size,
        "frames": summary.frames,
        "fps": job.fps,
        "files": {
            "png": optional_path(summary.png.as_deref()),
            "video": optional_path(summary.video.as_deref()),
            "frames": summary.frame_files.iter().map(|p| path(p)).collect::<Vec<_>>(),
            "metrics": optional_path(summary.metrics.as_ref().map(|m| m.path.as_path())),
            "recipe": optional_path(summary.recipe.as_deref()),
        },
        "metrics": metrics,
        "secs": number(summary.secs),
    })
}

/// `gallery --json`.
pub fn gallery(summary: &GallerySummary, seed: u64, size: [u32; 2], frames: u32) -> Value {
    let images: Vec<Value> = summary
        .images
        .iter()
        .map(|(world, index, image)| json!({ "world": WORLDS[*world].id, "preset": preset(*world, *index), "path": path(image) }))
        .collect();
    json!({
        "ok": true,
        "command": "gallery",
        "seed": seed.to_string(),
        "size": size,
        "frames": frames,
        "images": images,
        "sheet": optional_path(summary.sheet.as_deref()),
        "secs": number(summary.secs),
    })
}

/// The parent of a child candidate.
fn parent(candidate: &Candidate) -> Option<usize> {
    match candidate.origin {
        Origin::Child { parent } => Some(parent),
        _ => None,
    }
}

/// The kept candidates of an exploration in rank order; `file` prints the
/// path of an image or recipe.
fn explore_kept(summary: &explore::Summary, file: impl Fn(&Path) -> Value) -> Vec<Value> {
    let file = |p: Option<&PathBuf>| p.map_or(Value::Null, |p| file(p));
    summary
        .kept
        .iter()
        .enumerate()
        .map(|(rank, &i)| {
            let c = &summary.candidates[i];
            json!({
                "rank": rank + 1,
                "candidate": c.index,
                "round": c.round,
                "origin": c.origin.name(),
                "parent": parent(c),
                "seed": c.seed.to_string(),
                "preset": preset(summary.world, c.preset),
                "novelty": number(c.novelty),
                "image": file(summary.images.get(rank)),
                "recipe": file(summary.recipes.get(rank)),
                "installed": optional_path(summary.installed.get(rank).map(|p| p.as_path())),
            })
        })
        .collect()
}

/// `explore --json`: the kept candidates in rank order, with their files.
pub fn explore(summary: &explore::Summary, job: &explore::ExploreJob) -> Value {
    let world = summary.world;
    json!({
        "ok": true,
        "command": "explore",
        "world": WORLDS[world].id,
        "base": {
            "preset": preset(world, summary.preset),
            "source": optional_path(job.recipe.as_ref().map(|r| r.path.as_path())),
            "set": settings(&job.sets),
            "seed": summary.base_seed.to_string(),
        },
        "seed": job.seed.to_string(),
        "select": job.select.to_string(),
        "evaluated": summary.candidates.len(),
        "inert": summary.candidates.iter().filter(|c| c.inert).count(),
        "kept": explore_kept(summary, path),
        "files": {
            "out_dir": path(&summary.out_dir),
            "csv": path(&summary.csv),
            "sheet": optional_path(summary.sheet.as_deref()),
            "all": summary.all.iter().map(|p| path(p)).collect::<Vec<_>>(),
            "manifest": path(&summary.manifest),
        },
        "library": optional_path(job.library.as_deref()),
        "secs": number(summary.secs),
    })
}

/// Version of the `run.json` layout; bumped only when fields change meaning or go away.
pub const MANIFEST_SCHEMA: u32 = 1;

/// `run.json`, the record an exploration leaves in its folder: the command
/// line, version, GPU and settings, what every column of candidates.csv
/// holds, the kept candidates, and `files`, every file the run wrote into the
/// folder. Those paths (and the kept candidates' images and recipes) are
/// relative to the folder, with `/` between their parts; the next run into
/// the folder removes exactly them. A failed run that wrote files leaves
/// `"complete": false`, its `error` and no `kept`.
pub fn explore_manifest(job: &explore::ExploreJob, record: &explore::Record) -> Value {
    let entry = &WORLDS[record.world];
    let fixed = explore::CSV_COLUMNS.iter().map(|(name, meaning)| json!({ "name": name, "meaning": meaning }));
    let descriptor = entry.metrics.iter().flat_map(|m| {
        explore::DESCRIPTOR_STATS.iter().map(move |(stat, meaning)| {
            json!({
                "name": format!("{}_{stat}", m.id),
                "meaning": meaning,
                "stat": stat,
                "metric": metric(m, entry.vital.contains(&m.id)),
            })
        })
    });
    let base = if job.recipe.is_some() || !job.sets.is_empty() { Origin::Recipe } else { Origin::Preset };
    let mut value = json!({
        "tool": explore::MANIFEST_TOOL,
        "schema": MANIFEST_SCHEMA,
        "version": env!("CARGO_PKG_VERSION"),
        "complete": record.outcome.is_ok(),
        "error": record.outcome.as_ref().err(),
        "argv": job.argv,
        "started": record.started,
        "secs": number(record.secs),
        "gpu": record.gpu,
        "world": { "index": record.world + 1, "id": entry.id, "name": entry.name },
        "base": {
            "origin": base.name(),
            "preset": preset(record.world, record.preset),
            "recipe": optional_path(job.recipe.as_ref().map(|r| r.path.as_path())),
            "set": settings(&job.sets),
            "seed": record.base_seed.to_string(),
        },
        "settings": {
            "seed": job.seed.to_string(),
            "runs": job.runs,
            "refine": job.refine,
            "children": job.children,
            "strength": number(job.strength),
            "keep": job.keep,
            "frames": job.frames,
            "size": job.size,
            "max_fps": number(job.max_fps),
            "select": job.select.to_string(),
            "keep_inert": job.inert,
            "all": job.all,
            "sheet": job.sheet,
            "install": optional_path(job.library.as_deref()),
        },
        "csv": { "file": "candidates.csv", "columns": fixed.chain(descriptor).collect::<Vec<_>>() },
        "files": record.files,
    });
    if let Ok(summary) = record.outcome {
        let count = |status: &str| summary.candidates.iter().filter(|c| c.status() == status).count();
        value["evaluated"] = json!(summary.candidates.len());
        value["inert"] = json!(count("inert"));
        value["failed"] = json!(count("failed"));
        let file = |p: &Path| explore::relative(&summary.out_dir, p).map_or_else(|| path(p), Value::String);
        value["kept"] = Value::Array(explore_kept(summary, file));
    }
    value
}

/// The `provenance` object of a kept recipe: which exploration found it and
/// how. Seeds are strings; `parent` is the candidate a child perturbs, whose
/// seed it reuses.
pub fn explore_provenance(job: &explore::ExploreJob, candidates: &[Candidate], index: usize) -> Value {
    let c = &candidates[index];
    json!({
        "tool": explore::MANIFEST_TOOL,
        "version": env!("CARGO_PKG_VERSION"),
        "run_seed": job.seed.to_string(),
        "select": job.select.to_string(),
        "candidate": c.index,
        "round": c.round,
        "rank": c.rank,
        "novelty": number(c.novelty),
        "origin": c.origin.name(),
        "parent": parent(c),
        "parent_seed": parent(c).map(|p| candidates[p].seed.to_string()),
    })
}

/// `time` in UTC as RFC 3339, to the second: "2026-09-22T14:59:49Z".
pub fn utc_time(time: std::time::SystemTime) -> String {
    let secs = time.duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs() as i64);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Howard Hinnant's civil-from-days algorithm.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!("{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem / 60 % 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{Sample, MAX_METRICS, MAX_SERIES};
    use std::path::PathBuf;

    #[test]
    fn list_json_describes_every_world_without_a_gpu() {
        let value = list_json(None, &["agx", "aces"]);
        // Through text and back, as a script would read it.
        let value: Value = serde_json::from_str(&value.to_string()).unwrap();
        assert_eq!(value["schema"], 1);
        assert_eq!(value["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(value["recipe_version"], 1);
        assert!(value["library_dir"].is_string());
        assert_eq!(value["tonemaps"], json!(["agx", "aces"]));
        let worlds = value["worlds"].as_array().unwrap();
        assert_eq!(worlds.len(), WORLDS.len());
        for (i, (world, entry)) in worlds.iter().zip(WORLDS).enumerate() {
            assert_eq!(world["index"], i + 1);
            assert_eq!(world["id"], entry.id);
            assert_eq!(world["aliases"], json!(entry.aliases));
            let presets = world["presets"].as_array().unwrap();
            assert_eq!(presets.len(), (entry.presets)().len());
            assert_eq!(presets[0]["index"], 1);
            assert_eq!(presets[0]["name"], (entry.presets)()[0]);
            assert_eq!(presets[0]["slug"], headless::slug((entry.presets)()[0]));
            let metrics = world["metrics"].as_array().unwrap();
            assert_eq!(metrics.len(), entry.metrics.len());
            for (m, desc) in metrics.iter().zip(entry.metrics) {
                assert_eq!(m["id"], desc.id);
                assert_eq!(m["vital"], entry.vital.contains(&desc.id));
                match m["unit"].as_str().unwrap() {
                    "fraction" => assert_eq!(m["range"], json!([0.0, 1.0])),
                    "scalar" => assert!(m["range"].is_null()),
                    other => panic!("unexpected unit {other}"),
                }
                assert!(!m["hint"].as_str().unwrap().is_empty() && !m["label"].as_str().unwrap().is_empty());
            }
            assert!(!world["palettes"].as_array().unwrap().is_empty());
        }
        // Physarum lists its own palettes (which its presets use) before the shared ones.
        let physarum = &worlds[0]["palettes"];
        assert_eq!(physarum[1], "Frost");
        assert!(physarum.as_array().unwrap().contains(&json!("Bioluminescence")));

        let lenia = list_json(Some(2), &[]);
        assert_eq!(lenia["worlds"].as_array().unwrap().len(), 1);
        assert_eq!(lenia["worlds"][0]["id"], "lenia");
        assert_eq!(lenia["worlds"][0]["palette_setting"], "palettes");
    }

    #[test]
    fn times_are_utc_in_rfc_3339() {
        let at = |secs: u64| utc_time(std::time::UNIX_EPOCH + std::time::Duration::from_secs(secs));
        assert_eq!(at(0), "1970-01-01T00:00:00Z");
        assert_eq!(at(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(at(1_790_000_000), "2026-09-21T14:13:20Z");
        assert_eq!(at(4_102_444_799), "2099-12-31T23:59:59Z");
    }

    #[test]
    fn list_text_shows_aliases_numbered_presets_and_measurements() {
        let text = list_text(Some(3), &["agx"]);
        assert!(text.contains("4. Reaction-Diffusion (reaction-diffusion)"), "{text}");
        assert!(text.contains("Aliases: rd, gray-scott"), "{text}");
        assert!(text.contains(" 2. Mitosis"), "{text}");
        assert!(text.contains("alive*"), "vital measurements are marked: {text}");
        assert!(text.contains("v_drift "), "{text}");
        assert!(text.contains("Palettes (recipe field `palette`)"), "{text}");
        assert!(!text.contains("Physarum"), "only the chosen world: {text}");
        let all = list_text(None, &["agx"]);
        assert!(all.contains("Colour schemes (recipe field `params.colors`, by 0-based index)"));
        for entry in WORLDS {
            assert!(all.contains(&format!("{} ({})", entry.name, entry.id)), "{}", entry.id);
        }
    }

    #[test]
    fn render_results_use_string_seeds_one_based_presets_and_every_file() {
        let mut values = [[0.0; MAX_METRICS]; MAX_SERIES];
        values[0][0] = 0.25;
        values[0][1] = f32::NAN;
        let summary = RenderSummary {
            world: 3,
            preset: 1,
            size: [320, 180],
            frames: 60,
            modified: true,
            png: Some(PathBuf::from("r/mitosis.png")),
            video: None,
            frame_files: vec![PathBuf::from("frames/reaction-diffusion_00030.png")],
            metrics: Some(headless::MetricsLog {
                path: PathBuf::from("r/mitosis.csv"),
                rows: 60,
                series: None,
                last: Some(Sample { frame: 59, time: 59.0 / 60.0, series: 1, values }),
            }),
            recipe: Some(PathBuf::from("r/mitosis.json")),
            secs: 0.5,
        };
        let sets = vec!["params.feed=0.031".parse().unwrap(), "palette=Frost".parse().unwrap()];
        let job = RenderJob { seed: u64::MAX, fps: 30, sets, ..RenderJob::new("rd") };
        let value = render(&summary, &job);
        assert_eq!(value["ok"], true);
        assert_eq!(value["command"], "render");
        assert_eq!(value["world"], "reaction-diffusion");
        assert_eq!(value["preset"], json!({ "index": 2, "name": "Mitosis", "slug": "mitosis" }));
        assert_eq!(value["seed"], "18446744073709551615");
        assert_eq!((value["size"].clone(), value["fps"].clone()), (json!([320, 180]), json!(30)));
        assert_eq!(value["files"]["png"], "r/mitosis.png");
        assert!(value["files"]["video"].is_null());
        assert_eq!(value["files"]["frames"][0], "frames/reaction-diffusion_00030.png");
        assert_eq!(value["files"]["metrics"], "r/mitosis.csv");
        assert_eq!(value["files"]["recipe"], "r/mitosis.json");
        assert_eq!((value["modified"].clone(), value["source"].clone()), (json!(true), Value::Null));
        assert_eq!(value["set"], json!(["params.feed=0.031", "palette=Frost"]));
        assert_eq!(value["metrics"]["rows"], 60);
        let last = &value["metrics"]["last"][0];
        assert_eq!(last["alive"], 0.25);
        assert_eq!(number(0.1589202).to_string(), "0.1589202", "f32 values keep their own digits");
        assert!(number(f32::INFINITY).is_null());
        assert!(last["body"].is_null(), "non-finite measurements become null");
        assert_eq!(last.as_object().unwrap().len(), WORLDS[3].metrics.len());
        let files: Vec<&Path> = summary.files();
        assert_eq!(files.len(), 4);
        assert_eq!(files[3], Path::new("r/mitosis.json"), "the recipe is written last");
    }
}
