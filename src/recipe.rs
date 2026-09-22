//! Recipes on the command line: `--recipe FILE` (a library or explore `.json`
//! file, or a PNG written by Primordia, which carries its recipe), `--set
//! KEY=VALUE` edits, and `primordia recipe`, which prints a complete recipe.
//!
//! A `--set` key is a path into a recipe's `settings` (the JSON of
//! [`WorldSettings`]): object keys and list indices joined by dots, such as
//! `params.feed`, `palette` or `params.kernels.0.mu`. The value is JSON, or
//! plain text when it does not parse as JSON (`palette=Frost`). The `post.*`
//! keys edit the world's own look, and a recipe's look follows them.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context as _, Result};
use serde_json::Value;

use crate::failure::Failure;
use crate::gpu::Gpu;
use crate::headless;
use crate::library::{self, SavedWorld, WorldSettings};
use crate::post::PostSettings;
use crate::world::{self, Camera, World, WORLDS};

/// One `--set KEY=VALUE`.
#[derive(Clone, Debug, PartialEq)]
pub struct Setting {
    /// Dotted path into the recipe's settings, e.g. `params.feed`.
    pub key: String,
    /// The value: its JSON, or a string when the text is not JSON.
    pub value: Value,
    /// The value as typed, used as it is where the setting holds text.
    pub text: String,
}

impl FromStr for Setting {
    type Err = String;

    fn from_str(arg: &str) -> Result<Self, String> {
        let (key, text) = arg
            .split_once('=')
            .ok_or_else(|| format!("expected KEY=VALUE, such as params.feed=0.031 or palette=Frost (got '{arg}')"))?;
        let parts: Vec<&str> = key.split('.').map(str::trim).collect();
        if parts.iter().any(|part| part.is_empty()) {
            return Err(format!("'{key}' is not a setting: use keys joined by dots, such as params.feed"));
        }
        let value = serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()));
        Ok(Self { key: parts.join("."), value, text: text.to_string() })
    }
}

impl fmt::Display for Setting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.key, self.text)
    }
}

/// What a JSON value is, in the words of an error message.
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "nothing (null)",
        Value::Bool(_) => "true or false",
        Value::Number(_) => "a number",
        Value::String(_) => "text",
        Value::Array(_) => "a list",
        Value::Object(_) => "a group of settings",
    }
}

fn normalize(s: &str) -> String {
    s.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_lowercase()).collect()
}

/// The key in `keys` that `query` looks like a typo of.
fn suggest<'a>(query: &str, keys: &[&'a str]) -> Option<&'a str> {
    let q = normalize(query);
    keys.iter()
        .map(|key| (*key, strsim::jaro_winkler(&q, &normalize(key))))
        .filter(|(_, score)| *score >= 0.8)
        .min_by(|a, b| b.1.total_cmp(&a.1))
        .map(|(key, _)| key)
}

/// Sets `setting` inside `root` (the JSON of a world's settings): the path
/// must exist, and the new value must be of the old one's kind (a text setting
/// takes the value as typed, so a number typed for it stays text).
fn edit(root: &mut Value, setting: &Setting, world: usize) -> std::result::Result<(), String> {
    let key = &setting.key;
    let path: Vec<&str> = key.split('.').collect();
    if path[0] == "world" {
        return Err(format!(
            "'world' is the world a recipe belongs to ({}) and cannot be set; use --world or another recipe",
            WORLDS[world].id
        ));
    }
    let mut here = root;
    for (depth, part) in path.iter().enumerate() {
        let parent = path[..depth].join(".");
        here = match here {
            Value::Object(fields) => {
                if !fields.contains_key(*part) {
                    let keys: Vec<&str> =
                        fields.keys().map(String::as_str).filter(|k| depth > 0 || *k != "world").collect();
                    let owner = if parent.is_empty() { "the settings have".to_string() } else { format!("{parent} has") };
                    let prefix = if parent.is_empty() { String::new() } else { format!("{parent}.") };
                    let hint = suggest(part, &keys).map(|s| format!("; did you mean '{prefix}{s}'?")).unwrap_or_default();
                    return Err(format!(
                        "unknown setting '{key}' for {}{hint} ({owner}: {}; `primordia recipe -w {}` shows every setting)",
                        WORLDS[world].name,
                        keys.join(", "),
                        WORLDS[world].id
                    ));
                }
                fields.get_mut(*part).expect("the key exists")
            }
            Value::Array(items) => {
                let count = items.len();
                match part.parse::<usize>() {
                    Ok(index) if index < count => &mut items[index],
                    _ if count == 0 => return Err(format!("setting '{key}': {parent} is an empty list")),
                    _ => {
                        return Err(format!(
                            "setting '{key}': {parent} is a list of {count}, so '{part}' must be an index from 0 to {}",
                            count - 1
                        ));
                    }
                }
            }
            other => return Err(format!("setting '{key}': {parent} is {}, with nothing inside it", kind(other))),
        };
    }
    let value = match (&*here, &setting.value) {
        (Value::String(_), value) if !value.is_string() => Value::String(setting.text.clone()),
        (old, new) if std::mem::discriminant(old) == std::mem::discriminant(new) || old.is_null() || new.is_null() => {
            new.clone()
        }
        (old, new) => {
            // Settings are f32s widened to f64 in the JSON: show them with their own digits.
            let now = match old.as_f64() {
                Some(f) if old.is_f64() && f64::from(f as f32) == f => (f as f32).to_string(),
                _ => old.to_string(),
            };
            let now = if now.len() <= 40 { format!(" (it is {now} now)") } else { String::new() };
            return Err(format!("setting '{key}' takes {}{now}, not {} ({})", kind(old), kind(new), setting.text));
        }
    };
    *here = value;
    Ok(())
}

/// Checks the palette names of settings JSON against the world's registered
/// palettes and spells them as registered (`palette=frost` picks "Frost").
fn check_palettes(value: &mut Value, world: usize) -> std::result::Result<(), String> {
    let entry = &WORLDS[world];
    let names = (entry.palettes)();
    let slots: Vec<&mut Value> = match (entry.palette_setting, value) {
        ("palette", Value::Object(fields)) => fields.get_mut("palette").into_iter().collect(),
        ("palettes", Value::Object(fields)) => match fields.get_mut("palettes") {
            Some(Value::Array(items)) => items.iter_mut().collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    for name in slots {
        let Some(given) = name.as_str() else { continue };
        match names.iter().find(|n| n.eq_ignore_ascii_case(given)) {
            Some(registered) => *name = Value::String((*registered).to_string()),
            None => {
                let hint = suggest(given, &names).map(|s| format!("; did you mean '{s}'?")).unwrap_or_default();
                return Err(format!(
                    "unknown palette '{given}' for {}{hint} (palettes: {})",
                    entry.name,
                    names.join(", ")
                ));
            }
        }
    }
    Ok(())
}

/// The index in `WORLDS` of the world `settings` belong to.
fn world_of(settings: &WorldSettings) -> usize {
    world::find(settings.world_id()).expect("every recipe world is registered")
}

/// `settings` with `sets` applied in order. Every problem is invalid input.
pub fn apply(settings: &WorldSettings, sets: &[Setting]) -> Result<WorldSettings> {
    let world = world_of(settings);
    let mut value = serde_json::to_value(settings).context("serialising the recipe")?;
    let mut edited = settings.clone();
    for setting in sets {
        edit(&mut value, setting, world).map_err(|e| Failure::Usage.error(e))?;
        if setting.key.starts_with("palette") {
            check_palettes(&mut value, world).map_err(|e| Failure::Usage.error(e))?;
        }
        if !library::numbers_fit(&value) {
            return Err(Failure::Usage.error(format!(
                "setting '{setting}' is out of range: numbers must fit a 32-bit float"
            )));
        }
        edited = serde_json::from_value(value.clone()).map_err(|e| {
            Failure::Usage.error(format!("setting '{setting}' does not fit a {} recipe: {e}", WORLDS[world].name))
        })?;
    }
    Ok(edited)
}

/// Applies `sets` to a recipe: its settings, and the `post.*` ones to its look
/// too, which is what a render shows. The recipe counts as modified.
pub fn apply_to_recipe(saved: &mut SavedWorld, sets: &[Setting]) -> Result<()> {
    if sets.is_empty() {
        return Ok(());
    }
    saved.settings = apply(&saved.settings, sets)?;
    saved.look = follow_look(saved.look, sets, world_of(&saved.settings))?;
    saved.modified = true;
    Ok(())
}

/// `look` with the `post.*` settings of `sets` applied.
fn follow_look(look: PostSettings, sets: &[Setting], world: usize) -> Result<PostSettings> {
    let mut value = serde_json::to_value(look).context("serialising the look")?;
    for setting in sets {
        if setting.key == "post" {
            value = setting.value.clone();
        } else if let Some(key) = setting.key.strip_prefix("post.") {
            let inner = Setting { key: key.to_string(), ..setting.clone() };
            edit(&mut value, &inner, world).map_err(|e| Failure::Usage.error(e))?;
        }
    }
    serde_json::from_value(value).map_err(|e| Failure::Usage.error(format!("the edited look does not fit: {e}")))
}

/// Applies `sets` to a running world and restarts it from `seed`, under
/// [`world::guarded`] so that values the world rejects leave the GPU usable.
pub fn apply_to_world(gpu: &Gpu, world: &mut dyn World, sets: &[Setting], seed: u64) -> Result<()> {
    if sets.is_empty() {
        return Ok(());
    }
    let edited = apply(&world.settings()?, sets)?;
    let name = world.name();
    world::guarded(gpu, || world.restore_settings(gpu, &edited, seed))?
        .map_err(|e| Failure::Usage.tag(e.context(format!("{name} rejected the settings"))))
}

/// " with N settings changed" for a log line, or nothing without `--set`.
pub fn changed(sets: &[Setting]) -> String {
    match sets.len() {
        0 => String::new(),
        1 => " with 1 setting changed".to_string(),
        n => format!(" with {n} settings changed"),
    }
}

/// A recipe named on the command line (`--recipe FILE`).
#[derive(Clone, Debug)]
pub struct Recipe {
    pub path: PathBuf,
    pub saved: SavedWorld,
}

impl Recipe {
    /// Reads and checks the recipe in `path` (see [`library::load`]).
    pub fn load(path: &Path) -> Result<Self> {
        Ok(Self { path: path.to_owned(), saved: library::load(path)? })
    }
}

/// What a command starts from, resolved and checked before any GPU work.
#[derive(Clone, Debug)]
pub enum Source {
    /// A recipe, already given its seed and edited by `--set`.
    Recipe(Box<SavedWorld>),
    /// Preset `preset` (0-based; `None` = the first) of world `world`, which
    /// `--set` edits once the world exists.
    Preset { world: usize, preset: Option<usize> },
}

impl Source {
    /// The `recipe`, run from `seed` and edited by `sets`, or else the world and
    /// preset names. Every failure is invalid input.
    pub fn resolve(
        recipe: Option<&Recipe>,
        world: &str,
        preset: Option<&str>,
        seed: u64,
        sets: &[Setting],
    ) -> Result<Self> {
        let Some(recipe) = recipe else {
            let world = world::resolve(world)?;
            let preset = preset.map(|p| world::resolve_preset(world, p)).transpose()?;
            return Ok(Self::Preset { world, preset });
        };
        let mut saved = recipe.saved.clone();
        let presets = (WORLDS[world_of(&saved.settings)].presets)();
        if saved.preset >= presets.len() {
            return Err(Failure::Usage.error(format!(
                "{} names preset {} of {}, which has presets 1-{}",
                recipe.path.display(),
                saved.preset + 1,
                saved.settings.world_id(),
                presets.len()
            )));
        }
        saved.seed = seed;
        apply_to_recipe(&mut saved, sets)?;
        Ok(Self::Recipe(Box::new(saved)))
    }

    /// Index into `WORLDS`.
    pub fn world(&self) -> usize {
        match self {
            Self::Recipe(saved) => world_of(&saved.settings),
            Self::Preset { world, .. } => *world,
        }
    }

    /// The 0-based preset, when known before the world exists.
    pub fn preset(&self) -> Option<usize> {
        match self {
            Self::Recipe(saved) => Some(saved.preset),
            Self::Preset { preset, .. } => *preset,
        }
    }

    /// Creates the world for an output of `size` pixels, running from `seed`
    /// (a recipe carries its own) with the `sets` of a preset applied.
    pub fn create(&self, gpu: &Gpu, size: [u32; 2], seed: u64, sets: &[Setting]) -> Result<Box<dyn World>> {
        match self {
            Self::Recipe(saved) => Ok(saved.instantiate_at(gpu, size)?.1),
            Self::Preset { world, preset } => {
                let mut world = world::create_at(gpu, *world, size, *preset, seed)?;
                apply_to_world(gpu, &mut *world, sets, seed)?;
                Ok(world)
            }
        }
    }

    /// The look a render starts from: the recipe's, or the world's own.
    pub fn look(&self, world: &dyn World) -> PostSettings {
        match self {
            Self::Recipe(saved) => saved.look,
            Self::Preset { .. } => world.post_settings(),
        }
    }

    /// The camera a render starts from: the recipe's, or the whole world.
    pub fn camera(&self) -> Camera {
        match self {
            Self::Recipe(saved) => saved.camera,
            Self::Preset { .. } => Camera::default(),
        }
    }

    /// The recipe of `world` as it runs now, from `seed` at `size` with `look`
    /// and `camera`: what `--save-recipe` writes and a PNG carries.
    pub fn snapshot(
        &self,
        world: &dyn World,
        seed: u64,
        size: [u32; 2],
        look: PostSettings,
        camera: Camera,
        sets: &[Setting],
    ) -> Result<SavedWorld> {
        let (name, modified) = match self {
            Self::Recipe(saved) => (saved.name.clone(), saved.modified),
            Self::Preset { .. } => {
                let preset = world.presets().get(world.preset()).copied().unwrap_or("custom");
                (format!("{} · {preset} (seed {seed})", world.name()), !sets.is_empty())
            }
        };
        Ok(SavedWorld {
            version: library::RECIPE_VERSION,
            name,
            seed,
            output_size: size,
            preset: world.preset(),
            modified,
            settings: world.settings()?,
            look,
            camera,
        })
    }
}

/// Writes `saved` as a recipe file in the library's format (pretty JSON).
pub fn write(path: &Path, saved: &SavedWorld) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
    }
    let mut json = serde_json::to_string_pretty(saved).context("serialising the recipe")?;
    json.push('\n');
    std::fs::write(path, json).with_context(|| format!("writing {}", path.display()))
}

/// A recipe file path given on the command line must end in `.json` (the
/// library only reads those) and its folder must be writable.
pub fn check_recipe_path(path: &Path, option: &str) -> Result<()> {
    if path.extension().is_none_or(|e| !e.eq_ignore_ascii_case("json")) {
        return Err(Failure::Usage.error(format!("{option} must name a .json file (got '{}')", path.display())));
    }
    headless::check_writable_file(path)
}

/// `primordia recipe`: the complete recipe of a preset or a recipe file, after `--set`.
#[derive(Clone, Debug)]
pub struct RecipeJob {
    pub world: String,
    pub preset: Option<String>,
    pub seed: u64,
    /// The recipe's output size.
    pub size: [u32; 2],
    pub recipe: Option<Recipe>,
    pub sets: Vec<Setting>,
    /// Write the recipe to this file instead of printing it.
    pub out: Option<PathBuf>,
}

/// The finished recipe, and the file it was written to.
#[derive(Clone, Debug)]
pub struct RecipeSummary {
    /// Index into `WORLDS`.
    pub world: usize,
    pub saved: SavedWorld,
    pub file: Option<PathBuf>,
}

/// Resolves and checks `job` without a GPU.
fn plan(job: &RecipeJob) -> Result<Source> {
    headless::check_size(job.size)?;
    if let Some(out) = &job.out {
        check_recipe_path(out, "-o/--out")?;
    }
    Source::resolve(job.recipe.as_ref(), &job.world, job.preset.as_deref(), job.seed, &job.sets)
}

pub fn build(job: &RecipeJob) -> Result<RecipeSummary> {
    plan(job)?;
    let gpu = headless::open_gpu()?;
    build_with(&gpu, job)
}

/// Creates the world on `gpu`, so that the world itself checks every value,
/// and reads its complete recipe back.
pub fn build_with(gpu: &Gpu, job: &RecipeJob) -> Result<RecipeSummary> {
    let source = plan(job)?;
    let world = source.create(gpu, job.size, job.seed, &job.sets)?;
    let saved = source.snapshot(&*world, job.seed, job.size, source.look(&*world), source.camera(), &job.sets)?;
    gpu.wait_idle();
    if let Some(problem) = gpu.fatal_error() {
        return Err(Failure::Gpu.error(format!("GPU error while building the recipe: {problem}")));
    }
    if let Some(out) = &job.out {
        write(out, &saved)?;
        log::info!("wrote {}", out.display());
    }
    Ok(RecipeSummary { world: source.world(), saved, file: job.out.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> SavedWorld {
        serde_json::from_str(include_str!("../tests/fixtures/reaction-diffusion.json")).unwrap()
    }

    fn set(arg: &str) -> Setting {
        arg.parse().unwrap_or_else(|e| panic!("{arg}: {e}"))
    }

    #[test]
    fn settings_parse_json_values_with_plain_text_as_a_fallback() {
        assert_eq!(set("params.feed=0.031").value, json!(0.031));
        assert_eq!(set(" params . kill =0.06").key, "params.kill");
        assert_eq!(set("palette=Frost").value, json!("Frost"));
        assert_eq!(set("palette=\"Frost\"").value, json!("Frost"));
        assert_eq!(set("params.atlas=true").value, json!(true));
        assert_eq!(set("palettes=[\"Frost\",\"Ember\",\"Aurora\"]").value, json!(["Frost", "Ember", "Aurora"]));
        assert_eq!(set("name=a=b").value, json!("a=b"), "only the first = splits");
        assert_eq!(set("palette=").value, json!(""));
        assert_eq!(set("params.feed=0.031").to_string(), "params.feed=0.031");
        assert_eq!(changed(&[]), "");
        assert_eq!(changed(&[set("palette=Moss")]), " with 1 setting changed");
        assert_eq!(changed(&[set("palette=Moss"), set("params.kill=0.06")]), " with 2 settings changed");
        for bad in ["params.feed", "=1", "params..feed=1", "params.=1", ".feed=1"] {
            assert!(bad.parse::<Setting>().is_err(), "{bad:?}");
        }
    }

    #[test]
    fn settings_edit_recipes_by_dotted_path() {
        let mut saved = fixture();
        let sets = [set("params.feed=0.031"), set("palette=ember"), set("params.material=Nacre"), set("post.exposure=1.5")];
        apply_to_recipe(&mut saved, &sets).unwrap();
        let WorldSettings::ReactionDiffusion { params, palette, post } = &saved.settings else { panic!("world changed") };
        assert_eq!((params.feed, palette.as_str(), post.exposure), (0.031, "Ember", 1.5), "palettes as registered");
        assert_eq!(saved.look.exposure, 1.5, "the look follows post.*");
        assert_eq!(saved.look.bloom_threshold, fixture().look.bloom_threshold, "only the edited key changes");
        assert!(saved.modified);
        // Text settings take the value as typed; integers take integers.
        let edited = apply(&fixture().settings, &[set("params.seeding=Nois"), set("params.steps_per_frame=12")]);
        let error = format!("{:#}", edited.unwrap_err());
        assert!(error.contains("unknown variant `Nois`") && !error.contains("takes"), "{error}");
        let edited = apply(&fixture().settings, &[set("params.steps_per_frame=12")]).unwrap();
        let WorldSettings::ReactionDiffusion { params, .. } = &edited else { panic!() };
        assert_eq!(params.steps_per_frame, 12);
        let mut value = json!({ "palette": "Moss", "name": "x" });
        edit(&mut value, &set("name=123"), 3).unwrap();
        assert_eq!(value["name"], json!("123"), "a number typed for text stays text");
        // No settings, no change.
        let mut untouched = fixture();
        apply_to_recipe(&mut untouched, &[]).unwrap();
        assert_eq!(serde_json::to_value(&untouched).unwrap(), serde_json::to_value(fixture()).unwrap());
    }

    #[test]
    fn bad_settings_name_the_valid_keys_and_are_invalid_input() {
        let cases = [
            ("params.fed=1", "unknown setting 'params.fed' for Reaction-Diffusion; did you mean 'params.feed'?"),
            ("params.fed=1", "(params has: activity, aniso, atlas, aura_tint"),
            ("params.fed=1", "`primordia recipe -w reaction-diffusion` shows every setting"),
            ("palete=Frost", "did you mean 'palette'? (the settings have: palette, params, post;"),
            ("world=lenia", "'world' is the world a recipe belongs to (reaction-diffusion) and cannot be set"),
            ("params.feed.x=1", "params.feed is a number, with nothing inside it"),
            ("params.feed=abc", "setting 'params.feed' takes a number (it is 0.049 now), not text (abc)"),
            ("params.atlas=yes", "takes true or false"),
            ("params.steps_per_frame=1.5", "setting 'params.steps_per_frame=1.5' does not fit a Reaction-Diffusion recipe"),
            ("params.material=Glas", "unknown variant `Glas`"),
            ("palette=Frost", "unknown palette 'Frost' for Reaction-Diffusion (palettes: Bioluminescence, Ember,"),
            ("palette=Auroa", "unknown palette 'Auroa' for Reaction-Diffusion; did you mean 'Aurora'?"),
            ("params.feed=1e300", "is out of range: numbers must fit a 32-bit float"),
            ("post={}", "setting 'post={}' does not fit"),
        ];
        for (arg, expected) in cases {
            let error = apply(&fixture().settings, &[set(arg)]).expect_err(arg);
            assert!(format!("{error:#}").contains(expected), "{arg}: {error:#}");
            assert_eq!(crate::failure::exit_code(&error), 2, "{arg}");
        }
        // The error names the setting that failed, not the first one.
        let error = apply(&fixture().settings, &[set("params.feed=0.03"), set("params.kil=1")]).unwrap_err();
        assert!(error.to_string().contains("'params.kil'"), "{error}");
    }

    #[test]
    fn palette_names_are_checked_and_spelled_as_registered() {
        let mut lenia = json!({ "palettes": ["frost", "EMBER", 3], "post": {} });
        check_palettes(&mut lenia, 0).unwrap();
        assert_eq!(lenia["palettes"], json!(["frost", "EMBER", 3]), "Physarum names one palette, not a list");
        check_palettes(&mut lenia, 2).expect_err("Frost is Physarum's own palette");
        lenia["palettes"][0] = json!("aurora");
        check_palettes(&mut lenia, 2).unwrap();
        assert_eq!(lenia["palettes"], json!(["Aurora", "Ember", 3]));
        let mut physarum = json!({ "palette": "frost" });
        check_palettes(&mut physarum, 0).unwrap();
        assert_eq!(physarum["palette"], "Frost");
        let mut particles = json!({ "params": { "colors": 99 } });
        check_palettes(&mut particles, 1).unwrap();
    }

    #[test]
    fn list_indices_are_checked() {
        let mut value = json!({ "params": { "kernels": [{ "mu": 0.1 }, { "mu": 0.2 }], "none": [] } });
        edit(&mut value, &set("params.kernels.1.mu=0.3"), 2).unwrap();
        assert_eq!(value["params"]["kernels"][1]["mu"], json!(0.3));
        edit(&mut value, &set("params.kernels.0={\"mu\":0.5}"), 2).unwrap();
        assert_eq!(value["params"]["kernels"][0]["mu"], json!(0.5));
        for (arg, expected) in [
            ("params.kernels.2.mu=1", "params.kernels is a list of 2, so '2' must be an index from 0 to 1"),
            ("params.kernels.first.mu=1", "'first' must be an index from 0 to 1"),
            ("params.none.0=1", "params.none is an empty list"),
            ("params.kernels.0.sigma=1", "unknown setting 'params.kernels.0.sigma' for Lenia"),
            ("params.kernels=1", "takes a list (it is [{\"mu\":0.5},{\"mu\":0.3}] now), not a number (1)"),
        ] {
            let error = edit(&mut value.clone(), &set(arg), 2).expect_err(arg);
            assert!(error.contains(expected), "{arg}: {error}");
        }
    }

    #[test]
    fn recipe_sources_resolve_names_seeds_and_settings_without_a_gpu() {
        let recipe = Recipe { path: PathBuf::from("reef.json"), saved: fixture() };
        let source = Source::resolve(Some(&recipe), "ignored", None, 7, &[set("params.kill=0.05")]).unwrap();
        let Source::Recipe(saved) = &source else { panic!("expected the recipe") };
        assert_eq!((saved.seed, source.world(), source.preset()), (7, 3, Some(0)));
        assert_eq!(source.camera(), fixture().camera);

        let preset = Source::resolve(None, "rd", Some("mito"), 1, &[set("params.anything=1")]).unwrap();
        assert!(matches!(preset, Source::Preset { world: 3, preset: Some(1) }), "{preset:?}");
        assert_eq!(preset.camera(), Camera::default());
        let error = Source::resolve(None, "physarm", None, 1, &[]).unwrap_err();
        assert!(error.to_string().contains("did you mean 'physarum'?"), "{error}");

        let mut bad = recipe.clone();
        bad.saved.preset = 99;
        let error = Source::resolve(Some(&bad), "rd", None, 1, &[]).unwrap_err();
        assert!(error.to_string().contains("names preset 100 of reaction-diffusion, which has presets 1-10"), "{error}");
        assert_eq!(crate::failure::exit_code(&error), 2);
    }

    #[test]
    fn recipe_files_must_be_json_in_a_writable_folder() {
        let dir = tempfile::tempdir().unwrap();
        assert!(check_recipe_path(&dir.path().join("a").join("r.JSON"), "--save-recipe").is_ok());
        let error = check_recipe_path(&dir.path().join("r.txt"), "--save-recipe").unwrap_err();
        assert!(error.to_string().contains("--save-recipe must name a .json file"), "{error}");
        assert_eq!(crate::failure::exit_code(&error), 2);
        let path = dir.path().join("out").join("r.json");
        write(&path, &fixture()).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with("}\n") && text.contains("\n  \"seed\": 18446744073709551615,"), "{text}");
        let back = library::load(&path).unwrap();
        assert_eq!(serde_json::to_value(back).unwrap(), serde_json::to_value(fixture()).unwrap());
    }

    #[test]
    fn gpu_recipe_command_prints_every_setting_and_rejects_values_the_world_refuses() {
        let Some((_guard, gpu)) = crate::gpu::test_gpu() else { return };
        let dir = tempfile::tempdir().unwrap();
        let job = RecipeJob {
            world: "rd".into(),
            preset: Some("mitosis".into()),
            seed: 9,
            size: [64, 48],
            recipe: None,
            sets: vec![set("params.feed=0.031"), set("post.exposure=1.25")],
            out: Some(dir.path().join("mito.json")),
        };
        let summary = build_with(&gpu, &job).unwrap();
        let saved = &summary.saved;
        assert_eq!((summary.world, saved.preset, saved.seed, saved.output_size), (3, 1, 9, [64, 48]));
        assert!(saved.modified);
        assert_eq!(saved.name, "Reaction-Diffusion · Mitosis (seed 9)");
        let WorldSettings::ReactionDiffusion { params, post, .. } = &saved.settings else { panic!() };
        assert_eq!((params.feed, post.exposure, saved.look.exposure), (0.031, 1.25, 1.25));
        let written = library::load(&dir.path().join("mito.json")).unwrap();
        assert_eq!(serde_json::to_value(&written).unwrap(), serde_json::to_value(saved).unwrap());

        // An unedited preset is not modified, and a recipe file keeps its name and look.
        let plain = build_with(&gpu, &RecipeJob { sets: Vec::new(), out: None, ..job.clone() }).unwrap();
        assert!(!plain.saved.modified);
        let from_file = RecipeJob {
            recipe: Some(Recipe::load(&dir.path().join("mito.json")).unwrap()),
            sets: vec![set("params.kill=0.061")],
            out: None,
            ..job.clone()
        };
        let edited = build_with(&gpu, &from_file).unwrap().saved;
        assert_eq!((edited.name.as_str(), edited.look.exposure), (saved.name.as_str(), 1.25));
        let WorldSettings::ReactionDiffusion { params, .. } = &edited.settings else { panic!() };
        assert_eq!((params.feed, params.kill), (0.031, 0.061));

        // The world's own checks run on the GPU and are invalid input; the GPU stays usable.
        let error = build_with(&gpu, &RecipeJob { sets: vec![set("params.steps_per_frame=0")], ..job.clone() })
            .expect_err("zero steps");
        assert!(format!("{error:#}").contains("Reaction-Diffusion rejected the settings: Invalid step count"), "{error:#}");
        assert_eq!(crate::failure::exit_code(&error), 2);
        let mut file = from_file.clone();
        file.sets = vec![set("params.scale=9")];
        let error = build_with(&gpu, &file).expect_err("a scale out of range");
        assert!(format!("{error:#}").contains("Invalid reaction-diffusion rates or scale"), "{error:#}");
        assert_eq!(crate::failure::exit_code(&error), 2);
        assert!(build_with(&gpu, &RecipeJob { sets: Vec::new(), out: None, ..job.clone() }).is_ok());
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    }
}
