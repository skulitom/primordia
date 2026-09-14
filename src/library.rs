//! Named, portable recipes. Each save is an independent, atomically written JSON file.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

use crate::post::PostSettings;
use crate::world::{Camera, lenia, particle_life, physarum, reaction_diffusion, symbiosis};

/// Recipes contain parameters, not snapshots or embedded images.
const MAX_SAVE_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "world", rename_all = "kebab-case")]
pub enum WorldSettings {
    Physarum { params: physarum::Params, population: physarum::Population, palette: String, post: PostSettings },
    ParticleLife { params: Box<particle_life::Params>, ground: u32, post: PostSettings },
    Lenia { params: lenia::Params, palettes: [String; 3], post: PostSettings },
    ReactionDiffusion { params: reaction_diffusion::Params, palette: String, post: PostSettings },
    Symbiosis { params: symbiosis::Params, palette: String, post: PostSettings },
}

impl WorldSettings {
    pub fn world_id(&self) -> &'static str {
        match self {
            Self::Physarum { .. } => "physarum",
            Self::ParticleLife { .. } => "particle-life",
            Self::Lenia { .. } => "lenia",
            Self::ReactionDiffusion { .. } => "reaction-diffusion",
            Self::Symbiosis { .. } => "symbiosis",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SavedWorld {
    pub version: u32,
    pub name: String,
    pub seed: u64,
    /// Original creation size, independent of later window resizing.
    pub output_size: [u32; 2],
    pub preset: usize,
    pub modified: bool,
    pub settings: WorldSettings,
    pub look: PostSettings,
    pub camera: Camera,
}

impl SavedWorld {
    /// Build separately and contain GPU validation/allocation errors so the
    /// caller can retain its running world if a save cannot be restored.
    pub fn instantiate(&self, gpu: &crate::gpu::Gpu) -> Result<(usize, Box<dyn crate::world::World>)> {
        self.validate()?;
        gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let restored = (|| -> Result<_> {
            let (index, mut world) =
                crate::world::create(gpu, self.settings.world_id(), self.output_size, None, self.seed)?;
            ensure!(self.preset < world.presets().len(), "Unknown saved preset");
            world.load_preset(gpu, self.preset, self.seed);
            world.restore_settings(gpu, &self.settings, self.seed)?;
            Ok((index, world))
        })();
        let validation = pollster::block_on(gpu.device.pop_error_scope());
        let oom = pollster::block_on(gpu.device.pop_error_scope());
        if let Some(error) = oom {
            bail!("Not enough GPU memory to load this world: {error}");
        }
        if let Some(error) = validation {
            bail!("This save is incompatible with this GPU: {error}");
        }
        restored
    }

    fn validate(&self) -> Result<()> {
        ensure!(self.version == 1, "This save requires a different version of Primordia");
        validate_name(&self.name)?;
        ensure!(self.output_size.iter().all(|n| (16..=16384).contains(n)), "Invalid saved world dimensions");
        ensure!(self.camera.zoom.is_finite() && (0.5..=64.0).contains(&self.camera.zoom), "Invalid saved zoom");
        ensure!(self.camera.center.iter().all(|v| v.is_finite()), "Invalid saved camera position");
        Ok(())
    }
}

fn validate_name(name: &str) -> Result<()> {
    ensure!(!name.trim().is_empty(), "Give this world a name");
    ensure!(name.chars().count() <= 100, "Use a name of 100 characters or fewer");
    Ok(())
}

pub struct Entry {
    pub path: PathBuf,
    pub saved: SavedWorld,
}

pub struct Library {
    pub directory: PathBuf,
    pub entries: Vec<Entry>,
    pub warnings: Vec<String>,
}

impl Library {
    pub fn open(directory: PathBuf) -> Self {
        let mut library = Self { directory, entries: Vec::new(), warnings: Vec::new() };
        library.refresh();
        library
    }

    pub fn refresh(&mut self) {
        self.entries.clear();
        self.warnings.clear();
        if let Err(e) = self.read_entries() {
            self.warnings.push(format!("Could not read library: {e:#}"));
        }
    }

    fn read_entries(&mut self) -> Result<()> {
        let files = match std::fs::read_dir(&self.directory) {
            Ok(files) => files,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        for file in files {
            let path = file?.path();
            if path.extension().is_none_or(|ext| ext != "json") {
                continue;
            }
            match read_save(&path) {
                Ok(saved) => self.entries.push(Entry { path, saved }),
                Err(e) => self
                    .warnings
                    .push(format!("Skipped {}: {e:#}", path.file_name().unwrap_or_default().to_string_lossy())),
            }
        }
        self.entries
            .sort_by(|a, b| a.saved.name.to_lowercase().cmp(&b.saved.name.to_lowercase()).then(a.path.cmp(&b.path)));
        Ok(())
    }

    pub fn save(&mut self, mut saved: SavedWorld) -> Result<()> {
        saved.name = saved.name.trim().to_owned();
        saved.validate()?;
        std::fs::create_dir_all(&self.directory).context("Creating the library folder")?;
        // Persist without replacing any existing save, even from another running app.
        let mut file = tempfile::NamedTempFile::new_in(&self.directory)?;
        serde_json::to_writer_pretty(&mut file, &saved)?;
        file.write_all(b"\n")?;
        file.as_file().sync_all()?;
        let name = file.path().file_name().context("Missing save filename")?.to_string_lossy();
        let path = self.directory.join(format!("world-{name}.json"));
        file.persist_noclobber(path).map_err(|e| e.error).context("Writing saved world")?;
        self.refresh();
        Ok(())
    }

    pub fn rename(&mut self, index: usize, name: &str) -> Result<()> {
        validate_name(name)?;
        let entry = self.entries.get(index).context("Save no longer exists")?;
        let mut saved = entry.saved.clone();
        saved.name = name.trim().to_owned();
        write_save(&entry.path, &saved)?;
        self.refresh();
        Ok(())
    }

    pub fn delete(&mut self, index: usize) -> Result<()> {
        let entry = self.entries.get(index).context("Save no longer exists")?;
        std::fs::remove_file(&entry.path).context("Removing saved world")?;
        self.refresh();
        Ok(())
    }
}

fn read_save(path: &Path) -> Result<SavedWorld> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.take(MAX_SAVE_BYTES + 1).read_to_end(&mut bytes)?;
    ensure!(bytes.len() as u64 <= MAX_SAVE_BYTES, "Saved settings file is too large");
    let header: serde_json::Value = serde_json::from_slice(&bytes)?;
    // JSON supports larger finite numbers than f32. Reject them before serde
    // narrows them to infinities in camera, simulation or lighting parameters.
    fn check_numbers(value: &serde_json::Value) -> bool {
        match value {
            serde_json::Value::Number(n) => n.as_f64().is_some_and(|n| n.abs() <= f64::from(f32::MAX)),
            serde_json::Value::Array(values) => values.iter().all(check_numbers),
            serde_json::Value::Object(fields) => fields.values().all(check_numbers),
            _ => true,
        }
    }
    ensure!(check_numbers(&header), "Saved settings contain a number outside the supported range");
    if header.get("version").and_then(|v| v.as_u64()) != Some(1) {
        bail!("Unsupported save version");
    }
    let saved: SavedWorld = serde_json::from_value(header)?;
    saved.validate()?;
    Ok(saved)
}

fn write_save(path: &Path, saved: &SavedWorld) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().context("Missing library folder")?)?;
    serde_json::to_writer_pretty(&mut file, saved)?;
    file.write_all(b"\n")?;
    file.as_file().sync_all()?;
    file.persist(path).map_err(|e| e.error).context("Writing saved world")?;
    Ok(())
}

pub fn default_directory() -> PathBuf {
    if let Some(path) = std::env::var_os("PRIMORDIA_LIBRARY_DIR") {
        return path.into();
    }
    if cfg!(target_os = "windows") {
        if let Some(path) = std::env::var_os("APPDATA") {
            return PathBuf::from(path).join("Primordia/library");
        }
    } else if cfg!(target_os = "macos") {
        if let Some(path) = std::env::var_os("HOME") {
            return PathBuf::from(path).join("Library/Application Support/Primordia/library");
        }
    } else {
        if let Some(path) = std::env::var_os("XDG_DATA_HOME") {
            return PathBuf::from(path).join("primordia/library");
        }
        if let Some(path) = std::env::var_os("HOME") {
            return PathBuf::from(path).join(".local/share/primordia/library");
        }
    }
    PathBuf::from("library")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_survives_restart_and_handles_duplicates_rename_delete_and_bad_files() {
        let dir = tempfile::tempdir().unwrap();
        let mut library = Library::open(dir.path().to_owned());
        let saved: SavedWorld =
            serde_json::from_str(include_str!("../tests/fixtures/reaction-diffusion.json")).unwrap();
        library.save(saved.clone()).unwrap();
        library.save(saved.clone()).unwrap();
        let mut reopened = Library::open(dir.path().to_owned());
        assert_eq!(reopened.entries.len(), 2);
        assert_eq!(serde_json::to_value(&reopened.entries[0].saved).unwrap(), serde_json::to_value(&saved).unwrap());
        assert_ne!(reopened.entries[0].path, reopened.entries[1].path);
        assert!(reopened.rename(0, "  ").is_err());
        reopened.rename(0, "My discovery").unwrap();
        assert!(reopened.entries.iter().any(|e| e.saved.name == "My discovery"));
        reopened.delete(0).unwrap();
        std::fs::write(dir.path().join("broken.json"), "{oops").unwrap();
        std::fs::write(dir.path().join("future.json"), r#"{"version":99}"#).unwrap();
        reopened.refresh();
        assert_eq!(reopened.entries.len(), 1);
        assert_eq!(reopened.warnings.len(), 2);
    }

    #[test]
    fn every_world_restores_a_serialized_recipe_on_the_gpu() {
        use crate::gpu::Gpu;
        use crate::world::{self, WORLDS};
        let _guard = crate::gpu::test_lock();
        let gpu = pollster::block_on(Gpu::new(Gpu::create_instance(), None)).unwrap();
        for entry in WORLDS {
            let (_, mut original) = world::create(&gpu, entry.id, [320, 240], None, 42).unwrap();
            let settings = original.settings().unwrap();
            let json = serde_json::to_string(&settings).unwrap();
            let decoded: WorldSettings = serde_json::from_str(&json).unwrap();
            // Load another preset, then restore the recipe to ensure restoration
            // replaces the actual parameter state (matrices, palettes, templates).
            original.load_preset(&gpu, 1, 99);
            original.restore_settings(&gpu, &decoded, 42).unwrap();
            assert_eq!(
                serde_json::to_value(original.settings().unwrap()).unwrap(),
                serde_json::to_value(&settings).unwrap(),
                "{}",
                entry.id
            );
            gpu.wait_idle();
            assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
        }
    }

    #[test]
    fn oversized_and_unrepresentable_settings_are_rejected_before_loading() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        let mut value: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/reaction-diffusion.json")).unwrap();
        value["settings"]["params"]["scale"] = serde_json::json!(1e100);
        std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(read_save(&path).unwrap_err().to_string().contains("supported range"));
        std::fs::File::create(&path).unwrap().set_len(MAX_SAVE_BYTES + 1).unwrap();
        assert!(read_save(&path).unwrap_err().to_string().contains("too large"));
    }

    fn recipe(world: &dyn crate::world::World, seed: u64) -> SavedWorld {
        SavedWorld {
            version: 1,
            name: "Test specimen".into(),
            seed,
            output_size: [320, 240],
            preset: world.preset(),
            modified: false,
            settings: world.settings().unwrap(),
            look: world.post_settings(),
            camera: Camera::default(),
        }
    }

    #[test]
    fn every_preset_loads_through_the_library_and_disabled_lenia_channels_survive() {
        let _guard = crate::gpu::test_lock();
        let gpu = pollster::block_on(crate::gpu::Gpu::new(crate::gpu::Gpu::create_instance(), None)).unwrap();
        for entry in crate::world::WORLDS {
            let mut original = (entry.create)(&gpu, [320, 240], 42);
            for (preset, name) in original.presets().iter().enumerate() {
                original.load_preset(&gpu, preset, 42);
                let saved = recipe(&*original, 42);
                let decoded: SavedWorld = serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
                let (index, restored) = decoded.instantiate(&gpu).unwrap();
                assert_eq!(crate::world::WORLDS[index].id, entry.id);
                assert_eq!(restored.preset(), preset);
                assert_eq!(
                    serde_json::to_value(restored.settings().unwrap()).unwrap(),
                    serde_json::to_value(&saved.settings).unwrap(),
                    "{} / {name}",
                    entry.name
                );
                gpu.wait_idle();
                assert!(gpu.fatal_error().is_none(), "{} / {name}: {:?}", entry.name, gpu.fatal_error());
            }
        }
        let (_, original) = crate::world::create(&gpu, "lenia", [320, 240], Some("Necklaces"), 42).unwrap();
        let mut saved = recipe(&*original, 42);
        if let WorldSettings::Lenia { params, .. } = &mut saved.settings {
            assert!(params.kernels.iter().any(|k| k.source > 0 || k.target > 0));
            params.channels = 1;
        }
        let (_, mut restored) = saved.instantiate(&gpu).unwrap();
        assert_eq!(
            serde_json::to_value(restored.settings().unwrap()).unwrap(),
            serde_json::to_value(&saved.settings).unwrap()
        );
        // Exercise the folded kernels on the GPU after restore, too.
        let frame = crate::world::Frame {
            gpu: &gpu,
            time: 0.0,
            dt: 1.0 / 60.0,
            frame: 0,
            view: crate::world::ViewXform::fit(restored.size(), [320, 240], &Camera::default()),
            target_size: [320, 240],
            pointer: None,
        };
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        restored.step(&frame, &mut encoder);
        gpu.queue.submit([encoder.finish()]);
        gpu.wait_idle();
        assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
    }

    #[test]
    fn invalid_recipes_leave_the_gpu_available_for_a_valid_load() {
        let _guard = crate::gpu::test_lock();
        let gpu = pollster::block_on(crate::gpu::Gpu::new(crate::gpu::Gpu::create_instance(), None)).unwrap();
        for entry in crate::world::WORLDS {
            let original = (entry.create)(&gpu, [320, 240], 42);
            let saved = recipe(&*original, 42);
            let mut bad = saved.clone();
            match &mut bad.settings {
                WorldSettings::Physarum { params, .. } => params.steps_per_frame = 0,
                WorldSettings::ParticleLife { params, .. } => params.color_shift = usize::MAX,
                WorldSettings::Lenia { params, .. } => params.kernels[0].source = usize::MAX,
                WorldSettings::ReactionDiffusion { params, .. } => params.scale = 1e20,
                WorldSettings::Symbiosis { params, .. } => params.steps = 0,
            }
            assert!(bad.instantiate(&gpu).is_err(), "{} accepted invalid settings", entry.name);
            bad = saved.clone();
            bad.preset = usize::MAX;
            assert!(bad.instantiate(&gpu).is_err());
            let (_, restored) = saved.instantiate(&gpu).unwrap();
            assert_eq!(restored.id(), original.id());
            gpu.wait_idle();
            assert!(gpu.fatal_error().is_none(), "{:?}", gpu.fatal_error());
        }
        assert!(crate::world::create(&gpu, "rd", [u32::MAX; 2], None, 42).is_err());
        assert!(gpu.fatal_error().is_none());
    }
}
