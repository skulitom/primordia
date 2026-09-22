//! Failure classes: what went wrong decides the exit status and the `kind` of a
//! `--json` error object (see the "Exit status" section of `primordia --help`).
//!
//! Errors stay ordinary `anyhow` errors. The code that detects a failure
//! attaches its class ([`Failure::tag`], [`Failure::error`]) and `main` reads
//! it back with [`classify`]. Name lookups ([`crate::world::NameError`]) count
//! as invalid input without a tag; everything unclassified exits with 1.

use std::fmt;

use serde_json::{json, Value};

use crate::world::{Miss, NameError};

/// Why a command failed, when it is more specific than "something went wrong".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Failure {
    /// Invalid input: arguments, names, recipes, sizes or output paths (exit status 2).
    Usage,
    /// No usable GPU, or the GPU failed during the run (exit status 3).
    Gpu,
    /// ffmpeg is missing or failed (exit status 4).
    Ffmpeg,
}

impl Failure {
    pub fn exit_code(self) -> i32 {
        match self {
            Failure::Usage => 2,
            Failure::Gpu => 3,
            Failure::Ffmpeg => 4,
        }
    }

    /// `kind` of a `--json` error object.
    pub fn kind(self) -> &'static str {
        match self {
            Failure::Usage => "usage",
            Failure::Gpu => "gpu",
            Failure::Ffmpeg => "ffmpeg",
        }
    }

    /// Marks `error` as this class of failure; its message is unchanged.
    pub fn tag(self, error: impl Into<anyhow::Error>) -> anyhow::Error {
        anyhow::Error::new(Tagged { class: self, error: error.into() })
    }

    /// A new error of this class.
    pub fn error(self, message: impl fmt::Display + fmt::Debug + Send + Sync + 'static) -> anyhow::Error {
        self.tag(anyhow::Error::msg(message))
    }
}

/// An error with a [`Failure`] class. It displays exactly like the error it
/// wraps: its own text is that error's outermost message, and `source`
/// continues with the rest of that error's chain.
#[derive(Debug)]
struct Tagged {
    class: Failure,
    error: anyhow::Error,
}

impl fmt::Display for Tagged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)
    }
}

impl std::error::Error for Tagged {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.error.source()
    }
}

/// The first `T` in `error`'s chain, looking inside tagged errors as well.
pub fn find<T: std::error::Error + 'static>(error: &anyhow::Error) -> Option<&T> {
    for cause in error.chain() {
        if let Some(found) = cause.downcast_ref::<T>() {
            return Some(found);
        }
        if let Some(tagged) = cause.downcast_ref::<Tagged>() {
            // The tagged error's chain is the rest of this one.
            return find(&tagged.error);
        }
    }
    None
}

/// The class of `error`: its outermost tag, or [`Failure::Usage`] for a name
/// that did not resolve.
pub fn classify(error: &anyhow::Error) -> Option<Failure> {
    if let Some(tagged) = find::<Tagged>(error) {
        return Some(tagged.class);
    }
    find::<NameError>(error).map(|_| Failure::Usage)
}

/// Exit status for `error`: its class's, or 1 for any other failure.
pub fn exit_code(error: &anyhow::Error) -> i32 {
    classify(error).map_or(1, Failure::exit_code)
}

/// The `--json` result of a failed `command`:
/// `{"ok":false,"command":…,"error":{"kind","message","exit_code",…}}`. A name
/// that did not resolve adds `available` and, when there is one,
/// `suggestion` or the ambiguous `candidates`.
pub fn json(command: &str, error: &anyhow::Error) -> Value {
    let kind = match classify(error) {
        Some(class) => class.kind(),
        None if find::<std::io::Error>(error).is_some() => "io",
        None => "other",
    };
    let mut object = json!({ "kind": kind, "message": format!("{error:#}"), "exit_code": exit_code(error) });
    if let Some(name) = find::<NameError>(error) {
        object["what"] = json!(name.what);
        object["query"] = json!(name.query);
        object["available"] = json!(name.available);
        match &name.miss {
            Miss::Unknown { suggestion: Some(s) } => object["suggestion"] = json!(s),
            Miss::Ambiguous(candidates) => object["candidates"] = json!(candidates),
            _ => {}
        }
    }
    json!({ "ok": false, "command": command, "error": object })
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context as _;

    #[test]
    fn tags_survive_context_and_keep_the_message() {
        let io = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let error = Failure::Usage.tag(anyhow::Error::new(io).context("creating out/x.png"));
        assert_eq!(format!("{error:#}"), "creating out/x.png: denied");
        assert_eq!(error.to_string(), "creating out/x.png");
        let wrapped = Err::<(), _>(error).context("rendering").unwrap_err();
        assert_eq!(format!("{wrapped:#}"), "rendering: creating out/x.png: denied");
        assert_eq!(classify(&wrapped), Some(Failure::Usage));
        assert_eq!(exit_code(&wrapped), 2);
        assert!(find::<std::io::Error>(&wrapped).is_some(), "the io error inside the tag is still found");

        // The outermost tag wins; untagged errors are "other", or "io" for an io::Error.
        let nested = Failure::Ffmpeg.tag(Failure::Gpu.error("inner"));
        assert_eq!((classify(&nested), exit_code(&nested)), (Some(Failure::Ffmpeg), 4));
        assert_eq!(format!("{nested:#}"), "inner");
        assert_eq!(exit_code(&anyhow::anyhow!("plain")), 1);
        assert_eq!(json("render", &anyhow::anyhow!("plain"))["error"]["kind"], "other");
        let io = anyhow::Error::new(std::io::Error::other("disk full")).context("writing a.csv");
        assert_eq!(json("render", &io)["error"]["kind"], "io");
        assert_eq!(Failure::Gpu.error("no adapter").to_string(), "no adapter");
        assert_eq!(exit_code(&Failure::Gpu.error("no adapter")), 3);
    }

    #[test]
    fn json_errors_carry_the_kind_exit_code_and_name_details() {
        let error = anyhow::Error::new(crate::world::resolve("physarm").unwrap_err());
        let value = json("render", &error);
        assert_eq!(value["ok"], false);
        assert_eq!(value["command"], "render");
        assert_eq!(value["error"]["kind"], "usage");
        assert_eq!(value["error"]["exit_code"], 2);
        assert_eq!(value["error"]["suggestion"], "physarum");
        assert_eq!(value["error"]["available"][0], "physarum");
        assert!(value["error"]["message"].as_str().unwrap().contains("did you mean 'physarum'"));

        let ambiguous = anyhow::Error::new(crate::world::resolve("p").unwrap_err()).context("explore");
        let value = json("explore", &ambiguous);
        assert_eq!(value["error"]["candidates"], json!(["physarum", "particle-life"]));
        assert_eq!(value["error"]["exit_code"], 2);
    }
}
