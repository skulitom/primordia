//! Embeds the icon, version information and an application manifest into the
//! Windows exe, so Explorer, the taskbar and the file's Properties show
//! Primordia rather than a generic program.
//!
//! This needs the Windows SDK's resource compiler (rc.exe), which comes with
//! the MSVC build tools. Without it the build still succeeds, with a warning,
//! and the exe simply has no icon; the window icon (see `ui::ICON`) does not
//! depend on this.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=assets/primordia.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        windows_resources();
    }
}

/// Common Controls 6, so the error message box (src/console.rs) is drawn in
/// the current Windows style rather than the Windows 95 one, and the usual
/// declarations that the app runs unelevated on Windows 10 and 11.
#[cfg(windows)]
const MANIFEST: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0"
        processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
    </dependentAssembly>
  </dependency>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
</assembly>
"#;

#[cfg(windows)]
fn windows_resources() {
    let root = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_default());
    let icon = root.join("assets").join("primordia.ico");
    let mut resource = winresource::WindowsResource::new();
    resource
        .set_icon(&icon.to_string_lossy())
        // Task Manager and the "Open with" list show the description as the app's name.
        .set("FileDescription", "Primordia")
        .set("ProductName", "Primordia")
        .set("OriginalFilename", "primordia.exe")
        .set("LegalCopyright", "Copyright (c) 2026 The Primordia contributors. MIT licence.")
        .set("Comments", "GPU artificial-life laboratory")
        .set_manifest(MANIFEST);
    if let Err(e) = resource.compile() {
        println!("cargo:warning=the exe will have no icon or version information: {e}");
    }
}

/// Cross-compiling from another system: its resource compiler is not assumed.
#[cfg(not(windows))]
fn windows_resources() {}
