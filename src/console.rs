//! The console window of a double-clicked Windows exe.
//!
//! Primordia is a console program, so `render`, `list` and the other
//! subcommands print into the terminal that started them. Started from
//! Explorer, a shortcut or the Start menu, Windows opens a console just for it:
//! a black window of log lines beside the app that closes, together with any
//! error message, the moment the process exits. [`detach`] closes that console
//! when this process is the only one attached to it (a terminal's shell would
//! be attached too), and [`show_error`] then reports fatal errors in a message
//! box instead. On other platforms both do nothing.

#[cfg(windows)]
mod imp {
    use std::io::IsTerminal as _;
    use std::sync::atomic::{AtomicBool, Ordering};

    use windows_sys::Win32::System::Console::{
        FreeConsole, GetConsoleProcessList, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW};

    /// Set once the console is gone and errors need a message box.
    static DETACHED: AtomicBool = AtomicBool::new(false);

    pub fn detach() {
        // Room for two: all that matters is whether anyone else is attached.
        let mut processes = [0u32; 2];
        // SAFETY: the buffer is valid for `processes.len()` entries.
        let attached = unsafe { GetConsoleProcessList(processes.as_mut_ptr(), processes.len() as u32) };
        // 0: no console at all (or the call failed); 2 or more: shared with a terminal.
        if attached != 1 {
            return;
        }
        let consoles = [
            (STD_INPUT_HANDLE, std::io::stdin().is_terminal()),
            (STD_OUTPUT_HANDLE, std::io::stdout().is_terminal()),
            (STD_ERROR_HANDLE, std::io::stderr().is_terminal()),
        ];
        // SAFETY: no arguments; afterwards the old console handles are invalid.
        if unsafe { FreeConsole() } == 0 {
            return;
        }
        // Forget the dead console handles, so later log lines are dropped quietly
        // (the standard library treats a missing handle as a closed stream).
        // Streams that were redirected to a file or pipe keep working.
        for (stream, console) in consoles {
            if console {
                // SAFETY: a null handle is a valid "no handle" value for SetStdHandle.
                unsafe { SetStdHandle(stream, std::ptr::null_mut()) };
            }
        }
        DETACHED.store(true, Ordering::Relaxed);
        // A panic would otherwise close the window without a word. The box comes
        // before the hook installed earlier, which may end the process.
        let earlier_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            show_error(&format!("Primordia hit an internal error.\n\n{info}"));
            earlier_hook(info);
        }));
    }

    pub fn show_error(message: &str) {
        if !DETACHED.load(Ordering::Relaxed) {
            return;
        }
        let text = wide(&format!(
            "{message}\n\nFor the full log, start Primordia from a terminal (Command Prompt or PowerShell)."
        ));
        let caption = wide("Primordia");
        // On its own thread, so the box's message loop cannot call back into the
        // app's window (this may run inside a panic on the event-loop thread).
        let shown = std::thread::spawn(move || {
            // SAFETY: both strings are NUL-terminated UTF-16 that outlive the call.
            unsafe {
                MessageBoxW(
                    std::ptr::null_mut(),
                    text.as_ptr(),
                    caption.as_ptr(),
                    MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
                )
            };
        });
        let _ = shown.join();
    }

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn detach() {}

    pub fn show_error(_message: &str) {}
}

/// Closes the console Windows opened for a double-clicked exe, when no other
/// process shares it. Call it only for the interactive app: the subcommands'
/// output belongs in that console.
pub fn detach() {
    imp::detach();
}

/// Shows a fatal error in a message box if [`detach`] closed the console, and
/// otherwise does nothing (print the error to stderr as well either way).
pub fn show_error(message: &str) {
    imp::show_error(message);
}
