<!-- What does this change, and why? Link the issue it fixes, if there is one. -->

## Checks

- [ ] `cargo test --locked` passes, GPU tests included (say below if you had to use `PRIMORDIA_GPU_TESTS=skip`)
- [ ] `cargo clippy --all-targets --locked -- -D warnings` is clean on Rust 1.86, and on current stable if you have it
- [ ] The code matches the style around it; `cargo fmt` was not run (the tree is formatted by hand)
- [ ] Saved recipes still load: no serialised field renamed or removed, and new fields have serde defaults
- [ ] The README or docs describe any new or changed flags, controls or behaviour
- [ ] For visual changes: before and after screenshots, with the command and seed that made them

Tested on: <!-- OS, GPU and backend, e.g. Windows 11, RTX 4090, Vulkan -->
