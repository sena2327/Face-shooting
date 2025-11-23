# Repository Guidelines

## Project Structure & Module Organization
This repository mixes Python-based gaze-tracking shooters with an optional C++ prototype. `main.py` launches `detect_face.py` plus `single_shooter.py` or `pvp_shooter.py` via `--pvp/--local`. Support utilities (`udp_gaze_shooter.py`, `face_overlay_util.py`, `img_face.py`) remain for focused experiments. Shared gameplay and rendering helpers sit under `src/` (`game_objects.py`, `bullet_render.py`, `hud.py`, `net.py`). Art, calibration, and audio assets stay inside `img/`, `sound/`, and `blue_cap.csv`, while `build/` holds the CMake-generated `gaze_shoot_game` binaries to keep the Python workspace clean.

## Build, Test, and Development Commands
- `python3 main.py [--pvp|--local] --base-gaze-port 5005 --base-listen-port 6000` orchestrates both detect and shooter processes.
- `python3 detect_face.py --flip --udp --udp-host 127.0.0.1 --udp-port 5005` validates camera capture plus the UDP gaze stream in isolation.
- `python3 pvp_shooter.py --gaze-port 5005 --listen-port 6000 --peer HOST:PORT --capture-opponent` spins up one peer endpoint; run it on each host.
- `cmake -S . -B build && cmake --build build --config Release` rebuilds the optional C++ prototype inside `build/`.

## Coding Style & Naming Conventions
Match the typed, 4-space style visible in `main.py` and `src/*`: keep modules import-safe, use `snake_case` names, reserve `CamelCase` for classes, and `UPPER_SNAKE` for config constants like `CAM_INDEX`. Keep networking helpers (`src/net.py`) pure functions so they can be unit-tested. Stick to C++17 as enforced by `CMakeLists.txt` and prefer clang-format's default brace style for the `gaze_shoot_game.cpp` experiment.

## Testing Guidelines
`pytest` drives regressions. Place new tests beside the logic under test (e.g., `tests/` or near `src/`). `test_face_overlay.py` can be invoked with `python3 -m pytest test_face_overlay.py` after swapping the webcam feed for recorded frames stored in `img/`. Cover UDP serialization and collision math before touching rendering code, and document any reliance on external hardware.

## Commit & Pull Request Guidelines
Git history is unavailable in this workspace, but maintainers typically use short, imperative messages that call out the subsystem, e.g., `feat(net): add UDP bullet broadcast`. Keep subject lines ≤72 characters and explain calibration or networking changes in the body. Pull requests should include run/test output, linked issues, and screenshots or screen recordings when HUD elements change.

## Security & Configuration Tips
Keep UDP gaze ports on localhost (default 5005/6000) unless both peers are trusted; document any firewall or NAT rules inside your PR description. Avoid committing personal calibration data—replace `blue_cap.csv` and headgear textures with sanitized samples before publishing.
