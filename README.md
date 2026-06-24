# StarcraftLLM

A small first step toward a strategy-driven StarCraft agent.

The repo has two runnable paths:

1. a fast browser canvas prototype for movement logic;
2. a real StarCraft II API bot that starts SC2 and issues an in-game move command.

## Real StarCraft II movement MVP

On macOS, install StarCraft II with the Blizzard/Battle.net app first, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_sc2_movement.py --check
python scripts/run_sc2_movement.py --strategy "move worker 35 42"
```

What this does:

- starts a real StarCraft II custom game through the SC2 API;
- plays Terran against a very easy Zerg computer;
- selects the starting worker units;
- issues a real in-game move command to the requested map coordinate;
- keeps the game open briefly so the movement can be seen.

Supported MVP strategy commands:

```text
move worker 35 42
move marine 35 42
move 35 42
```

Notes:

- This targets StarCraft II because Blizzard's SC2 API supports macOS clients.
- Classic StarCraft/Brood War bot control is not the macOS-friendly path for this MVP.
- If your SC2 install is in a custom location, set `SC2PATH` to the StarCraft II install directory.
- The default map is `Abyssal Reef LE`; if your install does not have that map, pass another installed map with `--map "Map Name"`.

## Browser prototype

Open the visual prototype directly:

```bash
open index.html
```

You can then:

- click the map to move the Marine;
- type `move Marine 620 360` and press **명령 실행**;
- type the shorter `move 240 160` command.

## Development

Run all local tests:

```bash
npm run test:all
```

Or run each suite separately:

```bash
npm test
python3 -m unittest discover -s tests -v
```

## Current scope

Implemented now:

- browser `Unit.moveTo(x, y)` movement logic;
- browser `GameWorld.moveUnit(unitId, x, y)` command surface;
- real SC2 `move worker/marine x y` strategy parser;
- real SC2 bot runner that issues movement commands through the StarCraft II API;
- local detection for common macOS SC2 install paths.

Not implemented yet:

- full build-order or combat strategy execution;
- natural-language strategy planning;
- computer vision;
- Brood War/BWAPI integration.
