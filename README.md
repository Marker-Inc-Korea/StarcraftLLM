# StarcraftLLM

A small first step toward a strategy-driven StarCraft agent.

The repo has two runnable paths:

1. a fast browser canvas prototype for movement logic;
2. a real StarCraft II API bot that starts SC2 and executes a small deterministic strategy plan.

## Real StarCraft II movement MVP

On macOS, install StarCraft II with the Blizzard/Battle.net app first. The SC2 API also needs local map files under the install directory; the Battle.net app may not create this folder automatically. Download an official Blizzard map pack from <https://github.com/Blizzard/s2client-proto#map-packs> and extract it into `/Applications/StarCraft II/Maps/`. The default `AbyssalReefLE` map is in the Ladder 2017 Season 1 pack.

Then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_sc2_movement.py --check
python scripts/run_sc2_movement.py --strategy "move worker 35 42"
python scripts/run_sc2_movement.py --strategy "move worker 35 42; wait 1; move worker 45 42"
python scripts/run_sc2_movement.py --strategy "일꾼으로 정찰해" --print-plan
python scripts/run_sc2_movement.py --print-state --fast
python scripts/run_sc2_movement.py --strategy "일꾼으로 정찰해"
```

What this does:

- starts a real StarCraft II custom game through the SC2 API;
- plays Terran against a very easy Zerg computer;
- selects the starting worker units;
- executes each primitive action in the requested strategy plan;
- issues real in-game move commands to the requested map coordinates;
- keeps the game open briefly after the plan finishes so the movement can be seen.

Supported MVP strategy actions:

```text
move worker 35 42
move marine 35 42
move 35 42
wait 1
```

Multiple actions can be separated by semicolons, newlines, or `then`:

```text
move worker 35 42; wait 1; move worker 45 42
move worker 35 42 then wait 1 then move worker 45 42
```

This deterministic plan format is the next integration seam for an LLM: the LLM can translate a higher-level strategy into these primitive actions, and the SC2 executor can run the result without interpreting free-form text during gameplay.

The same plan can be provided as JSON, which is the intended future LLM output contract:

```json
{
  "actions": [
    {"type": "move", "unit": "worker", "x": 35, "y": 42},
    {"type": "wait", "seconds": 1},
    {"type": "move", "unit": "worker", "x": 45, "y": 42}
  ]
}
```

A small rule-based intent translator is also available before real LLM integration:

```text
일꾼으로 정찰해
scout with worker
마린 전진
```

Use `--print-plan` to inspect the canonical JSON without launching SC2. Use `--print-state` to start SC2, capture the initial observation, print a JSON game-state summary, and exit.

Example state summary:

```json
{
  "minerals": 50,
  "vespene": 0,
  "supply": {"used": 12, "cap": 15, "left": 3},
  "workers": 12,
  "townhalls": 1,
  "army": {},
  "known_enemy_units": 0,
  "game_time_seconds": 0.0
}
```

Notes:

- This targets StarCraft II because Blizzard's SC2 API supports macOS clients.
- Classic StarCraft/Brood War bot control is not the macOS-friendly path for this MVP.
- If your SC2 install is in a custom location, set `SC2PATH` to the StarCraft II install directory.
- `--check` verifies both the SC2 app path and the API `Maps` directory.
- The default map is `AbyssalReefLE`; if your install does not have that map, pass another installed map with `--map "Map Name"`.
- If launch fails with an SC2 API websocket timeout, open StarCraft II once from Battle.net, finish first-run setup/login/update prompts, quit the game, and rerun the script.

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
- real SC2 `move worker/marine x y` and `wait seconds` strategy-plan parser;
- canonical JSON StrategyPlan parser/serializer for future LLM output;
- tiny rule-based intent translator for examples like `일꾼으로 정찰해`;
- real SC2 bot runner that executes sequential movement/wait plans through the StarCraft II API;
- `--print-state` game-state summary JSON for future LLM observation input;
- local detection for common macOS SC2 install paths.

Not implemented yet:

- full build-order or combat strategy execution;
- real LLM-backed natural-language strategy planning;
- computer vision;
- Brood War/BWAPI integration.
