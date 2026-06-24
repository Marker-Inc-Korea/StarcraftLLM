# StarcraftLLM

A small first step toward a strategy-driven StarCraft agent: a browser-based unit movement prototype.

## Run on macOS

Open the visual prototype directly:

```bash
open index.html
```

You can then:

- click the map to move the Marine;
- type `move Marine 620 360` and press **명령 실행**;
- type the shorter `move 240 160` command.

## Development

Run the unit tests with the Node.js version already available on the machine:

```bash
npm test
```

## Current scope

Implemented now:

- a `Unit.moveTo(x, y)` movement function;
- a `GameWorld.moveUnit(unitId, x, y)` command surface;
- simple text command parsing for `move`;
- a canvas view so movement is visible.

Not implemented yet:

- connection to the real StarCraft game process;
- build-order or combat strategy execution;
- computer vision or external game APIs.
