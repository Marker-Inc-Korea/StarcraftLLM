import test from "node:test";
import assert from "node:assert/strict";
import { GameWorld, Unit, parseMoveCommand } from "../src/game.js";

test("moveUnit stores a clamped target for a unit", () => {
  const world = new GameWorld({
    width: 100,
    height: 80,
    units: [new Unit({ id: "marine-1", x: 10, y: 10 })]
  });

  const result = world.moveUnit("marine-1", 140, -20);

  assert.deepEqual(result.target, { x: 100, y: 0 });
  assert.deepEqual(world.getUnit("marine-1").target, { x: 100, y: 0 });
});

test("unit advances toward target without overshooting", () => {
  const unit = new Unit({ id: "marine-1", x: 0, y: 0, speed: 10 });
  unit.moveTo(15, 0);

  unit.update(1);
  assert.equal(unit.x, 10);
  assert.equal(unit.y, 0);
  assert.equal(unit.isMoving, true);

  unit.update(1);
  assert.equal(unit.x, 15);
  assert.equal(unit.y, 0);
  assert.equal(unit.isMoving, false);
});

test("parseMoveCommand supports explicit and default marine commands", () => {
  assert.deepEqual(parseMoveCommand("move Marine 620 360"), {
    unitId: "marine-1",
    x: 620,
    y: 360
  });
  assert.deepEqual(parseMoveCommand("move 240 160"), {
    unitId: "marine-1",
    x: 240,
    y: 160
  });
});
