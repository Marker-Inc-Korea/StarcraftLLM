const EPSILON = 0.001;

export class Unit {
  constructor({ id, name, x, y, speed = 140, radius = 14, color = "#60a5fa" }) {
    if (!id) throw new Error("Unit id is required");
    this.id = id;
    this.name = name ?? id;
    this.x = x;
    this.y = y;
    this.speed = speed;
    this.radius = radius;
    this.color = color;
    this.target = { x, y };
  }

  moveTo(x, y) {
    this.target = { x: Number(x), y: Number(y) };
  }

  update(deltaSeconds) {
    const dx = this.target.x - this.x;
    const dy = this.target.y - this.y;
    const distance = Math.hypot(dx, dy);

    if (distance <= EPSILON) {
      this.x = this.target.x;
      this.y = this.target.y;
      return false;
    }

    const step = Math.min(distance, this.speed * deltaSeconds);
    this.x += (dx / distance) * step;
    this.y += (dy / distance) * step;
    return step < distance;
  }

  get isMoving() {
    return Math.hypot(this.target.x - this.x, this.target.y - this.y) > EPSILON;
  }
}

export class GameWorld {
  constructor({ width, height, units = [] }) {
    this.width = width;
    this.height = height;
    this.units = new Map(units.map((unit) => [unit.id, unit]));
  }

  addUnit(unit) {
    this.units.set(unit.id, unit);
  }

  getUnit(id) {
    return this.units.get(id);
  }

  moveUnit(id, x, y) {
    const unit = this.getUnit(id);
    if (!unit) throw new Error(`Unknown unit: ${id}`);

    const targetX = clamp(Number(x), 0, this.width);
    const targetY = clamp(Number(y), 0, this.height);
    unit.moveTo(targetX, targetY);
    return { unit, target: { x: targetX, y: targetY } };
  }

  update(deltaSeconds) {
    for (const unit of this.units.values()) {
      unit.update(deltaSeconds);
    }
  }
}

export function parseMoveCommand(command, defaultUnitId = "marine-1") {
  const parts = command.trim().split(/\s+/);
  if (parts[0]?.toLowerCase() !== "move") {
    throw new Error("Only move commands are supported for this prototype");
  }

  if (parts.length === 3) {
    const [x, y] = parts.slice(1).map(Number);
    assertCoordinates(x, y);
    return { unitId: defaultUnitId, x, y };
  }

  if (parts.length === 4) {
    const unitId = normalizeUnitName(parts[1]);
    const [x, y] = parts.slice(2).map(Number);
    assertCoordinates(x, y);
    return { unitId, x, y };
  }

  throw new Error("Use: move Marine 620 360 or move 620 360");
}

export function createDefaultWorld() {
  return new GameWorld({
    width: 800,
    height: 520,
    units: [
      new Unit({
        id: "marine-1",
        name: "Marine",
        x: 120,
        y: 120,
        speed: 150,
        color: "#38bdf8"
      })
    ]
  });
}

function normalizeUnitName(name) {
  if (name.toLowerCase() === "marine") return "marine-1";
  return name;
}

function assertCoordinates(x, y) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new Error("Coordinates must be numbers");
  }
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
