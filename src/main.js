import { createDefaultWorld, parseMoveCommand } from "./game.js";

const canvas = document.querySelector("#game");
const context = canvas.getContext("2d");
const commandInput = document.querySelector("#command");
const runButton = document.querySelector("#run-command");
const statusOutput = document.querySelector("#status");
const world = createDefaultWorld();

let lastTimestamp = performance.now();

runButton.addEventListener("click", () => runTextCommand(commandInput.value));
commandInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") runTextCommand(commandInput.value);
});
canvas.addEventListener("click", (event) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = Math.round((event.clientX - rect.left) * scaleX);
  const y = Math.round((event.clientY - rect.top) * scaleY);
  moveMarine(x, y);
});

function runTextCommand(command) {
  try {
    const parsed = parseMoveCommand(command);
    moveMarine(parsed.x, parsed.y, parsed.unitId);
  } catch (error) {
    statusOutput.value = `명령 오류: ${error.message}`;
  }
}

function moveMarine(x, y, unitId = "marine-1") {
  const { target } = world.moveUnit(unitId, x, y);
  statusOutput.value = `Marine 이동 중 → (${Math.round(target.x)}, ${Math.round(target.y)})`;
}

function frame(timestamp) {
  const deltaSeconds = Math.min((timestamp - lastTimestamp) / 1000, 0.05);
  lastTimestamp = timestamp;
  world.update(deltaSeconds);
  draw();
  requestAnimationFrame(frame);
}

function draw() {
  drawTerrain();
  for (const unit of world.units.values()) {
    drawTarget(unit);
    drawUnit(unit);
  }
}

function drawTerrain() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#0f172a";
  context.fillRect(0, 0, canvas.width, canvas.height);

  context.strokeStyle = "#1e293b";
  context.lineWidth = 1;
  for (let x = 0; x <= canvas.width; x += 40) {
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, canvas.height);
    context.stroke();
  }
  for (let y = 0; y <= canvas.height; y += 40) {
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(canvas.width, y);
    context.stroke();
  }
}

function drawTarget(unit) {
  context.strokeStyle = unit.isMoving ? "#fbbf24" : "#475569";
  context.lineWidth = 2;
  context.beginPath();
  context.arc(unit.target.x, unit.target.y, 10, 0, Math.PI * 2);
  context.stroke();

  context.beginPath();
  context.moveTo(unit.target.x - 14, unit.target.y);
  context.lineTo(unit.target.x + 14, unit.target.y);
  context.moveTo(unit.target.x, unit.target.y - 14);
  context.lineTo(unit.target.x, unit.target.y + 14);
  context.stroke();
}

function drawUnit(unit) {
  context.fillStyle = "rgb(0 0 0 / 0.35)";
  context.beginPath();
  context.ellipse(unit.x + 4, unit.y + 7, unit.radius + 4, 7, 0, 0, Math.PI * 2);
  context.fill();

  context.fillStyle = unit.color;
  context.beginPath();
  context.arc(unit.x, unit.y, unit.radius, 0, Math.PI * 2);
  context.fill();

  context.strokeStyle = "#e0f2fe";
  context.lineWidth = 3;
  context.beginPath();
  context.moveTo(unit.x + unit.radius * 0.4, unit.y - unit.radius * 0.2);
  context.lineTo(unit.x + unit.radius + 9, unit.y - unit.radius * 0.35);
  context.stroke();

  context.fillStyle = "#e5e7eb";
  context.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
  context.textAlign = "center";
  context.fillText(unit.name, unit.x, unit.y - unit.radius - 8);
}

requestAnimationFrame(frame);
