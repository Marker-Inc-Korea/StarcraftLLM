# StarcraftLLM

StarcraftLLM은 자연어 목표를 **검증 가능한 StarCraft II 명령 계획**으로 바꾸고, 통과한 계획만 BurnySC2 봇이 실행하도록 만드는 프로젝트입니다.

핵심 경계는 다음과 같습니다.

```text
자연어 목표 -> LLM/룰 플래너 -> StrategyPlan -> 로컬 검증기 -> BurnySC2 실행기
```

LLM이 임의의 Python 코드나 SC2 API를 직접 호출하지 않습니다. 대신 공개된 명령 함수와 카탈로그에서만 행동을 선택하고, 자원·보급·선행 건물·애드온·업그레이드 조건을 실행 전에 검증합니다.

## LLM 명령 표면

`starcraft_llm/commands.py`는 LLM이 호출할 수 있는 18개 안전 명령 함수를 제공합니다.

| 영역 | 함수 | 의미 |
| --- | --- | --- |
| 이동/전투 | `move`, `attack_move`, `attack_enemy`, `patrol`, `hold_position`, `stop` | 유닛 그룹 이동과 기본 교전 명령 |
| 집결/대기 | `rally`, `wait`, `wait_until` | 생산 건물 집결지 및 상태 기반 대기 |
| 경제 | `gather`, `distribute_workers` | 미네랄/가스 채취와 일꾼 재분배 |
| 생산/건설 | `train`, `build`, `expand`, `build_addon` | 유닛 생산, 건물·확장·애드온 건설 |
| 테크/유지 | `morph`, `research`, `repair` | 사령부 변환, 업그레이드 연구, 수리 |

`llm_command_function_schemas()`는 이 함수들을 provider-neutral JSON function declaration으로 반환합니다. `strategy_plan_from_function_calls()`는 일반 함수 호출과 OpenAI 스타일의 중첩 함수 호출 payload를 동일한 `StrategyPlan`으로 변환합니다.

```python
from starcraft_llm.commands import (
    llm_command_function_schemas,
    strategy_plan_from_function_calls,
)

function_schemas = llm_command_function_schemas()

plan = strategy_plan_from_function_calls(
    [
        {"name": "build", "arguments": {"building": "supply_depot"}},
        {
            "name": "wait_until",
            "arguments": {
                "condition": "structure_ready",
                "target": "supply_depot",
                "at_least": 1,
            },
        },
        {"name": "build", "arguments": {"building": "barracks"}},
    ]
)
```

함수 호출 결과도 항상 기존 wire contract인 `{"actions": [...]}`로 직렬화할 수 있으므로 Gemini, OpenAI, 로컬 모델, HTTP planner가 같은 검증기와 실행기를 공유할 수 있습니다.

## Terran 명령 카탈로그

`starcraft_llm/command_catalog.py`가 LLM prompt, 파서, 검증기, 실행기의 단일 명칭/비용/선행 조건 소스입니다. 영어·한국어 별칭을 canonical snake_case 키로 정규화합니다.

- 표준 유닛 17종: SCV, Marine, Marauder, Reaper, Ghost, Hellion, Hellbat, Widow Mine, Cyclone, Siege Tank, Thor, Viking, Medivac, Liberator, Raven, Banshee, Battlecruiser
- 건물 13종: Command Center, Supply Depot, Refinery, Barracks, Engineering Bay, Bunker, Missile Turret, Sensor Tower, Factory, Ghost Academy, Starport, Armory, Fusion Core
- 애드온 6종: Barracks/Factory/Starport의 Tech Lab과 Reactor
- 사령부 변환 2종: Orbital Command, Planetary Fortress
- Terran 업그레이드 31종: 보병·차량·함선 공방업과 Stimpack, Combat Shield, Cloaking, Yamato 등

카탈로그 메타데이터는 [SC2 5.0.15](https://news.blizzard.com/en-us/article/24225313/starcraft-ii-5-0-15-patch-notes) 기준으로 고정되어 있고, 모든 enum 이름은 설치된 BurnySC2의 `UnitTypeId`/`UpgradeId`와 대조합니다.

## StrategyPlan 입력 형식

### 1. DSL

여러 행동은 세미콜론, 줄바꿈 또는 `then`으로 연결합니다.

```text
gather minerals 8; wait until minerals 100; build supply depot
build supply depot; wait until structure supply depot ready; build barracks
build factory; wait until structure factory ready; addon factory tech lab; wait until structure factory tech lab ready; train siege tank 2
addon barracks tech lab; wait until structure barracks tech lab ready; research stimpack; wait until upgrade stimpack complete
expand; wait until townhalls 2; distribute workers 2
rally barracks 45 42; patrol marine 55 45; hold marine
```

### 2. JSON

```json
{
  "actions": [
    {"type": "gather", "unit": "worker", "resource": "minerals", "workers": 8},
    {"type": "wait_until", "condition": "minerals", "at_least": 100},
    {"type": "build", "building": "supply_depot", "worker": "worker", "count": 1},
    {"type": "wait_until", "condition": "structure_ready", "target": "supply_depot", "at_least": 1},
    {"type": "build", "building": "barracks", "worker": "worker", "count": 1}
  ]
}
```

### 3. 자연어 의도

로컬 `rule` planner는 자주 쓰는 한국어/영어 의도를 deterministic하게 번역합니다.

```text
일꾼으로 정찰해
미네랄 캐
공성 전차 생산
앞마당 확장
병영 기술실 건설
스팀팩 연구
병영 수리
```

더 긴 목표와 빌드 오더는 `gemini` planner가 동일한 카탈로그와 `StrategyPlan` schema를 사용해 생성합니다.

## 검증과 실행

`starcraft_llm/validator.py`는 최대 10개 행동을 순서대로 시뮬레이션합니다.

- 좌표 범위, 유한 숫자, 대기 시간, count 상한 검증
- 현재/계획된 minerals, vespene, supply 사용량 추적
- worker, townhall, 생산 건물, 선행 건물, Tech Lab/Reactors 검증
- 건설 시작과 `wait_until structure_ready`를 구분
- 업그레이드 비용, 연구 건물, 선행 레벨 및 완료 대기 검증
- gas 채취, expansion, morph, repair, rally 대상 검증
- 알 수 없는 명령·유닛·건물·업그레이드 거부

`starcraft_llm/sc2_bot.py`는 검증된 행동을 BurnySC2 API로 실행합니다. 건설 위치와 확장 위치는 executor가 선택하며, 생산 가능한 건물과 필요한 애드온을 자동으로 찾습니다.

관측 payload에는 다음이 포함됩니다.

- minerals, vespene
- supply used/cap/left
- worker 수, townhall 수, army unit counts
- structures / structures_ready / structures_pending
- 완료된 upgrades
- known enemy unit 수, game time seconds

## 실행 방법

### Python 환경

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### StarCraft II 설치 확인

SC2 API용 맵 파일이 필요합니다. Blizzard map pack(<https://github.com/Blizzard/s2client-proto#map-packs>)을 받아 StarCraft II의 `Maps/` 아래에 압축 해제하세요. 기본 맵은 `AbyssalReefLE`입니다.

```bash
python scripts/run_sc2_movement.py --check
```

설치 위치가 다르면 `SC2PATH`를 지정합니다.

```bash
export SC2PATH="/path/to/StarCraft II"
```

### 실제 봇 실행 예시

```bash
python scripts/run_sc2_movement.py --strategy "build supply depot; wait until structure supply depot ready; build barracks" --print-plan
python scripts/run_sc2_movement.py --strategy "expand; wait until townhalls 2; distribute workers" --print-plan
python scripts/run_sc2_movement.py --strategy "병영 기술실 건설" --print-plan
python scripts/run_sc2_movement.py --print-state --fast
```

### Gemini planner

```bash
export GEMINI_API_KEY="your-key-here"
python scripts/run_sc2_movement.py --planner gemini --observe-before-plan --strategy "2병영을 짓고 기술실을 단 뒤 스팀팩을 연구해" --print-plan
```

로컬 테스트용 key 파일은 `.secrets/gemini_api_key.txt`에 둘 수 있습니다. API key, key가 포함된 화면이나 로그는 커밋하지 마세요.

### 브라우저 이동 프로토타입

별도의 캔버스 기반 Marine 이동 프로토타입도 유지됩니다.

```bash
open index.html
```

## 테스트

```bash
npm test
python3 -m unittest discover -s tests -v
npm run test:all
```

테스트는 브라우저 이동, DSL/JSON/자연어 파싱, 함수 schema/adapter, 카탈로그 별칭, 상태 시뮬레이션 검증, fake BurnySC2 실행을 다룹니다.

## 현재 제한

- `openai` planner와 외부 `server` planner 클라이언트는 아직 연결되지 않았지만, 공통 함수 schema와 adapter는 준비되어 있습니다.
- 실시간 멀티 사이클 closed-loop replanning, 적 조합에 따른 전략 전환, 완성형 래더 AI는 범위 밖입니다.
- 건물 배치는 안전한 근처 위치를 찾는 단순 정책이며 wall-off나 지형 최적화는 하지 않습니다.
- scan, MULE, siege/unsiege, cloak, lift/land, load/unload처럼 대상·에너지·현재 모드 검증이 필요한 개별 능력은 아직 공개 명령 표면에 포함하지 않습니다.
- 현재 카탈로그와 검증기는 Terran 중심입니다.

## 디렉터리 구조

```text
src/                  브라우저 캔버스 이동 prototype
test/                 JavaScript 테스트
starcraft_llm/        command catalog, StrategyPlan, planner, validator, SC2 bot
tests/                Python 단위 테스트
scripts/              SC2 실행 entrypoint
docs/                 개발 원칙 문서
```

## 한 줄 요약

StarcraftLLM은 **LLM의 목표를 허용 목록 기반 Terran 명령 함수로 제한하고, 계획 전체를 검증한 뒤 실제 StarCraft II API로 실행하는 안전한 명령 계층**입니다.
