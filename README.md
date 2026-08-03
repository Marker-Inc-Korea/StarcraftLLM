# StarcraftLLM

StarcraftLLM은 자연어 목표를 **검증 가능한 StarCraft II 명령 계획**으로 바꾸고, 통과한 계획만 BurnySC2 봇이 실행하도록 만드는 프로젝트입니다.

핵심 경계는 다음과 같습니다.

```text
자연어 목표 -> LLM/룰 플래너 -> StrategyPlan -> 로컬 검증기 -> BurnySC2 실행기
```

LLM이 임의의 Python 코드나 SC2 API를 직접 호출하지 않습니다. 대신 공개된 명령 함수, Terran 카탈로그, 정적 능력 allowlist에서만 행동을 선택하고, 자원·보급·선행 건물·애드온·업그레이드·대상 형태를 실행 전에 검증합니다.

## LLM 명령 표면

기존 18개 매크로/제어 명령만으로는 표준 Terran 게임을 표현하기에 부족했습니다. Orbital 스캔/MULE, Siege Tank 모드 전환, Medivac 드랍, Raven/Battlecruiser 능력, lift/land, load/unload, nuke, cancel/salvage 같은 핵심 Terran 상호작용이 모두 빠져 있었기 때문입니다.

`starcraft_llm/commands.py`는 현재 **41개 provider-neutral 함수 호출 명령**을 노출합니다. 기존 18개 함수 이름과 좌표 호출은 유지하면서, 정확한 tag 기반 actor/target 선택, 도착·처치 완료 동기화, bounded 카이팅, 지속 생산, 자원 반환, wall placement, add-on swap과 능력 명령을 확장했습니다.

| 영역 | 함수 | 의미 |
| --- | --- | --- |
| 이동/전투 | `move`, `move_target`, `move_and_wait`, `attack_move`, `attack_enemy`, `attack_target`, `focus_fire`, `kite`, `patrol`, `hold_position`, `stop` | 지점 이동, 도착 확인, 아군 추종, 적 type/selector/tag 공격, 처치 확인, 쿨다운 기반 bounded 카이팅 |
| 집결/대기 | `rally`, `wait`, `wait_until` | 생산 건물·벙커 집결지와 자원/병력/적/위치/수송/피격 상태 기반 대기 |
| 경제 | `gather`, `return_cargo`, `distribute_workers` | 특정 자원 tag 채취, 화물 반환과 일꾼 재분배 |
| 생산/건설 | `train`, `produce_until`, `maintain_production`, `build`, `expand`, `build_addon` | 고정 batch, 목표 수까지 blocking 생산, 이후 행동과 병행하는 background 생산, 건물·확장·애드온 건설 |
| 테크/유지 | `morph`, `research`, `repair` | 사령부 변환, 업그레이드 연구, 수리 |
| 정적 능력 | `use_ability` | `ABILITY_SPECS`에 등록된 Terran 능력만 직접 사용 |
| 능력 래퍼 | `scan`, `call_down_mule`, `supply_drop`, `transform`, `lift`, `land`, `land_on_addon`, `load`, `unload`, `cancel`, `salvage`, `build_nuke`, `launch_nuke`, `replan` | 자주 쓰는 능력, add-on swap, 개별 수송/하차를 typed wrapper로 표현 |

`llm_command_function_schemas()`는 이 41개 함수를 JSON function declaration으로 반환합니다. `strategy_plan_from_function_calls()`는 일반 함수 호출과 OpenAI 스타일 중첩 함수 호출 payload를 동일한 `StrategyPlan`으로 변환합니다.

```python
from starcraft_llm.commands import strategy_plan_from_function_calls

plan = strategy_plan_from_function_calls(
    [
        {"name": "scan", "arguments": {"location": "enemy_main"}},
        {"name": "call_down_mule", "arguments": {"location": "nearest_mineral"}},
        {"name": "transform", "arguments": {"ability": "siege_mode", "actor": "siege_tank"}},
    ]
)
```

함수 호출 결과도 항상 기존 wire contract인 `{"actions": [...]}`로 직렬화할 수 있으므로 Gemini, OpenAI, 로컬 모델, HTTP planner가 같은 검증기와 실행기를 공유할 수 있습니다.

## Terran 명령 카탈로그

`starcraft_llm/command_catalog.py`가 LLM prompt, 파서, 검증기, 실행기의 단일 명칭/비용/선행 조건 소스입니다. 영어·한국어 별칭을 canonical snake_case 키로 정규화합니다.

- 생산 가능한 표준 유닛 17종: SCV, Marine, Marauder, Reaper, Ghost, Hellion, Hellbat, Widow Mine, Cyclone, Siege Tank, Thor, Viking, Medivac, Liberator, Raven, Banshee, Battlecruiser
- 소환 제어 유닛 2종: MULE(이동·채취·수리)과 Raven Auto Turret(공격·집중 공격·hold/stop)
- 비행 이동 건물 5종: lift된 Command Center, Orbital Command, Barracks, Factory, Starport (`move`/`patrol`/`hold`/`stop`; grounded form은 실행기에서 제외)
- 건물 13종: Command Center, Supply Depot, Refinery, Barracks, Engineering Bay, Bunker, Missile Turret, Sensor Tower, Factory, Ghost Academy, Starport, Armory, Fusion Core
- 애드온 6종: Barracks/Factory/Starport의 Tech Lab과 Reactor
- 사령부 변환 2종: Orbital Command, Planetary Fortress
- Terran 업그레이드 31종: 보병·차량·함선 공방업과 Stimpack, Combat Shield, Cloaking, Yamato 등
- Terran 능력 83종: 아래 allowlist만 `use_ability` 또는 typed wrapper로 사용 가능

카탈로그 메타데이터는 [SC2 5.0.15](https://news.blizzard.com/en-us/article/24225313/starcraft-ii-5-0-15-patch-notes) 기준으로 고정되어 있고, 모든 enum 이름은 프로젝트 `.venv`에 설치된 BurnySC2의 `UnitTypeId`/`UpgradeId`/`AbilityId`와 대조합니다. BurnySC2가 없는 별도 Python 환경에서만 enum 대조 테스트가 skip됩니다.

### 능력 allowlist

`use_ability`는 raw `AbilityId` 문자열을 받지 않습니다. 아래 canonical key만 받으며, 각 key는 내부적으로 BurnySC2 `AbilityId` enum 이름과 대상 형태(`none`, `point`, `unit`, MULE용 `mineral`)를 가집니다.

| 범주 | 능력 key |
| --- | --- |
| 보병/은폐/유닛 능력 | `stim_marine`, `stim_marauder`, `kd8_charge`, `ghost_cloak_on`, `ghost_cloak_off`, `ghost_hold_fire_on`, `ghost_hold_fire_off`, `ghost_snipe`, `ghost_emp`, `ghost_nuke_call_down`, `banshee_cloak_on`, `banshee_cloak_off`, `medivac_afterburners`, `medivac_heal` |
| 모드 전환 | `morph_hellbat`, `morph_hellion`, `siege_mode`, `unsiege_mode`, `thor_high_impact_mode`, `thor_explosive_mode`, `viking_assault_mode`, `viking_fighter_mode`, `liberator_ag_mode`, `liberator_aa_mode`, `lower_supply_depot`, `raise_supply_depot` |
| 고급 전투 능력 | `widow_mine_burrow_down`, `widow_mine_burrow_up`, `widow_mine_attack`, `cyclone_lock_on`, `cyclone_cancel_lock_on`, `raven_auto_turret`, `raven_interference_matrix`, `raven_anti_armor_missile`, `battlecruiser_tactical_jump`, `battlecruiser_yamato` |
| Orbital Command | `scan`, `call_down_mule`, `supply_drop` |
| MULE | `mule_gather`, `mule_repair` |
| 건물 lift/land | `lift_command_center`, `land_command_center`, `lift_orbital_command`, `land_orbital_command`, `lift_barracks`, `land_barracks`, `lift_factory`, `land_factory`, `lift_starport`, `land_starport` |
| 수송 | `load_command_center`, `load_all_command_center`, `unload_all_command_center`, `unload_unit_command_center`, `load_bunker`, `unload_all_bunker`, `unload_unit_bunker`, `load_medivac`, `unload_all_medivac`, `unload_unit_medivac` |
| 핵 | `build_nuke`, `launch_nuke` |
| 취소 | `cancel_any`, `cancel_build_in_progress`, `cancel_queue_1`, `cancel_queue_5`, `cancel_queue_addon`, `cancel_slot`, `cancel_slot_queue_cancel_to_selection`, `cancel_slot_queue_passive`, `cancel_slot_queue_passive_cancel_to_selection`, `cancel_addon_barracks`, `cancel_addon_factory`, `cancel_addon_starport`, `cancel_morph_orbital`, `cancel_morph_planetary_fortress`, `cancel_morph_thor_explosive_mode`, `cancel_lock_on`, `cancel_nuke`, `cancel_last` |
| 회수 | `salvage_bunker`, `salvage_sensor_tower` |

### 대상 형태와 선택 제한

능력 명령은 명시적인 target shape를 사용합니다.

| 형태 | 입력 | 예시 |
| --- | --- | --- |
| 대상 없음 | `ability`, `actor`, optional `selection`, `queued` | `{"type":"use_ability","ability":"stim_marine","actor":"marine"}` |
| 지점 대상 | `location` 또는 `x`/`y` | `{"type":"scan","location":"enemy_main"}` |
| 유닛 대상 | `target_unit`, 관측된 `target_tag`, 또는 semantic selector | `{"type":"attack_target","unit":"viking","target_unit":"nearest_enemy_air"}` |
| 미네랄 대상 | 미네랄을 찾을 `location` 또는 `x`/`y` anchor | `{"type":"call_down_mule","location":"nearest_mineral"}` |

유닛 대상 selector에는 `nearest_enemy`, `nearest_enemy_structure`, 지상/공중/생체/기계/거대/탐지기 필터, `lowest_health_enemy`, `highest_energy_enemy`, `nearest_friendly`, `lowest_health_friendly`, `highest_energy_friendly`, `damaged_friendly`, `any_friendly`가 포함됩니다. 능력별 `target_filter`가 Snipe/Medivac Heal의 생체 유닛, Interference Matrix의 기계 또는 사이오닉 유닛, MULE 수리의 기계 대상, Supply Drop의 보급고, Bunker/Medivac/Command Center의 적재 가능 유닛을 검증기와 실행기 양쪽에서 제한합니다.

Semantic location allowlist는 정확히 다음 20개입니다.

```text
own_main, own_natural, own_third, own_ramp,
own_ramp_depot_1, own_ramp_depot_2, own_ramp_depot_middle,
own_ramp_barracks, own_ramp_barracks_with_addon,
enemy_main, enemy_natural, enemy_third,
map_center, frontline, retreat, proxy, next_expansion,
nearest_enemy, nearest_enemy_structure, nearest_mineral
```

`selection`은 persistent control group이 아니라 단일 action 안에서만 쓰는 bounded selector입니다.

```json
{"selection": {"mode": "highest_energy", "count": 1, "tags": [12345]}}
```

허용 mode는 `all`, `ready`, `idle`, `closest`, `lowest_health`, `highest_energy`입니다. `selection.tags`는 관측 payload의 실제 unit tag로 actor를 정확히 고릅니다. `target_selection`, `producer_selection`, `researcher_selection`도 같은 형식을 사용합니다. `selection.count`와 유닛 생산 count 상한은 200입니다. 전체 plan은 최대 24 actions, 건물·애드온·확장 count는 최대 20, worker 배정·수리 count는 최대 100입니다. 좌표는 0~256 범위, 단순 `wait`는 최대 30초이며 동적 대기·생산 정책은 timeout을 반드시 갖고 최대 1200초로 제한됩니다. 긴 반복은 action 복제 대신 `count`, `produce_until`, `maintain_production`, bounded `wait_until`, `replan` checkpoint로 표현합니다.

`wait_until`은 자원/보급/게임 시간뿐 아니라 `army_supply`, 적 유닛·건물 수, idle 생산 건물, 생산 가능 슬롯, 수송 화물, 특정 위치 주변 아군·적, 기지 피격 상태를 관측합니다. 위치 조건에는 `radius`, 모든 동적 조건에는 `timeout_seconds`와 `on_timeout`(`replan` 또는 `fail`)을 지정할 수 있습니다.

건설은 기본 `placement_mode="near"` 외에 `placement_mode="exact"`, `max_distance`, `reserve_addon_space`를 지원합니다. own-ramp semantic slot과 함께 쓰면 depot/barracks wall 위치를 정확히 요청할 수 있습니다. `land_on_addon`은 관측된 add-on type/tag의 `add_on_land_position`으로 생산 건물을 내려 add-on swap을 수행합니다.

## StrategyPlan 입력 형식

### 1. DSL

여러 행동은 세미콜론, 줄바꿈 또는 `then`으로 연결합니다.

```text
gather minerals 8; wait until minerals 100; build supply depot
build factory; wait until structure factory ready; addon factory tech lab; wait until structure factory tech lab ready; train siege tank 2
use ability stim marine with marine count 8; attack marine enemy
scan enemy main; mule nearest mineral; supply drop supply depot
transform siege tank siege; lift barracks; land barracks proxy
load medivac marine 8; unload medivac enemy main
cancel queue 1; salvage bunker; build nuke; launch nuke enemy main; replan ability unavailable
```

### 2. JSON

```json
{
  "actions": [
    {"type": "scan", "location": "enemy_main"},
    {"type": "call_down_mule", "location": "nearest_mineral"},
    {"type": "use_ability", "ability": "raven_auto_turret", "actor": "raven", "location": "frontline", "selection": {"mode": "closest", "count": 1}},
    {"type": "use_ability", "ability": "battlecruiser_yamato", "actor": "battlecruiser", "target": "nearest_enemy_structure"}
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

## 대표 전략 예시

아래 예시는 모두 `strategy_plan_from_dict()`와 `validate_strategy_plan()`으로 검증되는 shape입니다. 실제 성공 여부는 맵, 상대, 현재 관측, 컨트롤 품질에 따라 달라집니다.

### Proxy Barracks pressure

```json
{
  "actions": [
    {"type": "build", "building": "barracks", "location": "proxy", "selection": {"mode": "closest", "count": 1}},
    {"type": "train", "unit": "marine", "count": 3},
    {"type": "rally", "building": "barracks", "location": "enemy_natural"},
    {"type": "attack", "unit": "marine", "location": "enemy_main", "selection": {"count": 3}}
  ]
}
```

### Medivac drop

```json
{
  "actions": [
    {"type": "load", "transport": "medivac", "unit": "marine", "count": 8},
    {"type": "use_ability", "ability": "medivac_afterburners", "actor": "medivac", "selection": {"count": 1}},
    {"type": "move_and_wait", "unit": "medivac", "location": "enemy_main", "timeout_seconds": 90},
    {"type": "unload", "transport": "medivac", "location": "enemy_main"}
  ]
}
```

### Orbital scan and MULE via function calls

```json
[
  {"name": "scan", "arguments": {"location": "enemy_main"}},
  {"name": "call_down_mule", "arguments": {"location": "nearest_mineral"}},
  {"name": "supply_drop", "arguments": {"target_unit": "supply_depot"}}
]
```

### Siege, air control, and nuke

```json
{
  "actions": [
    {"type": "transform", "actor": "siege_tank", "mode": "siege", "selection": {"count": 2}},
    {"type": "use_ability", "ability": "liberator_ag_mode", "actor": "liberator", "location": "frontline"},
    {"type": "use_ability", "ability": "raven_interference_matrix", "actor": "raven", "target": "nearest_enemy"},
    {"type": "use_ability", "ability": "battlecruiser_tactical_jump", "actor": "battlecruiser", "location": "enemy_main"},
    {"type": "build_nuke"},
    {"type": "use_ability", "ability": "ghost_cloak_on", "actor": "ghost"},
    {"type": "launch_nuke", "location": "enemy_main"}
  ]
}
```

## 검증과 실행

`starcraft_llm/validator.py`는 최대 24개 행동을 순서대로 시뮬레이션합니다.

- 좌표 범위, 유한 숫자, 대기 시간, count 상한 검증
- 현재/계획된 minerals, vespene, supply 사용량 추적
- worker, townhall, 생산 건물, 선행 건물, Tech Lab/Reactors 검증
- 건설 시작과 `wait_until structure_ready`를 구분
- 업그레이드 비용, 연구 건물, 선행 레벨 및 완료 대기 검증
- gas 채취, expansion, morph, repair, rally 대상 검증
- blocking/background 생산 정책의 절대 목표 수, 자원·가스·보급 reserve, timeout 검증
- 도착 확인, 처치 확인, bounded 카이팅, 위치/적/수송/피격 조건의 target shape 검증
- ability key, actor, target shape, semantic location, selector, queued flag 검증
- 알 수 없는 명령·유닛·건물·업그레이드·능력 거부

능력의 에너지, 쿨다운, 현재 form, 주문 가능 여부는 정적 검증만으로 확정하지 않습니다. `starcraft_llm/sc2_bot.py`가 실행 직전에 BurnySC2 `get_available_abilities`(주입 runtime에서는 동등한 query hook)로 live availability를 확인하고, 후보가 여러 기면 실제 사용 가능한 source를 먼저 고른 뒤 명령을 발행합니다. 공성/비행/매설/하강 형태의 별도 `UnitTypeId`도 canonical actor로 합쳐 선택합니다.

능력이 아직 불가능하거나 대상 semantic location을 해결할 수 없으면 executor는 같은 action을 deterministic하게 재시도합니다. 기본 재시도 창은 30 in-game seconds입니다. 그 뒤 `original_strategy`가 있고 replan 한도(최대 2회)가 남아 있으면 새 관측으로 재계획하고, 한도를 넘으면 게임을 떠나 terminal failure로 종료합니다. `replan` action은 의도적인 bounded closed-loop checkpoint입니다.

관측 payload에는 다음이 포함됩니다.

- minerals, vespene
- supply used/cap/left
- worker 수, townhall 수, army unit counts
- structures / structures_ready / structures_pending
- 완료된 upgrades
- known enemy unit 수, game time seconds, 적 유닛/구조물과 중립 자원별 관측 정보
- ability-relevant `unit_observations`: tag, 위치, health, energy, ready/flying/burrowed/loaded/idle, cargo, orders
- 정밀 제어 관측: add-on tag, passenger tags/types, biological/mechanical/psionic/massive/detector flags, weapon cooldown
- allowlisted `semantic_locations` snapshot

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
python scripts/run_sc2_movement.py --strategy "scan enemy main; mule nearest mineral" --print-plan
python scripts/run_sc2_movement.py --strategy "load medivac marine 8; unload medivac enemy main" --print-plan
python scripts/run_sc2_movement.py --strategy "transform siege tank siege; launch nuke enemy main" --print-plan
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

테스트는 브라우저 이동, DSL/JSON/자연어 파싱, 함수 schema/adapter, 카탈로그 별칭, complete Terran ability surface, 상태 시뮬레이션 검증, fake BurnySC2 실행을 다룹니다.

## 호환성과 현재 제한

- 기존 18개 함수 이름, JSON root shape `{"actions": [...]}`, JSON action array shortcut, OpenAI-style nested function-call adapter는 유지됩니다.
- `openai` planner와 외부 `server` planner 클라이언트는 아직 연결되지 않았지만, 공통 함수 schema와 adapter는 준비되어 있습니다.
- 현재 범위는 **Terran standard melee command surface**입니다. 표준 Terran 유닛·건물·업그레이드·능력 표현을 넓힌 것이며, 임의 SC2 API 호출, 비-Terran 종족, 커스텀 모드, 임의 adaptive ladder AI를 의미하지 않습니다.
- 함수 표면은 bounded 멀티 프레임 생산·이동·전투·조건 관측을 제공하지만, 적 조합에 따른 장기 전략 선택과 완성형 래더 승률은 플래너/정책 품질의 문제이며 보장하지 않습니다.
- wall slot과 exact placement는 지원하지만, 해당 ramp 속성을 제공하지 않는 맵이나 막힌 위치에서는 재시도/재계획하며 모든 맵의 최적 wall을 보장하지 않습니다.
- SC2가 플레이어 명령으로 노출하지 않는 자동 공격/내부 ability는 함수로 가장하지 않습니다. 다만 BurnySC2가 실제 플레이어 명령으로 노출하는 매설 Widow Mine의 명시적 target fire는 `widow_mine_attack`으로 제공합니다.
- BurnySC2가 여러 queue-cancel enum을 같은 `CANCEL_LAST` 동작으로 redirect하는 경우에는 관측 order ID로 임의 queue index 하나만 취소한다고 보장하지 않습니다.
- live ability availability는 executor가 확인하지만, 그것이 전투 micro 품질이나 승리를 보장하지 않습니다.

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

StarcraftLLM은 **LLM의 목표를 허용 목록 기반 Terran 명령 함수와 정적 능력 카탈로그로 제한하고, 계획 전체를 검증한 뒤 실제 StarCraft II API로 실행하는 안전한 명령 계층**입니다.
