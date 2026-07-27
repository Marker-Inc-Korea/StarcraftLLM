# StarcraftLLM

StarcraftLLM은 **자연어 전략 지시를 안전한 StarCraft II 실행 계획으로 바꾸고, 실제 SC2 API 봇이 그 계획을 수행하는 것**을 목표로 하는 MVP 프로젝트입니다.

현재는 완성형 게임 AI가 아니라, LLM이 만들 수 있는 전략 출력을 제한된 `StrategyPlan` JSON/DSL로 정규화한 뒤 검증하고 실행하는 **초기 통합 실험**입니다. 즉, “LLM이 자유 텍스트로 게임을 직접 조작한다”가 아니라 “LLM 또는 룰 기반 플래너가 작은 명령 목록을 만들고, 로컬 검증기를 통과한 명령만 봇이 실행한다”는 구조입니다.

## 프로젝트 목적

이 프로젝트가 검증하려는 핵심 질문은 다음과 같습니다.

1. **전략 지시 입력**: 한국어/영어 자연어 또는 간단한 DSL/JSON으로 스타크래프트 전략을 입력한다.
2. **계획 생성**: 입력을 `move`, `wait`, `gather`, `train`, `build`, `attack` 같은 작은 원자 행동으로 변환한다.
3. **안전 검증**: 실행 전에 행동 개수, 좌표 범위, 대기 시간, 자원/보급/건물 선행 조건을 검증한다.
4. **실제 실행**: 검증된 계획을 StarCraft II API를 통해 실제 커스텀 게임에서 수행한다.
5. **LLM 확장 지점 확보**: 향후 OpenAI/server 기반 플래너나 반복 재계획 루프를 붙일 수 있도록 `StrategyPlan` 계약을 먼저 고정한다.

## 현재 구현된 범위

### 1. 브라우저 이동 프로토타입

빠르게 확인할 수 있는 캔버스 기반 유닛 이동 프로토타입이 있습니다.

- `index.html`에서 직접 실행 가능
- 캔버스 클릭으로 Marine 이동
- `move Marine 620 360`, `move 240 160` 텍스트 명령 지원
- `src/game.js`에 `Unit`, `GameWorld`, `parseMoveCommand` 구현
- Node 테스트(`test/game.test.mjs`)로 이동/좌표 클램프/명령 파싱 검증

### 2. StrategyPlan 모델과 파서

`starcraft_llm/strategy.py`에 실제 SC2 봇이 실행할 수 있는 작은 전략 계획 모델이 구현되어 있습니다.

지원되는 원자 행동:

- `move`: worker/marine을 좌표로 이동
- `attack`: worker/marine attack-move
- `attack_enemy`: 현재 보이는 적 공격
- `wait`: 짧은 시간 대기
- `wait_until`: 자원, 보급, 건물 상태, 유닛 수 조건 대기
- `gather`: 미네랄/가스 채취
- `train`: SCV/Marine 생산, 반복 생산 수량 지원
- `build`: Supply Depot, Barracks, Refinery 건설

입력 형식은 세 가지를 지원합니다.

1. 간단한 DSL

```text
move worker 35 42; wait 1; move worker 45 42
gather minerals; wait until minerals 100; build supply depot
build supply depot; wait until structure supply depot ready; build barracks
train marine 2; wait until unit marine 2; attack marine enemy
```

2. JSON 계획

```json
{
  "actions": [
    {"type": "gather", "unit": "worker", "resource": "minerals"},
    {"type": "train", "unit": "scv", "count": 1},
    {"type": "wait_until", "condition": "minerals", "at_least": 100},
    {"type": "build", "building": "supply_depot", "worker": "worker"}
  ]
}
```

3. 아주 작은 자연어 의도 번역

```text
일꾼으로 정찰해
scout with worker
마린 전진
미네랄 캐
마린 생산
보급고 건설
```

### 3. 플래너 인터페이스

`starcraft_llm/planner.py`에 전략 텍스트를 `StrategyPlan`으로 바꾸는 플래너 인터페이스가 있습니다.

구현됨:

- `rule`: 기본값. 로컬 deterministic parser/intent translator 사용
- `gemini`: Gemini API를 호출해 같은 `StrategyPlan` JSON 계약으로 응답을 받음
- `--observe-before-plan`: SC2를 먼저 시작해 초기 게임 상태를 요약한 뒤 플래너에 전달

예약만 되어 있음:

- `openai`: 인터페이스 모드는 있으나 아직 미구현
- `server`: 외부 HTTP planner 모드는 있으나 아직 미구현

### 4. 실제 StarCraft II API 봇

`starcraft_llm/sc2_bot.py`와 `scripts/run_sc2_movement.py`에 실제 SC2 커스텀 게임 실행 경로가 구현되어 있습니다.

현재 봇이 할 수 있는 일:

- macOS/Windows/Linux의 일반적인 SC2 설치 경로 탐지
- `--check`로 SC2 앱과 Maps 디렉터리 확인
- Terran 대 매우 쉬운 Zerg 컴퓨터 커스텀 게임 시작
- 기본 맵: `AbyssalReefLE`
- worker/marine 선택 후 이동/공격 명령 실행
- 미네랄/가스 채취 명령 실행
- SCV/Marine 생산 명령 실행
- Supply Depot/Barracks/Refinery 건설 명령 실행
- 건설 시작 여부를 확인한 뒤 다음 행동으로 진행
- 조건 대기(`wait_until`) 기반으로 자원, 보급, 건물 ready/pending/count, 유닛 수를 확인
- 계획 완료 후 게임을 자동 종료

### 5. 게임 상태 요약

`starcraft_llm/game_state.py`에 LLM/플래너에 넘길 작은 관측 payload가 구현되어 있습니다.

포함 정보:

- minerals, vespene
- supply used/cap/left
- worker 수, townhall 수
- army 유닛 카운트
- structures / structures_ready / structures_pending
- known enemy unit 수
- game time seconds

`--print-state`로 실제 SC2를 시작한 뒤 초기 상태를 JSON으로 출력할 수 있습니다.

### 6. 실행 전 검증기

`starcraft_llm/validator.py`의 `validate_strategy_plan` 함수가 봇 실행 전에 계획을 검증합니다.

검증 범위:

- 빈 계획 거부
- 최대 행동 수 제한
- 이동/공격 좌표 범위 제한
- 너무 긴 `wait` 거부
- 지원하지 않는 유닛/조건/건물 거부
- worker 없이 채취하는 계획 거부
- refinery 없이 gas 채취하는 계획 거부
- 자원/보급 부족 상태에서 생산/건설하는 계획 거부
- Supply Depot 없이 Barracks를 짓는 계획 거부
- Barracks 없이 Marine을 생산하는 계획 거부
- `wait_until` 이후의 자원/유닛/건물 상태를 단순 시뮬레이션해 후속 행동 가능 여부 확인

### 7. 테스트

현재 테스트는 JavaScript 브라우저 로직과 Python SC2 계획/검증/봇 로직을 모두 다룹니다.

- `npm test`: 브라우저 prototype 로직 테스트
- `python3 -m unittest discover -s tests -v`: Python 전략 파서, 플래너, 검증기, SC2 봇 fake runtime 테스트
- `npm run test:all`: 전체 테스트 실행

## 실행 방법

### Python 환경 준비

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### StarCraft II 설치 확인

macOS 기준으로 StarCraft II를 Battle.net 앱으로 설치한 뒤, SC2 API용 맵 파일이 필요합니다. 공식 Blizzard map pack(<https://github.com/Blizzard/s2client-proto#map-packs>)을 받아 `/Applications/StarCraft II/Maps/` 아래에 압축 해제하세요. 기본 `AbyssalReefLE` 맵은 Ladder 2017 Season 1 pack에 포함되어 있습니다.

```bash
python scripts/run_sc2_movement.py --check
```

SC2 설치 위치가 다르면 `SC2PATH`를 설정할 수 있습니다.

```bash
export SC2PATH="/path/to/StarCraft II"
```

### 실제 SC2 봇 실행 예시

```bash
python scripts/run_sc2_movement.py --strategy "move worker 35 42"
python scripts/run_sc2_movement.py --strategy "move worker 35 42; wait 1; move worker 45 42"
python scripts/run_sc2_movement.py --strategy "gather minerals; train scv"
python scripts/run_sc2_movement.py --strategy "gather minerals; wait until minerals 100; build supply depot" --fast --stop-after 1
python scripts/run_sc2_movement.py --strategy "build supply depot; wait until structure supply depot ready; build barracks; wait until structure barracks ready; train marine; wait until unit marine 1; attack marine 55 45" --print-plan
python scripts/run_sc2_movement.py --strategy "일꾼으로 정찰해" --print-plan
python scripts/run_sc2_movement.py --print-state --fast
```

### Gemini planner 사용

`--planner gemini`는 Gemini API를 호출해 canonical `StrategyPlan` JSON을 생성합니다. 기본 모델은 `gemini-2.5-flash`입니다.

환경 변수로 API key를 지정합니다.

```bash
export GEMINI_API_KEY="your-key-here"
python scripts/run_sc2_movement.py --planner gemini --strategy "초반에 일꾼을 뽑고 미네랄을 캐" --print-plan
python scripts/run_sc2_movement.py --planner gemini --observe-before-plan --strategy "초반에 일꾼을 뽑고 미네랄을 캐" --fast --stop-after 1
```

로컬 테스트용으로만 git ignore된 파일에 둘 수도 있습니다.

```bash
mkdir -p .secrets
printf "%s" "your-key-here" > .secrets/gemini_api_key.txt
```

API key, key가 노출된 스크린샷, 터미널 로그는 커밋하지 마세요.

### 브라우저 프로토타입 실행

```bash
open index.html
```

## 현재 미구현/제한 사항

아직 구현되지 않은 것:

- OpenAI planner 연동
- server planner HTTP client 연동
- 실시간 멀티 사이클 closed-loop replanning
- 정교한 빌드 오더/멀티 건물 배치/확장/업그레이드/애드온/생산 큐
- 적 정찰 정보를 활용한 본격 전략 전환
- 컴퓨터 비전 기반 인식
- Brood War/BWAPI 연동
- 완성형 대전 AI 또는 래더 수준 의사결정

현재 설계상 제한:

- MVP 안전성을 위해 행동 수와 좌표 범위가 제한됩니다.
- 지원 유닛은 worker/marine 중심입니다.
- 지원 건물은 Supply Depot, Barracks, Refinery입니다.
- Terran 시작 상황을 중심으로 검증 로직이 작성되어 있습니다.
- `build` 위치는 executor가 안전한 근처 위치를 선택하는 단순 전략입니다.

## 개발/검증

전체 테스트:

```bash
npm run test:all
```

개별 테스트:

```bash
npm test
python3 -m unittest discover -s tests -v
```

## 디렉터리 구조

```text
src/                  브라우저 캔버스 이동 prototype
test/                 JavaScript 테스트
starcraft_llm/        StrategyPlan, planner, validator, SC2 bot 구현
tests/                Python 단위 테스트
scripts/              SC2 실행 entrypoint
docs/                 개발 원칙 문서
```

## 한 줄 요약

StarcraftLLM은 **LLM이 낼 수 있는 스타크래프트 전략을 작은 검증 가능한 계획으로 제한하고, 그 계획을 실제 StarCraft II API 봇으로 실행하기 위한 초기 MVP**입니다.
