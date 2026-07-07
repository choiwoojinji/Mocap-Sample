# 수신호 기반 공정 기계 안내 시뮬레이터 (mocap-sample)

## 프로젝트 개요

OpenCV와 MediaPipe를 이용한 모션/포즈 인식으로 **산업 현장 수신호(hand signal)** 를 감지하고,
머신러닝 학습을 통해 카메라에 인식되는 수신호를 분류하는 프로젝트.

**최종 목표:** 웹캠 앞에서 작업자가 수신호를 하면 Gazebo 시뮬레이션의 공정 기계가
이를 인식·예측하여 대응 행동을 취하는 것.
예: 기계가 접근하다가 stop 수신호 → 정지, slow → 감속, come → 접근.
**→ 2026-07-07 end-to-end 실증 완료** (웹캠 → ROS 2 → Gazebo 기계 반응).

**이 저장소의 범위는 수신호 인식(데이터 수집 → 학습 → 실시간 추론 → 결과 전달)까지.**
ROS 2 / Gazebo 기계 제어는 별도 프로젝트 **`../Mocap-Sim`** 이 담당한다.

## 시스템 아키텍처 (실증된 구성)

```
[Mocap-Sample — macOS/Windows]
카메라 입력 (OpenCV)
   ↓
손+포즈 랜드마크 추출 (MediaPipe HandLandmarker + PoseLandmarker)
   ↓
150차원 특징 시퀀스 버퍼링 (슬라이딩 윈도우 30프레임)
   ↓
LSTM 분류 + 안정화 필터 (신뢰도 임계값 + 연속 일치)
   ↓
rosbridge 웹소켓 전송              ← 이 저장소의 책임은 여기까지
   ↓
[Mocap-Sim — Ubuntu VM(UTM) 또는 네이티브 Ubuntu]
ROS 2 토픽 /hand_signal → signal_listener 노드 → /cmd_vel → Gazebo 공정 기계
```

- 시뮬 환경: Ubuntu 24.04 + ROS 2 Jazzy + Gazebo Harmonic.
  현재 UTM VM(arm64)에서 검증됨 — VM에서는 `LIBGL_ALWAYS_SOFTWARE=1`(CPU 렌더링) 필요.
  GPU 렌더링이 필요해지면(센서 시뮬레이션 등) 네이티브 Ubuntu + GPU로 이전.
- Mac 단독 데모도 가능: Gazebo를 brew로 네이티브 설치 + UDP 어댑터
  (`Mocap-Sim/tools/`, VSCode 태스크 "시뮬 데모").

## 기술 스택

| 구성 요소 | 기술 |
|---|---|
| 영상 입력/처리 | OpenCV |
| 랜드마크 추출 | MediaPipe HandLandmarker + PoseLandmarker (Tasks API) |
| 수신호 분류 모델 | PyTorch LSTM (2층, hidden 64) |
| 결과 전달 | rosbridge 웹소켓 (+ UDP 디버깅 백엔드) |
| 언어 | Python 3.12 (의존성은 pyproject.toml에 버전 고정) |

## 모듈 구성

```
mocap-sample/
├── doc/                    # 문서 (AGENT.md, design.md, ml-flow.md)
├── asset/                  # 수집한 수신호 랜드마크 데이터셋 (.npy) — 라벨 폴더 = 클래스
├── models/                 # MediaPipe 모델(.task) + 학습된 분류 모델(sign_classifier.pt)
└── src/
    ├── capture/            # 카메라 캡처 + 특징 추출기(extractor.py) + 시각화
    ├── dataset/            # 데이터 수집(collect.py), Jester 변환(import_jester.py)
    └── training/ inference/ # 학습(train.py) / 실시간 추론(predict.py) + 전달(publisher.py)
```

## 개발 단계 (전체 완료)

1. ~~랜드마크 추출 검증~~ ✅
2. ~~데이터 수집 도구~~ ✅ (자동 연속 수집 + 품질 필터 + 손 이탈 경고)
3. ~~모델 학습~~ ✅ (자체 수집 6클래스, test 77.8% — 데이터 보강으로 개선 중)
4. ~~실시간 추론~~ ✅ (안정화 필터 + 사람 미감지 안전 가드)
5. ~~결과 전달~~ ✅ (rosbridge, Mocap-Sim과 연동 실증)

**남은 작업 = 품질 루프**: 라벨당 100개 목표로 데이터 보강(`back` 추가, idle에
손 안 보이는 변형 포함) → 재학습 → 실사용 오분류 관찰 → 반복 (doc/ml-flow.md 6절).

## 주요 설계 결정 사항

- **동적 수신호(시퀀스)** — 시퀀스당 30프레임, LSTM 시퀀스 분류
- **특징 = 양손 + 상체 포즈 150차원** (`src/capture/extractor.py`)
  - 왼손 63 + 오른손 63 (21랜드마크 × xyz) + 상체 8관절(어깨·팔꿈치·손목·골반) × xyz = 24
  - 포즈를 넣은 이유: 산업 수신호는 팔 궤적이 본체이고, 목장갑 등으로 손 랜드마크가
    불안정한 환경에서도 포즈는 몸 전체 스케일이라 강인하다. 골반 포함으로 상반신 회전 표현.
  - 수집·변환·추론이 모두 이 모듈을 사용해야 학습-추론 특징 형식이 일치한다.
- **데이터 형식**: `asset/<라벨>/<타임스탬프>.npy`, shape (30, 150) — 감지 안 된 부분은 0
- **클래스 목록은 고정하지 않는다** — `asset/`의 라벨 폴더가 곧 클래스다.
  학습이 폴더를 스캔해 동적으로 구성하고 라벨 목록을 모델 파일에 함께 저장,
  추론은 그것을 읽는다. 코드에 클래스 이름 하드코딩 금지.
- **수신호 목록 (현재 수집분)** — 라벨명은 기계 명령어로 쓰도록 영어 snake_case:
  | 라벨 | 동작 | 기계 행동 |
  |---|---|---|
  | stop | 손바닥 정면으로 내밀기 | 즉시 정지 |
  | slow | 손바닥 아래로 눌러 내리기 | 감속 |
  | come | 손짓해 부르기 | 접근 |
  | back (미수집) | 손등으로 밀어내기 | 후진 |
  | left_go / right_go | 좌/우로 팔 젓기 | 선회 |
  | idle | 평상시 동작 (`--idle`로 수집, 필수 — 오작동 방지) | 직전 상태 유지 |
- **학습 데이터는 자체 수집 단독** — Jester(14.8만 클립) 실험으로 외부 데이터의 도메인 갭
  (동작 방식 차이, 셀피 뷰 좌우 반전)을 실측 확인하고 자체 수집으로 전환했다.
  외부 데이터셋(NATOPS 등)은 추후 다중 사용자 일반화 단계에서 사전학습 용도로만 검토.
  (Jester 변환 파이프라인 `import_jester.py`는 도구로 유지)
- **안전 규칙 (계층별)**
  - 추론: 신뢰도 임계값 미만 or 미학습 동작 → "unknown" / 사람(포즈) 미감지 → 모델 판단 무시하고 "unknown"
  - 기계(Mocap-Sim): unknown → 정지 / 하트비트 2초 끊김 → 무조건 정지 / idle → 직전 상태 유지
- **분류 결과 전달 = rosbridge 웹소켓** — `src/inference/publisher.py`
  - 메시지: std_msgs/String에 JSON `{"signal": "stop", "confidence": 0.93, "timestamp": ...}`
  - 토픽 `/hand_signal`, 상태 변화 시 즉시 + 1초 주기 하트비트, 서버 없어도 추론 계속(자동 재접속)
  - 디버깅용 UDP 백엔드(`--publish udp`) — Mac 단독 데모의 어댑터가 이걸 수신

## 환경 참고

- 개발 머신: **Windows / macOS 둘 다 지원** — MediaPipe, OpenCV, 학습·추론 모두 네이티브 실행
- 모든 코드는 크로스 플랫폼으로 작성 (경로는 `pathlib`, 카메라 백엔드는 OS별 분기, 셸 스크립트 대신 Python)
- 웹캠을 사용하는 코드는 Docker에 넣지 않는다 (Windows/macOS Docker 모두 카메라 접근 불가)
- 의존성 설치: 프로젝트 루트 `.venv` (Python 3.12)에 `pip install -e .` — VSCode 태스크 "프로젝트 셋업" 참고
