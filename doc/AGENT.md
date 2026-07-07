# AGENT.md

Claude Code 등 코딩 에이전트가 이 저장소에서 작업할 때 따라야 할 안내 문서.

## 프로젝트 개요

OpenCV와 MediaPipe 기반 **산업 현장 수신호(hand signal) 인식** 프로젝트. 카메라 영상에서 손 랜드마크를 추출하고, 머신러닝으로 수신호 동작을 학습·분류한다.

**최종 목표:** 웹캠 앞에서 작업자가 수신호를 하면 Gazebo 시뮬레이션의 공정 기계가 인식·대응하는 것 (예: 접근 중 '멈춰' → 정지, '천천히' → 감속). **이 저장소의 범위는 수신호 인식(데이터 수집 → 학습 → 실시간 추론)까지다.** ROS 2 / Gazebo 기계 제어는 별도 프로젝트에서 다루며, 이 저장소는 분류 결과를 외부로 전달하는 것까지만 책임진다.

**현재 상태:** 전체 파이프라인 완성 + ROS 2 연동 실증 완료 (2026-07-07). 웹캠 인식(자체 수집 6클래스, test 77.8%) → rosbridge → Ubuntu VM(UTM)의 ROS 2 Jazzy + Gazebo Harmonic 기계 제어까지 end-to-end 동작 확인. 시뮬레이터는 별도 프로젝트 `../Mocap-Sim`. 남은 작업: 데이터 보강(라벨당 100개 목표, `back` 라벨 추가, idle에 손 안 보이는 변형 포함) → 재학습으로 인식 정확도 향상. 상세 설계는 `doc/design.md`, ML 플로우는 `doc/ml-flow.md` 참고.

## 셋업

**Python 3.12 필수** (`pyproject.toml`에 고정). 의존성도 팀 기준 버전으로 `==` 고정되어 있으므로
버전을 임의로 올리거나 범위를 풀지 말 것. 프로젝트 루트에 `.venv` 가상환경을 만들고 그 안에 설치한다:

```bash
# macOS
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .

# Windows
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e .
```

VSCode에서는 `Cmd/Ctrl+Shift+B` (기본 빌드 태스크 "프로젝트 셋업") 한 번으로 venv 생성부터
패키지 설치까지 끝난다 (`.vscode/tasks.json`이 OS별 명령을 자동 선택). 셋업 후 F5 실행을 위해
인터프리터로 `.venv`를 선택할 것.

테스트와 린터는 아직 설정되어 있지 않다.

## 아키텍처

데이터가 파이프라인으로 흐르며, `src/` 하위 디렉토리 하나가 한 단계를 담당한다:

```
카메라 입력 (OpenCV)
   ↓
손/포즈 랜드마크 추출 (MediaPipe HandLandmarker + PoseLandmarker)
   ↓
랜드마크 시퀀스 전처리 (정규화, 시퀀스 버퍼링)
   ↓
수신호 분류 모델 추론 (학습된 ML 모델)
   ↓
분류 결과 출력/전달  ← 이 저장소의 책임은 여기까지
   ↓
(별도 프로젝트) ROS 2 → Gazebo 공정 기계 행동 실행 (정지/감속/후진 등)
```

### 모듈 구성

- `src/capture/` — 카메라 캡처 + MediaPipe 랜드마크 추출
- `src/dataset/` — 데이터 수집 스크립트 (수신호별 라벨링·저장)
- `src/training/` — 모델 학습 (PyTorch LSTM 시퀀스 분류)
- `src/inference/` — 실시간 수신호 인식 (추론) + 분류 결과를 외부(기계 프로젝트)로 전달하는 인터페이스
- `asset/`, `models/` — 수집한 랜드마크 데이터셋, 학습된 모델 파일

## 주요 제약 사항

- **범위:** ROS 2 / Gazebo 관련 코드는 이 저장소에 추가하지 않는다. 분류 결과를 외부로 전달하는 인터페이스(예: 소켓, rosbridge 클라이언트 등 — 방식 미정)까지만 이 저장소의 몫이다.
- **Windows와 macOS 양쪽에서 동일하게 동작해야 한다:**
  - 경로는 항상 `pathlib.Path`로 다루고, 경로 문자열 하드코딩(`/` 또는 `\` 구분자, 절대 경로) 금지.
  - 카메라는 `cv2.VideoCapture(index)` 기본 백엔드를 사용하되, OS별 백엔드 지정이 필요하면 `platform` 분기로 처리 (Windows: `CAP_DSHOW`/`CAP_MSMF`, macOS: `CAP_AVFOUNDATION`).
  - 셸 스크립트(.sh/.bat) 대신 Python 스크립트로 도구를 작성해 OS 의존을 없앤다.
  - 웹캠을 쓰는 코드는 Docker에 넣지 않는다 (Windows/macOS Docker 모두 카메라 접근 불가).
- **설계 결정** (`doc/design.md` 참고): 동적 수신호 30프레임 시퀀스, 특징은 양손+상체 포즈 150차원(`src/capture/extractor.py` — 수집·변환·추론이 모두 이 모듈을 사용해야 형식이 일치한다), 데이터는 `asset/<라벨>/*.npy` shape (30, 150). **클래스 목록은 하드코딩 금지** — `asset/`의 라벨 폴더에서 동적으로 구성하고 라벨 목록은 모델과 함께 저장·로드한다. 신뢰도가 낮으면 '정지'가 안전 기본값. 분류 결과 전달은 rosbridge 웹소켓(`src/inference/publisher.py`, 토픽 `/hand_signal`)로 결정됨.
- 이 저장소의 문서와 주석은 한국어로 작성한다.
