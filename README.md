# Mocap-Sample — 수신호 인식 (산업 수신호 → 공정 기계 제어)

웹캠 영상에서 작업자의 **수신호(hand signal)** 를 실시간 인식하고, 그 결과를 ROS 2로 전달해
Gazebo 시뮬레이션의 공정 기계를 제어하는 프로젝트의 **인식 파트**.
(기계 시뮬레이션 파트는 [`../Mocap-Sim`](../Mocap-Sim/README.md) 참고)

```
웹캠 → MediaPipe 랜드마크(손+상체 포즈) → LSTM 분류 → 안정화 필터
→ rosbridge → ROS 2 /hand_signal → Gazebo 공정 기계 (정지/감속/접근/선회)
```

## 셋업

Python 3.12 필요 (`.python-version`에 고정, MediaPipe 제약으로 3.13 미지원).

- **VSCode**: `Cmd/Ctrl+Shift+B` (기본 빌드 태스크 "프로젝트 셋업") 한 번으로
  venv 생성 → 패키지 설치 → 검증까지 자동. 이후 인터프리터로 `.venv` 선택
  (`⌘⇧P → Python: Select Interpreter`).
- **수동**:
  ```bash
  python3.12 -m venv .venv        # 또는: uv venv (.python-version 자동 적용)
  .venv/bin/python -m pip install -e .
  ```

## 사용법 (VSCode 태스크: Terminal → Run Task)

| 태스크 | 용도 |
|---|---|
| 실행: 랜드마크 시각화 | 웹캠 + 손/포즈 랜드마크 확인 (프레이밍 점검) |
| 실행: 데이터 수집 | 라벨 입력 → 자동 연속 수집 (라벨당 30개, START 버튼) |
| 실행: 모델 학습 | `asset/` 스캔 → LSTM 학습 → 혼동 행렬 → 모델 저장 |
| 실행: 실시간 추론 | 웹캠 실시간 인식 (확률 패널 UI) |
| 시뮬 데모: Gazebo + 인식기 (Mac) | Mac 단독 전체 데모 (Gazebo 네이티브 + UDP 어댑터) |

idle(신호없음) 수집은 터미널에서: `.venv/bin/python -m src.dataset.collect --label idle --idle`

**ROS 2 연동 실행** (Mocap-Sim이 떠 있을 때):
```bash
.venv/bin/python -m src.inference.predict --publish rosbridge --rosbridge-url ws://<시뮬머신IP>:9090
```

## 수신호 클래스

`asset/`의 라벨 폴더가 곧 클래스 (하드코딩 없음). 현재: **stop, slow, come, left_go, right_go, idle**
— 새 신호는 그 이름으로 수집하고 재학습하면 추가된다.

## 문서

- [doc/design.md](doc/design.md) — 아키텍처·설계 결정 (특징 150차원, 안전 규칙, 전달 방식)
- [doc/ml-flow.md](doc/ml-flow.md) — 데이터→학습→예측 플로우, 데이터셋 조사, 진행 체크리스트
- [doc/AGENT.md](doc/AGENT.md) — 코딩 에이전트용 안내 (셋업·제약·현재 상태)

## 상태

전체 파이프라인 + ROS 2 연동 실증 완료 (2026-07-07).
진행 중: 데이터 보강(라벨당 100개)으로 인식 정확도 개선 — [ml-flow.md 7절](doc/ml-flow.md) 체크리스트 참고.
