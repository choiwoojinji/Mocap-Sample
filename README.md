# Mocap-Sample

수화(수어) 인식 기반 로봇 제어 프로젝트. 카메라 영상에서 손/포즈 랜드마크를 추출해 수화 동작을 분류하고, 결과를 ROS 토픽으로 발행하여 Gazebo 시뮬레이터의 로봇을 제어한다. 상세 설계는 `doc/design.md` 참고.

## 셋업

Python 3.10–3.12 필요 (MediaPipe 제약으로 3.13 미지원). 버전은 `.python-version`(3.12)에 고정되어 있다.

```bash
# 가상환경 생성 (둘 중 하나)
python3.12 -m venv .venv
uv venv                      # uv 사용 시 .python-version 자동 적용

# 패키지 설치 (editable)
.venv/bin/python -m pip install -e .
```

VS Code는 워크스페이스의 `.venv`를 자동 감지하므로 별도 설정이 필요 없다. 인터프리터가 다른 것으로 잡혀 있으면 `⌘⇧P → Python: Select Interpreter`에서 `.venv`를 선택할 것.
