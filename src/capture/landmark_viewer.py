"""웹캠 영상에서 MediaPipe HandLandmarker 랜드마크를 실시간 시각화하는 검증 스크립트.

개발 단계 1 (doc/design.md): 카메라 입력 → 손 랜드마크 추출이 정상 동작하는지 확인한다.
mediapipe 0.10.35는 구형 solutions API가 제거되어 Tasks API(HandLandmarker)를 사용한다.
최초 실행 시 모델 파일(hand_landmarker.task, 약 8MB)을 models/ 폴더에 자동 다운로드한다.

실행: F5 또는 VSCode 태스크 "실행: 랜드마크 시각화" (종료: q 또는 ESC)
"""

import platform
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "hand_landmarker.task"

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def ensure_model() -> Path:
    """모델 파일이 없으면 다운로드한다."""
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"모델 다운로드 중... → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("모델 다운로드 완료")
    return MODEL_PATH


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """OS별 백엔드로 웹캠을 연다."""
    system = platform.system()
    if system == "Windows":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif system == "Darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    else:
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"카메라 {index}번을 열 수 없습니다. "
            "다른 앱이 카메라를 사용 중인지, OS 카메라 권한이 허용되어 있는지 확인하세요."
        )
    return cap


def create_landmarker() -> vision.HandLandmarker:
    """비디오 모드 HandLandmarker를 생성한다."""
    options = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def draw_landmarks(frame, hand_landmarks) -> None:
    """정규화 좌표(0~1)의 랜드마크를 프레임 위에 그린다."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, points[conn.start], points[conn.end], (0, 255, 0), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)


def main() -> None:
    landmarker = create_landmarker()
    cap = open_camera()
    start = time.monotonic()
    prev_time = start

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임을 읽지 못했습니다. 종료합니다.")
            break

        # 셀피 뷰로 좌우 반전 후 MediaPipe 입력용 RGB 이미지 생성
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 비디오 모드는 단조 증가하는 타임스탬프(ms)가 필요
        timestamp_ms = int((time.monotonic() - start) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        for hand in result.hand_landmarks:
            draw_landmarks(frame, hand)

        # FPS 표시
        now = time.monotonic()
        fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
        prev_time = now
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}  hands: {len(result.hand_landmarks)}  (q/ESC: quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Hand Landmarks", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
