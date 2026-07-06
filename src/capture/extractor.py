"""공용 특징 추출기: 양손 랜드마크 + 상체 포즈 → 고정 크기 벡터.

수집(collect)·변환(import_jester)·추론(predict)이 모두 이 모듈을 사용해
학습-추론 특징 형식을 일치시킨다.

특징 벡터 (FEATURE_DIM = 150):
    [왼손 21관절 × xyz = 63] [오른손 63] [상체 8관절 × xyz = 24]
    상체 관절: 어깨(11,12), 팔꿈치(13,14), 손목(15,16), 골반(23,24)
    — 골반 포함으로 어깨-골반 상대 관계(상반신 회전)를 표현할 수 있다.
    감지 안 된 손/포즈는 0으로 채움.

목장갑 등으로 손 랜드마크가 불안정한 환경에서도 포즈(팔 궤적)는 몸 전체
스케일로 추정되어 상대적으로 강인하다 (doc/design.md 참고).
"""

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from src.capture.landmark_viewer import MODEL_PATH as HAND_MODEL_PATH, ensure_model as ensure_hand_model

ROOT = Path(__file__).resolve().parents[2]
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = ROOT / "models" / "pose_landmarker.task"

LANDMARKS_PER_HAND = 21
HAND_DIM = LANDMARKS_PER_HAND * 3  # 63
UPPER_BODY_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24]  # 어깨/팔꿈치/손목/골반
POSE_DIM = len(UPPER_BODY_JOINTS) * 3  # 24
FEATURE_DIM = HAND_DIM * 2 + POSE_DIM  # 150

# 팔·몸통 그리기용 연결 (UPPER_BODY_JOINTS 인덱스 기준이 아니라 포즈 원본 인덱스)
ARM_CONNECTIONS = [(11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 24), (11, 23), (12, 24)]


def ensure_pose_model() -> Path:
    if not POSE_MODEL_PATH.exists():
        POSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"포즈 모델 다운로드 중... → {POSE_MODEL_PATH}")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("포즈 모델 다운로드 완료")
    return POSE_MODEL_PATH


class FeatureExtractor:
    """손 + 포즈 랜드마커를 함께 돌려 150차원 특징 벡터를 만든다 (VIDEO 모드)."""

    def __init__(self) -> None:
        hand_options = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_options)
        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_pose_model())),
            running_mode=vision.RunningMode.VIDEO,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_options)

    def detect(self, bgr_frame, timestamp_ms: int):
        """BGR 프레임에서 (손 결과, 포즈 결과)를 반환한다."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return (
            self.hands.detect_for_video(image, timestamp_ms),
            self.pose.detect_for_video(image, timestamp_ms),
        )

    @staticmethod
    def vector(hand_result, pose_result) -> np.ndarray:
        """검출 결과 → (150,) 벡터. 왼손 | 오른손 | 상체 포즈 순."""
        vec = np.zeros(FEATURE_DIM, dtype=np.float32)
        for hand_lms, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            slot = 0 if handedness[0].category_name == "Left" else 1
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms], dtype=np.float32).flatten()
            vec[slot * HAND_DIM : (slot + 1) * HAND_DIM] = coords
        if pose_result.pose_landmarks:
            pose_lms = pose_result.pose_landmarks[0]
            coords = np.array(
                [[pose_lms[j].x, pose_lms[j].y, pose_lms[j].z] for j in UPPER_BODY_JOINTS],
                dtype=np.float32,
            ).flatten()
            vec[HAND_DIM * 2 :] = coords
        return vec

    def close(self) -> None:
        self.hands.close()
        self.pose.close()


def draw_pose(frame, pose_result) -> None:
    """상체 관절(팔·어깨·골반)을 프레임 위에 그린다."""
    if not pose_result.pose_landmarks:
        return
    lms = pose_result.pose_landmarks[0]
    h, w = frame.shape[:2]
    pts = {j: (int(lms[j].x * w), int(lms[j].y * h)) for j in UPPER_BODY_JOINTS}
    for a, b in ARM_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (255, 200, 0), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 5, (255, 100, 0), -1)
