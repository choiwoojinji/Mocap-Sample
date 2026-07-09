"""학습용 특징 수준 도메인 랜덤화 (doc/ml-flow.md 2절).

악조건(저조도, 센서 노이즈, 연기·가림, 렌즈 왜곡, 위치·속도 변화)이 랜드마크
좌표에 남기는 효과를 학습 배치에 무작위 주입한다. 픽셀이 아니라 좌표를
변형하므로 계산이 싸고, 원본 데이터는 건드리지 않는다 (실시간 적용).

주의: 감지 안 된 지점(0 벡터)은 "없음"이라는 의미이므로 어떤 변형도
0을 깨뜨리지 않는다 — 존재하는 지점만 변형한다.

시퀀스 형식: (T, 150) = (T, 50포인트 × xyz). 앞 42포인트 = 양손, 뒤 8 = 상체 포즈.
"""

import numpy as np

HAND_POINTS = 42  # 21 × 2손
NUM_POINTS = 50   # 손 42 + 상체 포즈 8

# 변형별 적용 확률과 강도 (경험적 기본값 — 과하면 학습이 흔들린다)
P_TIME_WARP = 0.5
P_TRANSLATE = 0.5
P_SCALE = 0.5
P_DISTORT = 0.3
P_HAND_DROPOUT = 0.3
P_NOISE = 0.8

NOISE_STD = 0.01          # 좌표 노이즈 (센서 노이즈·저조도 떨림)
TRANSLATE_MAX = 0.06      # 화면 대비 이동량 (작업자 위치 변화)
SCALE_RANGE = (0.9, 1.1)  # 크기 (카메라 거리 변화)
WARP_RANGE = (0.8, 1.25)  # 시간 신축 (동작 속도 변화)
DISTORT_K = 0.15          # 배럴/핀쿠션 왜곡 계수 최대치 (렌즈 왜곡)
DROPOUT_FRAMES = (1, 6)   # 손 소실 프레임 수 범위 (연기·가림·감지 실패)


def _time_warp(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """재생 속도를 바꿨다가 원래 길이로 리샘플링한다."""
    t = len(seq)
    factor = rng.uniform(*WARP_RANGE)
    src = np.clip(np.linspace(0, (t - 1) * factor, t), 0, t - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, t - 1)
    w = (src - lo)[:, None]
    return (1 - w) * seq[lo] + w * seq[hi]


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """시퀀스 1개 (T, 150)에 무작위 악조건 변형을 적용한다."""
    seq = seq.copy()

    if rng.random() < P_TIME_WARP:
        seq = _time_warp(seq, rng)

    pts = seq.reshape(len(seq), NUM_POINTS, 3)
    present = (pts != 0).any(axis=2)  # (T, 50) — 존재하는 지점만 변형

    # 이동 (작업자 위치 변화)
    if rng.random() < P_TRANSLATE:
        offset = rng.uniform(-TRANSLATE_MAX, TRANSLATE_MAX, size=2)
        pts[..., 0][present] += offset[0]
        pts[..., 1][present] += offset[1]

    # 스케일 (카메라 거리 변화) — 화면 중앙(0.5, 0.5) 기준
    if rng.random() < P_SCALE:
        s = rng.uniform(*SCALE_RANGE)
        for axis in (0, 1):
            vals = pts[..., axis]
            vals[present] = (vals[present] - 0.5) * s + 0.5
        pts[..., 2][present] *= s

    # 렌즈 왜곡 (배럴/핀쿠션): r' = r(1 + k·r²), 화면 중앙 기준.
    # r²는 화면 안 최대 반경(0.5)으로 제한 — 화면 밖 추정 지점(예: 골반)이
    # 과하게 증폭되는 것을 막는다 (실제 렌즈 왜곡은 화면 안 현상).
    if rng.random() < P_DISTORT:
        k = rng.uniform(-DISTORT_K, DISTORT_K)
        dx = pts[..., 0] - 0.5
        dy = pts[..., 1] - 0.5
        factor = 1.0 + k * np.minimum(dx * dx + dy * dy, 0.5)
        pts[..., 0][present] = (dx * factor + 0.5)[present]
        pts[..., 1][present] = (dy * factor + 0.5)[present]

    # 손 소실 (연기·가림·순간 감지 실패): 임의 프레임의 손 블록을 0으로
    if rng.random() < P_HAND_DROPOUT:
        n = rng.integers(*DROPOUT_FRAMES)
        frames = rng.choice(len(pts), size=min(n, len(pts)), replace=False)
        pts[frames, :HAND_POINTS, :] = 0.0
        present[frames, :HAND_POINTS] = False

    # 좌표 노이즈 (센서 노이즈·저조도 떨림)
    if rng.random() < P_NOISE:
        noise = rng.normal(0.0, NOISE_STD, size=pts.shape).astype(np.float32)
        pts[present] += noise[present]

    return pts.reshape(len(seq), -1).astype(np.float32)


def augment_batch(batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """배치 (B, T, 150) 전체에 시퀀스별 독립 변형을 적용한다."""
    return np.stack([augment_sequence(s, rng) for s in batch])
