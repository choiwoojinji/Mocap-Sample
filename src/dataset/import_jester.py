"""Jester 데이터셋 → 학습 데이터(asset/) 변환 스크립트 (doc/ml-flow.md 1-2절).

Jester(20BN-Jester)는 웹캠 3인칭 동적 제스처 영상 데이터셋으로, 클립이
프레임 JPG 폴더(12fps, 약 36프레임)로 제공된다. 이 스크립트는 수신호에
해당하는 클래스만 골라 MediaPipe로 랜드마크를 추출하고, 우리 학습 형식
asset/<라벨>/*.npy (30, 126) 으로 저장한다.

사용법:
    python -m src.dataset.import_jester \
        --jester-dir <20bn-jester-v1 폴더> --labels-csv <jester-v1-train.csv> \
        [--max-per-class 100] [--out asset]

CSV 형식: "video_id;label" (Jester 배포본 그대로)

클래스 매핑(기본값, --mapping 으로 JSON 파일 지정 가능):
    Stop Sign          → 멈춰
    Pushing Hand Away  → 오지마
    Pulling Hand In    → 이리와
    No gesture         → 신호없음
    Doing other things → 신호없음

품질 보호: 손 감지 프레임이 절반 미만인 클립은 건너뛴다 (collect.py와 동일 기준).
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from src.capture.extractor import FeatureExtractor
from src.dataset.collect import MIN_DETECTED_RATIO, hand_detected_ratio

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "asset"
TARGET_FRAMES = 30
FRAME_INTERVAL_MS = 83  # Jester 12fps
VIDEO_GAP_MS = 1000  # 클립 사이 타임스탬프 간격 (트래킹 연결 끊기용)

DEFAULT_MAPPING = {
    "Stop Sign": "멈춰",
    "Pushing Hand Away": "오지마",
    "Pulling Hand In": "이리와",
    "No gesture": "신호없음",
    "Doing other things": "신호없음",
}


def resample(seq: np.ndarray, target: int) -> np.ndarray:
    """시퀀스를 시간 축 선형 보간으로 target 프레임으로 리샘플링한다."""
    if len(seq) == target:
        return seq.astype(np.float32)
    t_orig = np.linspace(0.0, 1.0, len(seq))
    t_new = np.linspace(0.0, 1.0, target)
    out = np.empty((target, seq.shape[1]), dtype=np.float32)
    for j in range(seq.shape[1]):
        out[:, j] = np.interp(t_new, t_orig, seq[:, j])
    return out


def extract_video(extractor: FeatureExtractor, video_dir: Path, ts_ms: int) -> tuple[np.ndarray | None, int]:
    """클립 폴더의 프레임들에서 특징 시퀀스를 추출한다. 반환: (시퀀스 또는 None, 다음 타임스탬프)."""
    frames = sorted(video_dir.glob("*.jpg"))
    if len(frames) < 5:
        return None, ts_ms
    vecs = []
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        hand_result, pose_result = extractor.detect(img, ts_ms)
        vecs.append(FeatureExtractor.vector(hand_result, pose_result))
        ts_ms += FRAME_INTERVAL_MS
    ts_ms += VIDEO_GAP_MS
    if not vecs:
        return None, ts_ms
    return np.stack(vecs), ts_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Jester → asset/ 변환")
    parser.add_argument("--jester-dir", type=Path, required=True,
                        help="클립 폴더들이 들어있는 디렉토리 (20bn-jester-v1)")
    parser.add_argument("--labels-csv", type=Path, required=True,
                        help="video_id;label 형식 CSV (jester-v1-train.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-per-class", type=int, default=100,
                        help="우리 라벨당 최대 변환 개수")
    parser.add_argument("--mapping", type=Path, default=None,
                        help="Jester 라벨 → 우리 라벨 매핑 JSON (기본 매핑 대체)")
    parser.add_argument("--idle-labels", nargs="*", default=["신호없음"],
                        help="손 감지 품질 필터를 면제할 라벨 (손이 없는 게 정상인 클래스)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mapping = DEFAULT_MAPPING
    if args.mapping:
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))

    # CSV에서 매핑 대상 클립만 수집하고 라벨별로 섞는다.
    # 두 형식 지원: 원본 "video_id;label" (헤더 없음), Kaggle 미러 "video_id,label,..." (헤더 있음)
    by_label: dict[str, list[str]] = {}
    with open(args.labels_csv, newline="", encoding="utf-8") as fp:
        first = fp.readline()
        delimiter = "," if "," in first else ";"
        has_header = "video_id" in first
        if not has_header:
            fp.seek(0)
        for row in csv.reader(fp, delimiter=delimiter):
            if len(row) < 2:
                continue
            video_id, jester_label = row[0].strip(), row[1].strip()
            our_label = mapping.get(jester_label)
            if our_label:
                by_label.setdefault(our_label, []).append(video_id)
    rng = random.Random(args.seed)
    for ids in by_label.values():
        rng.shuffle(ids)

    print("변환 대상:", {k: min(len(v), args.max_per_class) for k, v in by_label.items()})
    extractor = FeatureExtractor()
    ts_ms = 0
    t_start = time.monotonic()

    for our_label, ids in by_label.items():
        out_dir = args.out / our_label
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = skipped = 0
        for video_id in ids:
            if saved >= args.max_per_class:
                break
            video_dir = args.jester_dir / video_id
            if not video_dir.is_dir():
                continue
            seq, ts_ms = extract_video(extractor, video_dir, ts_ms)
            if seq is None:
                skipped += 1
                continue
            # idle 라벨은 손이 없는 게 정상이므로 감지율 필터를 적용하지 않는다
            if our_label not in args.idle_labels and hand_detected_ratio(seq) < MIN_DETECTED_RATIO:
                skipped += 1
                continue
            np.save(out_dir / f"jester_{video_id}.npy", resample(seq, TARGET_FRAMES))
            saved += 1
        print(f"{our_label}: 저장 {saved}개, 스킵 {skipped}개 (손 감지 부족/프레임 없음)")

    extractor.close()
    print(f"완료 ({time.monotonic() - t_start:.0f}초) → {args.out}")


if __name__ == "__main__":
    main()
