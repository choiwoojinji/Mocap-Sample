"""악조건 스트레스 테스트: 조건별로 test 셋을 오염시켜 모델 강건성을 측정한다.

학습과 동일한 분할(seed 42)의 test 셋에 augment.py의 변형을 조건별로 강제
적용(p=1)하고 정확도 하락을 비교한다. 무작위 변형이므로 여러 번 반복한 평균을 본다.

    python -m src.training.stress_test [--repeats 20]

주의: 여기서 측정하는 것은 "좌표가 오염됐을 때 분류기의 강건성"이다.
저조도·연기가 MediaPipe 추출 자체를 실패시키는 효과는 실측(불 끄기, 가림 등)으로
확인해야 한다 — 그 경우는 안전 가드(인식 불가 → 정지)가 담당한다.
"""

import argparse

import numpy as np
import torch

import src.training.augment as aug
from src.training.train import DEFAULT_DATA_DIR, DEFAULT_MODEL_PATH, SignLSTM, load_dataset, split_dataset

# 조건 이름 → augment 모듈 확률/강도 오버라이드
CONDITIONS = {
    "깨끗함 (기준)": None,
    "좌표 노이즈 (센서·저조도)": {"P_NOISE": 1.0, "NOISE_STD": 0.02},
    "손 소실 (연기·가림)": {"P_HAND_DROPOUT": 1.0, "DROPOUT_FRAMES": (3, 10)},
    "시간 신축 (동작 속도)": {"P_TIME_WARP": 1.0},
    "이동·스케일 (위치·거리)": {"P_TRANSLATE": 1.0, "P_SCALE": 1.0},
    "렌즈 왜곡": {"P_DISTORT": 1.0},
    "종합 (전부 동시)": {"P_NOISE": 1.0, "P_HAND_DROPOUT": 1.0, "P_TIME_WARP": 1.0,
                    "P_TRANSLATE": 1.0, "P_SCALE": 1.0, "P_DISTORT": 1.0},
}

ALL_KEYS = ["P_NOISE", "P_HAND_DROPOUT", "P_TIME_WARP", "P_TRANSLATE", "P_SCALE",
            "P_DISTORT", "NOISE_STD", "DROPOUT_FRAMES"]


def evaluate(model, X, y, device) -> float:
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="악조건 스트레스 테스트")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--repeats", type=int, default=20, help="무작위 변형 반복 횟수")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SignLSTM(cfg["input_dim"], cfg["hidden_dim"], len(ckpt["labels"]),
                     num_layers=cfg["num_layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X, y, labels = load_dataset(args.data_dir)
    assert labels == ckpt["labels"], f"데이터 라벨 {labels} ≠ 모델 라벨 {ckpt['labels']}"
    *_, X_test, y_test = split_dataset(X, y)
    print(f"\ntest 셋 {len(X_test)}개, 조건별 {args.repeats}회 반복 평균\n")

    defaults = {k: getattr(aug, k) for k in ALL_KEYS}
    zeros = {k: 0.0 for k in ALL_KEYS if k.startswith("P_")}

    print(f"{'조건':<24} 정확도")
    print("-" * 34)
    baseline = None
    for name, overrides in CONDITIONS.items():
        if overrides is None:
            acc = evaluate(model, X_test, y_test, device)
            baseline = acc
        else:
            for k, v in {**defaults, **zeros, **overrides}.items():
                setattr(aug, k, v)
            rng = np.random.default_rng(7)
            accs = [evaluate(model, aug.augment_batch(X_test, rng), y_test, device)
                    for _ in range(args.repeats)]
            for k, v in defaults.items():  # 원상 복구
                setattr(aug, k, v)
            acc = float(np.mean(accs))
        drop = "" if baseline is None or overrides is None else f"  ({(acc - baseline) * 100:+.1f}%p)"
        print(f"{name:<24} {acc * 100:5.1f}%{drop}")

    print("\n해석: 하락 폭이 큰 조건이 모델의 약점 — 해당 조건 증강 강도를 올리거나"
          "\n      그 조건의 실측 데이터를 보강한다. (MediaPipe 층 실패는 실측으로 확인)")


if __name__ == "__main__":
    main()
