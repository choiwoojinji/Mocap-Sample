"""수신호 분류 모델 학습 스크립트 (doc/ml-flow.md 4단계).

asset/의 라벨 폴더를 스캔해 클래스를 동적으로 구성하고 (하드코딩 금지),
LSTM 시퀀스 분류 모델을 학습한 뒤 라벨 목록과 함께 저장한다.

사용법:
    python -m src.training.train [--data-dir asset] [--epochs 300] [--hidden 64]

플로우:
    데이터 로드 → 층화 분할(70/15/15) → LSTM 학습 (early stopping)
    → test 평가 (혼동 행렬 포함) → models/sign_classifier.pt 저장 (라벨 목록 포함)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "asset"
DEFAULT_MODEL_PATH = ROOT / "models" / "sign_classifier.pt"


class SignLSTM(nn.Module):
    """랜드마크 시퀀스 (T, 126) → 수신호 클래스 분류."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])  # 마지막 타임스텝의 은닉 상태로 분류


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """asset/<라벨>/*.npy 를 스캔해 (X, y, 라벨목록)을 반환한다."""
    labels = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and list(d.glob("*.npy")))
    if len(labels) < 2:
        raise SystemExit(f"클래스가 2개 이상 필요합니다. 현재: {labels} ({data_dir})")
    xs, ys = [], []
    for idx, label in enumerate(labels):
        for f in sorted((data_dir / label).glob("*.npy")):
            seq = np.load(f)
            xs.append(seq.astype(np.float32))
            ys.append(idx)
    X = np.stack(xs)
    y = np.array(ys)
    for idx, label in enumerate(labels):
        n = int((y == idx).sum())
        print(f"  {label}: {n}개" + ("  ⚠ 10개 미만 — 추가 수집 권장" if n < 10 else ""))
    return X, y, labels


def split_dataset(X, y):
    """train 70 / val 15 / test 15 층화 분할."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate(model, X, y, device) -> tuple[float, np.ndarray]:
    """정확도와 예측값을 반환한다."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == y).mean()), pred


def main() -> None:
    parser = argparse.ArgumentParser(description="수신호 분류 모델 학습")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=30, help="early stopping 대기 epoch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"데이터 로드: {args.data_dir}")
    X, y, labels = load_dataset(args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y)
    print(f"분할: train {len(X_train)} / val {len(X_val)} / test {len(X_test)}  (device: {device})")

    model = SignLSTM(X.shape[2], args.hidden, len(labels), num_layers=args.layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_x = torch.from_numpy(X_train).to(device)
    train_y = torch.from_numpy(y_train).long().to(device)

    best_val, best_state, wait = 0.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train_x))
        for i in range(0, len(train_x), args.batch):
            idx = perm[i : i + args.batch]
            optimizer.zero_grad()
            loss = criterion(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()

        val_acc, _ = evaluate(model, X_val, y_val, device)
        if val_acc > best_val:
            best_val, wait = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if epoch % 10 == 0 or wait == 0:
            print(f"epoch {epoch:3d}  loss {loss.item():.4f}  val_acc {val_acc:.3f}  best {best_val:.3f}")
        if wait >= args.patience:
            print(f"early stopping (epoch {epoch}, {args.patience} epoch 동안 개선 없음)")
            break

    model.load_state_dict(best_state)
    test_acc, pred = evaluate(model, X_test, y_test, device)
    print(f"\n=== 최종 평가 (test) ===\n정확도: {test_acc:.3f}")
    print("혼동 행렬 (행=실제, 열=예측):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=labels, zero_division=0))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "labels": labels,
            "config": {
                "input_dim": X.shape[2],
                "hidden_dim": args.hidden,
                "num_layers": args.layers,
                "frames": X.shape[1],
            },
            "test_accuracy": test_acc,
        },
        args.model_out,
    )
    print(f"모델 저장: {args.model_out}  (라벨: {labels})")


if __name__ == "__main__":
    main()
