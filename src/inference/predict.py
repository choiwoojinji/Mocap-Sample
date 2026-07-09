"""실시간 수신호 추론 (doc/ml-flow.md 5단계).

웹캠에서 최근 30프레임 랜드마크를 슬라이딩 윈도우로 유지하며 매 프레임 분류하고,
안정화 필터(신뢰도 임계값 + 연속 K회 동일 예측)를 통과한 신호만 확정한다.
확정 신호가 없으면 "인식 불가" — 기계 쪽 안전 기본값은 '정지'다.

사용법:
    python -m src.inference.predict [--threshold 0.8] [--consecutive 5]

종료: q 또는 ESC
"""

import argparse
import platform
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from src.capture.extractor import HAND_DIM, FeatureExtractor
from src.capture.landmark_viewer import HandWarning, open_camera
from src.dataset.collect import draw_detections, process_frame
from src.inference.publisher import make_publisher
from src.training.train import SignLSTM

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "sign_classifier.pt"

# 한글 오버레이용 폰트 (OS별) — cv2.putText는 한글을 못 그리므로 PIL 사용
FONT_CANDIDATES = {
    "Darwin": ["/System/Library/Fonts/Supplemental/AppleGothic.ttf",
               "/System/Library/Fonts/AppleSDGothicNeo.ttc"],
    "Windows": ["C:/Windows/Fonts/malgun.ttf"],
    "Linux": ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf"],
}


def find_font() -> str | None:
    for path in FONT_CANDIDATES.get(platform.system(), []):
        if Path(path).exists():
            return path
    return None


def make_text_drawer():
    """한글 텍스트 렌더러를 반환한다. 여러 텍스트를 한 번의 변환으로 그린다.

    draw(frame, items) — items: [(text, (x, y), BGR색, 크기), ...]
    폰트가 없으면 cv2 기본(영문 대체)으로 폴백.
    """
    font_path = find_font()
    if font_path is None:
        def draw(frame, items):
            for text, xy, color, size in items:
                cv2.putText(frame, text.encode("ascii", "replace").decode(), xy,
                            cv2.FONT_HERSHEY_SIMPLEX, size / 32, color, 2)
            return frame
        return draw

    from PIL import Image, ImageDraw, ImageFont
    fonts: dict[int, "ImageFont.FreeTypeFont"] = {}

    def draw(frame, items):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        for text, xy, color, size in items:
            font = fonts.setdefault(size, ImageFont.truetype(font_path, size))
            d.text(xy, text, font=font, fill=(color[2], color[1], color[0]))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return draw


# 확률 패널 레이아웃
BAR_X, BAR_TOP, BAR_W, BAR_H, BAR_GAP = 210, 120, 320, 30, 12


def draw_probability_panel(frame, labels, probs, top: int | None,
                           confirmed_idx: int | None, threshold: float):
    """클래스별 확률 막대 패널을 그린다. 텍스트 항목 목록을 반환한다 (draw_text용)."""
    panel_h = len(labels) * (BAR_H + BAR_GAP) + BAR_GAP
    x0, y0 = 10, BAR_TOP - BAR_GAP
    x1, y1 = BAR_X + BAR_W + 70, BAR_TOP + panel_h - BAR_GAP
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    texts = []
    for i, label in enumerate(labels):
        y = BAR_TOP + i * (BAR_H + BAR_GAP)
        p = float(probs[i]) if probs is not None else 0.0
        if confirmed_idx is not None and i == confirmed_idx:
            color = (0, 220, 0)      # 확정 = 초록
        elif top is not None and i == top:
            color = (0, 180, 255)    # 현재 1위 = 주황
        else:
            color = (180, 130, 60)   # 나머지 = 파랑
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + BAR_W, y + BAR_H), (70, 70, 70), -1)
        cv2.rectangle(frame, (BAR_X, y), (BAR_X + int(BAR_W * p), y + BAR_H), color, -1)
        tx = BAR_X + int(BAR_W * threshold)
        cv2.line(frame, (tx, y - 3), (tx, y + BAR_H + 3), (255, 255, 255), 1)  # 임계값선
        texts.append((label, (20, y + 2), color, 24))
        texts.append((f"{p * 100:4.0f}%", (BAR_X + BAR_W + 10, y + 2), (230, 230, 230), 24))
    return texts


def load_model(model_path: Path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SignLSTM(cfg["input_dim"], cfg["hidden_dim"], len(ckpt["labels"]),
                     num_layers=cfg["num_layers"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["labels"], cfg["frames"]


def main() -> None:
    parser = argparse.ArgumentParser(description="실시간 수신호 추론")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="확정에 필요한 최소 신뢰도")
    parser.add_argument("--consecutive", type=int, default=5,
                        help="확정에 필요한 연속 동일 예측 횟수")
    parser.add_argument("--ema", type=float, default=0.4,
                        help="확률 지수 평활 계수 (1.0 = 평활 끔) — 스파이크 노이즈 방어")
    parser.add_argument("--release-grace", type=float, default=1.0,
                        help="확정 해제 유예 시간(초) — 일시적 신뢰도 하락에 기계가 서지 않게")
    parser.add_argument("--publish", choices=["none", "udp", "rosbridge"], default="none",
                        help="확정 신호 외부 전달 방식 (기계 프로젝트 연동)")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5555)
    parser.add_argument("--rosbridge-url", default="ws://localhost:9090")
    parser.add_argument("--ros-topic", default="/hand_signal")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, labels, num_frames = load_model(args.model, device)
    print(f"모델 로드: {args.model.name}  라벨: {labels}")
    print(f"안정화: 신뢰도 ≥ {args.threshold}, 연속 {args.consecutive}회 일치 시 확정")

    publisher = make_publisher(args.publish, udp_host=args.udp_host, udp_port=args.udp_port,
                               rosbridge_url=args.rosbridge_url, ros_topic=args.ros_topic)
    draw_text = make_text_drawer()
    extractor = FeatureExtractor()
    cap = open_camera()
    t0 = time.monotonic()
    last_sent: tuple[str, float] = ("", 0.0)  # (마지막 전송 신호, 전송 시각)

    window: deque[np.ndarray] = deque(maxlen=num_frames)
    streak_label: int | None = None
    streak = 0
    confirmed = "인식 불가"
    confirmed_idx: int | None = None
    warning = HandWarning()
    smoothed: np.ndarray | None = None   # 확률 EMA 상태
    low_conf_since: float | None = None  # 해제 유예 타이머 시작 시각

    while True:
        frame, hand_result, pose_result = process_frame(cap, extractor, t0)
        if frame is None:
            print("프레임을 읽지 못했습니다. 종료합니다.")
            break
        draw_detections(frame, hand_result, pose_result)
        warning.update(hand_result)
        warning.draw(frame)
        window.append(FeatureExtractor.vector(hand_result, pose_result))

        probs, top = None, None
        status = "버퍼 채우는 중..."
        if len(window) == num_frames:
            arr = np.stack(window)

            # 안전 가드: 윈도우 대부분에서 사람(포즈)이 안 잡히면 모델 판단을 신뢰하지 않는다
            pose_ratio = float((arr[:, HAND_DIM * 2 :] != 0).any(axis=1).mean())
            if pose_ratio < 0.3:
                streak_label, streak = None, 0
                smoothed, low_conf_since = None, None  # 사람이 사라지면 평활·유예 상태도 초기화
                if confirmed != "인식 불가":
                    confirmed = "인식 불가"
                    confirmed_idx = None
                    print("확정: 인식 불가 (사람 미감지) → 안전 기본값(정지)")
                now = time.monotonic()
                if last_sent[0] != "unknown" or now - last_sent[1] >= 1.0:
                    publisher.publish("unknown", 0.0)
                    last_sent = ("unknown", now)
                panel_texts = draw_probability_panel(frame, labels, None, None,
                                                     None, args.threshold)
                frame = draw_text(frame, [
                    ("확정: 인식 불가", (10, 10), (80, 80, 255), 40),
                    ("사람 미감지", (10, 62), (80, 80, 255), 24),
                    *panel_texts,
                ])
                cv2.imshow("Sign Inference", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    break
                continue

            x = torch.from_numpy(arr[None]).to(device)
            with torch.no_grad():
                raw_probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            # 확률 EMA 평활: 1프레임 스파이크가 판단에 직접 튀지 않게
            smoothed = raw_probs if smoothed is None else \
                args.ema * raw_probs + (1 - args.ema) * smoothed
            probs = smoothed
            top = int(probs.argmax())
            conf = float(probs[top])
            status = f"연속 일치 {min(streak, args.consecutive)}/{args.consecutive}"

            # 안정화 필터: 임계값 + 연속 일치 (확정은 깐깐하게)
            if conf >= args.threshold:
                low_conf_since = None
                streak = streak + 1 if top == streak_label else 1
                streak_label = top
                if streak >= args.consecutive and confirmed != labels[top]:
                    confirmed = labels[top]
                    confirmed_idx = top
                    print(f"확정: {confirmed}  (신뢰도 {conf:.2f})")
            else:
                # 해제는 유예를 두고 (히스테리시스) — 일시적 하락에 기계가 서지 않게.
                # 사람 미감지 가드(위)는 이 유예와 무관하게 즉시 해제된다.
                streak_label, streak = None, 0
                now = time.monotonic()
                if confirmed != "인식 불가":
                    if low_conf_since is None:
                        low_conf_since = now
                    remain = args.release_grace - (now - low_conf_since)
                    if remain > 0:
                        status = f"신뢰 회복 대기 {remain:.1f}s (유지: {confirmed})"
                    else:
                        confirmed = "인식 불가"
                        confirmed_idx = None
                        low_conf_since = None
                        print("확정: 인식 불가 (유예 초과) → 안전 기본값(정지)")

        # 확정 상태 전송: 상태가 바뀌면 즉시, 같으면 1초 주기 하트비트
        signal = labels[confirmed_idx] if confirmed_idx is not None else "unknown"
        now = time.monotonic()
        if signal != last_sent[0] or now - last_sent[1] >= 1.0:
            conf_out = float(probs[confirmed_idx]) if (probs is not None and confirmed_idx is not None) else 0.0
            publisher.publish(signal, conf_out)
            last_sent = (signal, now)

        panel_texts = draw_probability_panel(frame, labels, probs, top,
                                             confirmed_idx, args.threshold)
        header_color = (0, 220, 0) if confirmed_idx is not None else (80, 80, 255)
        frame = draw_text(frame, [
            (f"확정: {confirmed}", (10, 10), header_color, 40),
            (status, (10, 62), (200, 200, 200), 24),
            *panel_texts,
        ])
        cv2.imshow("Sign Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    publisher.close()


if __name__ == "__main__":
    main()
