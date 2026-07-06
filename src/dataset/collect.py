"""수신호 랜드마크 시퀀스 수집 스크립트 (자동 연속 수집 + 화면 버튼).

개발 단계 2 (doc/design.md): 클래스(수신호)별로 특징 시퀀스를 라벨과 함께 저장한다.
특징은 양손 + 상체 포즈 150차원 (src/capture/extractor.py 참고).

사용법:
    python -m src.dataset.collect --label 멈춰 [--samples 30] [--frames 30]

조작 (화면 우상단 버튼 클릭 또는 키보드):
    START/PAUSE 버튼 (또는 SPACE) — 자동 수집 시작/일시정지.
        시작하면 "READY 카운트다운(기본 1.5초) → REC 녹화(30프레임) → 저장"이
        목표 개수를 채울 때까지 자동 반복된다. 카운트다운 동안 다음 동작을 준비하면 된다.
    QUIT 버튼 (또는 q/ESC) — 종료

품질 보호:
    손이 감지된 프레임이 절반 미만이면 저장하지 않고 SKIP 처리한다.
    (신호없음처럼 손이 없는 게 정상인 라벨은 --idle 로 필터를 끈다)

저장 형식:
    asset/<label>/<타임스탬프>.npy — shape (frames, 150)
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from src.capture.extractor import FEATURE_DIM, HAND_DIM, FeatureExtractor, draw_pose
from src.capture.landmark_viewer import HandWarning, draw_landmarks, open_camera

DATA_DIR = Path(__file__).resolve().parents[2] / "asset"
MIN_DETECTED_RATIO = 0.5  # 손 감지 프레임이 이 비율 미만이면 SKIP

WINDOW = "Collect"
BTN_W, BTN_H, BTN_MARGIN = 150, 50, 10



class ButtonBar:
    """창 우상단에 클릭 가능한 버튼을 그리고 클릭 이벤트를 받는다."""

    def __init__(self) -> None:
        self._clicked: str | None = None
        self._rects: dict[str, tuple[int, int, int, int]] = {}

    def attach(self, window: str) -> None:
        cv2.setMouseCallback(window, self._on_mouse)

    def _on_mouse(self, event, x, y, *_args) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (x1, y1, x2, y2) in self._rects.items():
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self._clicked = name

    def draw(self, frame, buttons: list[tuple[str, str, tuple[int, int, int]]]) -> None:
        """buttons: (이름, 표시 텍스트, BGR 색) 목록. 오른쪽부터 배치된다."""
        self._rects = {}
        w = frame.shape[1]
        x2 = w - BTN_MARGIN
        for name, label, color in buttons:
            x1 = x2 - BTN_W
            y1, y2 = BTN_MARGIN, BTN_MARGIN + BTN_H
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                frame, label, (x1 + 14, y1 + 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )
            self._rects[name] = (x1, y1, x2, y2)
            x2 = x1 - BTN_MARGIN

    def pop(self) -> str | None:
        clicked, self._clicked = self._clicked, None
        return clicked


def process_frame(cap, extractor: FeatureExtractor, t0: float):
    """프레임 1장을 읽어 검출까지 수행한다. 반환: (frame, 손결과, 포즈결과) 또는 (None,)*3."""
    ok, frame = cap.read()
    if not ok:
        return None, None, None
    frame = cv2.flip(frame, 1)  # 셀피 뷰
    timestamp_ms = int((time.monotonic() - t0) * 1000)
    hand_result, pose_result = extractor.detect(frame, timestamp_ms)
    return frame, hand_result, pose_result


def draw_detections(frame, hand_result, pose_result) -> None:
    draw_pose(frame, pose_result)
    for hand in hand_result.hand_landmarks:
        draw_landmarks(frame, hand)


def show(frame, hand_result, pose_result, text: str, color, bar: ButtonBar, buttons,
         progress: float | None = None,
         warning: HandWarning | None = None) -> str | None:
    """랜드마크·상태 텍스트·버튼을 그려 표시하고, 발생한 동작을 반환한다.

    progress가 주어지면(0~1) 화면 하단에 빨간 진행 바를 그린다 (녹화 진행 표시용).
    warning이 주어지면 손 미감지 상태를 갱신하고 이탈 경고를 그린다.
    반환: 'toggle' (START/PAUSE 버튼 또는 SPACE), 'quit' (QUIT 버튼 또는 q/ESC), None
    """
    draw_detections(frame, hand_result, pose_result)
    if warning is not None:
        warning.update(hand_result)
        warning.draw(frame)
    cv2.putText(frame, text, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    if progress is not None:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 14), (int(w * progress), h), (0, 0, 255), -1)
    bar.draw(frame, buttons)
    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    clicked = bar.pop()
    if key in (ord("q"), 27) or clicked == "quit":  # 27 = ESC
        return "quit"
    if key == ord(" ") or clicked == "toggle":
        return "toggle"
    return None


def hand_detected_ratio(seq: np.ndarray) -> float:
    """시퀀스에서 손(앞 126차원)이 감지된 프레임 비율."""
    return float((seq[:, : HAND_DIM * 2] != 0).any(axis=1).mean())


def record_sequence(cap, extractor, num_frames: int, t0: float,
                    bar: ButtonBar | None = None,
                    status: str = "",
                    warning: HandWarning | None = None) -> tuple[np.ndarray, str | None]:
    """num_frames 프레임 동안 시퀀스를 녹화한다. 반환: (시퀀스 (num_frames, 150), 동작).

    녹화 진행은 프레임 숫자 대신 하단 진행 바로 표시한다 (저장 개수와 혼동 방지).
    """
    buffer = []
    action = None
    while len(buffer) < num_frames:
        frame, hand_result, pose_result = process_frame(cap, extractor, t0)
        if frame is None:
            continue
        buffer.append(FeatureExtractor.vector(hand_result, pose_result))
        if bar is not None:
            a = show(frame, hand_result, pose_result, f"{status}  REC", (0, 0, 255),
                     bar, [("quit", "QUIT", (0, 0, 180))],
                     progress=len(buffer) / num_frames, warning=warning)
            action = a or action
    return np.stack(buffer), action


def countdown(cap, extractor, t0: float, seconds: float, status: str,
              bar: ButtonBar, warning: HandWarning | None = None) -> str | None:
    """다음 녹화 전 준비 카운트다운. 발생한 동작을 반환한다."""
    action = None
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        frame, hand_result, pose_result = process_frame(cap, extractor, t0)
        if frame is None:
            continue
        remain = end - time.monotonic()
        a = show(frame, hand_result, pose_result, f"{status}  READY {remain:.1f}s",
                 (0, 200, 255), bar,
                 [("quit", "QUIT", (0, 0, 180)), ("toggle", "PAUSE", (0, 140, 200))],
                 warning=warning)
        action = a or action
        if action == "quit":
            break
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description="수신호 특징 시퀀스 수집 (자동 연속)")
    parser.add_argument("--label", required=True, help="수신호 라벨 (예: 멈춰)")
    parser.add_argument("--samples", type=int, default=30, help="수집할 시퀀스 개수")
    parser.add_argument("--frames", type=int, default=30, help="시퀀스당 프레임 수")
    parser.add_argument("--prep", type=float, default=1.5, help="녹화 전 준비 시간(초)")
    parser.add_argument("--idle", action="store_true",
                        help="손 감지 품질 필터 끄기 (신호없음처럼 손이 없어도 되는 라벨)")
    args = parser.parse_args()

    out_dir = DATA_DIR / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.npy")))
    print(f"라벨 '{args.label}' — 기존 {existing}개, 목표 +{args.samples}개")
    print("화면의 START 버튼(또는 SPACE): 자동 수집 시작/일시정지, QUIT(또는 q/ESC): 종료")

    extractor = FeatureExtractor()
    cap = open_camera()
    cv2.namedWindow(WINDOW)
    bar = ButtonBar()
    bar.attach(WINDOW)
    warning = HandWarning(enabled=not args.idle)  # idle 수집은 손이 없는 게 정상

    t0 = time.monotonic()
    saved = 0
    skipped = 0
    collecting = False

    while saved < args.samples:
        status = f"[{args.label}] {existing + saved}/{existing + args.samples}"

        if not collecting:
            # 대기 화면: START 클릭(또는 SPACE)으로 자동 수집 시작
            frame, hand_result, pose_result = process_frame(cap, extractor, t0)
            if frame is None:
                print("프레임을 읽지 못했습니다. 종료합니다.")
                break
            action = show(frame, hand_result, pose_result,
                          status + "  SPACE=start  q=quit", (0, 255, 0), bar,
                          [("quit", "QUIT", (0, 0, 180)), ("toggle", "START", (0, 160, 0))],
                          warning=warning)
            if action == "quit":
                break
            if action == "toggle":
                collecting = True
            continue

        # 자동 수집: 카운트다운 → 녹화 → 저장, 반복
        action = countdown(cap, extractor, t0, args.prep, status, bar, warning)
        if action == "quit":
            break
        if action == "toggle":
            collecting = False
            print("일시정지 — START(또는 SPACE)로 재개")
            continue

        seq, action = record_sequence(cap, extractor, args.frames, t0, bar, status, warning)
        ratio = hand_detected_ratio(seq)
        if not args.idle and ratio < MIN_DETECTED_RATIO:
            skipped += 1
            print(f"SKIP (손 감지 {ratio * 100:.0f}% < 50%) — 손을 화면에 보이게 해주세요")
        else:
            path = out_dir / f"{int(time.time() * 1000)}.npy"
            np.save(path, seq)
            saved += 1
            print(f"저장 {saved}/{args.samples}: {path.name}  손 감지 {ratio * 100:.0f}%")
        if action == "quit":
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print(f"완료 — 저장 {saved}개, 스킵 {skipped}개, 총 {existing + saved}개 ({out_dir})")


if __name__ == "__main__":
    main()
