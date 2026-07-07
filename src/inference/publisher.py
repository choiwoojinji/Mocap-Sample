"""분류 결과 외부 전달 인터페이스 (doc/ml-flow.md 6단계).

이 저장소의 책임은 확정된 수신호를 외부(기계 시뮬레이터 프로젝트)로 내보내는
것까지다. ROS 2 노드 자체는 별도 프로젝트가 담당한다 (doc/AGENT.md 범위 참고).

백엔드 두 가지:
    rosbridge — rosbridge 웹소켓 서버로 ROS 2 토픽(std_msgs/String) 발행.
                기계 프로젝트(Docker/Ubuntu)에서 rosbridge_server를 띄워두면
                macOS의 추론 프로세스가 ROS 없이도 토픽을 발행할 수 있다.
    udp       — JSON을 UDP 데이터그램으로 송신. 의존성·서버 불필요, 디버깅용.

메시지 형식 (JSON, ROS에서는 std_msgs/String.data 안에 담김):
    {"signal": "stop", "confidence": 0.93, "timestamp": 1783300000.0}
    signal은 모델 라벨 그대로, 인식 불가 상태는 "unknown" (기계 쪽 안전 기본값 = 정지).
"""

import json
import socket
import time


class ConsolePublisher:
    """전송 없이 콘솔 출력만 (기본값)."""

    def publish(self, signal: str, confidence: float) -> None:
        pass  # predict.py가 이미 콘솔에 출력하므로 아무것도 하지 않는다

    def close(self) -> None:
        pass


class UdpPublisher:
    """JSON을 UDP로 송신한다."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5555) -> None:
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[publisher] UDP → {host}:{port}")

    def publish(self, signal: str, confidence: float) -> None:
        payload = json.dumps(
            {"signal": signal, "confidence": round(confidence, 3), "timestamp": time.time()},
            ensure_ascii=False,
        )
        self.sock.sendto(payload.encode("utf-8"), self.addr)

    def close(self) -> None:
        self.sock.close()


class RosbridgePublisher:
    """rosbridge 웹소켓으로 ROS 2 토픽(std_msgs/String)을 발행한다.

    서버가 없거나 끊겨도 추론을 막지 않는다 — 전송 실패는 경고만 남기고
    5초 간격으로 재접속을 시도한다.
    """

    RECONNECT_COOLDOWN = 5.0

    def __init__(self, url: str = "ws://localhost:9090", topic: str = "/hand_signal") -> None:
        self.url = url
        self.topic = topic
        self.ws = None
        self._last_attempt = 0.0
        self._connect()

    def _connect(self) -> None:
        import websocket

        self._last_attempt = time.monotonic()
        try:
            self.ws = websocket.create_connection(self.url, timeout=2)
            self.ws.send(json.dumps({
                "op": "advertise",
                "topic": self.topic,
                "type": "std_msgs/String",
            }))
            print(f"[publisher] rosbridge 연결됨 → {self.url}  토픽 {self.topic}")
        except Exception as e:
            self.ws = None
            print(f"[publisher] rosbridge 연결 실패 ({e}) — {self.RECONNECT_COOLDOWN}초 후 재시도")

    def publish(self, signal: str, confidence: float) -> None:
        if self.ws is None:
            if time.monotonic() - self._last_attempt >= self.RECONNECT_COOLDOWN:
                self._connect()
            if self.ws is None:
                return
        data = json.dumps(
            {"signal": signal, "confidence": round(confidence, 3), "timestamp": time.time()},
            ensure_ascii=False,
        )
        try:
            self.ws.send(json.dumps({
                "op": "publish",
                "topic": self.topic,
                "msg": {"data": data},
            }))
        except Exception as e:
            print(f"[publisher] rosbridge 전송 실패 ({e}) — 재접속 예정")
            self.ws = None

    def close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.send(json.dumps({"op": "unadvertise", "topic": self.topic}))
                self.ws.close()
            except Exception:
                pass
            self.ws = None


def make_publisher(kind: str, *, udp_host: str, udp_port: int,
                   rosbridge_url: str, ros_topic: str):
    if kind == "udp":
        return UdpPublisher(udp_host, udp_port)
    if kind == "rosbridge":
        return RosbridgePublisher(rosbridge_url, ros_topic)
    return ConsolePublisher()
