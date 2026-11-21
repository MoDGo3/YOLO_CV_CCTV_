from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import threading
import json
import time
from typing import Set, Any, Dict, Union, Optional, Iterable, Tuple
import paho.mqtt.client as mqtt
from config import Config

app = FastAPI(title="CCTV MQTT Dashboard")
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

MQTT_CLIENT = None
EVENT_LOOP: Optional[asyncio.AbstractEventLoop] = None
CONNECTED_WS: Set[WebSocket] = set()
MSG_BUFFER = []  # keep recent messages
BUFFER_SIZE = 200

DISPLAY_STATS = {
    "in_count": 0,
    "out_count": 0,
    "asia_count": 0,
    "europe_count": 0,
    "male_count": 0,
    "female_count": 0,
    "unknown_count": 0,
    "unknown_in": 0,
    "unknown_out": 0,
}


def _reset_display_stats() -> None:
    for key in DISPLAY_STATS.keys():
        DISPLAY_STATS[key] = 0


def _get_nested_value(source: Dict[str, Any], paths: Iterable[Tuple[str, ...]]) -> Any:
    for path in paths:
        current: Any = source
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    return None


def _compute_region_counts(ethnicity_counts: Dict[str, Any]) -> Tuple[int, int]:
    asia = 0
    europe = 0
    for key, value in ethnicity_counts.items():
        try:
            count = int(value)
        except Exception:
            continue
        key_lower = str(key).lower()
        if any(token in key_lower for token in ["white", "europe", "european"]):
            europe += count
        else:
            # default all other ethnicities into asia bucket to match overlay behaviour
            asia += count
    return asia, europe


def _apply_stats_snapshot(snapshot: Dict[str, Any]) -> Dict[str, int]:
    in_counts = snapshot.get("in_count", {})
    out_counts = snapshot.get("out_count", {})
    ethnicity_counts = snapshot.get("ethnicity_count", {})

    total_in = snapshot.get("total_in")
    if total_in is None:
        total_in = sum(int(v) for v in in_counts.values() if isinstance(v, (int, float)))

    total_out = snapshot.get("total_out")
    if total_out is None:
        total_out = sum(int(v) for v in out_counts.values() if isinstance(v, (int, float)))

    asia, europe = _compute_region_counts(ethnicity_counts)

    DISPLAY_STATS["in_count"] = int(total_in or 0)
    DISPLAY_STATS["out_count"] = int(total_out or 0)
    # รวม Male/Female/Unknown จากทั้ง IN และ OUT
    DISPLAY_STATS["male_count"] = int(in_counts.get("Male", 0)) + int(out_counts.get("Male", 0))
    DISPLAY_STATS["female_count"] = int(in_counts.get("Female", 0)) + int(out_counts.get("Female", 0))
    DISPLAY_STATS["unknown_count"] = int(in_counts.get("Unknown", 0)) + int(out_counts.get("Unknown", 0))
    # แยก Unknown IN และ Unknown OUT
    DISPLAY_STATS["unknown_in"] = int(in_counts.get("Unknown", 0))
    DISPLAY_STATS["unknown_out"] = int(out_counts.get("Unknown", 0))
    DISPLAY_STATS["asia_count"] = asia
    DISPLAY_STATS["europe_count"] = europe

    return DISPLAY_STATS.copy()


class MQTTBridge:
    def __init__(self, config: Config):
        self.config = config
        self.client = mqtt.Client()
        self.connected = False
        self.lock = threading.Lock()
        self.base_topic = config.MQTT_TOPIC
        self.command_topic = getattr(config, "MQTT_COMMAND_TOPIC", f"{self.base_topic}/commands")
        self.mirror_enabled = bool(
            getattr(config, "MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH", False)
            and isinstance(self.base_topic, str)
            and self.base_topic.startswith("/")
        )
        self.mirror_topic = self.base_topic[1:] if self.mirror_enabled else None
        stats_base = self.base_topic.rstrip("/")
        self.stats_topic = f"{stats_base}/stats"
        self.stats_mirror_topic = self.stats_topic[1:] if self.mirror_enabled else None

        # Assign callbacks
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client: mqtt.Client, userdata: Any, flags: Dict[str, Any], rc: int) -> None:
        if rc == 0:
            self.connected = True
            print("[dashboard] MQTT connected")
            # Subscribe to configured topic
            try:
                self.client.subscribe(self.base_topic, qos=self.config.MQTT_QOS)
                print(f"[dashboard] Subscribed to {self.base_topic}")
                if self.mirror_topic:
                    self.client.subscribe(self.mirror_topic, qos=self.config.MQTT_QOS)
                    print(f"[dashboard] Subscribed to mirror topic {self.mirror_topic}")
                self.client.subscribe(self.stats_topic, qos=self.config.MQTT_QOS)
                print(f"[dashboard] Subscribed to stats topic {self.stats_topic}")
                if self.stats_mirror_topic:
                    self.client.subscribe(self.stats_mirror_topic, qos=self.config.MQTT_QOS)
                    print(f"[dashboard] Subscribed to stats mirror topic {self.stats_mirror_topic}")
            except Exception as e:
                print(f"[dashboard] Subscribe failed: {e}")
        else:
            print(f"[dashboard] MQTT connect failed rc={rc}")

    def on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        self.connected = False
        print(f"[dashboard] MQTT disconnected rc={rc}")

    def on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        topic = msg.topic
        if self.mirror_topic and topic == self.mirror_topic:
            return
        is_stats_topic = topic == self.stats_topic or (
            self.stats_mirror_topic and topic == self.stats_mirror_topic
        )

        payload: Union[Dict[str, Any], str, None] = None
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            payload = msg.payload.decode("utf-8", errors="replace")

        wrapped: Dict[str, Any]

        if isinstance(payload, dict) and is_stats_topic:
            snapshot = payload.get("data", {})
            current_stats = _apply_stats_snapshot(snapshot)
            wrapped = {
                "topic": self.stats_topic,
                "payload": payload,
                "timestamp": time.time(),
                "stats": current_stats,
                "type": "stats_update",
            }
        else:
            if isinstance(payload, dict):
                event_type = payload.get("event_type") or payload.get("type")
                control_action = _get_nested_value(
                    payload,
                    [
                        ("action",),
                        ("data", "action"),
                    ],
                )

                if event_type == "system_control" and control_action == "reset":
                    _reset_display_stats()
                else:
                    direction_raw = _get_nested_value(
                        payload,
                        [
                            ("direction",),
                            ("summary", "direction"),
                            ("data", "movement", "direction"),
                            ("movement", "direction"),
                        ],
                    )
                    region_raw = _get_nested_value(
                        payload,
                        [
                            ("region",),
                            ("summary", "region"),
                            ("data", "region"),
                            ("data", "demographics", "region"),
                        ],
                    )
                    gender_raw = _get_nested_value(
                        payload,
                        [
                            ("gender",),
                            ("summary", "gender"),
                            ("data", "gender"),
                            ("data", "demographics", "gender"),
                        ],
                    )

                    direction = None
                    if isinstance(direction_raw, str):
                        d = direction_raw.strip().lower()
                        if d in {"in", "enter", "entered", "entering"}:
                            direction = "in"
                        elif d in {"out", "exit", "exited", "leaving", "leave"}:
                            direction = "out"
                    else:
                        direction = direction_raw

                    region = None
                    if isinstance(region_raw, str):
                        r = region_raw.strip().lower()
                        if r in {"asia", "asian"}:
                            region = "Asia"
                        elif r in {"europe", "european"}:
                            region = "Europe"
                    else:
                        region = region_raw

                    gender = None
                    if isinstance(gender_raw, str):
                        g = gender_raw.strip().lower()
                        if g in {"male", "man"}:
                            gender = "Male"
                        elif g in {"female", "woman", "lady"}:
                            gender = "Female"
                    else:
                        gender = gender_raw

                    if direction == "in":
                        DISPLAY_STATS["in_count"] += 1
                        if region == "Asia":
                            DISPLAY_STATS["asia_count"] += 1
                        elif region == "Europe":
                            DISPLAY_STATS["europe_count"] += 1

                        if gender == "Male":
                            DISPLAY_STATS["male_count"] += 1
                        elif gender == "Female":
                            DISPLAY_STATS["female_count"] += 1
                    elif direction == "out":
                        DISPLAY_STATS["out_count"] += 1

            wrapped = {
                "topic": topic,
                "payload": payload,
                "timestamp": time.time(),
                "stats": DISPLAY_STATS.copy(),
            }

        with self.lock:
            MSG_BUFFER.append(wrapped)
            if len(MSG_BUFFER) > BUFFER_SIZE:
                MSG_BUFFER.pop(0)

        for ws in list(CONNECTED_WS):
            if EVENT_LOOP is None or EVENT_LOOP.is_closed():
                continue
            try:
                asyncio.run_coroutine_threadsafe(
                    ws.send_text(json.dumps(wrapped, ensure_ascii=False)),
                    EVENT_LOOP,
                )
            except Exception:
                pass

    def publish_command(self, payload: Dict[str, Any]) -> bool:
        if not self.connected:
            return False
        try:
            result = self.client.publish(
                self.command_topic,
                json.dumps(payload, ensure_ascii=False),
                qos=self.config.MQTT_QOS,
            )
            if getattr(self.config, "MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH", False) and self.command_topic.startswith("/"):
                try:
                    self.client.publish(
                        self.command_topic[1:],
                        json.dumps(payload, ensure_ascii=False),
                        qos=self.config.MQTT_QOS,
                    )
                except Exception as exc:
                    print(f"[dashboard] Command mirror publish failed: {exc}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as exc:
            print(f"[dashboard] Failed to publish command: {exc}")
            return False

    def start(self):
        try:
            self.client.connect(self.config.MQTT_BROKER, self.config.MQTT_PORT, self.config.MQTT_KEEPALIVE)
            self.client.loop_start()
        except Exception as e:
            print(f"[dashboard] Failed to start MQTT client: {e}")

    def stop(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


@app.on_event("startup")
async def startup_event():
    global MQTT_CLIENT, EVENT_LOOP
    EVENT_LOOP = asyncio.get_running_loop()
    cfg = Config()
    MQTT_CLIENT = MQTTBridge(cfg)
    MQTT_CLIENT.start()


@app.on_event("shutdown")
def shutdown_event():
    global MQTT_CLIENT
    if MQTT_CLIENT:
        MQTT_CLIENT.stop()


# Benchmark results storage
BENCHMARK_RESULTS = {
    "last_updated": None,
    "inference_speed": {},
    "gender_metrics": {},
    "ethnicity_metrics": {},
    "system_info": {}
}


@app.get("/api/benchmark")
def get_benchmark_results():
    """Get latest benchmark results"""
    import os
    from pathlib import Path
    
    # Try to load from latest results file
    results_dir = Path("benchmark/results")
    if results_dir.exists():
        json_files = list(results_dir.glob("benchmark_results_*.json"))
        if json_files:
            latest_file = max(json_files, key=os.path.getmtime)
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                return {"error": str(e), "cached": BENCHMARK_RESULTS}
    
    return BENCHMARK_RESULTS


@app.post("/api/benchmark")
async def update_benchmark_results(data: dict):
    """Update benchmark results (called from benchmark script)"""
    global BENCHMARK_RESULTS
    BENCHMARK_RESULTS.update(data)
    BENCHMARK_RESULTS["last_updated"] = time.time()
    return {"status": "ok", "message": "Benchmark results updated"}


@app.get("/")
def index():
    with open("src/web/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/reset")
async def reset_dashboard():
    lock = MQTT_CLIENT.lock if MQTT_CLIENT else threading.Lock()
    with lock:
        MSG_BUFFER.clear()
        _reset_display_stats()

    payload_text = json.dumps(
        {
            "type": "reset",
            "timestamp": time.time(),
            "stats": DISPLAY_STATS.copy(),
        },
        ensure_ascii=False,
    )

    for ws in list(CONNECTED_WS):
        try:
            await ws.send_text(payload_text)
        except Exception:
            try:
                await ws.close()
            except Exception:
                pass
            CONNECTED_WS.discard(ws)

    if MQTT_CLIENT:
        MQTT_CLIENT.publish_command(
            {
                "action": "reset",
                "scope": "session",
                "source": "dashboard",
                "timestamp": time.time(),
            }
        )

    return {"status": "ok"}


@app.get("/metrics")
def get_metrics(limit: int = 50):
    lock = MQTT_CLIENT.lock if MQTT_CLIENT else threading.Lock()
    with lock:
        stats_copy = DISPLAY_STATS.copy()
        limit = max(0, limit)
        if limit == 0:
            messages_copy = []
        else:
            messages_copy = list(MSG_BUFFER[-limit:])
    return {
        "stats": stats_copy,
        "messages": messages_copy,
        "timestamp": time.time(),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    CONNECTED_WS.add(websocket)
    try:
        if MQTT_CLIENT:
            with MQTT_CLIENT.lock:
                for msg in MSG_BUFFER:
                    await websocket.send_text(json.dumps(msg, ensure_ascii=False))
        while True:
            data = await websocket.receive_text()
            if data and data.lower() == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
    except WebSocketDisconnect:
        CONNECTED_WS.discard(websocket)
    except Exception:
        CONNECTED_WS.discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
