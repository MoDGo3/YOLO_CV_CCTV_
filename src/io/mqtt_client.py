"""
MQTT Client - Real-time Data Publishing
ส่งข้อมูล People Tracking และ Demographics แบบ JSON ไปยัง IoT Dashboard
"""

import json
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
import paho.mqtt.client as mqtt
import numpy as np

class MQTTClient:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.command_handler = None
        self.command_topic = getattr(config, "MQTT_COMMAND_TOPIC", f"{config.MQTT_TOPIC}/commands")
        self._mirror_no_leading_slash = getattr(
            config, "MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH", False
        ) and isinstance(self.command_topic, str) and self.command_topic.startswith("/")
        
        # Statistics
        self.messages_sent = 0
        self.messages_failed = 0
        self.last_message_time = 0
        
        print(f"[INFO] Initializing MQTT Client")
        print(f"[INFO] Broker: {config.MQTT_BROKER}:{config.MQTT_PORT}")
        print(f"[INFO] Topic: {config.MQTT_TOPIC}")
        
        if config.MQTT_ENABLED:
            self.setup_client()
        
    def setup_client(self):
        """Setup MQTT client with callbacks"""
        try:
            self.client = mqtt.Client()
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_publish = self.on_publish
            self.client.on_log = self.on_log
            self.client.on_message = self.on_message
            
            # Set connection parameters
            self.client.keepalive = self.config.MQTT_KEEPALIVE
            
            # Authentication (username/password from .env)
            username = getattr(self.config, 'MQTT_USERNAME', None)
            password = getattr(self.config, 'MQTT_PASSWORD', None)
            if username:
                self.client.username_pw_set(username, password)
                print(f"[INFO] MQTT auth enabled (user: {username})")
            
            # TLS encryption (MQTT_USE_TLS=true in .env)
            if getattr(self.config, 'MQTT_USE_TLS', False):
                import ssl
                ca_certs = getattr(self.config, 'MQTT_CA_CERTS', None)
                self.client.tls_set(ca_certs=ca_certs, cert_reqs=ssl.CERT_REQUIRED,
                                    tls_version=ssl.PROTOCOL_TLS_CLIENT)
                if getattr(self.config, 'MQTT_TLS_INSECURE', False):
                    self.client.tls_insecure_set(True)
                    print("[WARNING] MQTT TLS hostname check disabled (testing only)")
                print("[INFO] MQTT TLS enabled")
            elif self.config.MQTT_BROKER not in ("localhost", "127.0.0.1"):
                print("[WARNING] MQTT is sending demographic data WITHOUT encryption "
                      f"to {self.config.MQTT_BROKER} - set MQTT_USE_TLS=true in .env")
            
            print("[OK] MQTT client configured")
            
        except Exception as e:
            print(f"[ERROR] MQTT client setup failed: {e}")
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback for when client connects to broker"""
        if rc == 0:
            self.connected = True
            self.reconnect_attempts = 0
            print("[OK] MQTT connected successfully")
            self._subscribe_command_topics()
        else:
            self.connected = False
            error_messages = {
                1: "Connection refused - incorrect protocol version",
                2: "Connection refused - invalid client identifier", 
                3: "Connection refused - server unavailable",
                4: "Connection refused - bad username or password",
                5: "Connection refused - not authorised"
            }
            error_msg = error_messages.get(rc, f"Connection failed with code {rc}")
            print(f"[ERROR] MQTT connection failed: {error_msg}")
            
    def on_disconnect(self, client, userdata, rc):
        """Callback for when client disconnects from broker"""
        self.connected = False
        if rc != 0:
            print(f"[WARNING] MQTT unexpected disconnection (code: {rc})")
            self.handle_reconnection()
        else:
            print("[INFO] MQTT disconnected gracefully")
            
    def on_publish(self, client, userdata, mid):
        """Callback for when message is published"""
        self.messages_sent += 1
        self.last_message_time = time.time()
        
    def on_log(self, client, userdata, level, buf):
        """Callback for logging MQTT events"""
        # Only log errors and warnings to avoid spam
        if level <= mqtt.MQTT_LOG_WARNING:
            print(f"[INFO] MQTT Log: {buf}")
    
    def on_message(self, client, userdata, message):
        """Handle inbound MQTT messages (commands)."""
        if not self.command_handler:
            return
        try:
            payload_str = message.payload.decode("utf-8")
            data = json.loads(payload_str)
        except Exception:
            data = {"raw_payload": message.payload.decode("utf-8", errors="replace")}
        try:
            self.command_handler(data, message=message)
        except Exception as exc:
            print(f"[WARNING] MQTT command handler error: {exc}")
    
    def connect(self):
        """Connect to MQTT broker"""
        if not self.config.MQTT_ENABLED:
            print("[INFO] MQTT is disabled in config")
            return
            
        if self.client is None:
            self.setup_client()
            
        try:
            print(f"[INFO] Connecting to MQTT broker: {self.config.MQTT_BROKER}:{self.config.MQTT_PORT}")
            self.client.connect(
                self.config.MQTT_BROKER, 
                self.config.MQTT_PORT, 
                self.config.MQTT_KEEPALIVE
            )
            
            # Start network loop in background thread
            self.client.loop_start()
            
            # Wait for connection confirmation
            timeout = 10
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
                
            if not self.connected:
                raise TimeoutError("MQTT connection timeout")
                
        except Exception as e:
            print(f"[ERROR] MQTT connection error: {e}")
            self.handle_reconnection()
    
    def set_command_handler(self, handler, topic: Optional[str] = None):
        """Register a callback to handle incoming commands."""
        self.command_handler = handler
        if topic:
            self.command_topic = topic
            self._mirror_no_leading_slash = (
                getattr(self.config, "MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH", False)
                and self.command_topic.startswith("/")
            )
        if self.connected:
            self._subscribe_command_topics()

    def _subscribe_command_topics(self):
        """Subscribe to command topics if handler is defined."""
        if not self.command_handler or not self.client or not self.connected:
            return
        try:
            if self.command_topic:
                self.client.subscribe(self.command_topic, qos=self.config.MQTT_QOS)
                print(f"[INFO] Subscribed to command topic: {self.command_topic}")
                if self._mirror_no_leading_slash:
                    mirror_topic = self.command_topic[1:]
                    self.client.subscribe(mirror_topic, qos=self.config.MQTT_QOS)
                    print(f"[INFO] Subscribed to compat command topic: {mirror_topic}")
        except Exception as exc:
            print(f"[WARNING] Failed to subscribe command topic: {exc}")
            
    def handle_reconnection(self):
        """Handle MQTT reconnection logic"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            print(f"[ERROR] Max MQTT reconnection attempts reached ({self.max_reconnect_attempts})")
            return
            
        self.reconnect_attempts += 1
        print(f"[INFO] MQTT Reconnecting... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
        
        # Wait before reconnecting
        wait_time = min(2 ** self.reconnect_attempts, 30)  # Exponential backoff, max 30s
        time.sleep(wait_time)
        
        self.connect()
        
    def publish_event(self, event_data: Dict[str, Any]):
        """Publish people tracking event to MQTT"""
        if not self.config.MQTT_ENABLED or not self.connected:
            return False
            
        try:
            # Normalize event_data if it is an object with to_dict
            if hasattr(event_data, 'to_dict'):
                event_data = event_data.to_dict()
            # Create MQTT message
            mqtt_message = self.create_mqtt_message(event_data)
            
            # Publish message
            topic_main = self.config.MQTT_TOPIC
            def _json_default(o):
                # Convert numpy types to native Python types
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, (np.ndarray,)):
                    return o.tolist()
                return str(o)

            result = self.client.publish(
                topic=topic_main,
                payload=json.dumps(mqtt_message, ensure_ascii=False, default=_json_default),
                qos=self.config.MQTT_QOS,
                retain=self.config.MQTT_RETAIN
            )
            
            # Check if message was queued successfully
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                # Optionally publish a JSON summary payload
                if getattr(self.config, 'MQTT_SEND_SUMMARY_LABEL', False):
                    summary = self._build_summary_json(event_data)
                    try:
                        topic_label = getattr(self.config, 'MQTT_SUMMARY_TOPIC', topic_main + "/label")
                        self.client.publish(
                            topic=topic_label,
                            payload=json.dumps(summary, ensure_ascii=False, default=_json_default),
                            qos=self.config.MQTT_QOS,
                            retain=False
                        )
                        # Mirror without leading slash if enabled
                        if getattr(self.config, 'MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH', False) and topic_label.startswith('/'):
                            self.client.publish(
                                topic=topic_label[1:],
                                payload=json.dumps(summary, ensure_ascii=False, default=_json_default),
                                qos=self.config.MQTT_QOS,
                                retain=False
                            )
                    except Exception as e:
                        # Non-critical
                        print(f"[WARNING] MQTT summary publish failed: {e}")
                # Mirror event without leading slash if enabled
                if getattr(self.config, 'MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH', False) and topic_main.startswith('/'):
                    try:
                        self.client.publish(
                            topic=topic_main[1:],
                            payload=json.dumps(mqtt_message, ensure_ascii=False, default=_json_default),
                            qos=self.config.MQTT_QOS,
                            retain=self.config.MQTT_RETAIN
                        )
                    except Exception as e:
                        print(f"[WARNING] MQTT mirror publish failed: {e}")
                return True
            else:
                self.messages_failed += 1
                print(f"[ERROR] MQTT publish failed: {mqtt.error_string(result.rc)}")
                return False
                
        except Exception as e:
            self.messages_failed += 1
            print(f"[ERROR] Error publishing MQTT message: {e}")
            return False
            
    def create_mqtt_message(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized MQTT message format"""
        
        # Base message structure
        mqtt_message = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": "CCTV_001",  # Can be configured
            "event_type": "person_tracking",
            "data": {
                "person_id": event_data.get("track_id"),
                "tracked_count": event_data.get("tracked_count"),
                "demographics": {
                    "gender": event_data.get("gender"),
                    "age": event_data.get("age"),
                    "confidence_scores": {
                        "gender": event_data.get("gender_confidence", 0.0),
                        "age": event_data.get("age_confidence", 0.0)
                    }
                },
                "movement": {
                    "status": event_data.get("status"),  # "entered" or "left"
                    "direction": event_data.get("direction", "unknown"),
                    "location": {
                        "bbox": event_data.get("bbox", []),
                        "center": event_data.get("center", [])
                    }
                }
            }
        }
        # Add human-readable label to message as well
        try:
            mqtt_message["label"] = self._build_summary_label(event_data)
            mqtt_message["summary"] = self._build_summary_json(event_data)
        except Exception:
            pass
        
        # Add ethnicity data if enabled and available
        if (self.config.ETHNICITY_ENABLED and 
            event_data.get("ethnicity") and 
            event_data.get("ethnicity_confidence", 0) >= self.config.ETHNICITY_ABSTAIN_THRESHOLD):
            
            mqtt_message["data"]["demographics"]["ethnicity"] = event_data.get("ethnicity")
            mqtt_message["data"]["demographics"]["group"] = event_data.get("group")
            mqtt_message["data"]["demographics"]["confidence_scores"]["ethnicity"] = event_data.get("ethnicity_confidence", 0.0)
        
        return mqtt_message

    def _build_summary_label(self, event_data: Dict[str, Any]) -> str:
        """Build a concise human-readable label: ID:{tid} | {gender} | {age_group} | {region}"""
        tid = event_data.get("track_id")
        gender = event_data.get("gender") or "Unknown"
        age = event_data.get("age")
        age_group = self._map_age_group(age)
        region = self._map_region(event_data.get("ethnicity"))
        return f"ID:{tid} | {gender} | {age_group} | {region}"

    def _map_age_group(self, age_val) -> str:
        if isinstance(age_val, (int, float)):
            try:
                a = int(age_val)
                if a <= 12:
                    return "Child"
                elif a <= 19:
                    return "Teen"
                elif a <= 39:
                    return "Adult"
                elif a <= 59:
                    return "Middle-age"
                else:
                    return "Elderly"
            except Exception:
                return "Unknown"
        return "Unknown"

    def _map_region(self, ethnicity) -> str:
        if not ethnicity:
            return "Unknown"
        el = str(ethnicity).lower()
        if any(k in el for k in ['asian', 'indian', 'middle eastern', 'latino', 'black', 'african']):
            return "Asian"
        if any(k in el for k in ['white', 'european']):
            return "Europe"
        return "Unknown"

    def _build_summary_json(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON summary with requested fields and extras."""
        return {
            "id": event_data.get("track_id"),
            "gender": event_data.get("gender") or "Unknown",
            "age_group": self._map_age_group(event_data.get("age")),
            "region": self._map_region(event_data.get("ethnicity")),
            "direction": event_data.get("direction", "unknown"),
            "status": event_data.get("status"),
            "timestamp": datetime.now().isoformat()
        }
        
    def publish_statistics(self, stats: Dict[str, Any]):
        """Publish system statistics"""
        if not self.config.MQTT_ENABLED or not self.connected:
            return False
            
        stats_message = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": "CCTV_001",
            "event_type": "system_statistics", 
            "data": stats
        }
        
        try:
            topic_stats = f"{self.config.MQTT_TOPIC}/stats"
            result = self.client.publish(
                topic=topic_stats,
                payload=json.dumps(stats_message, ensure_ascii=False),
                qos=self.config.MQTT_QOS,
                retain=True  # Retain stats for latest values
            )
            # Mirror without leading slash if enabled
            if getattr(self.config, 'MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH', False) and topic_stats.startswith('/'):
                try:
                    self.client.publish(
                        topic=topic_stats[1:],
                        payload=json.dumps(stats_message, ensure_ascii=False),
                        qos=self.config.MQTT_QOS,
                        retain=True
                    )
                except Exception as e:
                    print(f"[WARNING] MQTT stats mirror failed: {e}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
            
        except Exception as e:
            print(f"[ERROR] Error publishing statistics: {e}")
            return False

    def publish_control(self, action: str, **details) -> bool:
        """Publish control notifications (e.g., resets) to the main topic."""
        if not self.config.MQTT_ENABLED or not self.connected:
            return False

        control_event = {
            "timestamp": datetime.now().isoformat(),
            "camera_id": "CCTV_001",
            "event_type": "system_control",
            "action": action,
            "data": details,
        }

        try:
            payload = json.dumps(control_event, ensure_ascii=False)
            result = self.client.publish(
                topic=self.config.MQTT_TOPIC,
                payload=payload,
                qos=self.config.MQTT_QOS,
                retain=False,
            )
            if getattr(self.config, "MQTT_COMPAT_PUBLISH_NO_LEADING_SLASH", False) and self.config.MQTT_TOPIC.startswith("/"):
                try:
                    self.client.publish(
                        topic=self.config.MQTT_TOPIC[1:],
                        payload=payload,
                        qos=self.config.MQTT_QOS,
                        retain=False,
                    )
                except Exception as exc:
                    print(f"[WARNING] MQTT control mirror failed: {exc}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as exc:
            print(f"[ERROR] Error publishing control event: {exc}")
            return False
            
    def get_status(self) -> Dict[str, Any]:
        """Get MQTT client status"""
        return {
            "enabled": self.config.MQTT_ENABLED,
            "connected": self.connected,
            "broker": f"{self.config.MQTT_BROKER}:{self.config.MQTT_PORT}",
            "topic": self.config.MQTT_TOPIC,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "success_rate": (self.messages_sent / max(self.messages_sent + self.messages_failed, 1)) * 100,
            "last_message_time": self.last_message_time,
            "reconnect_attempts": self.reconnect_attempts
        }
        
    def disconnect(self):
        """Disconnect from MQTT broker"""
        if self.client and self.connected:
            print("[INFO] Disconnecting MQTT client...")
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print("[OK] MQTT disconnected")
            
    def __del__(self):
        """Destructor"""
        self.disconnect()
