import json
import time
import paho.mqtt.client as mqtt

broker = "localhost"
port = 1883
topic = "/cctv/people_tracking/stats"
payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "camera_id": "CCTV_001",
    "event_type": "system_statistics",
    "data": {
        "in_count": {
            "Male": 29,
            "Female": 26,
            "Unknown": 17,
        },
        "out_count": {
            "Male": 30,
            "Female": 28,
            "Unknown": 18,
        },
        "ethnicity_count": {
            "Asia": 54,
            "Europe": 18,
        },
        "total_in": 72,
        "total_out": 76,
    }
}

try:
    client = mqtt.Client()
    client.connect(broker, port, 10)
    client.loop_start()
    
    result = client.publish(topic, json.dumps(payload, ensure_ascii=False), qos=1, retain=True)
    result.wait_for_publish(timeout=5)
    
    print(f"[✓] SUCCESS! Published statistics to {broker}:{port}")
    print(f"    Topic: {topic}")
    print(f"    in_count: M={payload['data']['in_count']['Male']} F={payload['data']['in_count']['Female']} U={payload['data']['in_count']['Unknown']}")
    print(f"    out_count: M={payload['data']['out_count']['Male']} F={payload['data']['out_count']['Female']} U={payload['data']['out_count']['Unknown']}")
    print(f"    ethnicity_count: Asia={payload['data']['ethnicity_count']['Asia']} Europe={payload['data']['ethnicity_count']['Europe']}")
    
    client.loop_stop()
    client.disconnect()
    
except Exception as e:
    print(f"[✗] Failed: {e}")
