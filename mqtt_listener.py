import json
import threading
import paho.mqtt.client as mqtt

latest_reading = {}
lock = threading.Lock()

BROKER = "localhost"
PORT = 1883
TOPIC = "lab/iot/sensor"


def on_connect(client, userdata, flags, reason_code, properties):
    print("Connected to MQTT broker")
    client.subscribe(TOPIC)


def on_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode())

        with lock:
            latest_reading.clear()
            latest_reading.update(data)

    except (ValueError, TypeError):
        print("Invalid MQTT message")


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="streamlit-dashboard"
)

client.on_connect = on_connect
client.on_message = on_message


def start_listener():
    if not client.is_connected():
        client.connect(BROKER, PORT, 60)
        client.loop_start()


def get_latest_reading():
    with lock:
        return latest_reading.copy()
