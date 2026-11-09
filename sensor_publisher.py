import json
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "lab/iot/sensor"

client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="iot-sensor-01"
)

client.connect(BROKER, PORT, 60)
client.loop_start()

print("IoT Sensor Started")
print("Publishing sensor readings...")

try:
    count = 0

    while True:
        count += 1

        temperature = round(random.uniform(22, 30), 2)
        humidity = round(random.uniform(40, 65), 2)

        # Generate an abnormal reading every 10 messages
        if count % 10 == 0:
            temperature = round(random.uniform(40, 45), 2)
            humidity = round(random.uniform(75, 85), 2)

        data = {
            "device_id": "iot-sensor-01",
            "timestamp": datetime.now().isoformat(),
            "temperature": temperature,
            "humidity": humidity
        }

        client.publish(TOPIC, json.dumps(data))

        print("Published:", data)

        time.sleep(1)

except KeyboardInterrupt:
    print("\nSensor stopped.")

finally:
    client.loop_stop()
    client.disconnect()
