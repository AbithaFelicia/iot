import json
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPIC = "lab/iot/sensor"

# Illustrative normal sensor behaviour
NORMAL_TEMP = 26.0
NORMAL_HUMIDITY = 52.0

TEMP_LIMIT = 5.0
HUMIDITY_LIMIT = 15.0


def on_connect(client, userdata, flags, reason_code, properties):
    print("Digital Twin connected to MQTT broker")
    client.subscribe(TOPIC)


def on_message(client, userdata, message):
    try:
        data = json.loads(message.payload.decode())

        temperature = float(data["temperature"])
        humidity = float(data["humidity"])

        temp_deviation = abs(temperature - NORMAL_TEMP)
        humidity_deviation = abs(humidity - NORMAL_HUMIDITY)

        print("\n========== DIGITAL TWIN ==========")
        print("Device:", data["device_id"])
        print("Temperature:", temperature)
        print("Humidity:", humidity)

        if (temp_deviation > TEMP_LIMIT or
                humidity_deviation > HUMIDITY_LIMIT):
            print("STATUS: ANOMALY DETECTED")
        else:
            print("STATUS: NORMAL")

        print("=================================")

    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print("Invalid sensor message:", error)


client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2,
    client_id="digital-twin-monitor"
)

client.on_connect = on_connect
client.on_message = on_message

print("Starting Digital Twin monitor...")
client.connect(BROKER, PORT, 60)

try:
    client.loop_forever()
except KeyboardInterrupt:
    print("\nDigital Twin monitor stopped.")
    client.disconnect()
