import paho.mqtt.client as mqtt

def send_to_mqtt(broker, port, topic, payload, qos=1, retain=False):
    """
    Mengirimkan data ke broker MQTT.

    Parameters:
    - broker (str): Alamat broker MQTT.
    - port (int): Port broker MQTT.
    - topic (str): Topik di mana pesan akan dikirim.
    - payload (str): Pesan yang akan dikirim.
    - qos (int): Quality of Service level. Default adalah 1.
    - retain (bool): Apakah pesan akan dipertahankan di broker. Default adalah False.
    """

    client = mqtt.Client()

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Terhubung ke broker MQTT")
        else:
            print(f"Terhubung gagal dengan kode {rc}")

    def on_publish(client, userdata, mid):
        print(f"Pesan berhasil dikirim dengan id: {mid}")

    client.on_connect = on_connect
    client.on_publish = on_publish

    client.connect(broker, port, 60)

    client.loop_start()

    result = client.publish(topic, payload, qos, retain)

    result.wait_for_publish()

    client.loop_stop()

    client.disconnect()
