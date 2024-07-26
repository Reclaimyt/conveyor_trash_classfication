#include <ESP32Servo.h>
#include <PubSubClient.h>
#include <Arduino.h>
#include <WiFi.h>

// MQTT Configuration

const char* ssid = "Kiana";
const char* password = "1sampai8";
const char* mqtt_server = "test.mosquitto.org";
const int port = 1883;

const char* topic_conveyor = "/sic/kelompok15/conveyor";

String mqttMessage = "";

// Inisialisasi objek WiFi dan PubSubClient
WiFiClient espClient;
PubSubClient client(espClient);

// Define pin connections
#define IR_SENSOR_PIN 15
#define SERVO_PIN_A 2
#define SERVO_PIN_B 4

Servo servoA;
Servo servoB;

// Definisi pin untuk motor 1
#define MOTOR1_PIN1 22
#define MOTOR1_PIN2 21
#define MOTOR1_ENABLE 23

// Definisi pin untuk motor 2
#define MOTOR2_PIN1 19
#define MOTOR2_PIN2 18
#define MOTOR2_ENABLE 5

int motorSpeed = 98;

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Pesan diterima [");
  Serial.print(topic);
  Serial.print("] ");
  
  for (int i = 0; i < length ; i++) {
    mqttMessage += (char)payload[i];
  }

}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Menghubungkan ke ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi terhubung");
  Serial.println("Alamat IP: ");
  Serial.println(WiFi.localIP());
}

void reconnect() {
  // Loop sampai terhubung
  while (!client.connected()) {
    Serial.print("Menghubungkan ke broker MQTT...");

    if (client.connect("uPesy ESP32 Wroom DevKit")) {
      Serial.println("Terhubung");
      client.subscribe(topic_conveyor);
    } else {
      Serial.print("Gagal, rc=");
      Serial.print(client.state());
      Serial.println(" coba lagi dalam 5 detik");
      delay(5000);
    }

  }
}

void setup() {
    Serial.begin(115200);

    setup_wifi();
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);

    // Inisialisasi pin sebagai output
    pinMode(MOTOR1_PIN1, OUTPUT);
    pinMode(MOTOR1_PIN2, OUTPUT);
    pinMode(MOTOR1_ENABLE, OUTPUT);
    pinMode(MOTOR2_PIN1, OUTPUT);
    pinMode(MOTOR2_PIN2, OUTPUT);
    pinMode(MOTOR2_ENABLE, OUTPUT);

    // Matikan motor saat startup
    digitalWrite(MOTOR1_PIN1, LOW);
    digitalWrite(MOTOR1_PIN2, LOW);
    digitalWrite(MOTOR2_PIN1, LOW);
    digitalWrite(MOTOR2_PIN2, LOW);

    // Inisiasi Pin Servo
    servoA.attach(SERVO_PIN_A);
    servoB.attach(SERVO_PIN_B);

    pinMode(IR_SENSOR_PIN, INPUT);

    servoA.write(90);
    servoB.write(90);
}

void loop() {

    if (!client.connected()) {
        reconnect();
    }
    client.loop();

    digitalWrite(MOTOR1_PIN1, HIGH);
    digitalWrite(MOTOR1_PIN2, LOW);
    digitalWrite(MOTOR2_PIN1, HIGH);
    digitalWrite(MOTOR2_PIN2, LOW);
    analogWrite(MOTOR1_ENABLE, motorSpeed);
    analogWrite(MOTOR2_ENABLE, motorSpeed);
    delay(500);

    int irValue = digitalRead(IR_SENSOR_PIN);

    Serial.println(mqttMessage);

    if (irValue == LOW) {
        digitalWrite(MOTOR1_PIN1, LOW);
        digitalWrite(MOTOR1_PIN2, LOW);
        digitalWrite(MOTOR2_PIN1, LOW);
        digitalWrite(MOTOR2_PIN2, LOW);
        delay(1000);

        Serial.println("Object detected!");

        if(String(mqttMessage) == "B3"){
            Serial.println("Servo A Active.");
            servoA.write(145);
            servoB.write(90);
        } else if(String(mqttMessage) == "Anorganik") {
            Serial.println("Servo B Active.");
            servoB.write(150);
            servoA.write(90);
        }

        if(String(mqttMessage) == "Organik"){
            Serial.println("Two Servo Unactive.");
            servoA.write(90);
            servoB.write(90);
        }

        mqttMessage = "";

    } else {
        Serial.println("No object detected.");
    }
}