// Primer model - firmware de captura en cru per a la Seeed XIAO nRF52840 Sense.
//
// Llegeix l'acceleròmetre integrat (LSM6DS3) a 50 Hz i envia TOTES les mostres
// per Bluetooth de baix consum (BLE) a l'ordinador. Els índexs del model
// (MA1, MA2 i AR d'Iacovelli et al. 2019) es calculen a l'ordinador, perquè
// així podem provar variants sense tornar a programar la placa.
//
// Paquet BLE (20 bytes, cap en una notificació estàndard):
//   uint16  seq           comptador de paquets (dona la volta a 65535)
//   int16   x,y,z  x 3    tres mostres seguides, en mil·lèsimes de g (mg)
// 50 mostres/s -> ~17 paquets/s.
//
// Abans de compilar, a Arduino IDE:
//   1. Gestor de plaques: "Seeed nRF52 mbed-enabled Boards" ->
//      placa "Seeed XIAO BLE Sense - nRF52840".
//   2. Gestor de llibreries: "ArduinoBLE" i "Seeed Arduino LSM6DS3".
//
// Amb una sola placa, deixa NOM = "ICTUS-D" i canvia-la de braç entre sessions
// (vegeu el README). Quan tinguem més plaques, cada una porta el seu nom.

#include <ArduinoBLE.h>
#include <LSM6DS3.h>
#include <Wire.h>

const char NOM[] = "ICTUS-D";   // "ICTUS-D" braç dret, "ICTUS-E" braç esquerre, "ICTUS-P" pit

// Mateixos UUID que el prototip 1, amb una característica nova per a les dades en cru.
const char UUID_SERVEI[] = "7a1e0001-5c3b-4f2a-9d6e-2b8c4a10f001";
const char UUID_CRU[]    = "7a1e0003-5c3b-4f2a-9d6e-2b8c4a10f001";

const unsigned long PERIODE_US = 20000;   // 20 ms = 50 Hz
const int MOSTRES_PER_PAQUET = 3;

struct __attribute__((packed)) Paquet {
  uint16_t seq;
  int16_t m[MOSTRES_PER_PAQUET][3];       // mg
};

LSM6DS3 imu(I2C_MODE, 0x6A);
BLEService servei(UUID_SERVEI);
BLECharacteristic dadesCru(UUID_CRU, BLERead | BLENotify, sizeof(Paquet));

Paquet paquet;
int k = 0;
uint16_t seq = 0;
uint32_t mostra = 0;                      // comptador de mostres de 32 bits (no dona la volta)
unsigned long seguent = 0;

static int16_t a_mg(float g) {
  float v = g * 1000.0f;
  if (v > 32767.0f) v = 32767.0f;
  if (v < -32768.0f) v = -32768.0f;
  return (int16_t)lroundf(v);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, HIGH);        // a la XIAO, HIGH = LED apagat

  if (imu.begin() != 0) {
    Serial.println("Error: l'accelerometre no respon (la placa ha de ser la versio Sense)");
    while (true) {}
  }
  if (!BLE.begin()) {
    Serial.println("Error: el Bluetooth no arrenca");
    while (true) {}
  }

  BLE.setLocalName(NOM);
  BLE.setDeviceName(NOM);
  BLE.setConnectionInterval(6, 12);       // 7,5-15 ms: prou marge per a 17 paquets/s
  BLE.setAdvertisedService(servei);
  servei.addCharacteristic(dadesCru);
  BLE.addService(servei);
  BLE.advertise();

  Serial.println("idx,ax_mg,ay_mg,az_mg");
  seguent = micros();
}

void loop() {
  BLE.poll();

  // LED encès mentre hi ha un ordinador connectat
  digitalWrite(LED_BUILTIN, BLE.connected() ? LOW : HIGH);

  if ((long)(micros() - seguent) < 0) return;
  seguent += PERIODE_US;

  int16_t x = a_mg(imu.readFloatAccelX());
  int16_t y = a_mg(imu.readFloatAccelY());
  int16_t z = a_mg(imu.readFloatAccelZ());

  paquet.m[k][0] = x;
  paquet.m[k][1] = y;
  paquet.m[k][2] = z;

  // Còpia per USB: serveix per depurar i com a alternativa si el BLE falla
  // (receptor_serial.py). idx = número de mostra des de l'encesa.
  Serial.print(mostra++); Serial.print(',');
  Serial.print(x); Serial.print(',');
  Serial.print(y); Serial.print(',');
  Serial.println(z);

  k++;
  if (k < MOSTRES_PER_PAQUET) return;

  paquet.seq = seq++;
  if (BLE.connected()) {
    dadesCru.writeValue((const uint8_t *)&paquet, sizeof(paquet));
  }
  k = 0;
}
