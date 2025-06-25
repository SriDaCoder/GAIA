#include <ArduinoJson.h>

#define PH_PIN A0
#define TURBIDITY_PIN A1
#define DO_PIN A2
#define CONTAMINANT_PIN A3

void setup() {
  Serial.begin(9600);
}

void loop() {
  // Read raw analog values
  int phRaw = analogRead(PH_PIN);
  int turbidityRaw = analogRead(TURBIDITY_PIN);
  int doRaw = analogRead(DO_PIN);
  int contaminantRaw = analogRead(CONTAMINANT_PIN);

  // Convert to meaningful values (adjust based on your sensor specs)
  float pH = (phRaw / 1023.0) * 14.0;                   // scale 0–1023 to 0–14
  float turbidity = (turbidityRaw / 1023.0) * 100.0;    // %
  float dissolvedOxygen = (doRaw / 1023.0) * 14.0;      // mg/L
  float contaminants = (contaminantRaw / 1023.0) * 100; // ppm or %

  // Create JSON payload
  StaticJsonDocument<128> doc;
  doc["pH"] = pH;
  doc["turbidity"] = turbidity;
  doc["dissolved_oxygen"] = dissolvedOxygen;
  doc["contaminants"] = contaminants;

  // Send JSON via Serial
  serializeJson(doc, Serial);
  Serial.println();

  delay(1000); // Wait 1 second
}
