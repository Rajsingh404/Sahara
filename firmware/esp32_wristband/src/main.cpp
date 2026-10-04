// SAHARA wristband: BLE peripheral that turns sound alerts from the phone
// into a haptic pattern and LED flash per sound class.
// Protocol: docs/ble_protocol.md.
//
// BLE callbacks run on the NimBLE host task. They only queue events; all
// motor, LED and power work happens in loop().

#include <Arduino.h>
#include <NimBLEDevice.h>
#include <driver/rtc_io.h>
#include <esp_sleep.h>

#include "battery.h"
#include "ble_uuids.h"
#include "config.h"
#include "haptics.h"
#include "patterns.h"
#include "sound_classes.h"

namespace {

enum EventType : uint8_t { EV_CONNECTED, EV_DISCONNECTED, EV_ALERT };

struct Event {
  EventType type;
  uint8_t len;
  uint8_t data[ALERT_LEN];
};

QueueHandle_t gEvents;
NimBLEServer* gServer = nullptr;
NimBLECharacteristic* gStatusChar = nullptr;

bool gConnected = false;
uint32_t gAdvStartMs = 0;
bool gAdvSlow = false;
uint32_t gLastBatterySampleMs = 0;
uint32_t gLastBlinkMs = 0;
PowerState gPowerState = POWER_NORMAL;

uint8_t gLastStatus[STATUS_LEN] = {0};

bool gButtonDown = false;
bool gButtonRaw = false;
uint32_t gButtonChangeMs = 0;

// ---------------------------------------------------------------- BLE callbacks

void pushEvent(EventType type, const uint8_t* data = nullptr, size_t len = 0) {
  Event ev{type, 0, {0}};
  ev.len = len > sizeof(ev.data) ? sizeof(ev.data) : len;
  if (data != nullptr) memcpy(ev.data, data, ev.len);
  xQueueSend(gEvents, &ev, 0);
}

class ServerCallbacks : public NimBLEServerCallbacks {
  void onConnect(NimBLEServer* server, NimBLEConnInfo& info) override {
    // 30-50 ms interval, skip up to 2 events: worst-case alert delay ~150 ms.
    server->updateConnParams(info.getConnHandle(), 24, 40, 2, 400);
    pushEvent(EV_CONNECTED);
  }
  void onDisconnect(NimBLEServer*, NimBLEConnInfo&, int) override { pushEvent(EV_DISCONNECTED); }
};

class AlertCallbacks : public NimBLECharacteristicCallbacks {
  void onWrite(NimBLECharacteristic* c, NimBLEConnInfo&) override {
    const NimBLEAttValue v = c->getValue();
    pushEvent(EV_ALERT, v.data(), v.size());
  }
};

// ---------------------------------------------------------------- Status

// Notifies only when a field changed (battery %, charging, motor busy), as the protocol asks.
void publishStatus(bool force = false) {
  uint8_t flags = 0;
  if (battery::charging()) flags |= STATUS_FLAG_CHARGING;
  if (haptics::isPlaying()) flags |= STATUS_FLAG_MOTOR_BUSY;
  const uint8_t pct =
      battery::state() == POWER_NO_SENSOR ? STATUS_BATTERY_UNKNOWN : battery::percent();
  const uint8_t status[STATUS_LEN] = {SAHARA_PROTOCOL_VERSION, pct, flags, FIRMWARE_MINOR};

  if (!force && memcmp(status, gLastStatus, STATUS_LEN) == 0) return;
  memcpy(gLastStatus, status, STATUS_LEN);
  gStatusChar->setValue(status, STATUS_LEN);
  if (gConnected) gStatusChar->notify();
}

// ---------------------------------------------------------------- Advertising and power

void startAdvertising(bool slow) {
  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->stop();
  uint32_t ms = ADV_FAST_INTERVAL_MS;
  if (slow) ms = gPowerState == POWER_LOW ? ADV_SLOW_LOW_BATT_MS : ADV_SLOW_INTERVAL_MS;
  const uint16_t units = (uint16_t)(ms * 1000 / 625);  // 0.625 ms units.
  adv->setMinInterval(units);
  adv->setMaxInterval(units + units / 10);
  adv->start();
  gAdvSlow = slow;
  if (!slow) gAdvStartMs = millis();
  Serial.printf("[ble] advertising every %lu ms\n", (unsigned long)ms);
}

void applyPowerProfile() {
  const bool low = gPowerState == POWER_LOW;
  haptics::setStrengthScale(low ? MOTOR_LOW_BATT_SCALE : 100);
  haptics::setLedEnabled(!low);
  NimBLEDevice::setPower(low ? BLE_TX_POWER_LOW_DBM : BLE_TX_POWER_DBM);
  if (!gConnected && gAdvSlow) startAdvertising(true);
}

void waitForButtonRelease() {
  const uint32_t start = millis();
  while (digitalRead(PIN_BUTTON) == LOW && millis() - start < 5000) delay(10);
}

[[noreturn]] void enterDeepSleep() {
  Serial.println("[power] battery critical, deep sleep until button press");
  Serial.flush();
  haptics::stop();
  waitForButtonRelease();
  rtc_gpio_pullup_en((gpio_num_t)PIN_BUTTON);
  esp_sleep_enable_ext0_wakeup((gpio_num_t)PIN_BUTTON, 0);
  esp_deep_sleep_start();
}

bool shouldSleep() { return gPowerState == POWER_CRITICAL && !battery::charging(); }

// Woken by the button with a still-flat battery: flash once and go back to sleep.
void checkCriticalOnWake() {
  if (esp_sleep_get_wakeup_cause() != ESP_SLEEP_WAKEUP_EXT0 || !shouldSleep()) return;
  haptics::setIdleLed(true);
  delay(150);
  haptics::setIdleLed(false);
  enterDeepSleep();
}

void sampleBattery() {
  battery::sample();
  gLastBatterySampleMs = millis();
  const PowerState next = battery::state();
  if (next != gPowerState) {
    Serial.printf("[power] state %u -> %u (%u mV, %u%%)\n", gPowerState, next,
                  battery::millivolts(), battery::percent());
    gPowerState = next;
    applyPowerProfile();
    if (next == POWER_LOW) haptics::play(PATTERN_LOW_BATTERY, 255, false);
    if (next == POWER_CRITICAL) haptics::play(PATTERN_CRITICAL_BATTERY);
  }
  publishStatus();
}

// ---------------------------------------------------------------- Events

void handleAlert(const Event& ev) {
  // [version, command, class_id, intensity]
  if (ev.len < ALERT_LEN || ev.data[0] != SAHARA_PROTOCOL_VERSION) {
    Serial.println("[alert] ignored: bad length or version");
    return;
  }
  const uint8_t cmd = ev.data[1];
  const uint8_t classId = ev.data[2];
  const uint8_t intensity = ev.data[3];

  switch (cmd) {
    case CMD_ALERT:
      if (classId >= SOUND_CLASS_COUNT) {
        Serial.printf("[alert] ignored unknown class %u\n", classId);
        return;
      }
      haptics::play(patternForClass(classId), intensity);
      Serial.printf("[alert] %s at %u\n", patternForClass(classId).name, intensity);
      break;
    case CMD_TEST:
      haptics::play(PATTERN_TEST, intensity);
      break;
    case CMD_STOP:
      haptics::stop();
      break;
    default:
      Serial.printf("[alert] ignored unknown command 0x%02X\n", cmd);
  }
}

void handleEvent(const Event& ev) {
  switch (ev.type) {
    case EV_CONNECTED:
      gConnected = true;
      Serial.println("[ble] connected");
      haptics::setIdleLed(false);
      haptics::play(PATTERN_CONNECTED, 255, false);
      publishStatus(true);
      break;
    case EV_DISCONNECTED:
      gConnected = false;
      Serial.println("[ble] disconnected");
      // The user cannot hear the phone, so tell them alerts have stopped.
      haptics::play(PATTERN_LINK_LOST, 255, false);
      startAdvertising(false);
      break;
    case EV_ALERT:
      handleAlert(ev);
      break;
  }
}

// ---------------------------------------------------------------- Button

void pollButton() {
  const bool raw = digitalRead(PIN_BUTTON) == LOW;
  const uint32_t now = millis();
  if (raw != gButtonRaw) {
    gButtonRaw = raw;
    gButtonChangeMs = now;
  }
  if (raw == gButtonDown || now - gButtonChangeMs < BUTTON_DEBOUNCE_MS) return;
  gButtonDown = raw;
  if (!gButtonDown) return;
  // Press: dismiss the current alert, or give a short buzz to show the band is alive.
  if (haptics::isPlaying()) {
    haptics::stop();
  } else {
    haptics::play(PATTERN_TEST, 200);
  }
}

void setupBle() {
  NimBLEDevice::init(SAHARA_DEVICE_NAME);
  NimBLEDevice::setPower(BLE_TX_POWER_DBM);

  gServer = NimBLEDevice::createServer();
  gServer->setCallbacks(new ServerCallbacks());
  gServer->advertiseOnDisconnect(false);  // startAdvertising() picks the interval.

  NimBLEService* svc = gServer->createService(SAHARA_SERVICE_UUID);
  NimBLECharacteristic* alert = svc->createCharacteristic(
      SAHARA_ALERT_CHAR_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::WRITE_NR);
  alert->setCallbacks(new AlertCallbacks());
  gStatusChar = svc->createCharacteristic(SAHARA_STATUS_CHAR_UUID,
                                          NIMBLE_PROPERTY::READ | NIMBLE_PROPERTY::NOTIFY);
  gServer->start();

  NimBLEAdvertising* adv = NimBLEDevice::getAdvertising();
  adv->setName(SAHARA_DEVICE_NAME);
  adv->addServiceUUID(SAHARA_SERVICE_UUID);
  adv->enableScanResponse(true);

  Serial.printf("[ble] %s ready at %s\n", SAHARA_DEVICE_NAME,
                NimBLEDevice::getAddress().toString().c_str());
}

}  // namespace

void setup() {
  setCpuFrequencyMhz(CPU_FREQ_MHZ);
  Serial.begin(115200);
  pinMode(PIN_BUTTON, INPUT_PULLUP);

  gEvents = xQueueCreate(8, sizeof(Event));
  haptics::begin();
  battery::begin();
  gPowerState = battery::state();
  checkCriticalOnWake();

  setupBle();
  applyPowerProfile();
  publishStatus(true);
  startAdvertising(false);

  haptics::play(PATTERN_TEST, 200);  // Boot buzz.
  if (gPowerState == POWER_CRITICAL) haptics::play(PATTERN_CRITICAL_BATTERY);
  Serial.printf("[boot] SAHARA wristband fw 1.%d, battery %u mV (%u%%)\n", FIRMWARE_MINOR,
                battery::millivolts(), battery::percent());
}

void loop() {
  Event ev;
  while (xQueueReceive(gEvents, &ev, 0) == pdTRUE) handleEvent(ev);

  pollButton();
  haptics::update();
  publishStatus();  // Sends only on change (motor busy, charging).

  const uint32_t now = millis();

  // Read the battery only while the motor is off: motor current sags the cell.
  if (!haptics::isPlaying() && now - gLastBatterySampleMs >= BATTERY_SAMPLE_PERIOD_MS) {
    sampleBattery();
  }

  // Sleep once the critical-battery warning has finished.
  if (shouldSleep() && !haptics::isPlaying()) enterDeepSleep();

  if (!gConnected) {
    if (!gAdvSlow && now - gAdvStartMs >= ADV_FAST_WINDOW_MS) startAdvertising(true);
    if (!haptics::isPlaying()) {
      const uint32_t sinceBlink = now - gLastBlinkMs;
      if (sinceBlink >= IDLE_BLINK_PERIOD_MS) {
        gLastBlinkMs = now;
        haptics::setIdleLed(true);
      } else if (sinceBlink >= IDLE_BLINK_MS) {
        haptics::setIdleLed(false);
      }
    }
  }

  delay(5);
}
