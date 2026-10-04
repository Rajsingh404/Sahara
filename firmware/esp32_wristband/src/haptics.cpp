#include "haptics.h"

#include <Arduino.h>

#include "config.h"

namespace haptics {
namespace {

const Pattern* gPattern = nullptr;
uint8_t gIntensity = 255;
uint8_t gStep = 0;
uint8_t gCycle = 0;
bool gInGap = false;
uint32_t gStepStart = 0;
uint8_t gScale = 100;
bool gLedEnabled = true;

void writeLed(bool on) {
  digitalWrite(PIN_LED, (on == (LED_ACTIVE_HIGH != 0)) ? HIGH : LOW);
}

void writeMotor(uint8_t strength) {
  uint32_t duty = 0;
  if (strength > 0 && gIntensity > 0) {
    duty = (uint32_t)strength * gIntensity / 255 * MOTOR_MAX_DUTY / 255 * gScale / 100;
    if (duty < MOTOR_MIN_DUTY) duty = MOTOR_MIN_DUTY;
  }
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(PIN_MOTOR, duty);
#else
  ledcWrite(0, duty);
#endif
}

void output(uint8_t strength) {
  writeMotor(strength);
  writeLed(gLedEnabled && strength > 0);
}

void applyStep() {
  output(gInGap ? 0 : gPattern->steps[gStep].strength);
  gStepStart = millis();
}

}  // namespace

void begin() {
  pinMode(PIN_LED, OUTPUT);
  writeLed(false);
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(PIN_MOTOR, MOTOR_PWM_FREQ_HZ, MOTOR_PWM_BITS);
#else
  ledcSetup(0, MOTOR_PWM_FREQ_HZ, MOTOR_PWM_BITS);
  ledcAttachPin(PIN_MOTOR, 0);
#endif
  writeMotor(0);
}

bool play(const Pattern& p, uint8_t intensity, bool interrupt) {
  if (gPattern != nullptr && !interrupt) return false;
  gPattern = &p;
  gIntensity = intensity;
  gStep = 0;
  gCycle = 0;
  gInGap = false;
  applyStep();
  return true;
}

void stop() {
  gPattern = nullptr;
  output(0);
}

void update() {
  if (gPattern == nullptr) return;
  const uint32_t now = millis();
  const uint16_t dur = gInGap ? gPattern->gapMs : gPattern->steps[gStep].ms;
  if (now - gStepStart < dur) return;

  if (gInGap) {
    gInGap = false;
    gStep = 0;
    applyStep();
    return;
  }
  if (++gStep < gPattern->stepCount) {
    applyStep();
    return;
  }
  // End of one cycle.
  if (++gCycle >= gPattern->cycles) {
    stop();
    return;
  }
  gStep = 0;
  gInGap = gPattern->gapMs > 0;
  applyStep();
}

bool isPlaying() { return gPattern != nullptr; }

void setStrengthScale(uint8_t percent) {
  gScale = percent == 0 ? 1 : (percent > 100 ? 100 : percent);
}

void setLedEnabled(bool enabled) { gLedEnabled = enabled; }

void setIdleLed(bool on) {
  if (gPattern == nullptr) writeLed(on && gLedEnabled);
}

}  // namespace haptics
