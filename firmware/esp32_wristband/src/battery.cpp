#include "battery.h"

#include <Arduino.h>

#include "config.h"

namespace battery {
namespace {

struct CurvePoint {
  uint16_t mv;
  uint8_t pct;
};

// Typical single-cell LiPo resting-voltage curve.
const CurvePoint kCurve[] = {
  {4200, 100}, {4150, 95}, {4110, 90}, {4020, 80}, {3950, 70}, {3870, 60}, {3830, 50},
  {3790, 40},  {3750, 30}, {3700, 20}, {3600, 10}, {3500, 5},  {3300, 0},
};

uint16_t gMv = 0;
uint8_t gPct = 100;
PowerState gState = POWER_NORMAL;
uint8_t gCriticalReadings = 0;

uint8_t toPercent(uint16_t mv) {
  const size_t n = sizeof(kCurve) / sizeof(kCurve[0]);
  if (mv >= kCurve[0].mv) return 100;
  for (size_t i = 1; i < n; i++) {
    if (mv >= kCurve[i].mv) {
      const CurvePoint& hi = kCurve[i - 1];
      const CurvePoint& lo = kCurve[i];
      return lo.pct + (uint32_t)(mv - lo.mv) * (hi.pct - lo.pct) / (hi.mv - lo.mv);
    }
  }
  return 0;
}

}  // namespace

void begin() {
#if CHARGE_DETECT_FITTED
  pinMode(PIN_CHARGE, INPUT_PULLUP);
#endif
  analogSetPinAttenuation(PIN_BATTERY_ADC, ADC_11db);
  sample();
  sample();  // Twice, so a flat battery is recognised at boot.
}

void sample() {
#if !BATTERY_DIVIDER_FITTED
  gMv = 0;
  gPct = 100;
  gState = POWER_NO_SENSOR;
  return;
#endif
  uint32_t sum = 0;
  for (int i = 0; i < 16; i++) sum += analogReadMilliVolts(PIN_BATTERY_ADC);
  gMv = (uint16_t)(sum / 16 * BATTERY_DIVIDER_RATIO);

  if (gMv < BATTERY_NO_SENSOR_MV) {
    gPct = 100;
    gState = POWER_NO_SENSOR;
    return;
  }
  gPct = toPercent(gMv);
  // Two critical readings in a row before sleeping, so one noisy sample cannot.
  gCriticalReadings = gMv <= BATTERY_CRITICAL_MV ? gCriticalReadings + 1 : 0;
  if (gCriticalReadings >= 2) {
    gState = POWER_CRITICAL;
  } else if (gPct <= BATTERY_LOW_PCT || gCriticalReadings > 0) {
    gState = POWER_LOW;
  } else if (gState == POWER_LOW && gPct <= BATTERY_LOW_EXIT_PCT) {
    // Stay low until clearly recovered (charging), so the mode does not flap.
  } else {
    gState = POWER_NORMAL;
  }
}

uint16_t millivolts() { return gMv; }
uint8_t percent() { return gPct; }
PowerState state() { return gState; }

bool charging() {
#if CHARGE_DETECT_FITTED
  return digitalRead(PIN_CHARGE) == LOW;
#else
  return false;
#endif
}

}  // namespace battery
