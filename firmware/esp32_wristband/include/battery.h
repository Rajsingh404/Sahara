// Battery voltage, charge estimate and power state.
#pragma once

#include <stdint.h>

enum PowerState : uint8_t {
  POWER_NORMAL    = 0,
  POWER_LOW       = 1,
  POWER_CRITICAL  = 2,
  POWER_NO_SENSOR = 3,  // No battery divider wired (e.g. USB bench power).
};

namespace battery {

void begin();
// Takes a new reading. Call only while the motor is off: motor current sags the cell.
void sample();
uint16_t millivolts();
uint8_t percent();
PowerState state();
// True while the charger reports charging (needs CHARGE_DETECT_FITTED).
bool charging();

}  // namespace battery
