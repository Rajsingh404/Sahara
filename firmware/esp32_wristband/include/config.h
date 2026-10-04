// Hardware pins and tunables for the SAHARA wristband.
// Every board-specific value lives here. Change this file, not the code,
// when moving to a different ESP32 board.
#pragma once

#include <stdint.h>

// ---------------------------------------------------------------- Pins
// Defaults are for a generic ESP32 DevKit V1 (ESP32-WROOM-32).
#define PIN_MOTOR        25   // Gate of the motor MOSFET (PWM).
#define PIN_LED          2    // On-board blue LED on most DevKits; active high.
#define PIN_BUTTON       0    // BOOT button; active low. Must be an RTC GPIO (wakes from deep sleep).
#define PIN_BATTERY_ADC  34   // ADC1 input from the battery voltage divider.
#define PIN_CHARGE       27   // TP4056 CHRG through a diode; low while charging.

#define LED_ACTIVE_HIGH  1

// ---------------------------------------------------------------- Motor
#define MOTOR_PWM_FREQ_HZ    20000  // Above hearing range, so no whine.
#define MOTOR_PWM_BITS       8
#define MOTOR_MAX_DUTY       230    // Caps drive for a 3 V coin motor on a 4.2 V cell.
#define MOTOR_MIN_DUTY       90     // Below this a coin motor stalls; non-zero steps are raised to this.
#define MOTOR_LOW_BATT_SCALE 70     // Percent of normal strength while the battery is low.

// ---------------------------------------------------------------- Battery
// Set to 0 on a bare DevKit with nothing wired to PIN_BATTERY_ADC: a floating
// ADC pin reads noise and can trip the low/critical battery logic.
#define BATTERY_DIVIDER_FITTED    1
// Set to 1 once the TP4056 CHRG pin is wired to PIN_CHARGE (see README).
#define CHARGE_DETECT_FITTED      0
// Divider: VBAT -- 100k -- ADC pin -- 100k -- GND, so the pin sees half the cell voltage.
#define BATTERY_DIVIDER_RATIO     2.0f
#define BATTERY_NO_SENSOR_MV      2500   // Below this, assume no battery is wired (USB bench power).
#define BATTERY_LOW_PCT           20     // Enter low-power mode at or below this.
#define BATTERY_LOW_EXIT_PCT      25     // Hysteresis: leave low-power mode above this.
#define BATTERY_CRITICAL_MV       3400   // Warn, then deep sleep at or below this.
#define BATTERY_SAMPLE_PERIOD_MS  30000

// ---------------------------------------------------------------- BLE / power
#define BLE_TX_POWER_DBM          3      // Phone is usually on the body; full power is not needed.
#define BLE_TX_POWER_LOW_DBM      0
#define ADV_FAST_INTERVAL_MS      100    // Right after boot or a disconnect.
#define ADV_SLOW_INTERVAL_MS      1000   // After ADV_FAST_WINDOW_MS without a connection.
#define ADV_SLOW_LOW_BATT_MS      2000
#define ADV_FAST_WINDOW_MS        30000
#define CPU_FREQ_MHZ              80     // Lowest frequency the radio supports.

#define IDLE_BLINK_PERIOD_MS      5000   // Short LED blink while not connected to the phone.
#define IDLE_BLINK_MS             20

#define BUTTON_DEBOUNCE_MS        30

#define FIRMWARE_MINOR            1   // Reported in the status characteristic.
