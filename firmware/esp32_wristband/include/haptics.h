// Non-blocking player that drives the vibration motor and the LED from a Pattern.
#pragma once

#include "patterns.h"

namespace haptics {

void begin();
// Starts `p` with every step scaled by `intensity` (0..255).
// With `interrupt` false, it plays only if nothing else is playing.
// Returns true if the pattern started.
bool play(const Pattern& p, uint8_t intensity = 255, bool interrupt = true);
void stop();
// Call from loop(). Advances the current pattern.
void update();
bool isPlaying();
// Percent of normal strength (1..100). Used while the battery is low.
void setStrengthScale(uint8_t percent);
void setLedEnabled(bool enabled);
// Drives the LED directly when no pattern is playing (idle blink).
void setIdleLed(bool on);

}  // namespace haptics
