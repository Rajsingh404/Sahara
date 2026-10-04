// Haptic patterns: one distinct rhythm per sound class, plus device cues.
#pragma once

#include <stdint.h>

struct PatternStep {
  uint8_t strength;   // 0 = pause, 1..255 = motor strength (LED is on while > 0).
  uint16_t ms;
};

struct Pattern {
  const char* name;
  const PatternStep* steps;
  uint8_t stepCount;
  uint8_t cycles;     // How many times the steps play.
  uint16_t gapMs;     // Pause between cycles.
};

// Index by SoundClass.
const Pattern& patternForClass(uint8_t classId);

extern const Pattern PATTERN_TEST;
extern const Pattern PATTERN_LINK_LOST;
extern const Pattern PATTERN_CONNECTED;
extern const Pattern PATTERN_LOW_BATTERY;
extern const Pattern PATTERN_CRITICAL_BATTERY;
