// Sound class ids sent by the app. The order must match SOUND_CLASSES in
// src/config.py; scripts/check_classes.py fails the build if it does not.
#pragma once

#include <stdint.h>

enum SoundClass : uint8_t {
  SOUND_SMOKE_ALARM    = 0,
  SOUND_DOORBELL       = 1,
  SOUND_SIREN          = 2,
  SOUND_KNOCKING       = 3,
  SOUND_DOG_BARK       = 4,
  SOUND_BABY_CRY       = 5,
  SOUND_GLASS_BREAK    = 6,
  SOUND_APPLIANCE_BEEP = 7,
  SOUND_CLASS_COUNT    = 8,
};
