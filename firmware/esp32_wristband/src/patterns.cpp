#include "patterns.h"

#include "sound_classes.h"

#define COUNT(a) (uint8_t)(sizeof(a) / sizeof((a)[0]))

// Smoke alarm: the "temporal three" rhythm real fire alarms use. Three long buzzes.
static const PatternStep kSmokeAlarm[] = {
  {255, 500}, {0, 500}, {255, 500}, {0, 500}, {255, 500},
};
// Doorbell: "ding-dong", short strong then longer softer.
static const PatternStep kDoorbell[] = {
  {255, 150}, {0, 120}, {160, 350},
};
// Siren: a swell that rises and falls.
static const PatternStep kSiren[] = {
  {100, 150}, {150, 150}, {200, 150}, {255, 300}, {200, 150}, {150, 150}, {100, 150},
};
// Knocking: four quick taps.
static const PatternStep kKnocking[] = {
  {255, 60}, {0, 100}, {255, 60}, {0, 100}, {255, 60}, {0, 100}, {255, 60},
};
// Dog bark: two sharp medium pulses.
static const PatternStep kDogBark[] = {
  {220, 120}, {0, 150}, {220, 120},
};
// Baby cry: a wavering, uneven pulse.
static const PatternStep kBabyCry[] = {
  {180, 400}, {0, 150}, {120, 250}, {0, 150}, {180, 400},
};
// Glass break: one hard hit, then a scatter of tiny flutters.
static const PatternStep kGlassBreak[] = {
  {255, 400}, {0, 120}, {200, 40}, {0, 40}, {200, 40}, {0, 40}, {200, 40}, {0, 40}, {200, 40},
};
// Appliance beep: three light, short ticks.
static const PatternStep kApplianceBeep[] = {
  {130, 80}, {0, 150}, {130, 80}, {0, 150}, {130, 80},
};

static const PatternStep kTest[] = {{255, 300}};  // Protocol: one 300 ms pulse.
static const PatternStep kLinkLost[] = {{150, 600}, {0, 300}, {150, 600}};
static const PatternStep kConnected[] = {{150, 80}, {0, 80}, {150, 80}};
static const PatternStep kLowBattery[] = {{110, 100}, {0, 150}, {110, 100}, {0, 150}, {110, 100}};
static const PatternStep kCriticalBattery[] = {{110, 1000}};

static const Pattern kClassPatterns[SOUND_CLASS_COUNT] = {
  /* SOUND_SMOKE_ALARM    */ {"smoke_alarm",    kSmokeAlarm,    COUNT(kSmokeAlarm),    3, 1500},
  /* SOUND_DOORBELL       */ {"doorbell",       kDoorbell,      COUNT(kDoorbell),      2,  700},
  /* SOUND_SIREN          */ {"siren",          kSiren,         COUNT(kSiren),         3,  300},
  /* SOUND_KNOCKING       */ {"knocking",       kKnocking,      COUNT(kKnocking),      2,  600},
  /* SOUND_DOG_BARK       */ {"dog_bark",       kDogBark,       COUNT(kDogBark),       2,  600},
  /* SOUND_BABY_CRY       */ {"baby_cry",       kBabyCry,       COUNT(kBabyCry),       2,  600},
  /* SOUND_GLASS_BREAK    */ {"glass_break",    kGlassBreak,    COUNT(kGlassBreak),    2,  800},
  /* SOUND_APPLIANCE_BEEP */ {"appliance_beep", kApplianceBeep, COUNT(kApplianceBeep), 1,    0},
};

const Pattern PATTERN_TEST             = {"test",             kTest,            COUNT(kTest),            1, 0};
const Pattern PATTERN_LINK_LOST        = {"link_lost",        kLinkLost,        COUNT(kLinkLost),        1, 0};
const Pattern PATTERN_CONNECTED        = {"connected",        kConnected,       COUNT(kConnected),       1, 0};
const Pattern PATTERN_LOW_BATTERY      = {"low_battery",      kLowBattery,      COUNT(kLowBattery),      1, 0};
const Pattern PATTERN_CRITICAL_BATTERY = {"critical_battery", kCriticalBattery, COUNT(kCriticalBattery), 1, 0};

const Pattern& patternForClass(uint8_t classId) {
  return kClassPatterns[classId < SOUND_CLASS_COUNT ? classId : 0];
}
