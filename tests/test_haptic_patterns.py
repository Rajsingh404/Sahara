"""The phone (HapticPatterns.kt) and the wristband (patterns.cpp) must play the same pattern for each sound."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware/esp32_wristband/src/patterns.cpp"
APP = ROOT / "app/android/app/src/main/kotlin/com/sahara/app/core/HapticPatterns.kt"
# firmware array name -> app label
NAMES = {"SmokeAlarm": "smoke_alarm", "Doorbell": "doorbell", "Siren": "siren", "Knocking": "knocking",
         "DogBark": "dog_bark", "BabyCry": "baby_cry", "GlassBreak": "glass_break",
         "ApplianceBeep": "appliance_beep", "NameCall": "Detection.NAME_CALL"}


def firmware_patterns():
    src = FIRMWARE.read_text()
    steps = {m[1]: [(int(a), int(b)) for a, b in re.findall(r"\{(\d+),\s*(\d+)\}", m[2])]
             for m in re.finditer(r"static const PatternStep k(\w+)\[\] = \{(.*?)\};", src, re.S)}
    reps = {m[1]: (int(m[2]), int(m[3]))
            for m in re.finditer(r'\{"\w+",\s*k(\w+),\s*COUNT\(k\w+\),\s*(\d+),\s*(\d+)\}', src)}
    return {name: (steps[name], *reps[name]) for name in NAMES}


def app_patterns():
    src = APP.read_text()
    out = {}
    for m in re.finditer(r'^\s*("?[\w.]+"?) to p\((\d+), (\d+), (.*?)\),$', src, re.M):
        steps = [(int(a), int(b)) for a, b in re.findall(r"(\d+) to (\d+)", m[4])]
        out[m[1].strip('"')] = (steps, int(m[2]), int(m[3]))
    return out


def test_phone_and_band_patterns_match():
    fw, app = firmware_patterns(), app_patterns()
    assert len(app) == len(NAMES)
    for name, label in NAMES.items():
        assert app[label] == fw[name], label
