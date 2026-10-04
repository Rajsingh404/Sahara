"""Single source of truth for SAHARA class labels across dataset sources."""
from __future__ import annotations

from src.config import SOUND_CLASSES

# Negative / non-target audio (DESED soundscapes, indian_ambient/background/, etc.).
# FSD50K has no dedicated background class; we leave the hook empty for now.
BACKGROUND_LABEL = "background"

# Native labels per source. Checked against FSD50K.ground_truth/vocabulary.csv and
# AudioSet class_labels_indices.csv (2026-10-04). FSD50K has no smoke-alarm, baby-cry or
# Beep_buzzer label; those classes come from AudioSet and ESC-50.
# Deliberately left out:
#   FSD50K "Alarm"  - mostly telephones, car alarms and ringtones, not appliance beeps.
#   FSD50K "Glass"  - mostly clinking; only clips also tagged "Shatter" are glass_break.
#   AudioSet "Dog"  - growls and whimpers; only "Bark" is dog_bark.
CLASS_LABEL_MAP: dict[str, dict[str, list[str]]] = {
    "smoke_alarm": {
        "fsd50k": [],
        "audioset": ["/m/01y3hg"],  # Smoke detector, smoke alarm
        "esc50": [],
        "indian_ambient": ["smoke_alarm"],
        "synthetic_mixed": ["smoke_alarm"],
    },
    "doorbell": {
        "fsd50k": ["Doorbell"],
        "audioset": ["/m/03wwcy"],  # Doorbell
        "esc50": [],
        "indian_ambient": ["doorbell"],
        "synthetic_mixed": ["doorbell"],
    },
    "siren": {
        "fsd50k": ["Siren"],
        "audioset": ["/m/03kmc9"],  # Siren
        "esc50": ["siren"],
        "indian_ambient": ["siren"],
        "synthetic_mixed": ["siren"],
    },
    "knocking": {
        "fsd50k": ["Knock"],
        "audioset": ["/m/07r4wb8"],  # Knock
        "esc50": ["door_wood_knock"],
        "indian_ambient": ["knocking"],
        "synthetic_mixed": ["knocking"],
    },
    "dog_bark": {
        "fsd50k": ["Bark"],
        "audioset": ["/m/05tny_"],  # Bark
        "esc50": ["dog"],
        "indian_ambient": ["dog_bark"],
        "synthetic_mixed": ["dog_bark"],
    },
    "baby_cry": {
        "fsd50k": [],
        "audioset": ["/t/dd00002"],  # Baby cry, infant cry
        "esc50": ["crying_baby"],
        "indian_ambient": ["baby_cry"],
        "synthetic_mixed": ["baby_cry"],
    },
    "glass_break": {
        "fsd50k": ["Shatter"],
        "audioset": ["/m/07rn7sz"],  # Shatter
        "esc50": ["glass_breaking"],
        "indian_ambient": ["glass_break"],
        "synthetic_mixed": ["glass_break"],
    },
    "appliance_beep": {
        "fsd50k": ["Microwave_oven"],
        "audioset": ["/m/02fs_r", "/m/0fx9l", "/m/046dlr"],  # Beep, bleep; Microwave oven; Alarm clock
        "esc50": ["clock_alarm"],
        "indian_ambient": ["appliance_beep"],
        "synthetic_mixed": ["appliance_beep"],
    },
}

# DESED clips carrying these events are neither clean background nor a clean target
# (Dog includes barking, Alarm_bell_ringing includes smoke alarms and beeps), so they are dropped.
DESED_EXCLUDED_EVENTS = {"Dog", "Alarm_bell_ringing"}


def fsd50k_labels_for_class(class_name: str) -> set[str]:
    return set(CLASS_LABEL_MAP[class_name]["fsd50k"])


def esc50_categories_for_class(class_name: str) -> set[str]:
    return set(CLASS_LABEL_MAP[class_name]["esc50"])


def audioset_mids_for_class(class_name: str) -> set[str]:
    return set(CLASS_LABEL_MAP[class_name]["audioset"])


def all_target_classes() -> list[str]:
    return list(SOUND_CLASSES)


def is_background_label(label: str) -> bool:
    return label == BACKGROUND_LABEL

