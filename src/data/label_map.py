"""Single source of truth for SAHARA class labels across dataset sources."""
from __future__ import annotations

from src.config import SOUND_CLASSES

# Negative / non-target audio (DESED soundscapes, indian_ambient/background/, etc.).
# FSD50K has no dedicated background class; we leave the hook empty for now.
BACKGROUND_LABEL = "background"

CLASS_LABEL_MAP: dict[str, dict[str, list[str]]] = {
    "smoke_alarm": {
        "fsd50k": ["Smoke_detector_smoke_alarm"],
        "audioset": ["/m/01y3hg"],
        "desed": ["Smoke_detector_smoke_alarm"],
        "indian_ambient": ["smoke_alarm"],
        "synthetic_mixed": ["smoke_alarm"],
    },
    "doorbell": {
        "fsd50k": ["Doorbell"],
        "audioset": ["/m/03wwcy"],
        "desed": ["Doorbell"],
        "indian_ambient": ["doorbell"],
        "synthetic_mixed": ["doorbell"],
    },
    "siren": {
        "fsd50k": ["Siren"],
        "audioset": ["/m/03kmc9"],
        "desed": ["Siren"],
        "indian_ambient": ["siren"],
        "synthetic_mixed": ["siren"],
    },
    "knocking": {
        "fsd50k": ["Knock"],
        "audioset": ["/m/07qnq_y"],  # Knock (best AudioSet mid match)
        "desed": ["Knock"],
        "indian_ambient": ["knocking"],
        "synthetic_mixed": ["knocking"],
    },
    "dog_bark": {
        "fsd50k": ["Bark"],
        "audioset": ["/m/05tny_"],
        "desed": ["Dog", "Bark"],
        "indian_ambient": ["dog_bark"],
        "synthetic_mixed": ["dog_bark"],
    },
    "baby_cry": {
        "fsd50k": ["Baby_cry_infant_cry"],
        "audioset": ["/t/dd00002"],
        "desed": ["Baby_cry_infant_cry"],
        "indian_ambient": ["baby_cry"],
        "synthetic_mixed": ["baby_cry"],
    },
    "glass_break": {
        "fsd50k": ["Glass"],
        "audioset": ["/m/07rwj3x"],
        "desed": ["Glass"],
        "indian_ambient": ["glass_break"],
        "synthetic_mixed": ["glass_break"],
    },
    "appliance_beep": {
        "fsd50k": ["Beep_buzzer", "Alarm", "Microwave_oven"],
        "audioset": ["/m/03cl9h", "/m/029bxz"],  # Beep, bleep; Microwave oven
        "desed": ["Beep_buzzer", "Microwave_oven"],
        "indian_ambient": ["appliance_beep"],
        "synthetic_mixed": ["appliance_beep"],
    },
}

# AudioSet display-name fallbacks used when resolving mids from class_labels_indices.csv.
AUDISET_DISPLAY_NAMES: dict[str, set[str]] = {
    "smoke_alarm": {"Smoke detector, smoke alarm"},
    "doorbell": {"Doorbell"},
    "siren": {"Siren"},
    "knocking": {"Knock"},
    "dog_bark": {"Bark", "Dog"},
    "baby_cry": {"Baby cry, infant cry"},
    "glass_break": {"Glass", "Breaking", "Shatter"},
    "appliance_beep": {"Beep, bleep", "Microwave oven", "Alarm clock"},
}


def fsd50k_labels_for_class(class_name: str) -> set[str]:
    return set(CLASS_LABEL_MAP[class_name]["fsd50k"])


def audioset_mids_for_class(class_name: str) -> set[str]:
    return set(CLASS_LABEL_MAP[class_name]["audioset"])


def all_target_classes() -> list[str]:
    return list(SOUND_CLASSES)


def is_background_label(label: str) -> bool:
    return label == BACKGROUND_LABEL

