"""Home calibration maths (src/evaluation/home_calibration.py; mirrored in HomeCalibration.kt)."""
import numpy as np

from src.evaluation import policy_tuning as pt
from src.evaluation.home_calibration import calibrate, fire_level


def test_fire_level_is_the_highest_threshold_that_still_alerts():
    s = np.array([0.1, 0.7, 0.9, 0.6, 0.2], np.float32)
    for hits in (1, 2, 3):
        level = fire_level(s, hits)
        assert pt.count_alerts(s, level, hits) > 0
        assert pt.count_alerts(s, np.nextafter(level, 1, dtype=np.float32), hits) == 0
    assert fire_level(s, 1) == np.float32(0.9)
    assert fire_level(s, 2) == np.float32(0.7)
    assert fire_level(s, 3) == np.float32(0.6)


def test_calibrate_stays_between_floor_and_current():
    assert calibrate(0.95, 0.6, [0.9, 0.8]) < 0.8  # below the weakest recording
    assert calibrate(0.95, 0.79, [0.9, 0.8]) == 0.79  # but not below the floor
    assert calibrate(0.95, 0.6, [0.99]) == 0.95  # never raised above the current threshold
    assert calibrate(0.95, 0.6, []) == 0.95


def test_background_raises_the_threshold_again():
    t = calibrate(0.95, 0.5, [0.8], background_level=0.85)
    assert 0.85 < t <= 0.95
