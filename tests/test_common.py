import threading
import time

from src.data.common import TIMED_OUT, imap_with_timeout


def test_imap_with_timeout_keeps_order_and_skips_hung_items():
    hang = threading.Event()

    def fn(x):
        if x == 3:
            hang.wait()  # never set: simulates a stuck Drive read
        time.sleep(0.01)
        return x * 10

    start = time.time()
    out = list(imap_with_timeout(fn, range(8), workers=2, timeout=0.5))
    assert [i for i, _ in out] == list(range(8))
    assert out[3][1] is TIMED_OUT
    assert [r for i, r in out if i != 3] == [0, 10, 20, 40, 50, 60, 70]
    assert time.time() - start < 5


def test_imap_with_timeout_returns_exceptions():
    def fn(x):
        raise ValueError(x)

    (item, result), = list(imap_with_timeout(fn, [1], workers=1, timeout=1))
    assert item == 1 and isinstance(result, ValueError)
