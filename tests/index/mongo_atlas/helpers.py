import time
from typing import Callable


def assert_when_ready(callable: Callable, tries: int = 5, interval: float = 1):
    while True:
        try:
            callable()
        except AssertionError:
            tries -= 1
            if tries == 0:
                raise
            time.sleep(interval)
        else:
            return
