"""A per-provider self-throttle: refuse a request locally before the provider ever has to rate-limit it.

Not a precise sliding-window limiter — a burst can land either side of a window boundary. That is fine here;
its only job is to stay comfortably under a provider's published per-minute limit, with headroom to spare, so
the engine switches to the fallback provider instead of finding out about the limit from a 429.
"""

import time
from collections.abc import Callable


class TokenBucket:
    def __init__(self, max_per_minute: int, *, clock: Callable[[], float] = time.monotonic) -> None:
        """`max_per_minute <= 0` disables the limit: every request is allowed."""
        self._max = max_per_minute
        self._clock = clock
        self._window_start = clock()
        self._count = 0

    def try_acquire(self) -> bool:
        """Count one request against the current minute; False if that would exceed the budget."""
        if self._max <= 0:
            return True
        now = self._clock()
        if now - self._window_start >= 60.0:
            self._window_start = now
            self._count = 0
        if self._count >= self._max:
            return False
        self._count += 1
        return True
