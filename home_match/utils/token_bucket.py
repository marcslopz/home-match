"""Token Bucket implementation for rate limiting."""

import threading
import time
from typing import Optional


class TokenBucket:
    """
    Thread-safe token bucket implementation for rate limiting.

    The bucket starts at full capacity and refills at a constant rate.
    Tokens are consumed for each request, and the bucket blocks until
    sufficient tokens are available.
    """

    def __init__(self, capacity: float, refill_rate: float):
        """
        Initialize the token bucket.

        Args:
            capacity: Maximum number of tokens the bucket can hold
            refill_rate: Number of tokens added per second
        """
        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = capacity
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.time()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self._refill_rate

        self._tokens = min(self._capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    def available_tokens(self) -> float:
        """
        Get the current number of available tokens.

        Returns:
            Number of tokens currently available
        """
        with self._lock:
            self._refill()
            return self._tokens

    def try_consume(self, tokens: float) -> bool:
        """
        Try to consume tokens without blocking.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens available
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def consume(self, tokens: float, timeout: Optional[float] = None) -> bool:
        """
        Consume tokens, blocking until they're available or timeout occurs.

        Args:
            tokens: Number of tokens to consume
            timeout: Maximum time to wait in seconds (None = wait indefinitely)

        Returns:
            True if tokens were consumed, False if timeout occurred
        """
        if tokens == 0:
            return True

        start_time = time.time()

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True

                # Check timeout
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed >= timeout:
                        return False

                # Calculate how long to wait for tokens
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._refill_rate

                # Cap wait time at remaining timeout
                if timeout is not None:
                    remaining = timeout - (time.time() - start_time)
                    wait_time = min(wait_time, remaining)

            # Sleep outside the lock to allow other threads to proceed
            if wait_time > 0:
                time.sleep(min(wait_time, 0.1))  # Sleep in small increments
