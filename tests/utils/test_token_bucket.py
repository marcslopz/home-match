"""Tests for TokenBucket implementation."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed


from home_match.utils.token_bucket import TokenBucket


def test_token_bucket_initialization():
    """Test that a token bucket starts at full capacity."""
    bucket = TokenBucket(capacity=100, refill_rate=10)
    assert bucket.available_tokens() == 100


def test_consume_tokens_success():
    """Test consuming tokens when sufficient tokens are available."""
    bucket = TokenBucket(capacity=100, refill_rate=10)
    assert bucket.consume(tokens=30, timeout=0.1)
    available = bucket.available_tokens()
    assert abs(available - 70) < 0.1  # Allow for timing imprecision


def test_consume_tokens_blocks_when_insufficient():
    """Test that consume blocks when insufficient tokens are available."""
    bucket = TokenBucket(capacity=100, refill_rate=10)  # 10 tokens/sec
    bucket.consume(tokens=95, timeout=1)

    start = time.time()
    # Try to consume 20 tokens (only 5 available, need to wait ~1.5s for 15 more)
    result = bucket.consume(tokens=20, timeout=2)
    elapsed = time.time() - start

    assert result is True
    assert elapsed >= 1.0  # Should have waited for tokens to refill


def test_token_replenishment_over_time():
    """Test that tokens are replenished over time."""
    bucket = TokenBucket(capacity=100, refill_rate=10)  # 10 tokens/sec
    bucket.consume(tokens=50, timeout=0.1)
    available = bucket.available_tokens()
    assert abs(available - 50) < 0.1  # Allow for timing imprecision

    # Wait 2 seconds, should get 20 tokens back
    time.sleep(2)
    available = bucket.available_tokens()
    assert available >= 69  # 50 + 20 (allowing for slight timing variance)
    assert available <= 100


def test_token_replenishment_capped_at_capacity():
    """Test that token replenishment doesn't exceed capacity."""
    bucket = TokenBucket(capacity=100, refill_rate=10)
    bucket.consume(tokens=20, timeout=0.1)

    # Wait long enough to refill beyond capacity
    time.sleep(3)

    # Should be capped at capacity
    assert bucket.available_tokens() == 100


def test_try_consume_non_blocking():
    """Test that try_consume doesn't block and returns False when insufficient."""
    bucket = TokenBucket(capacity=100, refill_rate=10)
    bucket.consume(tokens=95, timeout=0.1)

    # Should return False immediately without blocking
    start = time.time()
    result = bucket.try_consume(tokens=20)
    elapsed = time.time() - start

    assert result is False
    assert elapsed < 0.1  # Should be nearly instant


def test_try_consume_success():
    """Test that try_consume succeeds when tokens are available."""
    bucket = TokenBucket(capacity=100, refill_rate=10)
    result = bucket.try_consume(tokens=30)
    assert result is True
    available = bucket.available_tokens()
    assert abs(available - 70) < 0.1  # Allow for timing imprecision


def test_concurrent_consumption_thread_safe():
    """Test that TokenBucket is thread-safe with concurrent consumers."""
    bucket = TokenBucket(capacity=1000, refill_rate=100)

    def consume_tokens(amount):
        return bucket.consume(tokens=amount, timeout=2)

    # Launch 20 threads each consuming 50 tokens
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(consume_tokens, 50) for _ in range(20)]
        results = [f.result() for f in as_completed(futures)]

    # All should succeed (20 * 50 = 1000 tokens total)
    assert all(results)
    # Should be close to 0 after consuming 1000 tokens
    assert bucket.available_tokens() < 10


def test_fractional_tokens():
    """Test that the bucket handles fractional token amounts."""
    bucket = TokenBucket(capacity=100.0, refill_rate=10.5)
    assert bucket.consume(tokens=25.5, timeout=0.1)
    available = bucket.available_tokens()
    assert abs(available - 74.5) < 0.1


def test_consume_with_timeout_fails():
    """Test that consume returns False when timeout is exceeded."""
    bucket = TokenBucket(capacity=10, refill_rate=1)  # 1 token/sec
    bucket.consume(tokens=10, timeout=0.1)

    # Try to consume 20 tokens with short timeout (would need 20 seconds)
    start = time.time()
    result = bucket.consume(tokens=20, timeout=0.5)
    elapsed = time.time() - start

    assert result is False
    assert 0.4 <= elapsed <= 0.7  # Should timeout around 0.5s


def test_consume_zero_tokens():
    """Test consuming zero tokens."""
    bucket = TokenBucket(capacity=100, refill_rate=10)
    initial = bucket.available_tokens()
    assert bucket.consume(tokens=0, timeout=0.1)
    assert bucket.available_tokens() == initial
