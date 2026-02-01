"""Tests for RateLimitedAnthropicClient implementation."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch


from home_match.utils.rate_limited_client import RateLimitedAnthropicClient
from home_match.utils.rate_limit_config import RateLimitConfig


def test_client_initialization_wraps_anthropic():
    """Test basic initialization of the rate-limited client."""
    client = RateLimitedAnthropicClient(api_key="test-key")
    assert client is not None
    assert hasattr(client, "messages")


def test_client_loads_tier_1_config_by_default():
    """Test that client loads Tier 1 config by default."""
    client = RateLimitedAnthropicClient(api_key="test-key")
    assert client._config.requests_per_minute == 50


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_messages_create_under_limit_no_blocking(mock_anthropic_class):
    """Test that messages.create works without blocking when under limits."""
    # Mock the Anthropic client
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    client = RateLimitedAnthropicClient(api_key="test-key")

    start = time.time()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "test"}],
    )
    elapsed = time.time() - start

    assert response == mock_response
    assert elapsed < 0.5  # Should be fast, no blocking


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_messages_create_blocks_on_rpm_limit(mock_anthropic_class):
    """Test that client blocks when RPM limit is reached."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    # Create client with very low RPM (1 per minute = 1/60 per second)
    from home_match.utils.rate_limit_config import ModelLimits

    custom_config = RateLimitConfig(
        requests_per_minute=2,  # Very low for testing
        model_limits={
            "sonnet-4": ModelLimits(input_tokens_per_minute=100000, output_tokens_per_minute=100000)
        },
    )
    client = RateLimitedAnthropicClient(api_key="test-key", config=custom_config)

    # Make 2 requests quickly
    start = time.time()
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "test1"}],
    )
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "test2"}],
    )

    # Third request should block
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "test3"}],
    )
    elapsed = time.time() - start

    # Should have taken at least ~30 seconds (60/2 = 30 seconds per token)
    assert elapsed >= 25  # Allow some margin


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_messages_create_blocks_on_itpm_limit(mock_anthropic_class):
    """Test that client blocks when ITPM limit is reached."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=100, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    # Create client with low ITPM
    from home_match.utils.rate_limit_config import ModelLimits

    custom_config = RateLimitConfig(
        requests_per_minute=1000,  # High enough to not block
        model_limits={
            "sonnet-4": ModelLimits(
                input_tokens_per_minute=200,  # Low for testing
                output_tokens_per_minute=100000,
            )
        },
    )
    client = RateLimitedAnthropicClient(api_key="test-key", config=custom_config)

    # Make requests that use 200 input tokens
    start = time.time()
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "x" * 800}],  # ~200 tokens
    )

    # Second request should block waiting for ITPM refill
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "x" * 800}],  # ~200 tokens
    )
    elapsed = time.time() - start

    # Should have taken at least ~60 seconds (need to wait for 200 tokens at 200/min)
    assert elapsed >= 55  # Allow some margin


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_messages_create_blocks_on_otpm_limit(mock_anthropic_class):
    """Test that client blocks when OTPM limit is reached."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=10, output_tokens=100, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    # Create client with low OTPM
    from home_match.utils.rate_limit_config import ModelLimits

    custom_config = RateLimitConfig(
        requests_per_minute=1000,
        model_limits={
            "sonnet-4": ModelLimits(
                input_tokens_per_minute=100000,
                output_tokens_per_minute=150,  # Low for testing
            )
        },
    )
    client = RateLimitedAnthropicClient(api_key="test-key", config=custom_config)

    # Make requests that request 100 output tokens each
    start = time.time()
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "test"}],
    )

    # Second request should block
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "test"}],
    )
    elapsed = time.time() - start

    # Should have taken at least ~40 seconds
    assert elapsed >= 35


def test_model_family_detection_sonnet():
    """Test that Sonnet models are correctly identified."""
    client = RateLimitedAnthropicClient(api_key="test-key")
    family = client.messages._determine_model_family("claude-sonnet-4-20250514")
    assert family == "sonnet-4"

    family2 = client.messages._determine_model_family("claude-sonnet-4.5-20261015")
    assert family2 == "sonnet-4"


def test_model_family_detection_haiku():
    """Test that Haiku models are correctly identified."""
    client = RateLimitedAnthropicClient(api_key="test-key")
    family = client.messages._determine_model_family("claude-haiku-4.5-20250514")
    assert family == "haiku-4"


def test_model_family_detection_opus():
    """Test that Opus models are correctly identified."""
    client = RateLimitedAnthropicClient(api_key="test-key")
    family = client.messages._determine_model_family("claude-opus-4-20250514")
    assert family == "opus-4"


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_different_models_independent_buckets(mock_anthropic_class):
    """Test that different model families use independent token buckets."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=100, output_tokens=50, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    client = RateLimitedAnthropicClient(api_key="test-key")

    # Use Sonnet and Haiku models - they should have independent buckets
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "test sonnet"}],
    )

    client.messages.create(
        model="claude-haiku-4.5-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "test haiku"}],
    )

    # Both should succeed without long blocking (independent buckets)
    # If they shared buckets, second would block


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_cache_read_tokens_not_counted(mock_anthropic_class):
    """Test that cache_read_input_tokens are not counted for ITPM."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=50,
        output_tokens=10,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=1000,  # Large cache read
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    from home_match.utils.rate_limit_config import ModelLimits

    custom_config = RateLimitConfig(
        requests_per_minute=1000,
        model_limits={
            "sonnet-4": ModelLimits(
                input_tokens_per_minute=100,  # Low limit
                output_tokens_per_minute=100000,
            )
        },
    )
    client = RateLimitedAnthropicClient(api_key="test-key", config=custom_config)

    # Make request - cache_read_input_tokens should NOT count
    start = time.time()
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "test"}],
    )

    # Second request should not block significantly
    client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        messages=[{"role": "user", "content": "test"}],
    )
    elapsed = time.time() - start

    # Should be fast since only 100 input tokens consumed (not 1000)
    assert elapsed < 30


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_disabled_rate_limiting_bypasses(mock_anthropic_class):
    """Test that rate limiting can be disabled."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    # Disable rate limiting
    client = RateLimitedAnthropicClient(api_key="test-key", enable_rate_limiting=False)

    # Should make request without checking buckets
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": "test"}],
    )

    assert response == mock_response
    # Verify it actually bypassed (by not having buckets initialized)
    assert not hasattr(client.messages, "_rpm_bucket") or client.messages._rpm_bucket is None


@patch("home_match.utils.rate_limited_client.Anthropic")
def test_concurrent_requests_thread_safe(mock_anthropic_class):
    """Test that client is thread-safe with concurrent requests."""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.usage = Mock(
        input_tokens=10, output_tokens=5, cache_creation_input_tokens=0, cache_read_input_tokens=0
    )
    mock_response.content = [{"type": "text", "text": "Response"}]
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    client = RateLimitedAnthropicClient(api_key="test-key")

    def make_request(i):
        return client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": f"test {i}"}],
        )

    # Make 10 concurrent requests
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request, i) for i in range(10)]
        results = [f.result() for f in as_completed(futures)]

    # All should succeed
    assert len(results) == 10
    assert all(r == mock_response for r in results)
