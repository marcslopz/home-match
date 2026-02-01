"""Tests for TokenEstimator implementation."""



from home_match.utils.token_estimator import TokenEstimator


def test_estimate_input_tokens_simple_message():
    """Test basic token estimation for a simple message."""
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    estimated = TokenEstimator.estimate_input_tokens(messages)

    # "Hello, how are you?" is ~20 chars, so ~5 tokens (chars/4)
    # Allow some overhead for message structure
    assert 3 <= estimated <= 15


def test_estimate_input_tokens_multiple_messages():
    """Test token estimation with multiple messages."""
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And what is 3+3?"},
    ]

    estimated = TokenEstimator.estimate_input_tokens(messages)

    # Total ~40 chars across all messages = ~10 tokens
    # Allow overhead for structure
    assert 8 <= estimated <= 25


def test_estimate_input_tokens_with_system_prompt():
    """Test token estimation including system prompt."""
    messages = [{"role": "user", "content": "Hello!"}]
    system = "You are a helpful assistant that responds concisely."

    estimated = TokenEstimator.estimate_input_tokens(messages, system=system)

    # System prompt (~60 chars = ~15 tokens) + message (~6 chars = ~1-2 tokens)
    assert 12 <= estimated <= 30


def test_estimate_input_tokens_empty():
    """Test token estimation with empty messages."""
    messages = []
    estimated = TokenEstimator.estimate_input_tokens(messages)
    assert estimated >= 0


def test_extract_actual_usage_from_response():
    """Test extracting usage stats from API response."""

    # Mock response object with usage attribute
    class MockUsage:
        input_tokens = 100
        output_tokens = 50
        cache_creation_input_tokens = 20
        cache_read_input_tokens = 30

    class MockResponse:
        usage = MockUsage()

    response = MockResponse()
    usage = TokenEstimator.extract_actual_usage(response)

    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["cache_creation_input_tokens"] == 20
    assert usage["cache_read_input_tokens"] == 30


def test_extract_actual_usage_without_cache():
    """Test extracting usage when no cache tokens are present."""

    # Mock response without cache tokens
    class MockUsage:
        input_tokens = 150
        output_tokens = 75
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0

    class MockResponse:
        usage = MockUsage()

    response = MockResponse()
    usage = TokenEstimator.extract_actual_usage(response)

    assert usage["input_tokens"] == 150
    assert usage["output_tokens"] == 75
    assert usage["cache_creation_input_tokens"] == 0
    assert usage["cache_read_input_tokens"] == 0


def test_calculate_billable_excludes_cache_reads():
    """Test that cache read tokens are excluded from billable input."""
    usage = {
        "input_tokens": 100,
        "cache_creation_input_tokens": 20,
        "cache_read_input_tokens": 50,  # Should NOT be counted
    }

    billable = TokenEstimator.calculate_billable_input_tokens(usage)

    # Only input_tokens + cache_creation_input_tokens
    assert billable == 120  # 100 + 20


def test_calculate_billable_includes_cache_creation():
    """Test that cache creation tokens are included in billable input."""
    usage = {
        "input_tokens": 100,
        "cache_creation_input_tokens": 30,  # Should be counted
        "cache_read_input_tokens": 0,
    }

    billable = TokenEstimator.calculate_billable_input_tokens(usage)

    assert billable == 130  # 100 + 30


def test_calculate_billable_no_cache():
    """Test billable calculation when no cache is used."""
    usage = {"input_tokens": 200, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}

    billable = TokenEstimator.calculate_billable_input_tokens(usage)

    assert billable == 200


def test_estimate_input_tokens_with_long_content():
    """Test estimation with longer content."""
    long_content = "This is a much longer message " * 50  # ~1500 chars
    messages = [{"role": "user", "content": long_content}]

    estimated = TokenEstimator.estimate_input_tokens(messages)

    # ~1500 chars / 4 = ~375 tokens
    assert 300 <= estimated <= 500


def test_estimate_input_tokens_with_structured_content():
    """Test estimation with structured content (text + images)."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "source": {"type": "base64", "data": "..."}},
            ],
        }
    ]

    estimated = TokenEstimator.estimate_input_tokens(messages)

    # Should count text portion: "What is in this image?" ~23 chars = ~5-6 tokens
    assert 5 <= estimated <= 15
