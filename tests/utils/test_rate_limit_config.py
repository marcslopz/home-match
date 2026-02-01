"""Tests for RateLimitConfig implementation."""


from home_match.utils.rate_limit_config import ModelLimits, RateLimitConfig


def test_tier_1_config_has_correct_limits():
    """Test that Tier 1 config has correct RPM limit."""
    config = RateLimitConfig.tier_1()
    assert config.requests_per_minute == 50


def test_tier_1_sonnet_limits():
    """Test that Tier 1 has correct limits for Sonnet models."""
    config = RateLimitConfig.tier_1()

    # Check sonnet-4 family
    assert "sonnet-4" in config.model_limits
    sonnet_limits = config.model_limits["sonnet-4"]
    assert sonnet_limits.input_tokens_per_minute == 30000
    assert sonnet_limits.output_tokens_per_minute == 8000


def test_tier_1_haiku_limits():
    """Test that Tier 1 has correct limits for Haiku models."""
    config = RateLimitConfig.tier_1()

    # Check haiku-4 family
    assert "haiku-4" in config.model_limits
    haiku_limits = config.model_limits["haiku-4"]
    assert haiku_limits.input_tokens_per_minute == 50000
    assert haiku_limits.output_tokens_per_minute == 10000


def test_tier_1_opus_limits():
    """Test that Tier 1 has correct limits for Opus models."""
    config = RateLimitConfig.tier_1()

    # Check opus-4 family
    assert "opus-4" in config.model_limits
    opus_limits = config.model_limits["opus-4"]
    assert opus_limits.input_tokens_per_minute == 30000
    assert opus_limits.output_tokens_per_minute == 8000


def test_custom_config_creation():
    """Test creating a custom rate limit config."""
    custom_limits = {
        "sonnet-4": ModelLimits(input_tokens_per_minute=100000, output_tokens_per_minute=20000),
        "haiku-4": ModelLimits(input_tokens_per_minute=200000, output_tokens_per_minute=40000),
    }

    config = RateLimitConfig(requests_per_minute=100, model_limits=custom_limits)

    assert config.requests_per_minute == 100
    assert config.model_limits["sonnet-4"].input_tokens_per_minute == 100000
    assert config.model_limits["sonnet-4"].output_tokens_per_minute == 20000
    assert config.model_limits["haiku-4"].input_tokens_per_minute == 200000
    assert config.model_limits["haiku-4"].output_tokens_per_minute == 40000


def test_model_limits_dataclass():
    """Test ModelLimits dataclass creation."""
    limits = ModelLimits(input_tokens_per_minute=50000, output_tokens_per_minute=10000)

    assert limits.input_tokens_per_minute == 50000
    assert limits.output_tokens_per_minute == 10000


def test_tier_1_has_all_model_families():
    """Test that Tier 1 config includes all expected model families."""
    config = RateLimitConfig.tier_1()

    expected_families = ["sonnet-4", "haiku-4", "opus-4"]
    for family in expected_families:
        assert family in config.model_limits, f"Missing model family: {family}"
