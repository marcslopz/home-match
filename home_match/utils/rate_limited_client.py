"""Rate-limited wrapper for Anthropic Claude API client."""

from typing import Any, Dict, Optional

from anthropic import Anthropic

from home_match.utils.rate_limit_config import RateLimitConfig
from home_match.utils.token_bucket import TokenBucket
from home_match.utils.token_estimator import TokenEstimator


class RateLimitedMessages:
    """
    Wrapper for Anthropic messages API with rate limiting.

    Intercepts messages.create() calls to apply rate limits.
    """

    def __init__(
        self, client: Anthropic, config: RateLimitConfig, enable_rate_limiting: bool = True
    ):
        """
        Initialize the rate-limited messages wrapper.

        Args:
            client: Anthropic client instance
            config: Rate limit configuration
            enable_rate_limiting: Whether to apply rate limiting
        """
        self._client = client
        self._config = config
        self._enable_rate_limiting = enable_rate_limiting

        if enable_rate_limiting:
            # Create RPM bucket (shared across all models)
            self._rpm_bucket = TokenBucket(
                capacity=config.requests_per_minute,
                refill_rate=config.requests_per_minute / 60.0,  # Per second
            )

            # Create ITPM/OTPM buckets per model family
            self._itpm_buckets: Dict[str, TokenBucket] = {}
            self._otpm_buckets: Dict[str, TokenBucket] = {}

            for family, limits in config.model_limits.items():
                self._itpm_buckets[family] = TokenBucket(
                    capacity=limits.input_tokens_per_minute,
                    refill_rate=limits.input_tokens_per_minute / 60.0,
                )
                self._otpm_buckets[family] = TokenBucket(
                    capacity=limits.output_tokens_per_minute,
                    refill_rate=limits.output_tokens_per_minute / 60.0,
                )
        else:
            self._rpm_bucket = None
            self._itpm_buckets = {}
            self._otpm_buckets = {}

    def _determine_model_family(self, model: str) -> str:
        """
        Determine model family from model string.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")

        Returns:
            Model family key (e.g., "sonnet-4", "haiku-4", "opus-4")
        """
        # Pattern: claude-{family}-{version}-{date}
        # Examples:
        # - claude-sonnet-4-20250514 -> sonnet-4
        # - claude-haiku-4.5-20250514 -> haiku-4
        # - claude-opus-4-20250514 -> opus-4

        model_lower = model.lower()

        if "sonnet" in model_lower:
            return "sonnet-4"
        elif "haiku" in model_lower:
            return "haiku-4"
        elif "opus" in model_lower:
            return "opus-4"
        else:
            # Default to sonnet if unknown
            return "sonnet-4"

    def _wait_for_rate_limits(
        self, model_family: str, estimated_input_tokens: int, max_output_tokens: int
    ) -> None:
        """
        Wait for rate limits to allow the request.

        Blocks until sufficient tokens/requests are available in all buckets.

        Args:
            model_family: Model family identifier
            estimated_input_tokens: Estimated input tokens for request
            max_output_tokens: Requested max_tokens
        """
        if not self._enable_rate_limiting:
            return

        # Wait for RPM bucket (1 request)
        self._rpm_bucket.consume(tokens=1, timeout=None)

        # Wait for ITPM bucket
        if model_family in self._itpm_buckets:
            self._itpm_buckets[model_family].consume(tokens=estimated_input_tokens, timeout=None)

        # Wait for OTPM bucket (reserve max_tokens)
        if model_family in self._otpm_buckets:
            self._otpm_buckets[model_family].consume(tokens=max_output_tokens, timeout=None)

    def _update_buckets_with_actual_usage(
        self, model_family: str, actual_usage: Dict[str, int]
    ) -> None:
        """
        Adjust buckets based on actual token usage.

        If actual usage differs from estimation, adjust the buckets
        to account for the difference.

        Args:
            model_family: Model family identifier
            actual_usage: Actual token usage from API response
        """
        if not self._enable_rate_limiting:
            return

        # Note: We already consumed estimated tokens pre-request.
        # If actual usage differs significantly from estimation, we could adjust here.
        # For simplicity, we rely on the estimation being close enough.
        # The buckets will naturally refill over time.

    def create(self, **kwargs) -> Any:
        """
        Create a message with rate limiting.

        This method wraps the Anthropic messages.create() call and applies
        rate limiting based on the configured limits.

        Args:
            **kwargs: Arguments passed to Anthropic messages.create()

        Returns:
            Message response from Anthropic API
        """
        if self._enable_rate_limiting:
            # Extract parameters
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])
            system = kwargs.get("system")
            max_tokens = kwargs.get("max_tokens", 1024)

            # Determine model family
            model_family = self._determine_model_family(model)

            # Estimate input tokens
            estimated_input = TokenEstimator.estimate_input_tokens(messages, system)

            # Wait for rate limits
            self._wait_for_rate_limits(model_family, estimated_input, max_tokens)

        # Make the actual API call
        response = self._client.messages.create(**kwargs)

        if self._enable_rate_limiting:
            # Extract actual usage and update buckets
            actual_usage = TokenEstimator.extract_actual_usage(response)
            self._update_buckets_with_actual_usage(model_family, actual_usage)

        return response


class RateLimitedAnthropicClient:
    """
    Rate-limited wrapper for Anthropic Claude API client.

    This client wraps the official Anthropic Python SDK and automatically
    applies rate limiting to stay within API tier limits.

    Example:
        >>> client = RateLimitedAnthropicClient(api_key="your-key")
        >>> response = client.messages.create(
        ...     model="claude-sonnet-4-20250514",
        ...     max_tokens=1024,
        ...     messages=[{"role": "user", "content": "Hello!"}]
        ... )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[RateLimitConfig] = None,
        enable_rate_limiting: bool = True,
    ):
        """
        Initialize the rate-limited Anthropic client.

        Args:
            api_key: Anthropic API key (reads from ANTHROPIC_API_KEY env var if not provided)
            config: Rate limit configuration (defaults to Tier 1)
            enable_rate_limiting: Whether to enable rate limiting (default: True)
        """
        self._client = Anthropic(api_key=api_key)
        self._config = config or RateLimitConfig.tier_1()
        self._enable_rate_limiting = enable_rate_limiting

        # Wrap messages API
        self._messages = RateLimitedMessages(
            client=self._client, config=self._config, enable_rate_limiting=enable_rate_limiting
        )

    @property
    def messages(self) -> RateLimitedMessages:
        """Get the rate-limited messages API."""
        return self._messages
