"""Token estimation and usage tracking for Claude API."""

from typing import Any, Dict, List, Optional


class TokenEstimator:
    """
    Utilities for estimating and tracking token usage.

    Provides methods to estimate input tokens before making a request
    and extract actual usage from API responses.
    """

    @staticmethod
    def estimate_input_tokens(messages: List[Dict[str, Any]], system: Optional[str] = None) -> int:
        """
        Estimate input tokens for a request.

        Uses a simple heuristic of ~4 characters per token, which is
        a reasonable approximation for English text.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system: Optional system prompt

        Returns:
            Estimated number of input tokens
        """
        total_chars = 0

        # Count system prompt characters
        if system:
            total_chars += len(system)

        # Count message content characters
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                # Handle structured content (text + images)
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))

        # Estimate tokens: ~4 chars per token, with some overhead
        estimated_tokens = int(total_chars / 4) + 5  # Add 5 tokens overhead

        return estimated_tokens

    @staticmethod
    def extract_actual_usage(response: Any) -> Dict[str, int]:
        """
        Extract actual token usage from API response.

        Args:
            response: Anthropic API response object

        Returns:
            Dictionary with usage stats:
            - input_tokens: Regular input tokens
            - output_tokens: Output tokens generated
            - cache_creation_input_tokens: Tokens used to create cache
            - cache_read_input_tokens: Tokens read from cache
        """
        usage = response.usage

        return {
            "input_tokens": getattr(usage, "input_tokens", 0),
            "output_tokens": getattr(usage, "output_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        }

    @staticmethod
    def calculate_billable_input_tokens(usage: Dict[str, int]) -> int:
        """
        Calculate billable input tokens from usage stats.

        IMPORTANT: cache_read_input_tokens do NOT count towards ITPM limits.
        Only input_tokens and cache_creation_input_tokens are billable.

        Args:
            usage: Usage dictionary from extract_actual_usage()

        Returns:
            Number of billable input tokens
        """
        input_tokens = usage.get("input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)

        # Explicitly exclude cache_read_input_tokens
        return input_tokens + cache_creation
