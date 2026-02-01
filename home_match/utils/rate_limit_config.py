"""Rate limit configuration for Claude API tiers."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ModelLimits:
    """Token limits for a specific model family."""

    input_tokens_per_minute: int
    output_tokens_per_minute: int


@dataclass
class RateLimitConfig:
    """
    Rate limit configuration for Claude API.

    Defines the requests per minute limit (shared across all models)
    and per-model token limits.
    """

    requests_per_minute: int
    model_limits: Dict[str, ModelLimits]

    @classmethod
    def tier_1(cls) -> "RateLimitConfig":
        """
        Create a Tier 1 rate limit configuration.

        Tier 1 limits (as of 2026-01):
        - RPM: 50 (shared across all models)
        - Sonnet 4.x: 30K ITPM, 8K OTPM
        - Haiku 4.5: 50K ITPM, 10K OTPM
        - Opus 4.x: 30K ITPM, 8K OTPM

        Returns:
            RateLimitConfig configured for Tier 1
        """
        return cls(
            requests_per_minute=50,
            model_limits={
                "sonnet-4": ModelLimits(
                    input_tokens_per_minute=30000, output_tokens_per_minute=8000
                ),
                "haiku-4": ModelLimits(
                    input_tokens_per_minute=50000, output_tokens_per_minute=10000
                ),
                "opus-4": ModelLimits(input_tokens_per_minute=30000, output_tokens_per_minute=8000),
            },
        )
