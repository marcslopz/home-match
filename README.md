# Home Match

AI-powered real estate property matcher that learns your preferences over time.

## Overview

Home Match is an intelligent agent that:
- 🏠 Monitors real estate portals (Idealista, etc.) on a recurring schedule
- 🤖 Uses AI to match properties with your preferences
- 📬 Notifies you about new matching listings
- 📊 Learns from your feedback to improve future recommendations

## Project Structure

```
home-match/
├── home_match/          # Main Python package
│   ├── agents/         # AI agents for monitoring and recommendations
│   ├── scrapers/       # Web scrapers for real estate portals
│   ├── models/         # Data models (properties, preferences, feedback)
│   └── utils/          # Utility functions
├── tests/              # Test suite
└── pyproject.toml      # Project configuration
```

## Setup

```bash
# Install dependencies
pip install -e .

# Run tests
pytest
```

## Usage

### Claude API Rate Limiter

Home Match includes a rate-limited wrapper for the Claude API that automatically respects tier limits.

```python
from home_match.utils import RateLimitedAnthropicClient
import os

# Initialize client (auto-loads Tier 1 config by default)
client = RateLimitedAnthropicClient(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Use like the standard Anthropic client
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "What is 2+2?"}
    ]
)

print(response.content)
```

#### Features

- **Automatic Rate Limiting**: Respects RPM (requests per minute) and token limits (ITPM/OTPM)
- **Token Bucket Algorithm**: Smooth rate limiting with automatic token replenishment
- **Per-Model Limits**: Different models (Sonnet, Haiku, Opus) have independent token buckets
- **Cache-Aware**: Correctly excludes `cache_read_input_tokens` from billable limits
- **Thread-Safe**: Safe for concurrent use across multiple threads
- **Tier Support**: Pre-configured for Claude API Tier 1, customizable for other tiers

#### Configuration

Customize limits for different tiers:

```python
from home_match.utils import RateLimitedAnthropicClient, RateLimitConfig, ModelLimits

# Create custom configuration (e.g., Tier 2)
custom_config = RateLimitConfig(
    requests_per_minute=100,
    model_limits={
        "sonnet-4": ModelLimits(
            input_tokens_per_minute=60000,
            output_tokens_per_minute=16000
        ),
        # ... other models
    }
)

client = RateLimitedAnthropicClient(
    api_key="your-key",
    config=custom_config
)
```

Disable rate limiting entirely:

```python
client = RateLimitedAnthropicClient(
    api_key="your-key",
    enable_rate_limiting=False  # Bypass all limits
)
```

### Other Usage

TBD

## License

TBD
