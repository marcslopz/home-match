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

TBD

## License

TBD
