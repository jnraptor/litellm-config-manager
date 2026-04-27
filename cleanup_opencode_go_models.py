#!/usr/bin/env python3
"""
OpenCode Go Model Cleanup and Cost Update Script for LiteLLM Config

This script validates OpenCode Go models in a LiteLLM config.yaml file against
the current OpenCode Go API and:
1. Removes any invalid model entries
2. Adds new models when requested

Supports both OpenAI-compatible models (openai/, dashscope/ prefixes) and
Anthropic-compatible models (anthropic/ prefix).

Usage:
    python cleanup_opencode_go_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_opencode_go_models.py --add-model "openai/glm-5"
    python cleanup_opencode_go_models.py --add-model "anthropic/minimax-m2.5"

Author: Generated for LiteLLM Config Management
"""

import sys

from cleanup_base import (
    ConfigDrivenModelCleaner,
    create_provider_main,
)


class OpenCodeGoModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for OpenCode Go models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the OpenCode Go model cleaner."""
        super().__init__("opencode-go", config_path, dry_run, verbose)


main = create_provider_main(
    OpenCodeGoModelCleaner,
    "Validate and cleanup OpenCode Go models in LiteLLM config",
    """
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model openai/glm-5    # Add new OpenAI-type model
  %(prog)s --add-model anthropic/minimax-m2.5  # Add new Anthropic-type model
    """,
)


if __name__ == "__main__":
    sys.exit(main())
