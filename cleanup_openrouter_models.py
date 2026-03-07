#!/usr/bin/env python3
"""
OpenRouter Model Cleanup and Cost Update Script for LiteLLM Config

This script validates OpenRouter models in a LiteLLM config.yaml file against
the current OpenRouter API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token)
3. Supports both regular and embedding models

Usage:
    python cleanup_openrouter_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_openrouter_models.py --add-model "qwen/qwen3-14b"

Author: Generated for LiteLLM Config Management
"""

import sys

from cleanup_base import (
    ConfigDrivenModelCleaner,
    create_provider_main,
)


class OpenRouterModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for OpenRouter models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the OpenRouter model cleaner."""
        super().__init__('openrouter', config_path, dry_run, verbose)


main = create_provider_main(
    OpenRouterModelCleaner,
    'Validate and cleanup OpenRouter models in LiteLLM config',
    """
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model qwen/qwen3-14b   # Add new model(s)
    """
)


if __name__ == "__main__":
    sys.exit(main())
