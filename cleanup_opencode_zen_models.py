#!/usr/bin/env python3
"""
OpenCode Zen Model Cleanup and Cost Update Script for LiteLLM Config

This script validates OpenCode Zen models in a LiteLLM config.yaml file against
the current OpenCode Zen API and:
1. Removes any invalid model entries
2. Adds new models when requested

Usage:
    python cleanup_opencode_zen_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_opencode_zen_models.py --add-model "glm-5.1"

Author: Generated for LiteLLM Config Management
"""

import sys

from cleanup_base import (
    ConfigDrivenModelCleaner,
    create_provider_main,
)


class OpenCodeZenModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for OpenCode Zen models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the OpenCode Zen model cleaner."""
        super().__init__("opencode-zen", config_path, dry_run, verbose)


main = create_provider_main(
    OpenCodeZenModelCleaner,
    "Validate and cleanup OpenCode Zen models in LiteLLM config",
    """
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model glm-5.1         # Add new model(s)
    """,
)


if __name__ == "__main__":
    sys.exit(main())
