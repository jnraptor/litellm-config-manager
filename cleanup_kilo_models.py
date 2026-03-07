#!/usr/bin/env python3
"""
Kilo Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Kilo models in a LiteLLM config.yaml file against
the current Kilo API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing
3. Adds free variants when adding new models

Usage:
    python cleanup_kilo_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_kilo_models.py --add-model "anthropic/claude-opus-4.6"

Author: Generated for LiteLLM Config Management
"""

import sys

from cleanup_base import (
    ConfigDrivenModelCleaner,
    create_provider_main,
)


class KiloModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Kilo models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Kilo model cleaner."""
        super().__init__('kilo', config_path, dry_run, verbose)


main = create_provider_main(
    KiloModelCleaner,
    'Validate and cleanup Kilo models in LiteLLM config',
    """
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model anthropic/claude-opus-4.6  # Add new model(s)
    """
)


if __name__ == "__main__":
    sys.exit(main())
