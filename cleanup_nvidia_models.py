#!/usr/bin/env python3
"""
Nvidia Model Cleanup Script for LiteLLM Config

This script validates Nvidia models in a LiteLLM config.yaml file against
the current Nvidia API and:
1. Removes any invalid model entries
2. Adds new models when requested

Note: Nvidia models are free, so all models use a nominal cost of 1e-09 for LiteLLM compatibility.

Usage:
    python cleanup_nvidia_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_nvidia_models.py --add-model "meta/llama-3.1-8b-instruct"

Author: Generated for LiteLLM Config Management
"""

import sys

from cleanup_base import (
    ConfigDrivenModelCleaner,
    create_provider_main,
)


class NvidiaModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Nvidia NIM models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Nvidia model cleaner."""
        super().__init__('nvidia', config_path, dry_run, verbose)


main = create_provider_main(
    NvidiaModelCleaner,
    'Validate and cleanup Nvidia NIM models in LiteLLM config',
    """
Note: All Nvidia models are free, so they use a nominal cost of 1e-09 for LiteLLM compatibility.

Examples:
  %(prog)s                                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml                             # Run cleanup on custom config file
  %(prog)s --dry-run                                    # Preview changes without modifying file
  %(prog)s --add-model "meta/llama-3.1-8b-instruct"     # Add a single model
    """
)


if __name__ == "__main__":
    sys.exit(main())
