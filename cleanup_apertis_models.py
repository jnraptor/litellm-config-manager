#!/usr/bin/env python3
"""
Apertis Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Apertis models in a LiteLLM config.yaml file against
the current Apertis API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token)

Usage:
    python cleanup_apertis_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_apertis_models.py --add-model "glm-5"

Author: Generated for LiteLLM Config Management
"""

import sys

from dotenv import load_dotenv
from cleanup_base import (
    ConfigDrivenModelCleaner,
    create_provider_main,
)

# Load environment variables from .env file
load_dotenv()


class ApertisModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Apertis models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Apertis model cleaner."""
        super().__init__("apertis", config_path, dry_run, verbose)


main = create_provider_main(
    ApertisModelCleaner,
    "Validate and cleanup Apertis models in LiteLLM config",
    """
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model glm-5            # Add new model(s)
    """,
)


if __name__ == "__main__":
    sys.exit(main())
