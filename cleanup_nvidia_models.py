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

import argparse
import sys
from typing import Dict, Any

from cleanup_base import (
    ConfigDrivenModelCleaner,
    setup_common_args,
    validate_model_name_arg,
)


class NvidiaModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Nvidia NIM models."""
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Nvidia model cleaner."""
        super().__init__('nvidia', config_path, dry_run, verbose)
    
    def parse_api_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a model from the Nvidia API response.
        
        Nvidia models are free and don't have pricing fields in the API response.
        We use the default_cost from the provider configuration.
        """
        return {
            'id': model['id'],
            'input_cost': self._pricing_config.get('default_cost', 1e-09),
            'output_cost': self._pricing_config.get('default_cost', 1e-09),
            'model_info': None
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup Nvidia NIM models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Note: All Nvidia models are free, so they use a nominal cost of 1e-09 for LiteLLM compatibility.

Examples:
  %(prog)s                                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml                             # Run cleanup on custom config file
  %(prog)s --dry-run                                    # Preview changes without modifying file
  %(prog)s --add-model "meta/llama-3.1-8b-instruct"     # Add a single model
        """
    )
    
    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)
    
    cleaner = NvidiaModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
