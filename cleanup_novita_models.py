#!/usr/bin/env python3
"""
Novita Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Novita models in a LiteLLM config.yaml file against
the current Novita API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing

Usage:
    python cleanup_novita_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_novita_models.py --add-model "deepseek/deepseek-r1"

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


class NovitaModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Novita models."""
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Novita model cleaner."""
        super().__init__('novita', config_path, dry_run, verbose)
    
    def parse_api_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a model from the Novita API response.
        
        Novita pricing uses credits per million tokens:
        - input_token_price_per_m: credits per million input tokens
        - output_token_price_per_m: credits per million output tokens
        
        Conversion: credits / 10000 / 1_000_000 = per token
        """
        model_info = {
            'id': model['id'],
            'input_cost': None,
            'output_cost': None,
            'model_info': None
        }
        
        # Get the divisor from config (10000 for Novita)
        divisor = self._pricing_config.get('divisor', 1)
        
        # Parse input cost
        input_price_per_m = model.get('input_token_price_per_m')
        if input_price_per_m is not None:
            try:
                # Convert from credits per million tokens to per token
                model_info['input_cost'] = round(float(input_price_per_m) / divisor / 1_000_000, 12)
            except (ValueError, TypeError):
                pass
        
        # Parse output cost
        output_price_per_m = model.get('output_token_price_per_m')
        if output_price_per_m is not None:
            try:
                # Convert from credits per million tokens to per token
                model_info['output_cost'] = round(float(output_price_per_m) / divisor / 1_000_000, 12)
            except (ValueError, TypeError):
                pass
        
        return model_info


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup Novita models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model deepseek/deepseek-r1  # Add new model(s)
        """
    )
    
    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)
    
    cleaner = NovitaModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
