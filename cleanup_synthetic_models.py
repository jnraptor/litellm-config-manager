#!/usr/bin/env python3
"""
Synthetic Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Synthetic models in a LiteLLM config.yaml file against
the current Synthetic API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token)

Usage:
    python cleanup_synthetic_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_synthetic_models.py --add-model "hf:zai-org/GLM-4.7"

Author: Generated for LiteLLM Config Management
"""

import argparse
import sys

from cleanup_base import (
    ConfigDrivenModelCleaner,
    setup_common_args,
    validate_model_name_arg,
)


class SyntheticModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Synthetic models."""
    
    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Synthetic model cleaner."""
        super().__init__('synthetic', config_path, dry_run, verbose)
    
    def generate_model_name(self, model_id: str) -> str:
        """
        Generate model name for Synthetic models.
        
        Applies the model name cleanup rules from configuration, which includes
        removing "hf:" and "zai-org/" prefixes.
        """
        # Apply cleanup rules from config (which includes removing hf: and zai-org/)
        clean_id = model_id.replace('/', '-').replace(':', '-')
        
        for cleanup_rule in self._model_name_cleanup:
            for replace_old, replace_new in cleanup_rule.get('replace', []):
                clean_id = clean_id.replace(replace_old, replace_new)
        
        model_name = f"{self._model_name_prefix}{clean_id}" if self._model_name_prefix else clean_id
        return model_name.lower()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup Synthetic models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model "hf:zai-org/GLM-4.7"  # Add new model(s)
        """
    )
    
    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)
    
    cleaner = SyntheticModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
