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

import argparse
import sys
from typing import Dict, Any, Optional, List

from cleanup_base import (
    ConfigDrivenModelCleaner,
    setup_common_args,
    validate_model_name_arg,
)


class KiloModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Kilo models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Kilo model cleaner."""
        super().__init__('kilo', config_path, dry_run, verbose)

    def check_for_free_variant(self, model_id: str,
                               api_models: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Check if a free variant of the model exists in the API.

        Args:
            model_id: The base model ID (e.g., "minimax/minimax-m2.5")
            api_models: Dict of available models from API

        Returns:
            The free variant model ID if it exists, None otherwise
        """
        free_variant_id = f"{model_id}:free"
        if free_variant_id in api_models:
            self.logger.info(f"Found free variant: {free_variant_id}")
            return free_variant_id
        return None

    def add_model_to_config(self, config: Dict[str, Any], model_ids: List[str],
                           api_models: Dict[str, Dict[str, Any]],
                           custom_model_name: Optional[str] = None) -> tuple[Dict[str, Any], List[str]]:
        """
        Add one or more models to the configuration, including free variants if they exist.

        Overrides the base implementation to handle Kilo's free variant logic.
        """
        added_models = []
        failed_models = []

        existing_models = self.extract_provider_models(config)
        existing_model_ids = [self.get_api_model_id(mid) for _, mid, _ in existing_models]
        existing_names = [name for _, _, name in existing_models]

        for model_id in model_ids:
            self.logger.info(f"Processing model: {model_id}")

            # Check if base model exists in API
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                self.logger.error(f"Model '{model_id}' not found in {self.PROVIDER_NAME} API")
                failed_models.append(model_id)
                continue

            # Check if model already exists in config
            if model_id in existing_model_ids:
                self.logger.warning(f"Model '{model_id}' already exists in configuration")
                failed_models.append(model_id)
                continue

            # Check for free variant
            free_variant_id = self.check_for_free_variant(model_id, api_models)

            # Generate model name for the base model
            if custom_model_name and len(model_ids) == 1:
                model_name = custom_model_name
            else:
                model_name = self.generate_model_name(model_id)

            # Handle name conflicts for base model
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1

            # Create and add base model entry
            model_entry = self.create_model_entry(model_id, model_info, model_name)
            config['model_list'].append(model_entry)
            added_models.append(model_id)
            existing_model_ids.append(model_id)
            existing_names.append(model_name)

            self.logger.info(f"Added model '{model_id}' with name '{model_name}'")
            input_cost = model_entry.get('litellm_params', {}).get('input_cost_per_token')
            output_cost = model_entry.get('litellm_params', {}).get('output_cost_per_token')
            if input_cost is not None:
                self.logger.info(f"  Input cost: {input_cost}")
            if output_cost is not None:
                self.logger.info(f"  Output cost: {output_cost}")

            # Add free variant if it exists
            if free_variant_id:
                # Check if free variant already exists in config
                if free_variant_id in existing_model_ids:
                    self.logger.warning(f"Free variant '{free_variant_id}' already exists in configuration")
                else:
                    free_model_info = self.find_model_in_api(free_variant_id, api_models)
                    if free_model_info:
                        # Use the same model name as the base model
                        free_model_entry = self.create_model_entry(free_variant_id, free_model_info, model_name)
                        config['model_list'].append(free_model_entry)
                        added_models.append(free_variant_id)
                        existing_model_ids.append(free_variant_id)

                        self.logger.info(f"Also added free version '{free_variant_id}' with same name '{model_name}'")
                        free_input_cost = free_model_entry.get('litellm_params', {}).get('input_cost_per_token')
                        free_output_cost = free_model_entry.get('litellm_params', {}).get('output_cost_per_token')
                        if free_input_cost is not None:
                            self.logger.info(f"  Input cost: {free_input_cost}")
                        if free_output_cost is not None:
                            self.logger.info(f"  Output cost: {free_output_cost}")

        if failed_models:
            self.logger.warning(f"Failed to add {len(failed_models)} model(s): {', '.join(failed_models)}")
        if added_models:
            self.logger.info(f"Successfully processed {len(added_models)} model(s) for addition")

        return config, added_models

    def preview_add_model(self, model_ids: List[str], api_models: Dict[str, Dict[str, Any]],
                         custom_model_name: Optional[str] = None) -> None:
        """Preview what would be added when adding models, including free variants."""
        config = self.load_config()
        existing_models = self.extract_provider_models(config)
        existing_model_ids = [self.get_api_model_id(mid) for _, mid, _ in existing_models]
        existing_names = [name for _, _, name in existing_models]

        valid_models = []
        failed_models = []
        previewed_additions = []

        for model_id in model_ids:
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                self.logger.error(f"Model not found in {self.PROVIDER_NAME} API: {model_id}")
                failed_models.append(model_id)
                continue

            if model_id in existing_model_ids:
                self.logger.warning(f"Model already exists in configuration: {model_id}")
                continue

            valid_models.append((model_id, model_info))

        if not valid_models and failed_models:
            self.logger.warning(f"No valid models to add. Failed: {', '.join(failed_models)}")
            return

        for model_id, model_info in valid_models:
            # Generate model name
            if custom_model_name and len(model_ids) == 1:
                model_name = custom_model_name
            else:
                model_name = self.generate_model_name(model_id)

            # Handle name conflicts
            original_name = model_name
            counter = 1
            while model_name in existing_names:
                model_name = f"{original_name}-{counter}"
                counter += 1

            inputs = {**api_models[model_id], 'model': model_id}
            input_cost = inputs.get('input_cost')
            output_cost = inputs.get('output_cost')

            self.logger.info(f"[DRY-RUN]   - Model '{model_id}' with name '{model_name}'")
            if input_cost is not None:
                self.logger.info(f"[DRY-RUN]     Input cost: {input_cost}")
            if output_cost is not None:
                self.logger.info(f"[DRY-RUN]     Output cost: {output_cost}")

            previewed_additions.append(model_id)

            # Check for free variant
            free_variant_id = self.check_for_free_variant(model_id, api_models)
            if free_variant_id and free_variant_id not in existing_model_ids:
                self.logger.info(f"[DRY-RUN]     Would also add free version: {free_variant_id}")
                previewed_additions.append(free_variant_id)

        if previewed_additions:
            self.logger.info(f"[DRY-RUN] Would add {len(previewed_additions)} model(s)")
        else:
            self.logger.info("[DRY-RUN] No valid models to add.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup Kilo models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model anthropic/claude-opus-4.6  # Add new model(s)
        """
    )

    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)

    cleaner = KiloModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
