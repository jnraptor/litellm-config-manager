#!/usr/bin/env python3
"""
Nano-GPT Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Nano-GPT models in a LiteLLM config.yaml file against
the current Nano-GPT API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing

The script fetches current pricing from https://nano-gpt.com/api/v1/models?detailed=true and automatically
updates any cost differences found in the configuration file.

Usage:
    python cleanup_nano_gpt_models.py [--config config.yaml] [--dry-run] [--verbose]

Author: Generated for LiteLLM Config Management
"""

import argparse
import logging
import sys
import yaml
import requests
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path


# Constants
NANOGPT_API_URL = "https://nano-gpt.com/api/v1/models?detailed=true"
DEFAULT_CONFIG_FILE = "config.yaml"


class NanoGPTModelCleaner:
    """Main class for cleaning up invalid Nano-GPT models from LiteLLM config."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)

        # Add handler to logger
        if not logger.handlers:
            logger.addHandler(handler)

        return logger

    def _costs_are_equal(self, cost1: float, cost2: float, rel_tol: float = 1e-9) -> bool:
        """
        Compare two cost values using relative tolerance for scientific notation.
        
        This method properly handles very small cost values in scientific notation
        (e.g., 6.0e-07 vs 4e-07) which would incorrectly round to 0.0 with the
        previous rounding approach.
        
        Args:
            cost1: First cost value
            cost2: Second cost value
            rel_tol: Relative tolerance (default: 1e-9)
            
        Returns:
            True if costs are equal within tolerance, False otherwise
        """
        # Exact equality check
        if cost1 == cost2:
            return True
        
        # Handle zero cases
        if cost1 == 0.0 or cost2 == 0.0:
            return cost1 == cost2
        
        # Use relative tolerance for non-zero values
        abs_diff = abs(cost1 - cost2)
        max_abs = max(abs(cost1), abs(cost2))
        return abs_diff <= rel_tol * max_abs

    def load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        try:
            self.logger.info(f"Loading configuration from {self.config_path}")

            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)

            if not config:
                raise ValueError("Configuration file is empty or invalid")

            if 'model_list' not in config:
                raise ValueError("Configuration file missing 'model_list' section")

            self.logger.debug(f"Loaded configuration with {len(config['model_list'])} models")
            return config

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise

    def extract_nano_gpt_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """
        Extract Nano-GPT models from the configuration.

        Returns:
            List of tuples: (index, model_id, model_name)
        """
        nano_gpt_models = []
        model_list = config.get('model_list', [])

        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue

            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            model_name = model_entry.get('model_name', 'unnamed')

            # Check if this is a Nano-GPT model (identified by api_base)
            api_base = litellm_params.get('api_base', '')
            if (model_id.startswith('openai/') and
                'NANOGPT_API_BASE' in str(api_base)):
                nano_gpt_models.append((index, model_id, model_name))
                self.logger.debug(f"Found Nano-GPT model: {model_id} (name: {model_name})")

        self.logger.info(f"Found {len(nano_gpt_models)} Nano-GPT models in configuration")
        return nano_gpt_models

    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch the list of available models with pricing from Nano-GPT API."""
        try:
            self.logger.info("Fetching available models with pricing from Nano-GPT API...")

            response = requests.get(NANOGPT_API_URL, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'data' not in data:
                raise ValueError("Invalid API response format: missing 'data' field")

            # Extract model IDs and pricing from the API response
            available_models = {}
            for model in data['data']:
                if isinstance(model, dict) and 'id' in model:
                    model_id = model['id']
                    model_info = {
                        'id': model_id,
                        'input_cost': None,
                        'output_cost': None
                    }

                    # Extract pricing information
                    pricing = model.get('pricing', {})
                    if isinstance(pricing, dict):
                        # Parse input cost (prompt)
                        prompt_cost = pricing.get('prompt')
                        if prompt_cost is not None:
                            try:
                                # Convert from per million tokens to per token
                                model_info['input_cost'] = float(prompt_cost) / 1000000
                            except (ValueError, TypeError):
                                self.logger.warning(f"Could not parse prompt cost for model {model_id}: {prompt_cost}")

                        # Parse output cost (completion)
                        completion_cost = pricing.get('completion')
                        if completion_cost is not None:
                            try:
                                # Convert from per million tokens to per token
                                model_info['output_cost'] = float(completion_cost) / 1000000
                            except (ValueError, TypeError):
                                self.logger.warning(f"Could not parse completion cost for model {model_id}: {completion_cost}")

                    available_models[model_id] = model_info
                    self.logger.debug(f"Available model: {model_id} with input_cost={model_info['input_cost']}, output_cost={model_info['output_cost']}")

            self.logger.info(f"Fetched {len(available_models)} available models from Nano-GPT API")
            return available_models

        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Invalid API response: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error fetching models: {e}")
            raise

    def validate_and_update_costs(self, config: Dict[str, Any], available_models: Dict[str, Dict[str, Any]]) -> Tuple[int, int, List[str]]:
        """
        Validate model entries against available models and update costs.

        Returns:
            Tuple of (models_removed, models_updated, changes_made)
        """
        nano_gpt_models = self.extract_nano_gpt_models(config)
        models_removed = 0
        models_updated = 0
        changes_made = []

        # Process in reverse order to maintain indices when removing
        for index, model_id, model_name in reversed(nano_gpt_models):
            model_entry = config['model_list'][index]
            litellm_params = model_entry.get('litellm_params', {})

            # Extract the Nano-GPT model ID (remove 'openai/' prefix)
            nano_gpt_model_id = model_id.replace('openai/', '')

            if nano_gpt_model_id not in available_models:
                # Model is no longer available
                self.logger.warning(f"Removing unavailable model: {model_id} (name: {model_name})")
                if not self.dry_run:
                    config['model_list'].pop(index)
                models_removed += 1
                changes_made.append(f"Removed unavailable model: {model_id}")
                continue

            # Check cost differences
            api_model_info = available_models[nano_gpt_model_id]
            current_input_cost = litellm_params.get('input_cost_per_token')
            current_output_cost = litellm_params.get('output_cost_per_token')

            api_input_cost = api_model_info['input_cost']
            api_output_cost = api_model_info['output_cost']

            # Handle free models (use 1.0e-09 as per existing pattern)
            if api_input_cost is None and api_output_cost is None:
                api_input_cost = 1.0e-09
                api_output_cost = 1.0e-09
                self.logger.debug(f"Model {model_id} appears to be free, using nominal costs")

            cost_differences = []

            # Compare costs using relative tolerance for scientific notation values
            if api_input_cost is not None:
                # Convert current cost to float if it's a string
                if current_input_cost is not None:
                    try:
                        current_float = float(current_input_cost)
                    except (ValueError, TypeError):
                        current_float = None
                else:
                    current_float = None

                if current_float is not None:
                    if not self._costs_are_equal(current_float, api_input_cost):
                        cost_differences.append(f"input_cost_per_token: {current_input_cost} -> {api_input_cost}")
                        if not self.dry_run:
                            litellm_params['input_cost_per_token'] = api_input_cost
                else:
                    # Current cost is None or invalid, always update
                    cost_differences.append(f"input_cost_per_token: {current_input_cost} -> {api_input_cost}")
                    if not self.dry_run:
                        litellm_params['input_cost_per_token'] = api_input_cost

            if api_output_cost is not None:
                # Convert current cost to float if it's a string
                if current_output_cost is not None:
                    try:
                        current_float = float(current_output_cost)
                    except (ValueError, TypeError):
                        current_float = None
                else:
                    current_float = None

                if current_float is not None:
                    if not self._costs_are_equal(current_float, api_output_cost):
                        cost_differences.append(f"output_cost_per_token: {current_output_cost} -> {api_output_cost}")
                        if not self.dry_run:
                            litellm_params['output_cost_per_token'] = api_output_cost
                else:
                    # Current cost is None or invalid, always update
                    cost_differences.append(f"output_cost_per_token: {current_output_cost} -> {api_output_cost}")
                    if not self.dry_run:
                        litellm_params['output_cost_per_token'] = api_output_cost

            if cost_differences:
                self.logger.info(f"Updating costs for model {model_id} (name: {model_name}): {', '.join(cost_differences)}")
                models_updated += 1
                changes_made.append(f"Updated costs for {model_id}: {', '.join(cost_differences)}")
            else:
                self.logger.debug(f"Costs are already up to date for model {model_id}")

        return models_removed, models_updated, changes_made

    def add_model_to_config(self, config: Dict[str, Any], model_id: str, available_models: Dict[str, Dict[str, Any]]) -> bool:
        """Add a new model to the configuration."""
        if model_id not in available_models:
            self.logger.error(f"Model {model_id} not found in available models")
            return False

        # Check if model already exists
        for model_entry in config['model_list']:
            if (isinstance(model_entry, dict) and
                model_entry.get('litellm_params', {}).get('model') == f"openai/{model_id}"):
                self.logger.warning(f"Model {model_id} already exists in configuration")
                return False

        api_model_info = available_models[model_id]

        # Create new model entry
        new_model_entry = {
            'model_name': model_id.replace('/', '-').replace('_', '-'),
            'litellm_params': {
                'model': f"openai/{model_id}",
                'api_base': "os.environ/NANOGPT_API_BASE",
                'api_key': "os.environ/NANOGPT_API_KEY"
            }
        }

        # Add costs if available
        if api_model_info['input_cost'] is not None:
            new_model_entry['litellm_params']['input_cost_per_token'] = api_model_info['input_cost']
        else:
            new_model_entry['litellm_params']['input_cost_per_token'] = 1.0e-09  # Free model

        if api_model_info['output_cost'] is not None:
            new_model_entry['litellm_params']['output_cost_per_token'] = api_model_info['output_cost']
        else:
            new_model_entry['litellm_params']['output_cost_per_token'] = 1.0e-09  # Free model

        if not self.dry_run:
            config['model_list'].append(new_model_entry)

        self.logger.info(f"Added new model: {model_id}")
        return True

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the configuration back to the YAML file."""
        if self.dry_run:
            self.logger.info("DRY RUN: Would save configuration to file")
            return

        try:
            self.logger.info(f"Saving configuration to {self.config_path}")

            # Create backup
            backup_path = self.config_path.with_suffix('.yaml.backup')
            if self.config_path.exists():
                self.config_path.rename(backup_path)
                self.logger.info(f"Created backup at {backup_path}")

            # Save new configuration
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)

            self.logger.info("Configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            # Restore backup if it exists
            if backup_path.exists():
                backup_path.rename(self.config_path)
                self.logger.info("Restored configuration from backup")
            raise

    def cleanup_models(self, add_models: Optional[List[str]] = None) -> Tuple[int, int, List[str]]:
        """
        Main cleanup method.

        Returns:
            Tuple of (models_removed, models_updated, changes_made)
        """
        try:
            # Load configuration
            config = self.load_config()

            # Fetch available models
            available_models = self.fetch_available_models()

            # Validate and update costs
            models_removed, models_updated, changes_made = self.validate_and_update_costs(config, available_models)

            # Add new models if requested
            if add_models:
                for model_id in add_models:
                    if self.add_model_to_config(config, model_id, available_models):
                        changes_made.append(f"Added new model: {model_id}")

            # Save configuration
            if changes_made and not self.dry_run:
                self.save_config(config)

            return models_removed, models_updated, changes_made

        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up and update Nano-GPT models in LiteLLM configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cleanup_nano_gpt_models.py
    python cleanup_nano_gpt_models.py --config my-config.yaml
    python cleanup_nano_gpt_models.py --dry-run --verbose
    python cleanup_nano_gpt_models.py --add-model "nvidia/nvidia-nemotron-nano-9b-v2"
    python cleanup_nano_gpt_models.py --add-model model1 model2 model3
        """
    )

    parser.add_argument(
        '--config', '-c',
        default=DEFAULT_CONFIG_FILE,
        help=f'Configuration file path (default: {DEFAULT_CONFIG_FILE})'
    )

    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Show what would be changed without making actual changes'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--add-model', '-a',
        nargs='+',
        help='Add new model(s) to configuration (space-separated list)'
    )

    args = parser.parse_args()

    try:
        cleaner = NanoGPTModelCleaner(
            config_path=args.config,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        models_removed, models_updated, changes_made = cleaner.cleanup_models(add_models=args.add_model)

        print(f"\nSummary:")
        print(f"Models removed: {models_removed}")
        print(f"Models updated: {models_updated}")
        print(f"Total changes: {len(changes_made)}")

        if changes_made:
            print(f"\nChanges made:")
            for change in changes_made:
                print(f"  - {change}")

        if args.dry_run and changes_made:
            print(f"\nDRY RUN: No actual changes were made")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())