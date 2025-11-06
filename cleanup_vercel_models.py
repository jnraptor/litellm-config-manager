#!/usr/bin/env python3
"""
Vercel AI Gateway Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Vercel AI Gateway models in a LiteLLM config.yaml file against
the current Vercel AI Gateway API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing

The script fetches current pricing from https://ai-gateway.vercel.sh/v1/models and automatically
updates any cost differences found in the configuration file.

Usage:
    python cleanup_vercel_models.py [--config config.yaml] [--dry-run] [--verbose]

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
VERCEL_API_URL = "https://ai-gateway.vercel.sh/v1/models"
DEFAULT_CONFIG_FILE = "config.yaml"


class VercelModelCleaner:
    """Main class for cleaning up invalid Vercel AI Gateway models from LiteLLM config."""
    
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
    
    def extract_vercel_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """
        Extract Vercel AI Gateway models from the configuration.
        
        Returns:
            List of tuples: (index, model_id, model_name)
        """
        vercel_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
                
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            
            # Check if this is a Vercel AI Gateway model
            if model_id.startswith('vercel_ai_gateway/'):
                model_name = model_entry.get('model_name', 'unnamed')
                vercel_models.append((index, model_id, model_name))
                self.logger.debug(f"Found Vercel model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(vercel_models)} Vercel models in configuration")
        return vercel_models
    
    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch the list of available models with pricing from Vercel AI Gateway API."""
        try:
            self.logger.info("Fetching available models with pricing from Vercel AI Gateway API...")
            
            # Make API request
            response = requests.get(VERCEL_API_URL, timeout=30)
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
                        'name': model.get('name', model_id),
                        'description': model.get('description', ''),
                        'input_cost': None,
                        'output_cost': None
                    }
                    
                    # Extract pricing information
                    pricing = model.get('pricing', {})
                    if isinstance(pricing, dict):
                        try:
                            if 'input' in pricing and pricing['input']:
                                # Vercel pricing is per token (not per million)
                                model_info['input_cost'] = float(pricing['input'])
                            
                            if 'output' in pricing and pricing['output']:
                                model_info['output_cost'] = float(pricing['output'])
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Invalid pricing for {model_id}: {e}")
                    
                    available_models[model_id] = model_info
            
            self.logger.info(f"Fetched {len(available_models)} available models from Vercel AI Gateway API")
            return available_models
            
        except requests.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing API response: {e}")
            raise
    
    def validate_models(self, config_models: List[Tuple[int, str, str]],
                       api_models: Dict[str, Dict[str, Any]]) -> List[Tuple[int, str, str]]:
        """Compare config models with API models and identify invalid ones."""
        # Convert API models to set for faster lookup
        api_models_set = set(api_models.keys())
        invalid_models = []
        
        for index, model_id, model_name in config_models:
            # Extract the part after "vercel_ai_gateway/"
            short_model_id = model_id.replace('vercel_ai_gateway/', '')
            
            if short_model_id not in api_models_set:
                invalid_models.append((index, model_id, model_name))
                self.logger.debug(f"Invalid model found: {model_id} (not in API response)")
        
        self.logger.info(f"Found {len(invalid_models)} invalid models out of {len(config_models)} total Vercel models")
        return invalid_models
    
    def validate_and_update_costs(self, config: Dict[str, Any],
                                api_models: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Validate and update model costs from API."""
        model_list = config.get('model_list', [])
        cost_changes = []
        
        for model_entry in model_list:
            if not isinstance(model_entry, dict):
                continue
                
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            
            # Only process Vercel models
            if not model_id.startswith('vercel_ai_gateway/'):
                continue
            
            # Extract the part after "vercel_ai_gateway/"
            short_model_id = model_id.replace('vercel_ai_gateway/', '')
            model_name = model_entry.get('model_name', 'unnamed')
            
            if short_model_id in api_models:
                api_model_info = api_models[short_model_id]
                api_input_cost = api_model_info.get('input_cost')
                api_output_cost = api_model_info.get('output_cost')
                
                # Get current costs from config
                current_input_cost = litellm_params.get('input_cost_per_token')
                current_output_cost = litellm_params.get('output_cost_per_token')
                
                # Handle free models: if API returns 0.0 or null, use 1e-09 for LiteLLM compatibility
                adjusted_input_cost = api_input_cost if api_input_cost is not None else 1e-09
                adjusted_output_cost = api_output_cost if api_output_cost is not None else 1e-09
                
                if adjusted_input_cost == 0.0:
                    adjusted_input_cost = 1e-09
                if adjusted_output_cost == 0.0:
                    adjusted_output_cost = 1e-09
                
                change_info = {
                    'model_id': model_id,
                    'model_name': model_name,
                    'changes': {}
                }
                
                input_changed = False
                output_changed = False
                
                # Compare input costs
                if api_input_cost is not None:
                    # Handle free models: if API returns 0.0, use 1e-09 for LiteLLM compatibility
                    adjusted_input_cost = 1e-09 if api_input_cost == 0.0 else api_input_cost

                    # Compare costs using relative tolerance for scientific notation values
                    if current_input_cost is not None:
                        if not self._costs_are_equal(current_input_cost, adjusted_input_cost):
                            input_changed = True
                            change_info['changes']['input_cost'] = {
                                'old': current_input_cost,
                                'new': adjusted_input_cost
                            }
                            litellm_params['input_cost_per_token'] = adjusted_input_cost
                            self.logger.debug(f"Input cost change for {model_id}: {current_input_cost} → {adjusted_input_cost}")
                    else:
                        # Current cost is None, always update
                        input_changed = True
                        change_info['changes']['input_cost'] = {
                            'old': current_input_cost,
                            'new': adjusted_input_cost
                        }
                        litellm_params['input_cost_per_token'] = adjusted_input_cost
                        self.logger.debug(f"Input cost change for {model_id}: {current_input_cost} → {adjusted_input_cost}")

                # Compare output costs
                if api_output_cost is not None:
                    # Handle free models: if API returns 0.0, use 1e-09 for LiteLLM compatibility
                    adjusted_output_cost = 1e-09 if api_output_cost == 0.0 else api_output_cost

                    # Compare costs using relative tolerance for scientific notation values
                    if current_output_cost is not None:
                        if not self._costs_are_equal(current_output_cost, adjusted_output_cost):
                            output_changed = True
                            change_info['changes']['output_cost'] = {
                                'old': current_output_cost,
                                'new': adjusted_output_cost
                            }
                            litellm_params['output_cost_per_token'] = adjusted_output_cost
                            self.logger.debug(f"Output cost change for {model_id}: {current_output_cost} → {adjusted_output_cost}")
                    else:
                        # Current cost is None, always update
                        output_changed = True
                        change_info['changes']['output_cost'] = {
                            'old': current_output_cost,
                            'new': adjusted_output_cost
                        }
                        litellm_params['output_cost_per_token'] = adjusted_output_cost
                        self.logger.debug(f"Output cost change for {model_id}: {current_output_cost} → {adjusted_output_cost}")

                # Record changes if any occurred
                if input_changed or output_changed:
                    cost_changes.append(change_info)
                    self.logger.info(f"Cost update for {model_id} (name: {model_name})")

                    if input_changed:
                        old_val = current_input_cost if current_input_cost is not None else "None"
                        new_val = change_info['changes']['input_cost']['new']

                        # Calculate percentage change
                        if current_input_cost is not None and current_input_cost != 0:
                            pct_change = ((new_val - current_input_cost) / current_input_cost) * 100
                            pct_str = f" ({pct_change:+.1f}%)"
                        else:
                            pct_str = ""

                        self.logger.info(f"  Input cost: {old_val} → {new_val}{pct_str}")

                    if output_changed:
                        old_val = current_output_cost if current_output_cost is not None else "None"
                        new_val = change_info['changes']['output_cost']['new']

                        # Calculate percentage change
                        if current_output_cost is not None and current_output_cost != 0:
                            pct_change = ((new_val - current_output_cost) / current_output_cost) * 100
                            pct_str = f" ({pct_change:+.1f}%)"
                        else:
                            pct_str = ""

                        self.logger.info(f"  Output cost: {old_val} → {new_val}{pct_str}")

        if cost_changes:
            self.logger.info(f"Identified {len(cost_changes)} models with cost updates")
        else:
            self.logger.info("No cost updates needed - all costs are current")

        return config, cost_changes
    
    def preview_changes(self, invalid_models: List[Tuple[int, str, str]]) -> None:
        """Preview what changes would be made in dry-run mode."""
        if not invalid_models:
            self.logger.info("[DRY-RUN] No invalid Vercel models found. No changes needed.")
            return
        
        self.logger.info(f"[DRY-RUN] Would remove the following invalid Vercel models:")
        for _, model_id, model_name in invalid_models:
            self.logger.info(f"  - Model: {model_id} (model_name: {model_name})")
        
        self.logger.info(f"[DRY-RUN] {len(invalid_models)} model entries would be removed from {self.config_path}")
        self.logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
    
    def remove_invalid_entries(self, config: Dict[str, Any],
                             invalid_models: List[Tuple[int, str, str]]) -> Dict[str, Any]:
        """Remove invalid model entries from the configuration."""
        if not invalid_models:
            self.logger.info("No invalid models to remove")
            return config
        
        # Sort indices in descending order to avoid index shifting issues
        invalid_indices = sorted([index for index, _, _ in invalid_models], reverse=True)
        
        model_list = config['model_list']
        removed_count = 0
        
        for index in invalid_indices:
            if 0 <= index < len(model_list):
                removed_model = model_list.pop(index)
                removed_count += 1
                
                # Log the removal
                model_name = removed_model.get('model_name', 'unnamed')
                model_id = removed_model.get('litellm_params', {}).get('model', 'unknown')
                self.logger.info(f"Removed invalid model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Removed {removed_count} invalid model entries from configuration")
        return config
    
    def add_model_to_config(self, config: Dict[str, Any], model_id: str) -> bool:
        """Add a new model to the configuration if it doesn't already exist."""
        try:
            # Fetch available models to get pricing info
            api_models = self.fetch_available_models()
            
            if model_id not in api_models:
                self.logger.error(f"Model '{model_id}' not found in available models")
                return False
            
            # Check if model already exists in config
            model_list = config.get('model_list', [])
            full_model_id = f"vercel_ai_gateway/{model_id}"
            
            for model_entry in model_list:
                litellm_params = model_entry.get('litellm_params', {})
                existing_model_id = litellm_params.get('model', '')
                if existing_model_id == full_model_id:
                    self.logger.warning(f"Model '{full_model_id}' already exists in configuration")
                    return False
            
            # Get model info from API
            api_model_info = api_models[model_id]
            
            # Generate model name
            base_name = model_id.replace('/', '-').replace(':', '-')
            model_name = f"vc-{base_name}"
            
            # Handle new model name conflicts
            counter = 1
            original_name = model_name
            while any(m.get('model_name') == model_name for m in model_list):
                model_name = f"{original_name}-{counter}"
                counter += 1
            
            # Get pricing information
            input_cost = api_model_info.get('input_cost')
            output_cost = api_model_info.get('output_cost')
            
            # Handle free models - if cost is 0.0 or None, use 1e-09 for LiteLLM compatibility
            if input_cost is None or input_cost == 0.0:
                input_cost = 1.0e-09
            if output_cost is None or output_cost == 0.0:
                output_cost = 1.0e-09
            
            # Create new model entry
            new_entry = {
                'model_name': model_name,
                'litellm_params': {
                    'model': full_model_id,
                    'input_cost_per_token': input_cost,
                    'output_cost_per_token': output_cost
                }
            }
            
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] Would add model: {full_model_id} (name: {model_name})")
                self.logger.info(f"[DRY-RUN]   Input cost: {input_cost}, Output cost: {output_cost}")
                return True
            
            # Add to model list
            model_list.append(new_entry)
            self.logger.info(f"Added new model: {full_model_id} (name: {model_name})")
            self.logger.info(f"  Input cost: {input_cost}, Output cost: {output_cost}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding model '{model_id}': {e}")
            return False
    
    def sort_model_list(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sort the model list alphabetically by model_name."""
        model_list = config.get('model_list', [])
        
        # Sort by model_name first, then by litellm_params.model as secondary key
        sorted_models = sorted(
            model_list, 
            key=lambda x: (
                x.get('model_name', ''), 
                x.get('litellm_params', {}).get('model', '')
            )
        )
        
        config['model_list'] = sorted_models
        self.logger.info("Sorted model list alphabetically")
        return config
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the updated configuration to file."""
        if self.dry_run:
            self.logger.debug("[DRY-RUN] Skipping config file save")
            return
        
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(config, file, sort_keys=False, default_flow_style=False,
                         allow_unicode=True, indent=2)
            
            if self.verbose:
                self.logger.debug(f"Updated configuration saved to {self.config_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            raise
    
    def run_cleanup(self, add_models: Optional[List[str]] = None) -> int:
        """Run the complete cleanup process."""
        try:
            # Load configuration
            config = self.load_config()
            
            # Fetch available models from API
            api_models = self.fetch_available_models()
            
            # Extract current Vercel models from config
            config_models = self.extract_vercel_models(config)
            
            # Validate models and identify invalid ones
            invalid_models = self.validate_models(config_models, api_models)
            
            # Preview changes if dry run
            if self.dry_run and invalid_models:
                self.preview_changes(invalid_models)
            
            # Remove invalid models
            if invalid_models:
                config = self.remove_invalid_entries(config, invalid_models)
            
            # Validate and update costs
            config, cost_changes = self.validate_and_update_costs(config, api_models)
            
            # Add new models if requested
            if add_models:
                self.logger.info(f"Adding {len(add_models)} new models...")
                for model_id in add_models:
                    self.add_model_to_config(config, model_id)
            
            # Sort model list
            config = self.sort_model_list(config)
            
            # Save configuration
            self.save_config(config)
            
            # Print summary
            total_changes = len(invalid_models) + len(cost_changes) + (len(add_models or []))
            if total_changes > 0:
                self.logger.info(f"Cleanup completed")
                self.logger.info(f"Models removed: {len(invalid_models)}")
                self.logger.info(f"Models updated: {len(cost_changes)}")
                self.logger.info(f"Models added: {len(add_models or [] if add_models else [])}")
                self.logger.info(f"Total changes: {total_changes}")
            else:
                self.logger.info("No changes needed - all Vercel models are valid and costs are current")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup Vercel AI Gateway models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run cleanup on default config.yaml
  %(prog)s --config my.yaml   # Run cleanup on custom config file
  %(prog)s --dry-run          # Preview changes without modifying file
  %(prog)s --verbose          # Show detailed logging output
  %(prog)s --add-model alibaba/qwen-3-14b     # Add new model(s)
  %(prog)s --add-model alibaba/qwen-3-14b alibaba/qwen-3-30b  # Add multiple models
        """
    )
    
    parser.add_argument(
        '--config',
        default=DEFAULT_CONFIG_FILE,
        help=f'Path to config file (default: {DEFAULT_CONFIG_FILE})'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying the config file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    parser.add_argument(
        '--add-model',
        nargs='+',
        metavar='MODEL_ID',
        help='Add new model(s) to configuration (e.g., alibaba/qwen-3-14b)'
    )
    
    args = parser.parse_args()
    
    # Create and run cleaner
    cleaner = VercelModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(args.add_model)


if __name__ == "__main__":
    sys.exit(main())
