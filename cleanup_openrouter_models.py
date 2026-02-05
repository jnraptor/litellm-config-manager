#!/usr/bin/env python3
"""
OpenRouter Model Cleanup and Cost Update Script for LiteLLM Config

This script validates OpenRouter models in a LiteLLM config.yaml file against
the current OpenRouter API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token)
3. Supports both regular and embedding models

Usage:
    python cleanup_openrouter_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_openrouter_models.py --add-model "qwen/qwen3-14b"

Author: Generated for LiteLLM Config Management
"""

import argparse
import sys
import requests
from typing import Dict, List, Tuple, Any, Optional

from cleanup_base import (
    BaseModelCleaner,
    setup_common_args,
    validate_model_name_arg,
    fetch_models_from_api,
    adjust_cost_for_free_model,
)
from typing import Set


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_EMBEDDINGS_API_URL = "https://openrouter.ai/api/v1/embeddings/models"


class OpenRouterModelCleaner(BaseModelCleaner):
    """Cleaner for OpenRouter models."""
    
    PROVIDER_NAME = "OpenRouter"
    API_URL = OPENROUTER_API_URL
    MODEL_PREFIX = "openrouter/"
    SPECIAL_MODELS: Set[str] = {"openrouter/free"}
    
    def extract_provider_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """Extract OpenRouter models from the configuration."""
        openrouter_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
            
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            
            if model_id.startswith(self.MODEL_PREFIX):
                model_name = model_entry.get('model_name', 'unnamed')
                openrouter_models.append((index, model_id, model_name))
                self.logger.debug(f"Found OpenRouter model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(openrouter_models)} OpenRouter models in configuration")
        return openrouter_models

    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch available models with pricing from OpenRouter API."""
        try:
            available_models = {}
            
            data = fetch_models_from_api(self.API_URL, self.logger)
            
            for model in data['data']:
                if isinstance(model, dict) and 'id' in model:
                    model_id = model['id']
                    model_info = {
                        'id': model_id,
                        'input_cost': None,
                        'output_cost': None,
                        'model_info': None
                    }
                    
                    pricing = model.get('pricing', {})
                    if isinstance(pricing, dict):
                        prompt_cost = pricing.get('prompt')
                        if prompt_cost is not None:
                            try:
                                model_info['input_cost'] = float(prompt_cost)
                            except (ValueError, TypeError):
                                self.logger.debug(f"Invalid prompt cost for {model_id}")
                        
                        completion_cost = pricing.get('completion')
                        if completion_cost is not None:
                            try:
                                model_info['output_cost'] = float(completion_cost)
                            except (ValueError, TypeError):
                                self.logger.debug(f"Invalid completion cost for {model_id}")
                    
                    available_models[model_id] = model_info
            
            # Fetch embedding models
            try:
                self.logger.debug("Fetching embedding models from OpenRouter API...")
                response = requests.get(OPENROUTER_EMBEDDINGS_API_URL, timeout=30)
                response.raise_for_status()
                embed_data = response.json()
                
                if 'data' in embed_data:
                    for model in embed_data['data']:
                        if isinstance(model, dict) and 'id' in model:
                            model_id = model['id']
                            model_info = {
                                'id': model_id,
                                'input_cost': None,
                                'output_cost': None,
                                'model_info': {'mode': 'embedding'}
                            }
                            
                            pricing = model.get('pricing', {})
                            if isinstance(pricing, dict):
                                prompt_cost = pricing.get('prompt')
                                if prompt_cost is not None:
                                    try:
                                        model_info['input_cost'] = float(prompt_cost)
                                    except (ValueError, TypeError):
                                        pass
                                
                                completion_cost = pricing.get('completion')
                                if completion_cost is not None:
                                    try:
                                        model_info['output_cost'] = float(completion_cost)
                                    except (ValueError, TypeError):
                                        pass
                            
                            available_models[model_id] = model_info
                    
                    self.logger.debug("Fetched embedding models from OpenRouter API")
            except requests.RequestException as e:
                self.logger.warning(f"Could not fetch embedding models: {e}")
            
            self.logger.info(f"Fetched {len(available_models)} available models from OpenRouter API")
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error fetching models from OpenRouter API: {e}")
            raise
    
    def get_api_model_id(self, model_id: str) -> str:
        """Extract the API model ID from the config model ID."""
        if model_id.startswith(self.MODEL_PREFIX):
            return model_id[len(self.MODEL_PREFIX):]
        return model_id
    
    def generate_model_name(self, model_id: str, prefix: str = "or-") -> str:
        """Generate model name with or- prefix for OpenRouter models."""
        return super().generate_model_name(model_id, prefix)
    
    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any],
                          model_name: str) -> Dict[str, Any]:
        """Create a new OpenRouter model entry for the config."""
        input_cost = adjust_cost_for_free_model(api_model_info.get('input_cost'))
        output_cost = adjust_cost_for_free_model(api_model_info.get('output_cost'))
        
        entry = {
            'model_name': model_name,
            'litellm_params': {
                'model': f'{self.MODEL_PREFIX}{model_id}'
            }
        }
        
        if input_cost is not None:
            entry['litellm_params']['input_cost_per_token'] = input_cost
        if output_cost is not None:
            entry['litellm_params']['output_cost_per_token'] = output_cost
        
        model_info = api_model_info.get('model_info')
        if model_info:
            entry['model_info'] = model_info
        
        return entry
    
    def check_for_free_variant(self, model_id: str, 
                               api_models: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        Check if a free variant of the model exists in the API.
        
        Args:
            model_id: The base model ID (e.g., "qwen/qwen3-next-80b-a3b-instruct")
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
                           custom_model_name: Optional[str] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Add one or more models to the configuration, including free variants if they exist.
        
        Args:
            config: The configuration dictionary
            model_ids: List of model IDs to add
            api_models: Dict of available models from API
            custom_model_name: Optional custom name (only for single model)
            
        Returns:
            Tuple of (updated_config, list_of_added_model_ids)
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
                        
                        self.logger.info(f"Added free variant '{free_variant_id}' with name '{model_name}'")
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
        invalid_models = []
        duplicate_models = []
        
        for model_id in model_ids:
            model_info = self.find_model_in_api(model_id, api_models)
            if not model_info:
                invalid_models.append(model_id)
                continue
            
            if model_id in existing_model_ids:
                duplicate_models.append(model_id)
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
            
            model_preview = {
                'id': model_id,
                'name': model_name,
                'input_cost': adjust_cost_for_free_model(model_info.get('input_cost')),
                'output_cost': adjust_cost_for_free_model(model_info.get('output_cost')),
                'free_variant': None
            }
            
            # Add free variant info if it exists
            if free_variant_id:
                if free_variant_id in existing_model_ids:
                    model_preview['free_variant'] = {'id': free_variant_id, 'status': 'already_exists'}
                else:
                    free_model_info = self.find_model_in_api(free_variant_id, api_models)
                    if free_model_info:
                        model_preview['free_variant'] = {
                            'id': free_variant_id,
                            'name': model_name,
                            'input_cost': adjust_cost_for_free_model(free_model_info.get('input_cost')),
                            'output_cost': adjust_cost_for_free_model(free_model_info.get('output_cost')),
                            'status': 'will_add'
                        }
            
            valid_models.append(model_preview)
            existing_names.append(model_name)
        
        if invalid_models:
            self.logger.error(f"[DRY-RUN] {len(invalid_models)} model(s) not found in {self.PROVIDER_NAME} API: {', '.join(invalid_models)}")
        
        if duplicate_models:
            self.logger.warning(f"[DRY-RUN] {len(duplicate_models)} model(s) already exist in configuration: {', '.join(duplicate_models)}")
        
        if valid_models:
            self.logger.info(f"[DRY-RUN] Would add {len(valid_models)} base model(s):")
            for model in valid_models:
                self.logger.info(f"[DRY-RUN]   - Model '{model['id']}' with name '{model['name']}'")
                self.logger.info(f"[DRY-RUN]     Input cost: {model['input_cost']}")
                self.logger.info(f"[DRY-RUN]     Output cost: {model['output_cost']}")
                
                if model['free_variant']:
                    if model['free_variant']['status'] == 'already_exists':
                        self.logger.info(f"[DRY-RUN]     Free variant '{model['free_variant']['id']}' already exists in config")
                    else:
                        self.logger.info(f"[DRY-RUN]     Free variant '{model['free_variant']['id']}' will also be added with name '{model['free_variant']['name']}'")
                        self.logger.info(f"[DRY-RUN]       Input cost: {model['free_variant']['input_cost']}")
                        self.logger.info(f"[DRY-RUN]       Output cost: {model['free_variant']['output_cost']}")
        else:
            self.logger.info("[DRY-RUN] No valid models to add.")

    def run_cleanup(self, add_models: Optional[List[str]] = None,
                   custom_model_name: Optional[str] = None) -> int:
        """Run the complete cleanup process."""
        try:
            config = self.load_config()
            api_models = self.fetch_available_models()
            
            if add_models:
                if self.dry_run:
                    self.preview_add_model(add_models, api_models, custom_model_name)
                    return 0
                else:
                    config, added_models = self.add_model_to_config(
                        config, add_models, api_models, custom_model_name)
                    if added_models:
                        config, was_sorted = self.sort_model_list(config)
                        self.save_config(config)
                        self.logger.info(f"✅ Successfully added {len(added_models)} model(s)")
                    else:
                        self.logger.warning("⚠️ No models were added")
                    return 0
            
            config_models = self.extract_provider_models(config)
            invalid_models = self.validate_models(config_models, api_models)
            config, cost_changes = self.validate_and_update_costs(config, config_models, api_models)
            config, was_sorted = self.sort_model_list(config)
            
            if self.dry_run:
                self.preview_sort_changes(config)
                self.preview_changes(invalid_models)
                self.preview_cost_changes(cost_changes)
            else:
                if invalid_models:
                    config = self.remove_invalid_entries(config, invalid_models)
                
                if invalid_models or cost_changes or was_sorted:
                    self.save_config(config)
            
            self.generate_report(invalid_models, cost_changes, was_sorted)
            return 0
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Validate and cleanup OpenRouter models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model qwen/qwen3-14b   # Add new model(s)
        """
    )
    
    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)
    
    cleaner = OpenRouterModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
