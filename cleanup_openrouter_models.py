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


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/models"
OPENROUTER_EMBEDDINGS_API_URL = "https://openrouter.ai/api/v1/embeddings/models"


class OpenRouterModelCleaner(BaseModelCleaner):
    """Cleaner for OpenRouter models."""
    
    PROVIDER_NAME = "OpenRouter"
    API_URL = OPENROUTER_API_URL
    MODEL_PREFIX = "openrouter/"
    
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
        return model_id.replace(self.MODEL_PREFIX, '')
    
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
