#!/usr/bin/env python3
"""
Vercel AI Gateway Model Cleanup and Cost Update Script for LiteLLM Config

This script validates Vercel AI Gateway models in a LiteLLM config.yaml file against
the current Vercel AI Gateway API and:
1. Removes any invalid model entries
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing

Usage:
    python cleanup_vercel_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_vercel_models.py --add-model "alibaba/qwen-3-14b"

Author: Generated for LiteLLM Config Management
"""

import argparse
import sys
from typing import Dict, List, Tuple, Any, Optional

from cleanup_base import (
    BaseModelCleaner,
    setup_common_args,
    validate_model_name_arg,
    fetch_models_from_api,
    adjust_cost_for_free_model,
)


VERCEL_API_URL = "https://ai-gateway.vercel.sh/v1/models"


class VercelModelCleaner(BaseModelCleaner):
    """Cleaner for Vercel AI Gateway models."""
    
    PROVIDER_NAME = "Vercel"
    API_URL = VERCEL_API_URL
    MODEL_PREFIX = "vercel_ai_gateway/"
    
    def extract_provider_models(self, config: Dict[str, Any]) -> List[Tuple[int, str, str]]:
        """Extract Vercel AI Gateway models from the configuration."""
        vercel_models = []
        model_list = config.get('model_list', [])
        
        for index, model_entry in enumerate(model_list):
            if not isinstance(model_entry, dict):
                continue
            
            litellm_params = model_entry.get('litellm_params', {})
            model_id = litellm_params.get('model', '')
            
            if model_id.startswith(self.MODEL_PREFIX):
                model_name = model_entry.get('model_name', 'unnamed')
                vercel_models.append((index, model_id, model_name))
                self.logger.debug(f"Found Vercel model: {model_id} (name: {model_name})")
        
        self.logger.info(f"Found {len(vercel_models)} Vercel models in configuration")
        return vercel_models

    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Fetch available models with pricing from Vercel AI Gateway API."""
        try:
            data = fetch_models_from_api(self.API_URL, self.logger)
            
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
                    
                    pricing = model.get('pricing', {})
                    if isinstance(pricing, dict):
                        try:
                            if 'input' in pricing and pricing['input']:
                                model_info['input_cost'] = float(pricing['input'])
                            if 'output' in pricing and pricing['output']:
                                model_info['output_cost'] = float(pricing['output'])
                        except (ValueError, TypeError) as e:
                            self.logger.warning(f"Invalid pricing for {model_id}: {e}")
                    
                    available_models[model_id] = model_info
            
            self.logger.info(f"Fetched {len(available_models)} available models from Vercel AI Gateway API")
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error fetching models from Vercel API: {e}")
            raise
    
    def get_api_model_id(self, model_id: str) -> str:
        """Extract the API model ID from the config model ID."""
        return model_id.replace(self.MODEL_PREFIX, '')
    
    def generate_model_name(self, model_id: str, prefix: str = "vc-") -> str:
        """Generate model name with vc- prefix for Vercel models."""
        return super().generate_model_name(model_id, prefix)
    
    def create_model_entry(self, model_id: str, api_model_info: Dict[str, Any],
                          model_name: str) -> Dict[str, Any]:
        """Create a new Vercel model entry for the config."""
        input_cost = adjust_cost_for_free_model(api_model_info.get('input_cost'))
        output_cost = adjust_cost_for_free_model(api_model_info.get('output_cost'))
        
        if input_cost is None:
            input_cost = 1.0e-09
        if output_cost is None:
            output_cost = 1.0e-09
        
        return {
            'model_name': model_name,
            'litellm_params': {
                'model': f'{self.MODEL_PREFIX}{model_id}',
                'input_cost_per_token': input_cost,
                'output_cost_per_token': output_cost
            }
        }
    
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
        description='Validate and cleanup Vercel AI Gateway models in LiteLLM config',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml             # Run cleanup on custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --add-model alibaba/qwen-3-14b  # Add new model(s)
        """
    )
    
    setup_common_args(parser)
    args = parser.parse_args()
    validate_model_name_arg(args, parser)
    
    cleaner = VercelModelCleaner(
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    return cleaner.run_cleanup(add_models=args.add_model, custom_model_name=args.model_name)


if __name__ == "__main__":
    sys.exit(main())
