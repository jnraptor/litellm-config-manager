#!/usr/bin/env python3
"""
Update Order Values in LiteLLM Config

This script updates the litellm_params/order value for all models in config.yaml
based on the provider configuration in providers.yaml.

- Synthetic provider models get order: 1 (highest priority)
- All other provider models get order: 2

Usage:
    python update_order_config.py [--config config.yaml] [--dry-run] [--verbose]
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("update_order_config")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


def load_providers_config(providers_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load the providers configuration file."""
    try:
        logger.info(f"Loading providers configuration from {providers_path}")
        with open(providers_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        if not config or 'providers' not in config:
            raise ValueError("Invalid providers config: missing 'providers' section")
        
        # Create a mapping of provider name to order
        provider_orders = {}
        for provider_name, provider_config in config.get('providers', {}).items():
            provider_orders[provider_name] = provider_config.get('order', 2)
        
        logger.info(f"Loaded {len(provider_orders)} provider configurations")
        return provider_orders
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in providers configuration: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Providers configuration file not found: {providers_path}")
        raise


def load_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """Load the LiteLLM configuration file."""
    try:
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        if not config:
            raise ValueError("Configuration file is empty or invalid")
        if 'model_list' not in config:
            raise ValueError("Configuration file missing 'model_list' section")
        
        logger.info(f"Loaded configuration with {len(config['model_list'])} models")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise


def determine_provider_for_model(model_entry: Dict[str, Any], 
                                 provider_orders: Dict[str, int],
                                 logger: logging.Logger) -> Optional[str]:
    """
    Determine which provider a model belongs to based on its configuration.
    
    Returns the provider name or None if it cannot be determined.
    """
    litellm_params = model_entry.get('litellm_params', {})
    model_id = litellm_params.get('model', '')
    api_base = str(litellm_params.get('api_base', ''))
    
    # Check for synthetic provider (order 1)
    if 'SYNTHETIC_API_BASE' in api_base:
        return 'synthetic'
    
    # Check for openrouter provider (order 2)
    if model_id.startswith('openrouter/'):
        return 'openrouter'
    
    # Check for nvidia provider (order 2)
    if model_id.startswith('nvidia_nim/'):
        return 'nvidia'
    
    # Check for vercel provider (order 2)
    if model_id.startswith('vercel_ai_gateway/'):
        return 'vercel'
    
    # Check for requesty provider (order 2)
    if 'router.requesty.ai' in api_base:
        return 'requesty'
    
    # Check for poe provider (order 2)
    if 'api.poe.com' in api_base:
        return 'poe'
    
    # Check for nano_gpt provider (order 2)
    if 'NANOGPT_API_BASE' in api_base:
        return 'nano_gpt'
    
    # Default to order 2 if provider cannot be determined
    logger.debug(f"Could not determine provider for model: {model_id}")
    return None


def update_order_values(config: Dict[str, Any], 
                       provider_orders: Dict[str, int],
                       logger: logging.Logger) -> tuple[Dict[str, Any], Dict[str, int]]:
    """
    Update the order values for all models in the configuration.
    
    Returns:
        Tuple of (updated_config, stats_dict)
    """
    model_list = config.get('model_list', [])
    stats = {
        'total_models': len(model_list),
        'updated_models': 0,
        'synthetic_models': 0,
        'other_models': 0,
        'unchanged_models': 0,
        'unknown_provider_models': 0
    }
    
    for model_entry in model_list:
        if not isinstance(model_entry, dict):
            continue
        
        litellm_params = model_entry.get('litellm_params', {})
        if not litellm_params:
            continue
        
        provider = determine_provider_for_model(model_entry, provider_orders, logger)
        
        if provider:
            order = provider_orders.get(provider, 2)
        else:
            order = 2  # Default to order 2
            stats['unknown_provider_models'] += 1
        
        current_order = litellm_params.get('order')
        
        if current_order != order:
            litellm_params['order'] = order
            stats['updated_models'] += 1
            model_name = model_entry.get('model_name', 'unnamed')
            model_id = litellm_params.get('model', '')
            logger.debug(f"Updated order for {model_name} ({model_id}): {current_order} → {order}")
        else:
            stats['unchanged_models'] += 1
        
        # Track statistics
        if provider == 'synthetic':
            stats['synthetic_models'] += 1
        elif provider:
            stats['other_models'] += 1
    
    return config, stats


def save_config(config: Dict[str, Any], config_path: str, logger: logging.Logger) -> None:
    """Save the updated configuration to file."""
    try:
        logger.info(f"Saving configuration to {config_path}")
        
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False,
                     allow_unicode=True, width=1000)
        
        logger.info("Configuration saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def generate_report(stats: Dict[str, int], logger: logging.Logger) -> None:
    """Generate a summary report of the update operation."""
    logger.info("=" * 60)
    logger.info("Order Value Update Summary")
    logger.info("=" * 60)
    logger.info(f"Total models processed: {stats['total_models']}")
    logger.info(f"Models updated: {stats['updated_models']}")
    logger.info(f"Models unchanged: {stats['unchanged_models']}")
    logger.info(f"  - Synthetic models (order 1): {stats['synthetic_models']}")
    logger.info(f"  - Other provider models (order 2): {stats['other_models']}")
    logger.info(f"  - Unknown provider models (order 2): {stats['unknown_provider_models']}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Update order values in LiteLLM config based on provider configuration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Update order values in default config.yaml
  %(prog)s --config my.yaml             # Update order values in custom config file
  %(prog)s --dry-run                    # Preview changes without modifying file
  %(prog)s --verbose                    # Enable verbose logging
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        default="config.yaml",
        help='Path to LiteLLM configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--providers', '-p',
        default="providers.yaml",
        help='Path to providers configuration file (default: providers.yaml)'
    )
    
    parser.add_argument(
        '--dry-run', '-d',
        action='store_true',
        help='Preview changes without modifying the configuration file'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging output'
    )
    
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    try:
        # Load configurations
        provider_orders = load_providers_config(args.providers, logger)
        config = load_config(args.config, logger)
        
        # Update order values
        config, stats = update_order_values(config, provider_orders, logger)
        
        # Display results
        generate_report(stats, logger)
        
        # Save if not dry run
        if not args.dry_run:
            if stats['updated_models'] > 0:
                save_config(config, args.config, logger)
                logger.info("✅ Order values updated successfully")
            else:
                logger.info("✅ No changes needed - all order values are already correct")
        else:
            logger.info("[DRY-RUN] No changes made to file. Use without --dry-run to apply changes.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to update order values: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
