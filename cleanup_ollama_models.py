#!/usr/bin/env python3
"""
Ollama Model Cleanup Script for LiteLLM Config

This script validates Ollama models in a LiteLLM config.yaml file against
the current Ollama API and:
1. Removes any invalid model entries
2. Adds new models when requested

Note: Ollama models don't have pricing information, so they use a nominal cost of 1e-09 for LiteLLM compatibility.

Usage:
    python cleanup_ollama_models.py [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_ollama_models.py --add-model "llama3.2"

Author: Generated for LiteLLM Config Management
"""

import sys
from typing import Dict, Any, Optional

import requests

from cleanup_base import (
    ConfigDrivenModelCleaner,
    create_provider_main,
)


class OllamaModelCleaner(ConfigDrivenModelCleaner):
    """Cleaner for Ollama models."""

    def __init__(self, config_path: str, dry_run: bool = False, verbose: bool = False):
        """Initialize the Ollama model cleaner."""
        super().__init__("ollama", config_path, dry_run, verbose)

    def fetch_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Fetch available models from Ollama API.

        Ollama API returns models in a "models" array with "name" field
        instead of the standard "data" array with "id" field.
        """
        try:
            self.logger.info(f"Fetching models from API: {self.API_URL}")
            headers = self._build_api_headers()

            # Direct fetch without using fetch_models_from_api since Ollama uses "models" not "data"
            response = requests.get(self.API_URL, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            available_models = {}
            # Ollama uses "models" array instead of "data"
            models_list = data.get("models", [])

            for model in models_list:
                if isinstance(model, dict) and "name" in model:
                    model_info = self.parse_api_model(model)
                    available_models[model_info["id"]] = model_info

            self.logger.info(
                f"Fetched {len(available_models)} available models from {self.PROVIDER_NAME} API"
            )
            return available_models

        except requests.RequestException as e:
            self.logger.error(
                f"Error fetching models from {self.PROVIDER_NAME} API: {e}"
            )
            raise
        except Exception as e:
            self.logger.error(
                f"Error fetching models from {self.PROVIDER_NAME} API: {e}"
            )
            raise

    def parse_api_model(self, model: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a model from the Ollama API response.

        Ollama uses "name" field instead of "id" for the model identifier.
        """
        # Ollama uses "name" field as the model ID
        model_id = model.get("name", "")

        model_info = {
            "id": model_id,
            "input_cost": None,
            "output_cost": None,
            "model_info": model.get("model_info"),
        }

        # Check for default cost (Ollama models don't have pricing)
        default_cost = self._pricing_config.get("default_cost")
        if default_cost is not None:
            model_info["input_cost"] = float(default_cost)
            model_info["output_cost"] = float(default_cost)

        return model_info


main = create_provider_main(
    OllamaModelCleaner,
    "Validate and cleanup Ollama models in LiteLLM config",
    """
Note: Ollama models don't have pricing information in the API, so they use a nominal cost of 1e-09 for LiteLLM compatibility.

Examples:
  %(prog)s                                              # Run cleanup on default config.yaml
  %(prog)s --config my.yaml                             # Run cleanup on custom config file
  %(prog)s --dry-run                                    # Preview changes without modifying file
  %(prog)s --add-model "llama3.2"                       # Add a single model
    """,
)


if __name__ == "__main__":
    sys.exit(main())
