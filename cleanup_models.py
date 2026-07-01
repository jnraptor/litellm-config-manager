#!/usr/bin/env python3
"""
Unified Model Cleanup and Cost Update Script for LiteLLM Config

This script validates models from multiple providers in a LiteLLM config.yaml file
against their current APIs and:
1. Removes any invalid model entries (except special models)
2. Updates model costs (input_cost_per_token and output_cost_per_token) when they differ from API pricing
3. Sorts the model list alphabetically
4. Adds new models when requested
5. Supports both regular and embedding models (where applicable)
6. Supports mapped models for simplified multi-provider addition
7. Supports deleting models by model_name from the configuration
8. Supports deleting providers, removing their models from config.yaml and
   disabling them in providers.yaml
9. Prunes ``special_models`` entries in ``providers.yaml`` that are now
   available through the provider's normal models source

Supported providers: openrouter, requesty, vercel, poe, nvidia, kilo, ollama, opencode-zen, opencode-go, all

Usage:
    python cleanup_models.py --provider openrouter [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_models.py --provider all [--config config.yaml] [--dry-run] [--verbose]
    python cleanup_models.py --provider requesty --add-model "model1 model2"
    python cleanup_models.py --provider all --add-mapped-model glm-5 [--dry-run]
    python cleanup_models.py --provider all --delete-model "model_name" [--dry-run]
    python cleanup_models.py --provider all --delete-provider "openrouter" [--dry-run]

Author: Unified script for LiteLLM Config Management
"""

import argparse
import sys
import yaml
from typing import Dict, List, Tuple, Any, Optional, Set
from pathlib import Path

# Import shared utilities from cleanup_base
from cleanup_base import (
    setup_logging,
    ConfigDrivenModelCleaner,
    ProviderConfigLoader,
    ModelMappingLoader,
    sort_model_list as base_sort_model_list,
    ValidationReport,
    _print_validation_report,
)
from dotenv import load_dotenv

# Import provider-specific cleaners
from cleanup_ollama_models import OllamaModelCleaner


class UnifiedModelCleaner:
    """
    Orchestrator for cleaning up invalid models across multiple providers.

    Delegates all provider-specific logic to ConfigDrivenModelCleaner instances,
    one per provider. This class handles multi-provider orchestration, config
    loading/saving, and result reporting.
    """

    def __init__(
        self,
        config_path: str,
        provider_names: List[str],
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.config_path = Path(config_path)
        self.dry_run = dry_run
        self.verbose = verbose
        self.logger = setup_logging(verbose, "UnifiedModelCleaner")
        self.provider_names = provider_names

        # Create per-provider cleaners that handle all provider-specific logic
        # Use custom cleaners for providers that need special handling
        self.cleaners: Dict[str, ConfigDrivenModelCleaner] = {}
        for name in provider_names:
            if name == "ollama":
                self.cleaners[name] = OllamaModelCleaner(config_path, dry_run, verbose)
            else:
                self.cleaners[name] = ConfigDrivenModelCleaner(
                    name, config_path, dry_run, verbose
                )

    def load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file."""
        try:
            self.logger.info(f"Loading configuration from {self.config_path}")
            if not self.config_path.exists():
                raise FileNotFoundError(
                    f"Configuration file not found: {self.config_path}"
                )

            with open(self.config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            if not config:
                raise ValueError("Configuration file is empty or invalid")
            if "model_list" not in config:
                raise ValueError("Configuration file missing 'model_list' section")

            self.logger.debug(
                f"Loaded configuration with {len(config['model_list'])} models"
            )
            return config

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save the updated configuration back to the file (with backup)."""
        if self.dry_run:
            self.logger.info("DRY RUN: Would save configuration to file")
            return

        backup_path = self.config_path.with_suffix(".yaml.backup")
        try:
            self.logger.info(f"Saving configuration to {self.config_path}")
            if self.config_path.exists():
                self.config_path.rename(backup_path)
                self.logger.info(f"Created backup at {backup_path}")

            with open(self.config_path, "w", encoding="utf-8") as file:
                yaml.dump(config, file, default_flow_style=False, sort_keys=False)
            self.logger.info("Configuration saved successfully")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            if backup_path.exists():
                backup_path.rename(self.config_path)
                self.logger.info("Restored configuration from backup")
            raise

    def sort_model_list(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Sort the model list by model_name alphabetically."""
        if "model_list" not in config or not config["model_list"]:
            self.logger.info("No model list found or model list is empty")
            return config, False

        model_list = config["model_list"]
        sorted_model_list, was_sorted = base_sort_model_list(model_list, self.logger)

        if was_sorted:
            config["model_list"] = sorted_model_list

        return config, was_sorted

    def cleanup_provider(
        self,
        provider_name: str,
        add_models: Optional[List[str]] = None,
        custom_model_name: Optional[str] = None,
    ) -> Tuple[int, int, List[str]]:
        """Clean up models for a specific provider, delegating to its cleaner instance."""
        cleaner = self.cleaners[provider_name]
        try:
            config = self.load_config()

            if add_models:
                api_models = cleaner.fetch_available_models()
                if self.dry_run:
                    cleaner.preview_add_model(add_models, api_models, custom_model_name)
                    return 0, 0, [f"Previewed {len(add_models)} models for addition"]
                else:
                    config, added_models = cleaner.add_model_to_config(
                        config, add_models, api_models, custom_model_name
                    )
                    if added_models:
                        config, _ = self.sort_model_list(config)
                        self.save_config(config)
                        self.logger.info(
                            f"✅ Successfully added {len(added_models)} model(s)"
                        )
                    else:
                        self.logger.warning("⚠️ No models were added")
                    return 0, 0, [f"Added {len(added_models)} model(s)"]

            config, was_sorted = self.sort_model_list(config)
            api_models = cleaner.fetch_available_models()
            provider_models = cleaner.extract_provider_models(config)

            if not provider_models:
                if not self.dry_run and was_sorted:
                    self.save_config(config)
                self.logger.info(f"No {provider_name} models found to process")
                return 0, 0, []

            invalid_models = cleaner.validate_models(provider_models, api_models)
            free_order = cleaner.defaults.get("free_order")
            config, cost_changes, order_changed = cleaner.validate_and_update_costs(
                config, provider_models, api_models, cleaner.PROVIDER_ORDER, free_order
            )
            pruned_special_models = self._prune_provider_special_models(
                provider_name, api_models
            )

            if self.dry_run:
                cleaner.preview_sort_changes(config)
                cleaner.preview_changes(invalid_models)
                cleaner.preview_cost_changes(cost_changes)
                if order_changed:
                    self.logger.info("[DRY-RUN] Model order values would be updated")
                if pruned_special_models:
                    self.logger.info(
                        f"[DRY-RUN] Would remove {len(pruned_special_models)} "
                        f"special_models now available in source: "
                        f"{', '.join(pruned_special_models)}"
                    )
            else:
                if invalid_models:
                    config = cleaner.remove_invalid_entries(config, invalid_models)
                if was_sorted or invalid_models or cost_changes or order_changed:
                    self.save_config(config)

            cleaner.generate_report(
                invalid_models, cost_changes, was_sorted, order_changed
            )

            models_removed = len(invalid_models)
            models_updated = len(cost_changes)
            changes_made = []
            if models_removed > 0:
                changes_made.append(f"Removed {models_removed} invalid models")
            if models_updated > 0:
                changes_made.append(f"Updated costs for {models_updated} models")
            if was_sorted:
                changes_made.append("Sorted model list")
            if pruned_special_models:
                changes_made.append(
                    f"Pruned {len(pruned_special_models)} special_models now in source"
                )

            return models_removed, models_updated, changes_made

        except Exception as e:
            self.logger.error(f"Cleanup failed for {provider_name}: {e}")
            raise

    def cleanup_all_providers(
        self,
        add_models: Optional[Dict[str, List[str]]] = None,
        custom_model_name: Optional[str] = None,
    ) -> Dict[str, Tuple[int, int, List[str]]]:
        """Clean up models for all configured providers."""
        results = {}
        for provider_name in self.provider_names:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"Processing provider: {provider_name}")
            self.logger.info(f"{'=' * 60}")

            provider_add_models = add_models.get(provider_name) if add_models else None
            try:
                models_removed, models_updated, changes_made = self.cleanup_provider(
                    provider_name, provider_add_models, custom_model_name
                )
                results[provider_name] = (models_removed, models_updated, changes_made)
            except Exception as e:
                self.logger.error(f"Error processing provider {provider_name}: {e}")
                results[provider_name] = (0, 0, [f"Error: {str(e)}"])

        return results

    def add_mapped_model(
        self,
        model_key: str,
        mapping_loader: ModelMappingLoader,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
        """
        Add a mapped model across all configured providers.

        Uses the models.yaml mapping to add the same logical model from multiple
        providers with a shared display name for load balancing.

        Args:
            model_key: The canonical model key from models.yaml
            mapping_loader: The ModelMappingLoader instance
            config: Optional pre-loaded config (will load if not provided)

        Returns:
            Tuple of (updated_config, dict of provider -> list of added model IDs)
        """
        if config is None:
            config = self.load_config()

        mapping = mapping_loader.get_model_mapping(model_key)
        if not mapping:
            raise ValueError(
                f"Model '{model_key}' not found in models.yaml. "
                f"Available: {', '.join(mapping_loader.list_mapped_models())}"
            )

        display_name = mapping.get("display_name", model_key)
        provider_mappings = mapping.get("providers", {})

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Adding mapped model: {model_key}")
        self.logger.info(f"Display name: {display_name}")
        self.logger.info(f"{'=' * 60}")

        added_models_by_provider: Dict[str, List[str]] = {}

        for provider_name in self.provider_names:
            provider_model_id = provider_mappings.get(provider_name)
            if not provider_model_id:
                self.logger.info(f"\nSkipping {provider_name}: No mapping defined")
                continue

            self.logger.info(f"\nProcessing {provider_name}: {provider_model_id}")

            cleaner = self.cleaners.get(provider_name)
            if not cleaner:
                self.logger.warning(f"No cleaner found for {provider_name}")
                continue

            try:
                api_models = cleaner.fetch_available_models()

                if self.dry_run:
                    self.logger.info(
                        f"  DRY RUN: Would add {provider_model_id} as '{display_name}'"
                    )
                    if provider_model_id in api_models:
                        added_models_by_provider[provider_name] = [provider_model_id]
                    else:
                        self.logger.warning(
                            f"  Model {provider_model_id} not found in {provider_name} API"
                        )
                else:
                    config, added = cleaner.add_model_to_config(
                        config, [provider_model_id], api_models, display_name
                    )
                    if added:
                        added_models_by_provider[provider_name] = added
                        self.logger.info(
                            f"  ✅ Added {len(added)} model(s) from {provider_name}"
                        )
                    else:
                        self.logger.warning(f"  ⚠️ No models added from {provider_name}")

            except Exception as e:
                self.logger.error(f"  Error adding model from {provider_name}: {e}")
                added_models_by_provider[provider_name] = [f"Error: {str(e)}"]

        # Sort after all additions
        if not self.dry_run and any(added_models_by_provider.values()):
            config, _ = self.sort_model_list(config)
            self.save_config(config)

        return config, added_models_by_provider

    def delete_model_from_config(
        self,
        config: Dict[str, Any],
        model_names: List[str],
    ) -> Tuple[Dict[str, Any], int]:
        """Remove all entries matching the given model_names."""
        model_list = config.get("model_list", [])
        original_count = len(model_list)
        model_names_set = set(model_names)

        new_model_list = [
            entry
            for entry in model_list
            if entry.get("model_name") not in model_names_set
        ]

        removed = original_count - len(new_model_list)
        config["model_list"] = new_model_list

        for name in model_names:
            count = sum(1 for entry in model_list if entry.get("model_name") == name)
            if count == 0:
                self.logger.warning(f"Model '{name}' not found in configuration")
            else:
                self.logger.info(f"Removed {count} entry(s) for model '{name}'")

        return config, removed

    def _get_cleaner(self, provider_name: str) -> ConfigDrivenModelCleaner:
        """Return the cleaner for a provider, creating it on demand if needed."""
        if provider_name not in self.cleaners:
            if provider_name == "ollama":
                self.cleaners[provider_name] = OllamaModelCleaner(
                    str(self.config_path), self.dry_run, self.verbose
                )
            else:
                self.cleaners[provider_name] = ConfigDrivenModelCleaner(
                    provider_name, str(self.config_path), self.dry_run, self.verbose
                )
        return self.cleaners[provider_name]

    def delete_provider_from_config(
        self,
        config: Dict[str, Any],
        provider_names: List[str],
    ) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
        """
        Remove all model entries that belong to the given providers.

        Uses each provider's configured detection rules (prefix or api_base)
        to identify matching entries in ``config.yaml``.
        """
        model_list = config.get("model_list", [])
        original_count = len(model_list)
        indices_to_remove: Set[int] = set()
        details: Dict[str, int] = {}

        for provider_name in provider_names:
            cleaner = self._get_cleaner(provider_name)
            provider_models = cleaner.extract_provider_models(config)
            indices = {index for index, _, _ in provider_models}
            indices_to_remove.update(indices)
            details[provider_name] = len(indices)
            self.logger.info(
                f"Found {len(indices)} entry(s) for provider '{provider_name}'"
            )

        new_model_list = [
            entry
            for index, entry in enumerate(model_list)
            if index not in indices_to_remove
        ]
        config["model_list"] = new_model_list
        removed = original_count - len(new_model_list)

        return config, removed, details

    def disable_providers(
        self,
        provider_names: List[str],
    ) -> List[str]:
        """
        Set ``enabled: false`` for the given providers in ``providers.yaml``.

        Honors ``dry_run`` by logging the intended change without writing the
        file. Returns the list of providers that were actually disabled.
        """
        loader = ProviderConfigLoader()
        disabled: List[str] = []

        for provider_name in provider_names:
            try:
                provider_cfg = loader.get_provider_config(provider_name)
            except ValueError:
                self.logger.warning(
                    f"Provider '{provider_name}' not found in providers.yaml"
                )
                continue

            if provider_cfg.get("enabled", True) is False:
                self.logger.info(f"Provider '{provider_name}' is already disabled")
            else:
                provider_cfg["enabled"] = False
                disabled.append(provider_name)
                self.logger.info(f"Disabled provider '{provider_name}'")

        if disabled and not self.dry_run:
            loader.save()
        elif disabled:
            self.logger.info(f"DRY RUN: Would disable providers: {', '.join(disabled)}")

        return disabled

    def _prune_provider_special_models(
        self,
        provider_name: str,
        api_models: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        Remove ``special_models`` entries that are now available from the
        provider's models source.

        Providers maintain a ``special_models`` list in ``providers.yaml``
        to pin models that aren't yet returned by the provider API or
        models.dev. Once a model becomes available through the normal
        source, its entry becomes redundant and is removed. ``dry_run``
        is honored — no file changes occur in that case.

        Returns the list of model IDs that were (or would be) removed.
        Returns an empty list if the provider has no ``special_models``,
        if nothing is redundant, or if the live ``providers.yaml`` could
        not be loaded (so we fail safely and don't break the cleanup).
        """
        loader = ProviderConfigLoader()
        try:
            provider_cfg = loader.get_provider_config(provider_name)
        except ValueError:
            return []
        if not provider_cfg.get("special_models"):
            return []

        removed = loader.prune_special_models(provider_name, api_models.keys())
        if removed and not self.dry_run:
            loader.save()
            self.logger.info(
                f"Pruned {len(removed)} special_models from {provider_name}: "
                f"{', '.join(removed)}"
            )
        elif removed:
            self.logger.info(
                f"Would prune {len(removed)} special_models from {provider_name}: "
                f"{', '.join(removed)}"
            )
        return removed

    def validate_config(
        self, config: Optional[Dict[str, Any]] = None
    ) -> ValidationReport:
        """
        Validate config.yaml structure without API calls (offline).

        Delegates to BaseModelCleaner.validate_config() via the first available
        cleaner instance. BaseModelCleaner.validate_config() is provider-agnostic
        and checks all entries structurally, including provider-specific api_key
        checks across all configured providers.

        Args:
            config: Optional configuration dict (loads from file if not provided)

        Returns:
            ValidationReport with all found issues
        """
        if config is None:
            config = self.load_config()

        if not self.cleaners:
            report = ValidationReport()
            report.total_entries = len(config.get("model_list", []))
            return report

        # BaseModelCleaner.validate_config() is provider-agnostic;
        # pick any cleaner to run structural validation.
        cleaner = next(iter(self.cleaners.values()))
        report = cleaner.validate_config(config)
        return report


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Clean up invalid models, update costs, sort model list, and add new models in LiteLLM configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 This script performs six main functions:
1. Sorts the model list alphabetically by model_name
2. Validates models against the current API and removes invalid entries
3. Updates model costs when they differ from API pricing
4. Adds one or more models to the configuration
5. Deletes models by model_name from the configuration
6. Deletes providers, removing their models from config.yaml and disabling
   them in providers.yaml

Supported providers: openrouter, vercel, poe, nvidia, kilo, ollama, fireworks, opencode-zen, opencode-go, all

Examples:
  %(prog)s --provider openrouter                           # Process OpenRouter models
  %(prog)s --provider all                                  # Process all providers
  %(prog)s --provider poe --add-model "model1 model2"      # Add models to poe
  %(prog)s --provider openrouter --dry-run --verbose       # Preview changes
  %(prog)s --provider all --delete-model "model1 model2"   # Delete models by name
  %(prog)s --provider all --delete-provider openrouter     # Remove OpenRouter models and disable provider

Mapped Model Addition (simplified multi-provider workflow):
  %(prog)s --provider all --add-mapped-model glm-5         # Add glm-5 from all providers
        """,
    )

    parser.add_argument(
        "--provider",
        "-p",
        required=True,
        help='Provider to process (or "all" for all providers)',
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Path to LiteLLM configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Preview all changes without modifying the configuration file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging output"
    )
    parser.add_argument(
        "--add-model", nargs="+", help="Add one or more models to the configuration"
    )
    parser.add_argument(
        "--model-name", help="Custom model name (only valid when adding a single model)"
    )
    parser.add_argument(
        "--add-mapped-model",
        help="Add a mapped model from models.yaml across configured providers (e.g., 'glm-5')",
    )
    parser.add_argument(
        "--models-config",
        default="models.yaml",
        help="Path to model mappings file (default: models.yaml)",
    )
    parser.add_argument(
        "--delete-model",
        nargs="+",
        help="Delete one or more models by model_name from the configuration",
    )
    parser.add_argument(
        "--delete-provider",
        nargs="+",
        help=(
            "Delete all models for one or more providers from the configuration "
            "and disable the providers in providers.yaml"
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate config.yaml structure without API calls (offline)",
    )

    args = parser.parse_args()

    try:
        load_dotenv(override=True)

        loader = ProviderConfigLoader()
        available_providers = loader.list_providers()
        all_providers = loader.list_providers(include_disabled=True)

        if args.provider.lower() == "all":
            provider_names = available_providers
        else:
            provider_names = [args.provider.lower()]

        for provider_name in provider_names:
            if provider_name not in available_providers:
                print(
                    f"Error: Unknown provider '{provider_name}'. Available: {', '.join(available_providers)}",
                    file=sys.stderr,
                )
                return 1

        add_models = None
        if args.add_model:
            add_models = {}
            for model_spec in args.add_model:
                if "|" in model_spec:
                    provider, model_id = model_spec.split("|", 1)
                    if provider not in provider_names:
                        print(
                            f"Warning: Provider '{provider}' not in selected providers, skipping '{model_id}'",
                            file=sys.stderr,
                        )
                        continue
                else:
                    if len(provider_names) == 1:
                        provider = provider_names[0]
                        model_id = model_spec
                    else:
                        print(
                            f"Error: Must specify provider for model '{model_spec}' when using multiple providers. Use format provider|model",
                            file=sys.stderr,
                        )
                        return 1

                if provider not in add_models:
                    add_models[provider] = []
                add_models[provider].append(model_id)

        cleaner = UnifiedModelCleaner(
            config_path=args.config,
            provider_names=provider_names,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )

        if args.validate:
            report = cleaner.validate_config()
            _print_validation_report(report)
            return 1 if report.has_errors else 0

        # Handle model deletion
        if args.delete_model:
            config = cleaner.load_config()
            _, removed = cleaner.delete_model_from_config(config, args.delete_model)
            if not args.dry_run:
                cleaner.save_config(config)
            print(f"\n{'=' * 60}")
            print("Model Deletion Summary")
            print(f"{'=' * 60}")
            print(f"Requested: {', '.join(args.delete_model)}")
            print(f"Removed: {removed} entries")
            if args.dry_run:
                print("\nDRY RUN: No actual changes were made")
            else:
                print("\nConfiguration saved successfully")
            return 0

        # Handle provider deletion
        if args.delete_provider:
            delete_provider_names = [p.lower() for p in args.delete_provider]
            for provider_name in delete_provider_names:
                if provider_name not in all_providers:
                    print(
                        f"Error: Unknown provider '{provider_name}'. Available: {', '.join(all_providers)}",
                        file=sys.stderr,
                    )
                    return 1

            config = cleaner.load_config()
            updated_config, removed, details = cleaner.delete_provider_from_config(
                config, delete_provider_names
            )
            disabled = cleaner.disable_providers(delete_provider_names)

            if not args.dry_run:
                cleaner.save_config(updated_config)

            print(f"\n{'=' * 60}")
            print("Provider Deletion Summary")
            print(f"{'=' * 60}")
            for provider_name, count in details.items():
                print(f"{provider_name}: {count} model entry(s) removed")
            print(f"\nTotal removed: {removed} entries")
            print(f"Providers disabled: {', '.join(disabled) if disabled else 'none'}")

            if args.dry_run:
                print("\nDRY RUN: No actual changes were made")
            else:
                print("\nConfiguration saved successfully")
            return 0

        # Handle mapped model addition
        if args.add_mapped_model:
            mapping_loader = ModelMappingLoader(args.models_config)
            config, added_by_provider = cleaner.add_mapped_model(
                args.add_mapped_model, mapping_loader
            )

            print(f"\n{'=' * 60}")
            print(f"Mapped Model Addition Summary: {args.add_mapped_model}")
            print(f"{'=' * 60}")

            total_added = 0
            for provider_name, added_models in added_by_provider.items():
                if added_models and not any(
                    str(m).startswith("Error:") for m in added_models
                ):
                    count = len(added_models)
                    total_added += count
                    print(f"\n{provider_name}: {count} model(s) added")
                    for model in added_models:
                        print(f"  - {model}")
                elif added_models and any(
                    str(m).startswith("Error:") for m in added_models
                ):
                    print(f"\n{provider_name}: Error adding model")
                    for msg in added_models:
                        print(f"  - {msg}")

            if args.dry_run:
                print("\nDRY RUN: No actual changes were made")
            else:
                print(f"\n✅ Total models added: {total_added}")

            return 0

        if len(provider_names) == 1:
            provider_add_models = (
                add_models.get(provider_names[0]) if add_models else None
            )
            models_removed, models_updated, changes_made = cleaner.cleanup_provider(
                provider_names[0], provider_add_models, args.model_name
            )

            print(f"\nSummary for {provider_names[0]}:")
            print(f"Models removed: {models_removed}")
            print(f"Models updated: {models_updated}")
            print(f"Total changes: {len(changes_made)}")

            if changes_made:
                print("\nChanges made:")
                for change in changes_made:
                    print(f"  - {change}")
        else:
            results = cleaner.cleanup_all_providers(add_models, args.model_name)

            print("\nSummary for all providers:")
            total_removed = 0
            total_updated = 0
            total_changes = 0

            for provider_name, (removed, updated, changes) in results.items():
                print(f"\n{provider_name}:")
                print(f"  Models removed: {removed}")
                print(f"  Models updated: {updated}")
                print(f"  Changes: {len(changes)}")

                total_removed += removed
                total_updated += updated
                total_changes += len(changes)

                if changes:
                    print("  Details:")
                    for change in changes:
                        print(f"    - {change}")

            print("\nOverall Summary:")
            print(f"Total models removed: {total_removed}")
            print(f"Total models updated: {total_updated}")
            print(f"Total changes: {total_changes}")

        if args.dry_run:
            print("\nDRY RUN: No actual changes were made")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
