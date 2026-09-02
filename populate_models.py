#!/usr/bin/env python3
"""
Populate models.yaml with new model(s) across all configured providers.

Fetches the available model list from every provider defined in
``providers.yaml`` and tries to locate each given model key in each one using
fuzzy matching (so ``glm-5.1`` matches ``glm-5-1``, ``glm-5p1``, etc.). Writes
the results back to ``models.yaml`` as new (or updated) canonical mappings,
commenting out any provider where the model was not found.

Usage:
    python populate_models.py minimax-m3
    python populate_models.py glm-5.1 --display-name "zai-glm-5.1" \\
        --description "GLM-5.1 by Z.ai"
    python populate_models.py minimax-m3 --provider openrouter,kilo
    python populate_models.py minimax-m3 --dry-run --verbose
    python populate_models.py minimax-m3 --force              # overwrite existing entry
    python populate_models.py minimax-m3 --skip-existing       # leave existing entry alone
    python populate_models.py minimax-m3 glm-5.1               # multiple models in one run

Provider model lists (and models.dev data) are cached on disk under
``.cache/api/`` for 10 minutes by default, so repeated runs do not re-download
them. Use ``--no-cache`` to bypass the cache and ``--cache-ttl`` to change the
freshness window.
"""

import argparse
import math
import re
import sys
from typing import Any

from dotenv import load_dotenv

from cleanup_base import (
    ConfigDrivenModelCleaner,
    ModelMappingLoader,
    ProviderConfigLoader,
    _models_dev_client,
    configure_disk_cache,
    setup_logging,
)
from cleanup_ollama_models import OllamaModelCleaner

# Vendor / model-family prefixes that are commonly prepended to a model id
# (e.g. "anthropic/claude-3", "z-ai/glm-5.1"). Stripped during normalization
# so that "z-ai/glm-5.1" and "accounts/fireworks/models/glm-5.1" both reduce
# to "glm51" for matching purposes.
VENDOR_PREFIXES = (
    "accounts/fireworks/models/",
    "openrouter/",
    "anthropic/",
    "google/",
    "meta-llama/",
    "mistralai/",
    "nvidia/",
    "qwen/",
    "deepseek/",
    "deepseek-ai/",
    "moonshotai/",
    "xiaomi/",
    "alibaba/",
    "z-ai/",
    "zai/",
    "minimax/",
    "meta/",
    "microsoft/",
    "cohere/",
    "perplexity/",
)

# Trailing tags / suffixes that some providers add and that we want to drop
# before comparing against the model key (e.g. Poe appends "-fw" or "-el",
# Ollama uses ":tag" syntax, and the ":free" variant is provided by some
# providers as a separate model).
TRAILING_SUFFIXES = (
    ":free",
    "-fw",
    "-el",
    "-t",
    "-it",
    "-instruct",
    "-chat",
    "-base",
)


def _normalize_for_match(value: str) -> str:
    """
    Reduce a model id or key to a canonical form for fuzzy comparison.

    Strips vendor prefixes, removes trailing tags, lowercases, and drops
    common separators (``.``, ``-``, ``+``, ``_``) so that "glm-5.1",
    "glm-5-1", and "glm-5p1" all collapse to the same string.
    """
    s = value.strip().lower()

    # Drop vendor prefix (everything up to and including the last '/')
    if "/" in s:
        s = s.rsplit("/", 1)[-1]

    # Drop trailing tag (e.g. ":free" or ":latest")
    if ":" in s:
        s = s.split(":", 1)[0]

    # Drop known short suffixes (Poe's "-fw", "-el", "-t", etc.)
    changed = True
    while changed:
        changed = False
        for suf in TRAILING_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf) + 1:
                s = s[: -len(suf)]
                changed = True

    # Collapse separators
    s = re.sub(r"[.\-+_]+", "", s)
    # Treat "p" (point) as a separator when it sits between digits, so
    # "5p1" / "2p5" (provider shorthand for "5.1" / "2.5") collapses the
    # same as "5.1" / "2.5".
    s = re.sub(r"(\d)p(\d)", r"\1\2", s)
    return s


def _strip_vendor_prefixes(value: str) -> str:
    """Strip any leading vendor prefix from a model id (e.g. 'z-ai/glm-5.1' -> 'glm-5.1')."""
    s = value
    changed = True
    while changed:
        changed = False
        for prefix in VENDOR_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix) :]
                changed = True
                break
    return s


def _strip_trailing_suffixes(value: str) -> str:
    """Strip any trailing :free, -fw, etc. that we know about."""
    s = value
    changed = True
    while changed:
        changed = False
        if ":" in s:
            head, tail = s.split(":", 1)
            if tail in ("free", "nitro", "preview", "beta", "latest"):
                s = head
                changed = True
                continue
        for suf in TRAILING_SUFFIXES:
            if s.endswith(suf) and len(s) > len(suf) + 1:
                s = s[: -len(suf)]
                changed = True
                break
    return s


def find_model_in_api(
    model_key: str,
    api_models: dict[str, dict[str, Any]],
) -> tuple[str | None, float, str]:
    """
    Find the best match for ``model_key`` in ``api_models``.

    Returns a tuple of (matched_id, score, match_type) where:
        - matched_id: the API model id that matched, or None
        - score: 0.0 (no match) to 1.0 (perfect)
        - match_type: explanation string for logging

    Matching tiers (highest score first). The function performs each tier
    as a separate pass over ``api_models`` so that a higher-quality match
    always wins over a lower-quality one, regardless of dict iteration
    order:

        1.0  - exact id match
        0.9  - id matches the model key with a vendor prefix stripped
        0.85 - normalized forms are equal
        0.75 - normalized forms equal with one trailing suffix stripped
        0.6  - model_key is contained in api id (or vice versa) after
               normalization (loose fallback; only used when no better
               match is available)
        0.0  - no match
    """
    if not api_models:
        return None, 0.0, "no api models"

    key_norm = _normalize_for_match(model_key)
    key_stripped = _strip_trailing_suffixes(_strip_vendor_prefixes(model_key)).lower()

    # Tier 1: exact match
    for api_id in api_models:
        if api_id == model_key:
            return api_id, 1.0, "exact"

    # Tier 2: vendor prefix stripped
    for api_id in api_models:
        api_stripped = _strip_vendor_prefixes(api_id).lower()
        if api_stripped == key_stripped or api_stripped == model_key.lower():
            return api_id, 0.9, "vendor-prefix-stripped"

    # Tier 3: normalized comparison
    for api_id in api_models:
        api_norm = _normalize_for_match(api_id)
        if api_norm == key_norm:
            return api_id, 0.85, "normalized"

    # Tier 4: normalized with one extra suffix strip
    for api_id in api_models:
        api_suf = _strip_trailing_suffixes(api_id)
        api_norm = _normalize_for_match(api_suf)
        if api_norm == key_norm:
            return api_id, 0.75, "normalized-with-suffix"

    # Tier 5: substring fallback (loose). Only match when the model key
    # is contained in the API model id, and require the key to be a
    # substantial portion of the api id. This avoids false positives such
    # as matching "kimi-k2.7" to "kimi-k2" because the shorter id happens
    # to be a substring of the key.
    if len(key_norm) >= 5:
        best_sub: tuple[str, float, str] | None = None
        for api_id in api_models:
            api_norm = _normalize_for_match(api_id)
            if key_norm in api_norm and len(key_norm) / len(api_norm) >= 0.5:
                if best_sub is None or len(api_norm) < len(
                    _normalize_for_match(best_sub[0])
                ):
                    best_sub = (api_id, 0.6, "substring")
        if best_sub is not None:
            return best_sub

    return None, 0.0, "no-match"


def filter_free_models(
    api_models: dict[str, dict[str, Any]], free_model_cost: float = 1.0e-09
) -> dict[str, dict[str, Any]]:
    """Return only models whose input and output costs are known to be free.

    ``free_model_handling`` normalizes zero pricing to the nominal free cost
    configured in providers.yaml, so accept both representations. Models with
    missing, invalid, or only partially available pricing are excluded.
    """
    free_costs = {0.0, float(free_model_cost)}
    free_models: dict[str, dict[str, Any]] = {}

    for model_id, model_info in api_models.items():
        if not isinstance(model_info, dict):
            continue
        try:
            input_cost = float(model_info["input_cost"])
            output_cost = float(model_info["output_cost"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            math.isfinite(input_cost)
            and math.isfinite(output_cost)
            and input_cost in free_costs
            and output_cost in free_costs
        ):
            free_models[model_id] = model_info

    return free_models


class ModelsPopulator:
    """
    Orchestrates populating ``models.yaml`` for canonical model keys.

    For each configured provider, fetches the available models (once per
    provider per run, reused across all ``populate()`` calls), runs fuzzy
    matching, and records the result. The collected mappings are then written
    back to ``models.yaml`` via :class:`ModelMappingLoader`.
    """

    def __init__(
        self,
        providers_config_path: str = "providers.yaml",
        models_config_path: str = "models.yaml",
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.dry_run = dry_run
        self.logger = setup_logging(verbose, "ModelsPopulator")
        self.providers_loader = ProviderConfigLoader(providers_config_path)
        self.mapping_loader = ModelMappingLoader(models_config_path)
        self._cleaners: dict[str, ConfigDrivenModelCleaner] = {}
        # Per-provider fetch results (or exceptions), so populating multiple
        # models in one run only hits each provider API once.
        self._api_models_cache: dict[str, Any] = {}

    def _get_cleaner(
        self, provider_name: str, config_path: str
    ) -> ConfigDrivenModelCleaner:
        if provider_name in self._cleaners:
            return self._cleaners[provider_name]
        if provider_name == "ollama":
            cleaner: ConfigDrivenModelCleaner = OllamaModelCleaner(
                config_path, self.dry_run, False
            )
        else:
            cleaner = ConfigDrivenModelCleaner(
                provider_name, config_path, self.dry_run, False
            )
        self._cleaners[provider_name] = cleaner
        return cleaner

    def _fetch_provider_models(
        self, provider_name: str, config_path: str
    ) -> dict[str, dict[str, Any]]:
        """
        Fetch (or reuse) the available-models list for a provider.

        Results are cached per provider for the lifetime of this populator so
        that populating multiple models in one run only hits each provider
        API once. Fetch failures are cached too and re-raised on subsequent
        requests for the same provider.
        """
        if provider_name in self._api_models_cache:
            cached = self._api_models_cache[provider_name]
            if isinstance(cached, Exception):
                raise cached
            return cached

        try:
            cleaner = self._get_cleaner(provider_name, config_path)
            api_models = cleaner.fetch_available_models()
        except Exception as exc:
            self._api_models_cache[provider_name] = exc
            raise

        self._api_models_cache[provider_name] = api_models
        return api_models

    def populate(
        self,
        model_key: str,
        display_name: str | None = None,
        description: str | None = None,
        provider_filter: list[str] | None = None,
        force: bool = False,
        skip_existing: bool = False,
    ) -> dict[str, Any]:
        """
        Populate the model mapping for ``model_key`` across all providers.

        Args:
            model_key: Canonical model key (e.g. ``minimax-m3``, ``glm-5.1``)
            display_name: Optional display name; defaults to ``model_key``
            description: Optional description; preserved if entry already exists
            provider_filter: Optional list of provider names to limit the search
            force: If True, overwrite an existing entry
            skip_existing: If True, do not touch a pre-existing entry

        Returns:
            The mapping data that was written (or would be written) to models.yaml
        """
        existing = self.mapping_loader.get_model_mapping(model_key)
        if existing and skip_existing:
            self.logger.info(
                f"Model '{model_key}' already exists in models.yaml; skipping"
            )
            return existing or {}

        if existing and not force:
            self.logger.info(
                f"Model '{model_key}' already exists; use --force to overwrite"
            )

        display_name = display_name or (existing or {}).get("display_name") or model_key
        description = description or (existing or {}).get("description") or ""

        providers = self.providers_loader.list_providers()
        if provider_filter:
            providers = [p for p in providers if p in provider_filter]

        config_path = "config.yaml"

        found: dict[str, str] = {}
        missing: list[str] = []
        matches: dict[str, tuple[str, float, str]] = {}

        for provider_name in providers:
            self.logger.info(f"Checking {provider_name}...")
            try:
                api_models = self._fetch_provider_models(provider_name, config_path)
            except Exception as exc:
                self.logger.warning(
                    f"  Failed to fetch models from {provider_name}: {exc}"
                )
                missing.append(provider_name)
                continue

            provider_config = self.providers_loader.get_provider_config(provider_name)
            if provider_config.get("free_models_only", False):
                before_count = len(api_models)
                api_models = filter_free_models(api_models)
                self.logger.info(
                    f"  {provider_name}: restricted to {len(api_models)} free models "
                    f"({before_count - len(api_models)} excluded)"
                )

            matched_id, score, match_type = find_model_in_api(model_key, api_models)
            if matched_id:
                # OpenCode Go uses the OpenAI route unless models.dev identifies
                # the model as Anthropic-compatible.
                if provider_name == "opencode-go":
                    npm = _models_dev_client.get_model_provider_npm(
                        "opencode-go", matched_id, self.logger
                    )
                    if npm == "@ai-sdk/anthropic":
                        matched_id = f"anthropic/{matched_id}"
                    else:
                        # @ai-sdk/openai, @ai-sdk/openai-compatible, or absent.
                        matched_id = f"openai/{matched_id}"
                self.logger.info(
                    f"  ✅ {provider_name}: {matched_id} (score={score:.2f}, {match_type})"
                )
                found[provider_name] = matched_id
                matches[provider_name] = (matched_id, score, match_type)
            else:
                self.logger.info(f"  ❌ {provider_name}: no match")
                missing.append(provider_name)

        # Carry over any pre-existing mappings the user wants to keep
        if existing and not force:
            for prov, mid in (existing.get("providers") or {}).items():
                if mid and prov not in found:
                    if provider_filter and prov not in provider_filter:
                        continue
                    self.logger.info(f"  ↻ Keeping existing {prov}: {mid}")
                    found[prov] = mid

        # Only include providers that actually have the model. Missing
        # providers are omitted from the file entirely (no null entries).
        providers_map = dict(found)

        mapping_data: dict[str, Any] = {
            "display_name": display_name,
            "description": description,
            "providers": providers_map,
        }

        self.mapping_loader.save(model_key, mapping_data, dry_run=self.dry_run)
        if self.dry_run:
            self.logger.info(
                f"DRY RUN: would write {len(found)} found / {len(missing)} missing"
            )
        else:
            self.logger.info(
                f"Wrote '{model_key}' to {self.mapping_loader._config_path} "
                f"({len(found)} found, {len(missing)} missing)"
            )

        return mapping_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Populate models.yaml with one or more models across all providers, "
            "using fuzzy matching to handle provider-specific naming variations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_keys",
        nargs="+",
        help="Canonical model key(s) to look up (e.g. minimax-m3 glm-5.1)",
    )
    parser.add_argument(
        "--display-name",
        help="Display name to use (default: model_key; single model only)",
    )
    parser.add_argument(
        "--description",
        help="Description for the model entry (single model only)",
    )
    parser.add_argument(
        "--provider",
        help="Comma-separated list of providers to search (default: all)",
    )
    parser.add_argument(
        "--providers-config",
        default="providers.yaml",
        help="Path to providers.yaml (default: providers.yaml)",
    )
    parser.add_argument(
        "--models-config",
        default="models.yaml",
        help="Path to models.yaml (default: models.yaml)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (used for cleaner instantiation)",
    )
    parser.add_argument(
        "--dry-run", "-d", action="store_true", help="Preview without writing"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing entry in models.yaml",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not touch a pre-existing entry",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk HTTP cache (fetch everything fresh)",
    )
    parser.add_argument(
        "--cache-ttl",
        type=float,
        default=None,
        metavar="SECONDS",
        help="On-disk HTTP cache TTL in seconds (default: 600)",
    )

    args = parser.parse_args()
    load_dotenv(override=True)

    if (args.display_name or args.description) and len(args.model_keys) > 1:
        parser.error(
            "--display-name and --description can only be used when populating "
            "a single model"
        )

    configure_disk_cache(
        enabled=False if args.no_cache else None,
        ttl=args.cache_ttl,
    )

    provider_filter: list[str] | None = None
    if args.provider:
        provider_filter = [p.strip() for p in args.provider.split(",") if p.strip()]

    populator = ModelsPopulator(
        providers_config_path=args.providers_config,
        models_config_path=args.models_config,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    results: list[tuple[str, dict[str, Any] | None]] = []
    failed = False
    for model_key in args.model_keys:
        try:
            mapping = populator.populate(
                model_key=model_key,
                display_name=args.display_name,
                description=args.description,
                provider_filter=provider_filter,
                force=args.force,
                skip_existing=args.skip_existing,
            )
            results.append((model_key, mapping))
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            results.append((model_key, None))
            failed = True

    for model_key, mapping in results:
        print(f"\n{'=' * 60}")
        print(f"Populate Summary: {model_key}")
        print(f"{'=' * 60}")
        if mapping is None:
            print("\nFailed (see error above)")
            continue
        providers_map = mapping.get("providers", {})
        found = [p for p, v in providers_map.items() if v]
        missing = [p for p, v in providers_map.items() if not v]
        if found:
            print(f"\nFound ({len(found)}):")
            for p in sorted(found):
                print(f"  {p}: {providers_map[p]}")
        if missing:
            print(f"\nNot found / commented ({len(missing)}):")
            for p in sorted(missing):
                print(f"  {p}")

    if args.dry_run:
        print("\nDRY RUN: No changes written")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
