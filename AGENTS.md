# AGENTS.md

This file provides guidance to OpenCode when working with code in this repository.

## Overview

This repository manages LiteLLM model configurations across multiple AI providers. See [CLAUDE.md](./CLAUDE.md) for comprehensive documentation including:

- Repository architecture and class hierarchy
- All available development commands
- Configuration management details
- Provider setup and model identification
- Complete testing instructions

## Quick Reference

### Python Virtual Env

Python should always be executed from venv.

```bash
source .venv/bin/activate  # On macOS/Linux
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=. --cov-report=term
```

### Test Files

- `test_cleanup_base.py` — Utility function tests
- `test_cleanup_coverage.py` — Model validation and cost tests
- `test_input_outputs.py` — Provider input/output validation
- `test_coverage_additional.py` — UnifiedModelCleaner, file I/O, free variants
- `test_validation.py` — Config validation (ValidationReport, validate_config, --validate CLI)
- `test_models_dev.py` — ModelsDevClient and models.dev cost augmentation tests
- `test_opencode_sync.py` — opencode.json regeneration (`build_opencode_models`, `regenerate_opencode_json`, save_config hook)
- `test_populate_models.py` — populate_models.py fuzzy matching, multi-model population, and YAML block editing

Note: `tests/conftest.py` disables the on-disk HTTP cache (`.cache/api/`) for all tests; cache-specific tests re-enable it with a per-test `tmp_path` directory.

### Test Coverage

Current coverage: ~72%

- cleanup_base.py: 74%
- cleanup_models.py: 25%
- All test files: >96%

## Important Files

- **CLAUDE.md** — Complete project documentation
- **providers.yaml** — Provider configuration (includes opencode-zen and opencode-go)
- **models.yaml** — Model mappings for multi-provider addition
- **config.yaml** — Main LiteLLM configuration
- **cleanup_base.py** — All shared logic (~2444 lines)
- **cleanup_models.py** — Unified cleanup script
- **cleanup_opencode_go_models.py** — OpenCode Go provider script (multi-prefix support)
- **populate_models.py** — Auto-populate `models.yaml` with one or more models across all providers using fuzzy matching

## Key Classes

- `BaseModelCleaner` — Abstract base for config operations
- `ConfigDrivenModelCleaner` — Provider-specific implementation
- `UnifiedModelCleaner` — Multi-provider orchestration (cleanup, add mapped models, delete models by name)
- `ModelMappingLoader` — Loads and saves model mappings from `models.yaml` (full rewrite via `yaml.dump`)
- `ModelsDevClient` — Fetches cost data (input, output, cache read/write) and modalities from models.dev API for providers without pricing
- `configure_disk_cache()` — Configures the on-disk HTTP cache in `cleanup_base.py` (default: on, 10-minute TTL, `.cache/api/`); CLI scripts expose `--no-cache` / `--cache-ttl`
- `build_opencode_models()` / `regenerate_opencode_json()` — module-level functions in `cleanup_base.py` that rebuild `provider.litellm.models` in `opencode.json` from `config.yaml` on every save (both unified and per-provider scripts)
- `ModelsPopulator` — Populates `models.yaml` for one or more model keys across all providers (each provider fetched once per run)

For full details, see [CLAUDE.md](./CLAUDE.md).
