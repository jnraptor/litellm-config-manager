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

### Test Coverage

Current coverage: ~63% (up from 30%)

- cleanup_base.py: 69%
- All test files: >98%

## Important Files

- **CLAUDE.md** — Complete project documentation
- **providers.yaml** — Provider configuration
- **models.yaml** — Model mappings for multi-provider addition
- **config.yaml** — Main LiteLLM configuration
- **cleanup_base.py** — All shared logic (~757 lines)
- **cleanup_models.py** — Unified cleanup script

## Key Classes

- `BaseModelCleaner` — Abstract base for config operations
- `ConfigDrivenModelCleaner` — Provider-specific implementation
- `UnifiedModelCleaner` — Multi-provider orchestration
- `ModelMappingLoader` — Loads model mappings from models.yaml

For full details, see [CLAUDE.md](./CLAUDE.md).
