# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a LiteLLM configuration management repository with Python scripts for managing model configurations across multiple AI providers: OpenRouter, Requesty, Novita AI, Nano-GPT, Vercel AI Gateway, Poe, Kilo, and Nvidia NIM. The main config file (`config.yaml`) contains LiteLLM model definitions with routing strategies and cost information.

## Development Commands

### Unified Script (Recommended)

```bash
# Validate models, update costs, and remove invalid entries for all providers
python cleanup_models.py --provider all [--config config.yaml] [--dry-run] [--verbose]

# Process specific provider
python cleanup_models.py --provider openrouter [--config config.yaml] [--dry-run] [--verbose]

# Add new models with automatic cost detection
python cleanup_models.py --provider openrouter --add-model model1 model2 [--dry-run]
python cleanup_models.py --provider nvidia --add-model meta/llama-3.1-8b-instruct [--dry-run]

# Add model with custom name (single model only)
python cleanup_models.py --provider openrouter --add-model gpt-4 --model-name "My GPT-4" [--dry-run]

# Add mapped model across multiple providers (simplified workflow)
python cleanup_models.py --provider all --add-mapped-model glm-5 [--dry-run]
```

### Mapped Model Addition (Multi-Provider Workflow)

Instead of manually adding the same model from different providers with different IDs, use `models.yaml` to define mappings:

**`models.yaml` Structure:**

```yaml
models:
  glm-5:
    display_name: "zai-glm-5"  # Common name across all providers
    description: "GLM-5 model by Z.ai"
    providers:
      openrouter: z-ai/glm-5
      requesty: zai/GLM-5
      novita: glm-5
```

**Benefits:**

- Define the model once in `models.yaml` with provider-specific IDs
- All instances share the same `model_name` for automatic load balancing
- Single command adds from all configured providers
- Works with `--dry-run` for preview
- Automatically handles free variants where supported (OpenRouter, Kilo)

### Provider-Specific Scripts

```bash
# All provider scripts support the same flags: [--config config.yaml] [--dry-run] [--verbose]
python cleanup_openrouter_models.py [--dry-run] [--verbose]
python cleanup_kilo_models.py --add-model "anthropic/claude-opus-4.6" [--dry-run]

# Multiple model addition (space-separated)
python cleanup_openrouter_models.py --add-model model1 model2 model3 [--dry-run]
```

### Dependencies

```bash
pip install -r requirements.txt
source .venv/bin/activate  # On macOS/Linux
```

**API Key Requirements:**

- **Requesty**: `REQUESTY_API_KEY`
- **Nano-GPT**: `NANOGPT_API_KEY`
- **Kilo**: `KILO_API_KEY`
- **OpenRouter, Novita, Vercel, Poe, Nvidia**: No API key required for model listing

### Testing Workflow

Always run with `--dry-run --verbose` first to preview changes:

```bash
python cleanup_models.py --provider all --dry-run --verbose
python cleanup_openrouter_models.py --dry-run --verbose
```

### Running Tests

The test suite uses pytest and includes coverage reporting:

```bash
# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ -v --cov=. --cov-report=term

# Run specific test file
pytest tests/test_input_outputs.py -v
pytest tests/test_cleanup_coverage.py -v

# Run tests matching a pattern
pytest tests/ -v -k "test_provider"
```

**Test Organization:**

- `test_cleanup_base.py` — Unit tests for utility functions (costs_are_equal, APIClient, etc.)
- `test_cleanup_coverage.py` — Tests for model validation, cost updating, model addition
- `test_input_outputs.py` — Integration tests validating expected input/output transformations from `tests/input-and-outputs.md`
- `test_coverage_additional.py` — Tests for UnifiedModelCleaner, file I/O, free variants, ModelMappingLoader

**Adding New Tests:**
To add a new provider test case, edit `tests/input-and-outputs.md`:

````markdown
## provider_name
### input
```json
{ ... raw API response JSON ... }
```
### output
```yaml
- model_name: expected-model-name
  litellm_params:
    model: provider/model-id
    order: 5
    input_cost_per_token: 1.0e-06
    output_cost_per_token: 3.0e-06
```
````

## Architecture

### Class Hierarchy

All logic lives in `cleanup_base.py`. The two entry points (`cleanup_models.py` and the 8 `cleanup_*_models.py` scripts) both delegate to the same base classes.

```
cleanup_base.py
├── BaseModelCleaner (abstract)        # load/save config, sort, validate, cost update
│   └── ConfigDrivenModelCleaner       # reads providers.yaml, implements all abstract methods
│       ├── OpenRouterModelCleaner     # cleanup_openrouter_models.py
│       ├── KiloModelCleaner           # cleanup_kilo_models.py
│       ├── NovitaModelCleaner         # cleanup_novita_models.py (custom parse_api_model)
│       └── ...all other providers
cleanup_models.py
└── UnifiedModelCleaner               # creates one ConfigDrivenModelCleaner per provider,
                                       # delegates all operations to them
```

### Key Components

**`cleanup_base.py`** (~1545 lines) — all shared logic:

- `get_nested_value(data, field_path)` — module-level utility for dot-notation dict access (e.g., `"pricing.prompt"`)
- `costs_are_equal()` — relative-tolerance comparison for scientific notation floats
- `BaseModelCleaner` — abstract base with YAML load/save, sort, validate, cost update, add model
- `ConfigDrivenModelCleaner` — reads `providers.yaml`; implements all abstract methods based on config; handles free variant logic via `_free_variant_suffix`
- `ModelMappingLoader` — loads `models.yaml` and provides canonical model mappings for multi-provider addition
- `create_provider_main(cleaner_class, description, epilog)` — factory that returns a `main()` function; used by all 8 provider scripts so each is ~48 lines

**`providers.yaml`** — single source of truth for all provider settings:

- `model_detection.type`: `"prefix"` (OpenRouter, Novita, Vercel, Nvidia, Nano-GPT) or `"api_base"` (Requesty, Poe, Kilo)
- `pricing.input_field` / `pricing.output_field`: dot-notation paths into the API response
- `pricing.is_per_million` + `pricing.divisor`: conversion to per-token cost
- `pricing.default_cost`: used when API has no pricing (Nvidia: `1.0e-09`)
- `free_variant_suffix`: `":free"` for OpenRouter and Kilo — triggers automatic `:free` variant addition when adding models
- `special_models`: model IDs exempt from removal validation

**`cleanup_models.py`** (~342 lines) — `UnifiedModelCleaner` delegates to per-provider `ConfigDrivenModelCleaner` instances; handles multi-provider orchestration, the `--provider all` flag, and mapped model additions via `--add-mapped-model`.

**`models.yaml`** — defines canonical model mappings for simplified multi-provider addition:

- Maps a canonical model key (e.g., `glm-5`) to provider-specific IDs
- Uses a shared `display_name` across all providers for load balancing
- Only adds models from providers listed in the mapping

### Model Identification

How each provider's models are detected in `config.yaml`:

- **OpenRouter**: `litellm_params.model` starts with `openrouter/`
- **Novita**: `litellm_params.model` starts with `novita/`
- **Vercel**: `litellm_params.model` starts with `vercel_ai_gateway/`
- **Nvidia NIM**: `litellm_params.model` starts with `nvidia_nim/`
- **Nano-GPT**: `litellm_params.model` starts with `nano-gpt/`
- **Requesty**: `litellm_params.api_base` contains `router.requesty.ai` + model starts with `openai/`
- **Poe**: `litellm_params.api_base` contains `api.poe.com` + model starts with `openai/`
- **Kilo**: `litellm_params.api_base` = `os.environ/KILO_API_BASE` + model starts with `openai/`

### Adding a New Provider

1. Add a block to `providers.yaml` with `api_url`, `model_detection`, `pricing`, `model_name_prefix`, `model_name_cleanup`, `order`
2. Create `cleanup_<provider>_models.py` — inherit `ConfigDrivenModelCleaner`, call `create_provider_main`
3. Test: `python cleanup_models.py --provider <name> --dry-run --verbose`
4. For custom pricing logic (e.g., non-standard API response format), override `parse_api_model()`

## Configuration Management

**Model Entry Structure:**

```yaml
- model_name: display-name
  litellm_params:
    model: provider/actual-model-id
    api_key: os.environ/API_KEY_NAME
    input_cost_per_token: 1.0e-07
    output_cost_per_token: 3.0e-07
    order: 5
```

**Free Model Handling:** Free models use `1.0e-09` costs for LiteLLM compatibility (LiteLLM requires non-zero costs). `adjust_cost_for_free_model()` converts `0.0` → `1.0e-09`.

**Free Variants (OpenRouter, Kilo):** When adding a model via `--add-model`, if a `<model-id>:free` variant exists in the API, it is automatically added with the same `model_name` for load balancing.

**Load Balancing:** Multiple entries sharing the same `model_name` distribute requests across providers.

**Embedding Models:** OpenRouter fetches from a separate embeddings endpoint; entries get `model_info.mode: embedding`.

## GitHub Actions

- **Schedule:** Weekly on Sundays (`0 0 * * 0`)
- **Manual trigger:** Workflow dispatch accepts `--add-model` and optional `--model-name`
- **Process:** dry-run → apply (remove invalid, update costs, sort) → add model if specified → auto-commit if changed
- Workflows: `cleanup-all-models-unified.yml` (unified) + one per provider
