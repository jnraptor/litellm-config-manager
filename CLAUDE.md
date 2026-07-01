# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a LiteLLM configuration management repository with Python scripts for managing model configurations across multiple AI providers: OpenRouter, Vercel AI Gateway, Poe, Kilo, Nvidia NIM, Ollama, Fireworks, OpenCode Zen, and OpenCode Go. The main config file (`config.yaml`) contains LiteLLM model definitions with routing strategies and cost information.

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

# Delete models by model_name
python cleanup_models.py --provider all --delete-model "model_name" [--dry-run]

# Delete a provider: remove its models from config.yaml and disable it in providers.yaml
python cleanup_models.py --provider all --delete-provider "openrouter" [--dry-run]

# Auto-populate models.yaml by fuzzy-matching a model across providers
python populate_models.py minimax-m3 [--dry-run] [--force] [--provider openrouter,kilo]

# List available mapped models
grep -E "^[a-z0-9-]+:$" models.yaml | tr -d ':'
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
```

**Benefits:**

- Define the model once in `models.yaml` with provider-specific IDs
- All instances share the same `model_name` for automatic load balancing
- Single command adds from all configured providers
- Works with `--dry-run` for preview
- Automatically handles free variants where supported (OpenRouter, Kilo)

### Auto-Populating `models.yaml` (`populate_models.py`)

Manually searching every provider's API for a new model is tedious, and
providers often use different naming conventions for the same model
(`glm-5.1` vs `glm-5-1` vs `glm-5p1`, or `minimax/minimax-m3` vs
`anthropic/minimax-m3`). `populate_models.py` fetches the model list from
every provider in `providers.yaml` and uses **tiered fuzzy matching** to find
the best match for a canonical key, writing the results back into
`models.yaml` (missing providers are left as commented-out entries).

```bash
# Dry-run preview
python populate_models.py minimax-m3 --dry-run

# Apply
python populate_models.py glm-5.1

# Limit to specific providers
python populate_models.py minimax-m3 --provider openrouter,kilo,vercel

# Overwrite an existing entry
python populate_models.py minimax-m3 --force

# Leave a pre-existing entry alone
python populate_models.py minimax-m3 --skip-existing
```

**Matching tiers** (highest score wins regardless of API order):

1. `1.00` — exact id match
2. `0.90` — id matches after stripping a vendor prefix (e.g. `z-ai/glm-5.1` ↔ `glm-5.1`)
3. `0.85` — normalized forms equal (case, separators, `p`-as-point)
4. `0.75` — normalized forms equal with one trailing suffix stripped (`:free`, `-fw`, `-el`, `-t`, `-it`)
5. `0.60` — substring fallback (only used when no better match exists; requires key to be a substantial portion of the api id)

`populate_models.py` rewrites the entire `models.yaml` file (via
`yaml.dump`), so any hand-written comments will be lost. A `.yaml.backup` is
written before each save. See `tests/test_populate_models.py` for the full
matching test matrix.

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

- **Kilo**: `KILO_API_KEY`
- **OpenCode Zen / OpenCode Go**: `OPENCODE_API_KEY`
- **OpenRouter, Vercel, Poe, Nvidia, Ollama**: No API key required for model listing
- **Fireworks**: Uses `FIREWORKS_AI_API_KEY`, required for model listing

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

# Run a single specific test
pytest tests/test_cleanup_base.py::test_costs_are_equal -v
```

**Test Organization:**

- `test_cleanup_base.py` — Unit tests for utility functions (costs_are_equal, APIClient, etc.)
- `test_cleanup_coverage.py` — Tests for model validation, cost updating, model addition
- `test_input_outputs.py` — Integration tests validating expected input/output transformations from `tests/input-and-outputs.md` (includes `opencode-go` and `opencode-go-anthropic` test cases)
- `test_coverage_additional.py` — Tests for UnifiedModelCleaner, file I/O, free variants, ModelMappingLoader
- `test_populate_models.py` — Tests for `populate_models.py` fuzzy matching tiers, `_format_model_block` / `_replace_model_block` YAML editing, `ModelMappingLoader.save()`, and `ModelsPopulator` end-to-end

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

All logic lives in `cleanup_base.py`. The two entry points (`cleanup_models.py` and the 11 `cleanup_*_models.py` scripts) both delegate to the same base classes.

```
cleanup_base.py
├── APIClient                          # HTTP fetch with retry and caching
├── ModelsDevClient                    # fetches cost data from models.dev/api.json
├── BaseModelCleaner (abstract)        # load/save config, sort, validate, cost update
│   └── ConfigDrivenModelCleaner       # reads providers.yaml, implements all abstract methods
│       ├── OpenRouterModelCleaner     # cleanup_openrouter_models.py
│       ├── KiloModelCleaner           # cleanup_kilo_models.py
│       ├── OllamaModelCleaner         # cleanup_ollama_models.py (custom fetch_available_models)
│       ├── FireworksModelCleaner      # cleanup_fireworks_models.py
│       └── ...all other providers (follow same pattern)
cleanup_models.py
└── UnifiedModelCleaner               # creates one ConfigDrivenModelCleaner per provider,
                                       # delegates all operations to them
populate_models.py
└── ModelsPopulator                    # fuzzy-matches a model key across providers and
                                       # writes results to models.yaml (full rewrite)
```

### Key Components

**`cleanup_base.py`** (~1545 lines) — all shared logic:

- `get_nested_value(data, field_path)` — module-level utility for dot-notation dict access (e.g., `"pricing.prompt"`)
- `costs_are_equal()` — relative-tolerance comparison for scientific notation floats
- `APIClient` — HTTP fetch with retry logic and response caching
- `ModelsDevClient` — fetches and caches cost data from `https://models.dev/api.json`; provides per-token costs (input, output, and cache read/write from `cost.cache_read` / `cost.cache_write`) for providers whose own APIs don't include pricing (e.g., Fireworks, OpenCode Zen, OpenCode Go)
- `BaseModelCleaner` — abstract base with YAML load/save, sort, validate, cost update, add model
- `ConfigDrivenModelCleaner` — reads `providers.yaml`; implements all abstract methods based on config; handles free variant logic via `_free_variant_suffix`; falls back to `ModelsDevClient` for pricing when `pricing.models_dev_id` is configured
- `ModelMappingLoader` — loads and saves `models.yaml`; `save()` rewrites the whole file via `yaml.dump` (hand-written comments will be lost)
- `ModelsPopulator` (`populate_models.py`) — fuzzy-matches a model key across all configured providers and writes the resulting mappings to `models.yaml` (via `ModelMappingLoader.save()`)
- `create_provider_main(cleaner_class, description, epilog)` — factory that returns a `main()` function; used by all 8 provider scripts so each is ~48 lines

**`providers.yaml`** — single source of truth for all provider settings:

- `model_detection.type`: `"prefix"` (OpenRouter, Vercel, Nvidia) or `"api_base"` (Requesty, Poe, Kilo, OpenCode Zen, OpenCode Go)
- `pricing.input_field` / `pricing.output_field`: dot-notation paths into the API response
- `pricing.cache_read_field` / `pricing.cache_write_field`: dot-notation paths for cache pricing (OpenRouter, Vercel, Kilo, Poe, Requesty); omit for providers that use models.dev or have no cache pricing
- `pricing.is_per_million` + `pricing.divisor`: conversion to per-token cost
- `pricing.default_cost`: used when API has no pricing (Nvidia: `1.0e-09`)
- `pricing.models_dev_id`: maps to a provider ID in `models.dev/api.json` for cost augmentation; used when provider API has no pricing (e.g., `"fireworks-ai"`, `"opencode"`, `"opencode-go"`); also sources `cost.cache_read` / `cost.cache_write` for cache fields
- `free_variant_suffix`: `":free"` for OpenRouter and Kilo — triggers automatic `:free` variant addition when adding models
- `special_models`: model IDs exempt from removal validation
- `model_prefixes`: optional list of `{prefix, api_base}` mappings for providers that serve models under multiple prefixes (e.g., OpenCode Go with `openai/`, `dashscope/`, `anthropic/`)

**`cleanup_models.py`** — `UnifiedModelCleaner` delegates to per-provider `ConfigDrivenModelCleaner` instances; handles multi-provider orchestration, the `--provider all` flag, model deletion via `--delete-model`, provider deletion via `--delete-provider`, and mapped model additions via `--add-mapped-model`.

**`models.yaml`** — defines canonical model mappings for simplified multi-provider addition:

- Maps a canonical model key (e.g., `glm-5`) to provider-specific IDs
- Uses a shared `display_name` across all providers for load balancing
- Only adds models from providers listed in the mapping

### Model Identification

How each provider's models are detected in `config.yaml`:

- **OpenRouter**: `litellm_params.model` starts with `openrouter/`
- **Vercel**: `litellm_params.model` starts with `vercel_ai_gateway/`
- **Nvidia NIM**: `litellm_params.model` starts with `nvidia_nim/`
- **Ollama**: `litellm_params.model` starts with `ollama_chat/`
- **Fireworks**: `litellm_params.model` starts with `fireworks_ai/`
- **Poe**: `litellm_params.api_base` contains `api.poe.com` + model starts with `openai/`
- **Kilo**: `litellm_params.api_base` = `os.environ/KILO_API_BASE` + model starts with `openai/`
- **OpenCode Zen**: `litellm_params.api_base` contains `opencode.ai/zen/v1` + model starts with `openai/`
- **OpenCode Go**: `litellm_params.api_base` contains `opencode.ai/zen/go` + model starts with `openai/`, `dashscope/`, or `anthropic/`

> **Note:** OpenCode Go is the first provider to support multiple model prefixes. The `model_prefixes` field in `providers.yaml` maps each prefix to its corresponding `api_base`:
> - `openai/` and `dashscope/` → `https://opencode.ai/zen/go/v1`
> - `anthropic/` → `https://opencode.ai/zen/go`

### Adding a New Provider

1. Add a block to `providers.yaml` with `api_url`, `model_detection`, `pricing`, `model_name_prefix`, `model_name_cleanup`, `order`
2. Create `cleanup_<provider>_models.py` — inherit `ConfigDrivenModelCleaner`, call `create_provider_main`
3. For multi-prefix providers (e.g., OpenCode Go), add `model_prefixes` to `providers.yaml` — a list of `{prefix, api_base}` mappings
4. Test: `python cleanup_models.py --provider <name> --dry-run --verbose`
5. For custom pricing logic (e.g., non-standard API response format), override `parse_api_model()`

**Multi-Prefix Support:** When a provider serves models under different API types (e.g., OpenAI-compatible and Anthropic-compatible), use the `model_prefixes` field instead of relying on a single `model_prefix`. This triggers multi-prefix detection in `is_api_base_model()`, `get_api_model_id()`, and `create_model_entry()` to correctly identify models and assign the appropriate `api_base` per prefix.

## Configuration Management

**Model Entry Structure:**

```yaml
- model_name: display-name
  litellm_params:
    model: provider/actual-model-id
    api_key: os.environ/API_KEY_NAME
    input_cost_per_token: 1.0e-07
    output_cost_per_token: 3.0e-07
    cache_creation_input_token_cost: 1.25e-06  # optional: cost to write to cache
    cache_read_input_token_cost: 1.0e-07       # optional: cost to read from cache
    order: 5
```

Cache fields are only present when the provider API reports cache pricing. They are removed automatically when the API stops reporting them (full sync).

**Free Model Handling:** Free models use `1.0e-09` costs for LiteLLM compatibility (LiteLLM requires non-zero costs). `adjust_cost_for_free_model()` converts `0.0` → `1.0e-09` for all cost fields — input, output, and both cache fields.

**Free Variants (OpenRouter, Kilo):** When adding a model via `--add-model`, if a `<model-id>:free` variant exists in the API, it is automatically added with the same `model_name` for load balancing.

**Load Balancing:** Multiple entries sharing the same `model_name` distribute requests across providers.

**Embedding Models:** OpenRouter fetches from a separate embeddings endpoint; entries get `model_info.mode: embedding`.

## GitHub Actions

- **Manual trigger:** Workflow dispatch accepts `--add-model` and optional `--model-name`
- **Process:** dry-run → apply (remove invalid, update costs, sort) → add model if specified → auto-commit if changed
- Workflows: `cleanup-all-models-unified.yml` (unified) + one per provider
