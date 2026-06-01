# LiteLLM Model Cleanup, Cost Update, and Model Addition Scripts

Comprehensive Python scripts that validate models from multiple providers (OpenRouter, Requesty, Vercel AI Gateway, Poe, Kilo, Nvidia NIM, Ollama, Fireworks, OpenCode Zen, and OpenCode Go) in LiteLLM configuration files against their respective APIs, remove invalid model entries, automatically update model costs, and provide an easy way to add new models.

## Overview

- Identifies models from multiple providers in your `config.yaml`
- Checks their validity against the current APIs
- Removes entries for invalid/deprecated models
- **Automatically updates costs** (`input_cost_per_token` and `output_cost_per_token`) when they differ from API pricing
- **Adds new models** with automatic cost detection and proper formatting
- **Custom model naming** when adding single models
- **Sorts model lists** alphabetically
- **Mapped model addition** — add the same model across multiple providers with one command via `models.yaml`
- **Model deletion** — remove models by `model_name` from the configuration
- **Auto-populate `models.yaml`** — fuzzy-match a model across all providers and update mappings in one shot
- Provides detailed logging with percentage-based cost change information
- Supports dry-run capabilities for safe previewing

## Quick Start

```bash
# Unified script (recommended) — process all providers
python cleanup_models.py --provider all --dry-run --verbose

# Process a specific provider
python cleanup_models.py --provider openrouter --dry-run

# Add new models
python cleanup_models.py --provider openrouter --add-model mistralai/mistral-medium mistralai/mistral-small

# Add the same model across all configured providers
python cleanup_models.py --provider all --add-mapped-model glm-5

# Delete models by model_name
python cleanup_models.py --provider all --delete-model "model-a" "model-b" --dry-run

# Auto-populate models.yaml by fuzzy-matching a model across all providers
python populate_models.py minimax-m3 --dry-run

# Apply changes (remove --dry-run)
python cleanup_models.py --provider all
```

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API keys (if required):**
   ```bash
   export REQUESTY_API_KEY="your-requesty-api-key"
   export KILO_API_KEY="your-kilo-api-key"
   export OPENCODE_API_KEY="your-opencode-api-key"
   ```

   Providers that require API keys:
   - **Requesty**: `REQUESTY_API_KEY`
   - **Kilo**: `KILO_API_KEY`
   - **OpenCode Zen / OpenCode Go**: `OPENCODE_API_KEY`
   - **OpenRouter, Vercel, Poe, Nvidia, Ollama, Fireworks**: No API key required for model listing

## Available Scripts

| Script | Purpose |
|--------|---------|
| `cleanup_models.py` | Unified script — process all providers or specific ones |
| `populate_models.py` | Auto-populate `models.yaml` by fuzzy-matching a model across providers |
| `cleanup_openrouter_models.py` | OpenRouter-specific |
| `cleanup_requesty_models.py` | Requesty-specific |
| `cleanup_vercel_models.py` | Vercel AI Gateway-specific |
| `cleanup_poe_models.py` | Poe-specific |
| `cleanup_kilo_models.py` | Kilo-specific |
| `cleanup_nvidia_models.py` | Nvidia NIM-specific |
| `cleanup_ollama_models.py` | Ollama-specific |
| `cleanup_fireworks_models.py` | Fireworks-specific |
| `cleanup_opencode_zen_models.py` | OpenCode Zen-specific |
| `cleanup_opencode_go_models.py` | OpenCode Go-specific (OpenAI + Anthropic compat) |

## Commands

### Unified Script (`cleanup_models.py`)

```bash
python cleanup_models.py --provider all                    # Process all providers
python cleanup_models.py --provider openrouter             # Process specific provider
python cleanup_models.py --provider requesty --config my-config.yaml
python cleanup_models.py --provider all --add-model model1 model2  # Add to all providers
python cleanup_models.py --provider all --add-mapped-model glm-5  # Add mapped model
python cleanup_models.py --provider openrouter --add-model gpt-4 --model-name "My GPT-4"
python cleanup_models.py --provider all --delete-model "model_name" --dry-run  # Preview deletion
python cleanup_models.py --provider all --dry-run          # Preview only
```

### Provider-Specific Scripts

```bash
python cleanup_openrouter_models.py                        # Process default config.yaml
python cleanup_openrouter_models.py --config /path/to/config.yaml
python cleanup_openrouter_models.py --dry-run              # Preview changes
python cleanup_openrouter_models.py --dry-run --verbose    # Detailed preview
python cleanup_openrouter_models.py --add-model model1 model2  # Add models
python cleanup_openrouter_models.py --add-model model --model-name "custom-name"
```

### Adding Multiple Models (Space-Separated)

```bash
python cleanup_openrouter_models.py --add-model mistralai/mistral-medium anthropic/claude-3-5-sonnet
python cleanup_requesty_models.py --add-model coding/gemini-2.5-flash smart-task
python cleanup_vercel_models.py --add-model alibaba/qwen-3-14b alibaba/qwen-3-30b
python cleanup_poe_models.py --add-model Claude-Sonnet-4.5 GPT-4-Turbo
python cleanup_nvidia_models.py --add-model meta/llama-3.1-8b-instruct nvidia/llama-3.1-nemotron-70b
python cleanup_opencode_go_models.py --add-model openai/glm-5 anthropic/minimax-m2.5
```

### Mapped Model Addition (Multi-Provider)

Define model mappings in `models.yaml`:

```yaml
models:
  glm-5:
    display_name: "zai-glm-5"
    description: "GLM-5 model by Z.ai"
    providers:
      openrouter: z-ai/glm-5
      kilo: z-ai/glm-5
      fireworks: accounts/fireworks/models/glm-5
```

Then add with one command:

```bash
python cleanup_models.py --provider all --add-mapped-model glm-5
```

### Auto-Populating `models.yaml` (`populate_models.py`)

Manually looking up a new model across every provider is tedious, and providers
often use different naming conventions for the same underlying model (e.g.
`glm-5.1` vs `glm-5-1` vs `glm-5p1`, or `minimax/minimax-m3` vs `anthropic/minimax-m3`).
`populate_models.py` fetches the model list from every provider defined in
`providers.yaml` and uses tiered fuzzy matching to find the best match for a
canonical model key, writing the results back into `models.yaml`.

```bash
# Populate (dry-run) — previews matches, does not write
python populate_models.py minimax-m3 --dry-run

# Apply
python populate_models.py glm-5.1

# Limit the search to specific providers
python populate_models.py minimax-m3 --provider openrouter,kilo,vercel

# Overwrite an existing entry
python populate_models.py minimax-m3 --force

# Leave a pre-existing entry alone
python populate_models.py minimax-m3 --skip-existing
```

Matching tiers (highest score wins regardless of API order):

1. `1.00` — exact id match
2. `0.90` — id matches with a vendor prefix stripped (e.g. `z-ai/glm-5.1` ↔ `glm-5.1`)
3. `0.85` — normalized forms are equal (case, separators, `p`-as-point)
4. `0.75` — normalized forms equal with one trailing suffix stripped (`:free`, `-fw`, `-el`, `-t`, `-it`)
5. `0.60` — substring fallback (only used when no better match exists)

`populate_models.py` rewrites the entire `models.yaml` file (via `yaml.dump`),
so any hand-written comments in it will be lost. A `.yaml.backup` is written
before each save.

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--provider PROVIDER` | Provider: `openrouter`, `requesty`, `vercel`, `poe`, `kilo`, `nvidia`, `ollama`, `fireworks`, `opencode-zen`, `opencode-go`, or `all` |
| `--config CONFIG` | Path to config file (default: `config.yaml`) |
| `--dry-run` | Preview changes without modifying the file |
| `--verbose` | Detailed logging with cost comparison and percentage changes |
| `--add-model MODEL_ID [MODEL_ID ...]` | Add one or more models (space-separated) |
| `--delete-model NAME [NAME ...]` | Delete one or more models by `model_name` |
| `--add-mapped-model NAME` | Add a model defined in `models.yaml` across all providers |
| `--model-name NAME` | Custom name for single-model additions |

### `populate_models.py` Options

| Option | Description |
|--------|-------------|
| `MODEL_KEY` (positional) | Canonical model key to look up (e.g. `minimax-m3`, `glm-5.1`) |
| `--display-name NAME` | Display name (default: `MODEL_KEY`) |
| `--description TEXT` | Description for the entry |
| `--provider a,b,c` | Limit search to specific providers (default: all in `providers.yaml`) |
| `--providers-config PATH` | Path to `providers.yaml` (default: `providers.yaml`) |
| `--models-config PATH` | Path to `models.yaml` (default: `models.yaml`) |
| `--config PATH` | Path to `config.yaml` (used to instantiate cleaners) |
| `--dry-run` | Preview without writing |
| `--verbose` | Verbose logging |
| `--force` | Overwrite an existing `models.yaml` entry |
| `--skip-existing` | Don't touch a pre-existing entry |

## Features

All scripts share these capabilities:

- **Dry-run mode**: Preview all changes (removals, cost updates, additions, sorting) before applying
- **Automatic cost updates**: Syncs `input_cost_per_token` and `output_cost_per_token` with current API pricing
- **Percentage-based logging**: Shows cost changes with percentage differences (e.g., `1e-06 → 3e-07 (-70.0%)`)
- **Free model handling**: Preserves `1e-09` costs for free models (LiteLLM compatibility)
- **Duplicate prevention**: Skips models that already exist in configuration
- **Smart naming**: Generates appropriate model names with conflict resolution
- **Custom naming**: Use `--model-name` when adding a single model
- **Preserves structure**: Maintains YAML formatting
- **Complete removal**: Removes entire model entries (not just references)
- **Model list sorting**: Alphabetically sorts models by `model_name` and `litellm_params.model`
- **Error handling**: Graceful handling of network issues, YAML parse errors, and missing files

### Provider-Specific Notes

- **OpenRouter**: Automatically adds both free (`:free`) and paid versions with the same name
- **OpenCode Go**: Supports multi-prefix detection (`openai/`, `dashscope/`, `anthropic/`) with correct API base routing
- **Nvidia NIM**: All models are free — uses `1e-09` cost for LiteLLM compatibility
- **Requesty**: Preserves existing model names for load balancing across providers

## How It Works

### Model Validation and Cost Updates
1. **Load Configuration**: Parse YAML config file
2. **Extract Models**: Identify provider-specific models by prefix or `api_base` pattern
3. **Fetch Available Models with Pricing**: Query the provider API
4. **Validate**: Compare config models against API models
5. **Update Costs**: Compare and update `input_cost_per_token` / `output_cost_per_token`
6. **Handle Free Models**: Preserve `1e-09` costs for free models
7. **Remove Invalid Entries**: Delete entire entries for deprecated models
8. **Sort**: Alphabetically sort model list
9. **Save**: Write updated configuration back to file
10. **Report**: Summary of removals, cost updates, and additions

### Model Addition
1. Fetch available models from API (once for efficiency)
2. For each model ID: validate against API, check for duplicates, generate name, apply costs
3. Save configuration once after processing all models
4. Report successful additions and failures

## Example Output

### Dry-Run with Cost Updates
```
2025-07-27 21:07:05 - INFO - Loading configuration from config.yaml
2025-07-27 21:07:05 - INFO - Found 28 OpenRouter models in configuration
2025-07-27 21:07:05 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:07:05 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:07:05 - INFO - Cost update for openrouter/qwen/qwen3-14b (name: or-qwen3-14b)
2025-07-27 21:07:05 - INFO -   Input cost: 7e-08 → 6e-08 (-14.3%)
2025-07-27 21:07:05 - INFO - Identified 1 models with cost updates
2025-07-27 21:07:05 - INFO - [DRY-RUN] Would update costs for 1 models
2025-07-27 21:07:05 - INFO - [DRY-RUN] No changes made to file. Use without --dry-run to apply changes.
```

### Adding New Models
```
2025-07-27 21:36:38 - INFO - Loading configuration from config.yaml
2025-07-27 21:36:38 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:36:38 - INFO - Added model 'qwen/qwen-2.5-72b-instruct' with name 'or-2.5-72b-instruct'
2025-07-27 21:36:38 - INFO -   Input cost: 1.01e-07
2025-07-27 21:36:38 - INFO -   Output cost: 1.01e-07
2025-07-27 21:36:38 - INFO - ✅ Successfully added 1 model(s)
```

### All Models Valid
```
2025-07-27 21:09:22 - INFO - No cost updates needed - all costs are current
2025-07-27 21:09:22 - INFO - ✅ All OpenRouter models are valid with current costs
```

## Safety Features

- **Dry-Run Mode**: Preview all changes before applying
- **Comprehensive Logging**: Track every action
- **Error Recovery**: Fails safely without corrupting files
- **Git-Friendly**: No backup files created (you're using git)

## Dependencies

- **PyYAML**: Safe YAML parsing and writing
- **requests**: HTTP API calls to provider APIs

## Best Practices

1. **Always dry-run first** with `--dry-run`
2. **Use `--verbose`** when troubleshooting
3. **Commit config to git** before running
4. **Run periodically** to keep costs current and remove deprecated models
5. **Use the unified script** (`cleanup_models.py --provider all`) for batch processing
6. **Custom model names** only work when adding a single model at a time
7. **Check model IDs** match the exact format from the provider's API

## Troubleshooting

- **"Configuration file not found"**: Use `--config` to specify the correct path
- **"Error fetching models from API"**: Check internet connectivity; API may be temporarily unavailable
- **"YAML parsing error"**: Validate your YAML syntax
- **"--model-name can only be used when adding a single model"**: Remove `--model-name` when adding multiple models

Run any script with `--help` to see all available options.

## Script Architecture

### Unified Script (`cleanup_models.py`)

Uses a provider-based architecture:

- **ProviderManager**: Loads provider configs from `providers.yaml`
- **ProviderStrategy**: Abstract base for provider-specific logic
  - `PrefixDetectionStrategy`: For prefix-based identification (OpenRouter, Vercel, Nvidia)
  - `ApiBaseDetectionStrategy`: For API base-based identification (Requesty, Poe)
- **UnifiedModelCleaner**: Orchestrates cleanup across all providers

### Provider-Specific Scripts

All scripts use `ConfigDrivenModelCleaner` (from `cleanup_base.py`), which loads configuration from `providers.yaml`:

- API URLs, model prefixes, pricing field mappings
- Model detection strategy (prefix or api_base)
- Model name prefixes and cleanup rules
- Special models to exclude from validation
- Provider order priority

Each script contains only:
- Provider-specific API response parsing
- Provider-specific business logic (e.g., OpenRouter free variants)
- Standard `main()` function with CLI argument handling

### Key Classes (in `cleanup_base.py`)

- `BaseModelCleaner` — abstract base for YAML load/save, sort, validate, cost update
- `ConfigDrivenModelCleaner` — reads `providers.yaml`, implements all abstract methods
- `UnifiedModelCleaner` — creates one `ConfigDrivenModelCleaner` per provider, delegates all operations

## License

Provided as-is for LiteLLM configuration management.
