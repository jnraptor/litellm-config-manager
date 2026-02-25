# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a LiteLLM configuration management repository with Python scripts for managing model configurations across multiple AI providers: OpenRouter, Requesty, Novita AI, Nano-GPT, Vercel AI Gateway, Poe, Kilo, and Nvidia NIM. The main config file (`config.yaml`) contains LiteLLM model definitions with routing strategies and cost information.

**Key Features:**
- Unified script (`cleanup_models.py`) for managing all providers via strategy pattern
- Provider-specific scripts for individual provider management
- Provider configuration via `providers.yaml`
- Custom model naming support
- Automatic model list sorting
- Cost synchronization with provider APIs
- Batch model addition
- Comprehensive dry-run and verbose modes

## Development Commands

### Unified Script (Recommended)
```bash
# Validate models, update costs, and remove invalid entries for all providers
python cleanup_models.py --provider all [--config config.yaml] [--dry-run] [--verbose]

# Process specific provider
python cleanup_models.py --provider openrouter [--config config.yaml] [--dry-run] [--verbose]
python cleanup_models.py --provider requesty [--config config.yaml] [--dry-run] [--verbose]
python cleanup_models.py --provider novita [--config config.yaml] [--dry-run] [--verbose]
python cleanup_models.py --provider nano_gpt [--config config.yaml] [--dry-run] [--verbose]
python cleanup_models.py --provider vercel [--config config.yaml] [--dry-run] [--verbose]
python cleanup_models.py --provider poe [--config config.yaml] [--dry-run] [--verbose]
python cleanup_models.py --provider kilo [--config config.yaml] [--dry-run] [--verbose]
python cleanup_models.py --provider nvidia [--config config.yaml] [--dry-run] [--verbose]

# Add new models with automatic cost detection
python cleanup_models.py --provider openrouter --add-model model1 model2 [--dry-run]
python cleanup_models.py --provider poe --add-model Claude-Sonnet-4.5 GPT-4-Turbo [--dry-run]
python cleanup_models.py --provider nvidia --add-model meta/llama-3.1-8b-instruct [--dry-run]

# Add model with custom name (single model only)
python cleanup_models.py --provider openrouter --add-model gpt-4 --model-name "My GPT-4" [--dry-run]
python cleanup_models.py --provider poe --add-model Claude-Sonnet-4.5 --model-name "my-claude" [--dry-run]
```

### Provider-Specific Scripts
```bash
# Validate models, update costs, and remove invalid entries
python cleanup_openrouter_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_requesty_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_novita_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_nano_gpt_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_vercel_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_poe_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_kilo_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_nvidia_models.py [--config config.yaml] [--dry-run] [--verbose]

# Add new models with automatic cost detection
python cleanup_openrouter_models.py --add-model "model/name" [--dry-run]
python cleanup_requesty_models.py --add-model "provider/model" [--dry-run]
python cleanup_novita_models.py --add-model "provider/model" [--dry-run]
python cleanup_nano_gpt_models.py --add-model "provider/model" [--dry-run]
python cleanup_vercel_models.py --add-model "model/name" [--dry-run]
python cleanup_poe_models.py --add-model "Model-Name" [--dry-run]
python cleanup_kilo_models.py --add-model "anthropic/claude-opus-4.6" [--dry-run]
python cleanup_nvidia_models.py --add-model "meta/llama-3.1-8b-instruct" [--dry-run]

# Add models with custom names (single model only)
python cleanup_openrouter_models.py --add-model "model/name" --model-name "my-custom-name" [--dry-run]
python cleanup_requesty_models.py --add-model "provider/model" --model-name "my-model" [--dry-run]
python cleanup_poe_models.py --add-model "Claude-Sonnet-4.5" --model-name "my-claude" [--dry-run]

# Multiple model addition (space-separated)
python cleanup_openrouter_models.py --add-model model1 model2 model3 [--dry-run]
python cleanup_vercel_models.py --add-model model1 model2 model3 [--dry-run]
python cleanup_poe_models.py --add-model Claude-Sonnet-4.5 GPT-4-Turbo Gemini-2.0-Flash [--dry-run]
```

### Dependencies Management
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create virtual environment (if needed)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Set up API keys (if required)
export REQUESTY_API_KEY="your-requesty-api-key"
export NANOGPT_API_KEY="your-nanogpt-api-key"
# Or use .env file
echo "REQUESTY_API_KEY=your-key" >> .env
echo "NANOGPT_API_KEY=your-key" >> .env
```

**API Key Requirements:**
- **Requesty**: Requires `REQUESTY_API_KEY` environment variable
- **Nano-GPT**: Requires `NANOGPT_API_KEY` environment variable
- **Kilo**: Requires `KILO_API_KEY` environment variable
- **OpenRouter, Novita, Vercel, Poe, Nvidia**: No API key required for model listing

### Testing Workflow

Always run with `--dry-run --verbose` first to preview changes before applying them:

```bash
# Test unified script across all providers
python cleanup_models.py --provider all --dry-run --verbose

# Test specific provider
python cleanup_models.py --provider openrouter --dry-run --verbose

# Test legacy scripts
python cleanup_openrouter_models.py --dry-run --verbose
```

**Before committing config changes:**
1. Run with `--dry-run --verbose` to review all changes
2. Review the output for unexpected model removals or cost changes
3. Apply changes without `--dry-run` only after validation

## Architecture

### Core Components

**Configuration File (`config.yaml`):**
- LiteLLM router configuration with model definitions
- Model routing strategies (cost-based-routing)
- Redis caching configuration
- Model cost specifications (`input_cost_per_token`, `output_cost_per_token`)

**Unified Cleanup Script:**
- `cleanup_models.py` - Single configurable script for all providers (~1,301 lines)
- `providers.yaml` - Provider configuration file with API endpoints and settings
- Strategy pattern implementation for provider-specific logic
- Supports all providers: openrouter, requesty, novita, nano_gpt, vercel, poe, kilo, nvidia

**Provider-Specific Cleanup Scripts:**
- `cleanup_openrouter_models.py` - OpenRouter API integration (~1,138 lines)
- `cleanup_requesty_models.py` - Requesty API integration (~1,034 lines)
- `cleanup_novita_models.py` - Novita API integration (~994 lines)
- `cleanup_poe_models.py` - Poe API integration (~957 lines)
- `cleanup_vercel_models.py` - Vercel AI Gateway API integration (~623 lines)
- `cleanup_nano_gpt_models.py` - Nano-GPT API integration (~563 lines)
- `cleanup_kilo_models.py` - Kilo API integration (~64 lines)
- `cleanup_nvidia_models.py` - Nvidia NIM API integration (~584 lines, free models only)

**GitHub Automation:**
- Weekly automated cleanup via GitHub Actions
- Manual model addition through workflow dispatch
- Automatic commits when changes are detected

**Base Module (`cleanup_base.py`):**
- Shared utilities for all cleanup scripts (~750 lines)
- Common logging setup with verbose support
- YAML file operations with error handling
- Cost comparison utilities using relative tolerance for scientific notation
- Model list sorting by `model_name` and `litellm_params.model`
- Base classes for provider-specific implementations
- CLI argument parsing utilities

### Script Architecture Pattern

#### Unified Script Architecture
The `cleanup_models.py` script uses a provider-agnostic approach that centralizes configuration and reduces code duplication:

**Provider Configuration Management:**
- `providers.yaml` defines all provider settings, API endpoints, and behavior
- Each provider has: API URL, model detection method, pricing field mapping, cost handling
- Strategy pattern for provider-specific model detection and pricing logic
- Supports two detection methods:
  - **Prefix-based**: Matches `litellm_params.model` prefix (OpenRouter, Novita, Vercel, Nvidia)
  - **API-base-based**: Matches `litellm_params.api_base` URL (Requesty, Nano-GPT, Poe, Kilo)

**Core Classes:**
- `ProviderConfig` - Data class encapsulating provider settings from providers.yaml
- `ProviderStrategy` - Abstract base class defining detection, API parsing, and naming logic
- `PrefixDetectionStrategy` - Implements detection for prefix-based providers
- `ApiBaseDetectionStrategy` - Implements detection for API-base-based providers
- `UnifiedModelCleaner` - Main orchestrator that processes models across all providers

**Key Methods:**
- `load_provider_config()` - Parses providers.yaml into ProviderConfig objects
- `extract_models_by_provider()` - Provider-agnostic model extraction using strategy
- `fetch_provider_models()` - Fetches and caches API data with provider-specific handling
- `parse_pricing_data()` - Extracts pricing based on provider's configuration
- `generate_model_name()` - Generates names using provider-specific prefixes and cleanup rules

**Model Identification:**
- **OpenRouter**: `litellm_params.model` starts with `openrouter/`
- **Requesty**: `litellm_params.api_base` contains `router.requesty.ai` and `litellm_params.model` starts with `openai/`
- **Novita**: `litellm_params.model` starts with `novita/`
- **Nano-GPT**: `litellm_params.model` starts with `openai/` and `litellm_params.api_base` contains `NANOGPT_API_BASE`
- **Vercel AI Gateway**: `litellm_params.model` starts with `vercel_ai_gateway/`
- **Poe**: `litellm_params.api_base` contains `api.poe.com` and `litellm_params.model` starts with `openai/`
- **Kilo**: `litellm_params.api_base` contains `os.environ/KILO_API_BASE` and `litellm_params.model` starts with `openai/`
- **Nvidia NIM**: `litellm_params.model` starts with `nvidia_nim/`

**Design Pattern Benefits:**
- Adding new providers only requires updating `providers.yaml`, no code changes needed (in most cases)
- Consistent behavior across all providers through unified interface
- Centralized cost comparison and validation logic

#### Legacy Script Architecture
Each legacy cleanup script follows the same pattern:
- `ModelCleaner` class with API integration
- Model extraction from config based on provider-specific patterns
- Cost validation and automatic updates
- Model addition functionality with duplicate prevention
- Comprehensive logging and dry-run capabilities

**Model Identification:**
- **OpenRouter**: `litellm_params.model` starts with `openrouter/`
- **Requesty**: `litellm_params.api_base` contains `router.requesty.ai`
- **Novita**: `litellm_params.model` starts with `novita/`
- **Nano-GPT**: `litellm_params.model` starts with `openai/` and `litellm_params.api_base` contains `NANOGPT_API_BASE`
- **Vercel AI Gateway**: `litellm_params.model` starts with `vercel_ai_gateway/`
- **Poe**: `litellm_params.api_base` contains `api.poe.com` and `litellm_params.model` starts with `openai/`
- **Kilo**: `litellm_params.api_base` contains `os.environ/KILO_API_BASE` and `litellm_params.model` starts with `openai/`
- **Nvidia NIM**: `litellm_params.model` starts with `nvidia_nim/`

**Key Methods:**
- `load_config()` / `save_config()` - YAML file operations
- `extract_*_models()` - Provider-specific model extraction
- `fetch_available_models()` - API integration for current models/pricing
- `validate_and_update_costs()` - Cost synchronization
- `add_model_to_config()` - New model addition with custom naming support
- `sort_model_list()` - Alphabetical sorting by model_name and litellm_params.model
- `preview_add_model()` - Dry-run preview for model additions
- `_costs_are_equal()` - Relative tolerance comparison for scientific notation

## Configuration Management

**Model Entry Structure:**
```yaml
- model_name: display-name
  litellm_params:
    model: provider/actual-model-id
    api_key: os.environ/API_KEY_NAME
    input_cost_per_token: 1.0e-07
    output_cost_per_token: 3.0e-07
```

**Free Model Handling:**
- Uses `1.0e-09` costs for free models (LiteLLM compatibility)
- Automatic detection and preservation of free model pricing
- **Nvidia NIM models**: All models are free, no pricing data in API - all use `1.0e-09` cost

**Load Balancing:**
- Multiple entries can share the same `model_name` for request distribution
- Requesty script preserves existing model names for load balancing

**Embedding Models:**
- OpenRouter supports embedding models via separate API endpoint
- Embedding models automatically tagged with `model_info.mode: embedding`

## Choosing Between Unified and Legacy Scripts

**Use the Unified Script (`cleanup_models.py`) for:**
- Processing multiple providers efficiently (`--provider all`)
- Standardized interface across all providers
- Easier configuration management via `providers.yaml`
- New feature development and improvements

**Use Legacy Scripts (`cleanup_*.py`) for:**
- Provider-specific edge cases or custom handling
- Backward compatibility with existing workflows
- Standalone provider management without dependencies
- Testing individual provider logic in isolation

**Key Difference:**
- Unified: Configuration-driven, provider-agnostic, extensible
- Legacy: Provider-specific, standalone, direct API integration

## GitHub Actions

**Automated Workflows:**
- `.github/workflows/cleanup-all-models-unified.yml` - Unified workflow for all providers
- `.github/workflows/cleanup-openrouter-models.yml`
- `.github/workflows/cleanup-requesty-models.yml`
- `.github/workflows/cleanup-novita-models.yml`
- `.github/workflows/cleanup-nano-gpt-models.yml`
- `.github/workflows/cleanup-vercel-models.yml`
- `.github/workflows/cleanup-poe-models.yml`
- `.github/workflows/cleanup-kilo-models.yml`
- `.github/workflows/cleanup-nvidia-models.yml`

**Schedule:** Weekly on Sundays (`0 0 * * 0`)

**Manual Triggers:** Workflow dispatch with model addition input parameter

**Process:**
1. Dry-run validation with verbose output
2. Sort model list alphabetically
3. Apply changes (remove invalid models, update costs)
4. Add new model if specified in manual trigger (with optional custom name)
5. Auto-commit changes if config.yaml modified

**Features:**
- Custom model naming with `--model-name` flag (single model only)
- Automatic model list sorting
- Batch model addition (space-separated)
- Cost synchronization with provider APIs
- Comprehensive dry-run and verbose modes

## Extending the System

**Adding a New Provider:**
1. Add provider configuration to `providers.yaml` with API details and detection method
2. Choose detection strategy: prefix-based or API-base-based
3. Set pricing field mappings and cost handling rules
4. Optional: Create legacy script if provider needs custom logic
5. Test with `python cleanup_models.py --provider <new_provider> --dry-run --verbose`

**Common Configuration Changes:**
- **Update API endpoint**: Edit `api_url` in `providers.yaml`
- **Change cost precision**: Modify `cost_comparison_precision` in defaults
- **Add special models**: Update `special_models` list for providers that need exceptions
- **Adjust model name cleanup**: Edit `model_name_cleanup` rules for better naming