# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a LiteLLM configuration management repository with Python scripts for managing model configurations across multiple AI providers: OpenRouter, Requesty, Novita AI, Nano-GPT, Vercel AI Gateway, Poe, and Nvidia NIM. The main config file (`config.yaml`) contains LiteLLM model definitions with routing strategies and cost information.

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
python cleanup_nvidia_models.py [--config config.yaml] [--dry-run] [--verbose]

# Add new models with automatic cost detection
python cleanup_openrouter_models.py --add-model "model/name" [--dry-run]
python cleanup_requesty_models.py --add-model "provider/model" [--dry-run]
python cleanup_novita_models.py --add-model "provider/model" [--dry-run]
python cleanup_nano_gpt_models.py --add-model "provider/model" [--dry-run]
python cleanup_vercel_models.py --add-model "model/name" [--dry-run]
python cleanup_poe_models.py --add-model "Model-Name" [--dry-run]
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
- **OpenRouter, Novita, Vercel, Poe, Nvidia**: No API key required for model listing

### Testing Workflow
Always run with `--dry-run --verbose` first to preview changes before applying them.

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
- Supports all providers: openrouter, requesty, novita, nano_gpt, vercel, poe, nvidia

**Provider-Specific Cleanup Scripts:**
- `cleanup_openrouter_models.py` - OpenRouter API integration (~1,138 lines)
- `cleanup_requesty_models.py` - Requesty API integration (~1,034 lines)
- `cleanup_novita_models.py` - Novita API integration (~994 lines)
- `cleanup_poe_models.py` - Poe API integration (~957 lines)
- `cleanup_vercel_models.py` - Vercel AI Gateway API integration (~623 lines)
- `cleanup_nano_gpt_models.py` - Nano-GPT API integration (~563 lines)
- `cleanup_nvidia_models.py` - Nvidia NIM API integration (~584 lines, free models only)

**GitHub Automation:**
- Weekly automated cleanup via GitHub Actions
- Manual model addition through workflow dispatch
- Automatic commits when changes are detected

### Script Architecture Pattern

#### Unified Script Architecture
The `cleanup_models.py` script uses a provider-agnostic approach:

**Provider Configuration:**
- `providers.yaml` defines all provider settings, API endpoints, and behavior
- Strategy pattern for provider-specific model detection and pricing logic
- Supports prefix-based and API-based model detection methods

**Core Components:**
- `ProviderConfig` - Data class for provider configuration
- `ProviderStrategy` - Abstract base class for provider-specific logic
- `PrefixDetectionStrategy` - For providers using model prefix detection
- `ApiBaseDetectionStrategy` - For providers using API base detection
- `UnifiedModelCleaner` - Main orchestrator class

**Key Methods:**
- `load_provider_config()` - Load provider configurations
- `extract_models_by_provider()` - Provider-agnostic model extraction
- `fetch_provider_models()` - API integration with provider-specific handling
- `parse_pricing_data()` - Provider-specific pricing extraction
- `generate_model_name()` - Provider-specific naming conventions

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

## GitHub Actions

**Automated Workflows:**
- `.github/workflows/cleanup-all-models-unified.yml` - Unified workflow for all providers
- `.github/workflows/cleanup-openrouter-models.yml`
- `.github/workflows/cleanup-requesty-models.yml`
- `.github/workflows/cleanup-novita-models.yml`
- `.github/workflows/cleanup-nano-gpt-models.yml`
- `.github/workflows/cleanup-vercel-models.yml`
- `.github/workflows/cleanup-poe-models.yml`
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