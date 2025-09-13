# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a LiteLLM configuration management repository with Python scripts for managing model configurations across OpenRouter, Requesty, Novita AI, and Nano-GPT providers. The main config file (`config.yaml`) contains LiteLLM model definitions with routing strategies and cost information.

## Development Commands

### Core Scripts
```bash
# Validate models, update costs, and remove invalid entries
python cleanup_openrouter_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_requesty_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_novita_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_nano_gpt_models.py [--config config.yaml] [--dry-run] [--verbose]

# Add new models with automatic cost detection
python cleanup_openrouter_models.py --add-model "model/name" [--dry-run]
python cleanup_requesty_models.py --add-model "provider/model" [--dry-run]
python cleanup_novita_models.py --add-model "provider/model" [--dry-run]
python cleanup_nano_gpt_models.py --add-model "provider/model" [--dry-run]

# Multiple model addition (space-separated)
python cleanup_openrouter_models.py --add-model model1 model2 model3 [--dry-run]
```

### Dependencies Management
```bash
# Install Python dependencies
pip install -r requirements.txt

# Create virtual environment (if needed)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

### Testing Workflow
Always run with `--dry-run --verbose` first to preview changes before applying them.

## Architecture

### Core Components

**Configuration File (`config.yaml`):**
- LiteLLM router configuration with model definitions
- Model routing strategies (cost-based-routing)
- Redis caching configuration
- Model cost specifications (`input_cost_per_token`, `output_cost_per_token`)

**Cleanup Scripts (4 parallel implementations):**
- `cleanup_openrouter_models.py` - OpenRouter API integration (~967 lines)
- `cleanup_requesty_models.py` - Requesty API integration (~966 lines)
- `cleanup_novita_models.py` - Novita API integration (~928 lines)
- `cleanup_nano_gpt_models.py` - Nano-GPT API integration (new)

**GitHub Automation:**
- Weekly automated cleanup via GitHub Actions
- Manual model addition through workflow dispatch
- Automatic commits when changes are detected

### Script Architecture Pattern

Each cleanup script follows the same pattern:
- `ModelCleaner` class with API integration
- Model extraction from config based on provider-specific patterns
- Cost validation and automatic updates
- Model addition functionality with duplicate prevention
- Comprehensive logging and dry-run capabilities

**Model Identification:**
- OpenRouter: `litellm_params.model` starts with `openrouter/`
- Requesty: `litellm_params.api_base` contains `router.requesty.ai`
- Novita: `litellm_params.model` starts with `novita/`
- Nano-GPT: `litellm_params.model` starts with `openai/` and `litellm_params.api_base` contains `NANOGPT_API_BASE`

**Key Methods:**
- `load_config()` / `save_config()` - YAML file operations
- `extract_*_models()` - Provider-specific model extraction
- `fetch_available_models()` - API integration for current models/pricing
- `validate_and_update_costs()` - Cost synchronization
- `add_model_to_config()` - New model addition

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

**Load Balancing:**
- Multiple entries can share the same `model_name` for request distribution
- Requesty script preserves existing model names for load balancing

## GitHub Actions

**Automated Workflows:**
- `.github/workflows/cleanup-openrouter-models.yml`
- `.github/workflows/cleanup-requesty-models.yml`
- `.github/workflows/cleanup-novita-models.yml`
- `.github/workflows/cleanup-nano-gpt-models.yml`

**Schedule:** Weekly on Sundays (`0 0 * * 0`)

**Manual Triggers:** Workflow dispatch with model addition input parameter

**Process:**
1. Dry-run validation with verbose output
2. Apply changes (remove invalid models, update costs)
3. Add new model if specified in manual trigger
4. Auto-commit changes if config.yaml modified