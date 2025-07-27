# OpenRouter Model Cleanup, Cost Update, and Model Addition Script

A comprehensive Python script that validates OpenRouter models in LiteLLM configuration files against the current OpenRouter API, removes invalid model entries, automatically updates model costs to match current pricing, and provides an easy way to add new OpenRouter models.

## Overview

This script helps maintain your LiteLLM configuration by:
- Identifying OpenRouter models in your `config.yaml`
- Checking their validity against the current OpenRouter API
- Removing entire model entries for invalid/deprecated models
- **Automatically updating model costs** (`input_cost_per_token` and `output_cost_per_token`) when they differ from API pricing
- **Adding new OpenRouter models** with automatic cost detection and proper formatting
- Providing detailed logging with percentage-based cost change information
- Supporting dry-run capabilities for safe previewing

## Features

- ✅ **Safe Operation**: Dry-run mode to preview changes before applying
- ✅ **Automatic Cost Updates**: Synchronizes `input_cost_per_token` and `output_cost_per_token` with current API pricing
- ✅ **Easy Model Addition**: Add new OpenRouter models with automatic cost detection and proper formatting
- ✅ **Dual Version Support**: Automatically adds both free and paid versions when available (with same model name)
- ✅ **Percentage-Based Logging**: Shows cost changes with percentage differences (e.g., "Input cost: 1e-06 → 3e-07 (-70.0%)")
- ✅ **Free Model Handling**: Properly handles free models by preserving `1e-09` costs for LiteLLM compatibility
- ✅ **Duplicate Prevention**: Prevents adding models that already exist in configuration
- ✅ **Smart Naming**: Generates appropriate model names with conflict resolution
- ✅ **Comprehensive Logging**: Detailed output with verbose mode for validation, cost updates, and model addition
- ✅ **Error Handling**: Robust error handling for network and file issues
- ✅ **No Authentication Required**: Uses public OpenRouter API endpoint
- ✅ **Preserves Structure**: Maintains YAML formatting and structure
- ✅ **Complete Removal**: Removes entire model entries (not just model references)

## Installation

1. **Clone or download the script files:**
   ```bash
   # Files needed:
   # - cleanup_openrouter_models.py
   # - requirements.txt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
# Process config.yaml in current directory (validates models + updates costs)
python cleanup_openrouter_models.py

# Process a specific config file
python cleanup_openrouter_models.py --config /path/to/your/config.yaml
```

### Adding New Models

```bash
# Add a new OpenRouter model to the configuration
python cleanup_openrouter_models.py --add-model "anthropic/claude-3-5-sonnet-20241022"

# Preview adding a model without making changes
python cleanup_openrouter_models.py --add-model "qwen/qwen-2.5-72b-instruct" --dry-run

# Add a model with verbose output
python cleanup_openrouter_models.py --add-model "meta-llama/llama-3.2-1b-instruct" --verbose
```

### Dry-Run Mode (Recommended First)

```bash
# Preview all changes (model removals + cost updates) without making changes
python cleanup_openrouter_models.py --dry-run

# Detailed preview with verbose logging and percentage changes
python cleanup_openrouter_models.py --dry-run --verbose
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--config CONFIG` | Path to LiteLLM configuration file (default: `config.yaml`) |
| `--dry-run` | Preview all changes (model removals, cost updates, and model additions) without modifying the configuration file |
| `--verbose` | Enable detailed logging output with cost comparison information and percentage changes |
| `--add-model MODEL_ID` | Add a new OpenRouter model to the configuration. Provide the model ID (e.g., "anthropic/claude-3-5-sonnet-20241022") |
| `--help` | Show help message and exit |

## Example Output

### Dry-Run Mode with Cost Updates
```
2025-07-27 21:07:05 - INFO - Loading configuration from config.yaml
2025-07-27 21:07:05 - INFO - Found 28 OpenRouter models in configuration
2025-07-27 21:07:05 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:07:05 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:07:05 - INFO - Identified 0 invalid OpenRouter models
2025-07-27 21:07:05 - INFO - Cost update for openrouter/qwen/qwen3-14b (name: or-qwen3-14b)
2025-07-27 21:07:05 - INFO -   Input cost: 7e-08 → 6e-08 (-14.3%)
2025-07-27 21:07:05 - INFO - Cost update for openrouter/qwen/qwen3-32b (name: or-qwen3-32b)
2025-07-27 21:07:05 - INFO -   Input cost: 1e-07 → 2.7e-08 (-73.0%)
2025-07-27 21:07:05 - INFO -   Output cost: 3e-07 → 2.7e-08 (-91.0%)
2025-07-27 21:07:05 - INFO - Cost update for openrouter/qwen/qwen3-coder (name: or-qwen3-coder)
2025-07-27 21:07:05 - INFO -   Input cost: 1e-06 → 3e-07 (-70.0%)
2025-07-27 21:07:05 - INFO -   Output cost: 5e-06 → 1.2e-06 (-76.0%)
2025-07-27 21:07:05 - INFO - Identified 3 models with cost updates
2025-07-27 21:07:05 - INFO - [DRY-RUN] No invalid OpenRouter models found. No changes needed.
2025-07-27 21:07:05 - INFO - [DRY-RUN] Would update costs for 3 models:
2025-07-27 21:07:05 - INFO - 💰 [DRY-RUN] Cost updates: 3 models would have cost changes
2025-07-27 21:07:05 - INFO - 📋 [DRY-RUN] Total changes identified: 3
2025-07-27 21:07:05 - INFO - [DRY-RUN] No changes made to file. Use without --dry-run to apply changes.
```

### Actual Execution with Cost Updates
```
2025-07-27 21:08:15 - INFO - Loading configuration from config.yaml
2025-07-27 21:08:15 - INFO - Found 28 OpenRouter models in configuration
2025-07-27 21:08:15 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:08:15 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:08:15 - INFO - Identified 0 invalid OpenRouter models
2025-07-27 21:08:15 - INFO - Cost update for openrouter/qwen/qwen3-coder (name: or-qwen3-coder)
2025-07-27 21:08:15 - INFO -   Input cost: 1e-06 → 3e-07 (-70.0%)
2025-07-27 21:08:15 - INFO -   Output cost: 5e-06 → 1.2e-06 (-76.0%)
2025-07-27 21:08:15 - INFO - Identified 1 models with cost updates
2025-07-27 21:08:15 - INFO - Saving updated configuration to config.yaml
2025-07-27 21:08:15 - INFO - Configuration saved successfully
2025-07-27 21:08:15 - INFO - ✅ Cost updates: 1 models had cost changes applied
2025-07-27 21:08:15 - INFO - ✅ Cleanup completed: 1 total changes applied
```

### All Models Valid with Current Costs
```
2025-07-27 21:09:22 - INFO - Loading configuration from config.yaml
2025-07-27 21:09:22 - INFO - Found 28 OpenRouter models in configuration
2025-07-27 21:09:22 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:09:22 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:09:22 - INFO - Identified 0 invalid OpenRouter models
2025-07-27 21:09:22 - INFO - No cost updates needed - all costs are current
2025-07-27 21:09:22 - INFO - ✅ All OpenRouter models are valid with current costs
```

### Adding New Models
```
2025-07-27 21:36:38 - INFO - Loading configuration from config.yaml
2025-07-27 21:36:38 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:36:38 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:36:38 - INFO - Found 29 OpenRouter models in configuration
2025-07-27 21:36:38 - INFO - Added model 'qwen/qwen-2.5-72b-instruct' with name 'or-2.5-72b-instruct'
2025-07-27 21:36:38 - INFO -   Input cost: 1.01e-07
2025-07-27 21:36:38 - INFO -   Output cost: 1.01e-07
2025-07-27 21:36:38 - INFO - Also added free version 'qwen/qwen-2.5-72b-instruct:free' with same name 'or-2.5-72b-instruct'
2025-07-27 21:36:38 - INFO - Saving updated configuration to config.yaml
2025-07-27 21:36:38 - INFO - Configuration saved successfully
2025-07-27 21:36:38 - INFO - ✅ Successfully added 2 model(s): qwen/qwen-2.5-72b-instruct, qwen/qwen-2.5-72b-instruct:free
```

### Adding Models (Dry-Run)
```
2025-07-27 21:36:29 - INFO - Loading configuration from config.yaml
2025-07-27 21:36:29 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:36:29 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:36:29 - INFO - Loading configuration from config.yaml
2025-07-27 21:36:29 - INFO - Found 29 OpenRouter models in configuration
2025-07-27 21:36:29 - INFO - [DRY-RUN] Would add model 'qwen/qwen-2.5-72b-instruct' with name 'or-2.5-72b-instruct'
2025-07-27 21:36:29 - INFO - [DRY-RUN]   Input cost: 1.01e-07
2025-07-27 21:36:29 - INFO - [DRY-RUN]   Output cost: 1.01e-07
2025-07-27 21:36:29 - INFO - [DRY-RUN] Would also add free version 'qwen/qwen-2.5-72b-instruct:free' with same name 'or-2.5-72b-instruct'
```

### Duplicate Model Detection
```
2025-07-27 21:34:49 - INFO - Loading configuration from config.yaml
2025-07-27 21:34:49 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:34:49 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:34:49 - INFO - Loading configuration from config.yaml
2025-07-27 21:34:49 - INFO - Found 29 OpenRouter models in configuration
2025-07-27 21:34:49 - WARNING - [DRY-RUN] Model 'meta-llama/llama-3.2-1b-instruct' already exists in configuration
```

## How It Works

### Model Validation and Cost Updates
1. **Load Configuration**: Parses the YAML configuration file safely
2. **Extract OpenRouter Models**: Identifies all models starting with `openrouter/`
3. **Fetch Available Models with Pricing**: Queries the OpenRouter API for current model list and pricing information
4. **Validate Models**: Compares config models against API response to identify invalid models
5. **Validate and Update Costs**: Compares current costs with API pricing and updates when differences are found
6. **Handle Free Models**: Preserves `1e-09` costs for free models (LiteLLM compatibility) when API returns `0.0`
7. **Remove Invalid Entries**: Removes entire model entries for invalid/deprecated models
8. **Save Configuration**: Writes the updated configuration back to file (if any changes were made)
9. **Generate Report**: Provides comprehensive summary of model removals and cost updates with percentage changes

### Model Addition (--add-model)
1. **Load Configuration**: Parses the YAML configuration file safely
2. **Fetch Available Models with Pricing**: Queries the OpenRouter API for current model list and pricing information
3. **Find Model in API**: Searches for the specified model ID in the API response
4. **Check for Duplicates**: Verifies the model doesn't already exist in the configuration
5. **Generate Model Name**: Creates an appropriate model name (e.g., "qwen/qwen-2.5-72b-instruct" → "or-2.5-72b-instruct")
6. **Handle Dual Versions**: Automatically adds both paid and free versions when available (with same model name)
7. **Apply Costs**: Sets appropriate costs based on API pricing (1e-09 for free models)
8. **Save Configuration**: Writes the updated configuration back to file
9. **Generate Report**: Provides summary of added models

## Model Identification

The script identifies OpenRouter models by looking for entries where:
- `litellm_params.model` starts with `openrouter/`
- Examples: `openrouter/qwen/qwen3-14b:free`, `openrouter/anthropic/claude-3.5-haiku`

## Validation Logic

### Model Validation
- Config models like `openrouter/qwen/qwen3-14b:free` are compared against API models like `qwen/qwen3-14b:free`
- The `openrouter/` prefix is stripped for comparison
- Models not found in the API response are marked as invalid

### Cost Validation and Updates
- The script fetches current pricing from the OpenRouter API (`pricing.prompt` and `pricing.completion` fields)
- Compares `input_cost_per_token` and `output_cost_per_token` in your config with API pricing
- **Automatically updates costs** when differences are detected
- **Free model handling**: When API returns `0.0` for free models, the script preserves `1e-09` costs for LiteLLM compatibility
- **Percentage tracking**: All cost changes are logged with percentage differences for easy impact assessment

## Error Handling

The script handles various error conditions:

- **File Not Found**: Clear error if config file doesn't exist
- **YAML Parse Errors**: Detailed error messages for malformed YAML
- **Network Issues**: Graceful handling of API connectivity problems
- **Invalid API Response**: Validation of API response format
- **File Write Errors**: Error handling for file permission issues

## Safety Features

- **Dry-Run Mode**: Preview changes before applying them
- **Comprehensive Logging**: Track exactly what the script is doing
- **Error Recovery**: Script fails safely without corrupting files
- **Git-Friendly**: Since you mentioned using git, no backup files are created

## Dependencies

- **PyYAML**: For safe YAML parsing and writing
- **requests**: For HTTP API calls to OpenRouter

## Troubleshooting

### Common Issues

1. **"Configuration file not found"**
   - Ensure the config file exists in the specified path
   - Use `--config` to specify the correct path

2. **"Error fetching models from OpenRouter API"**
   - Check internet connectivity
   - OpenRouter API might be temporarily unavailable
   - Try again after a few minutes

3. **"YAML parsing error"**
   - The config file might have syntax errors
   - Validate your YAML syntax

### Getting Help

Run the script with `--help` to see all available options:
```bash
python cleanup_openrouter_models.py --help
```

## Best Practices

1. **Always run with `--dry-run` first** to preview changes
2. **Use `--verbose` for detailed information** when troubleshooting
3. **Commit your config to git before running** (as you mentioned using git)
4. **Run periodically** to keep your configuration up to date
5. **Check the logs** to understand what models were removed and why

## Script Architecture

The script is organized into a main `OpenRouterModelCleaner` class with the following key methods:

### Core Methods
- `load_config()`: Safe YAML loading with error handling
- `extract_openrouter_models()`: Find OpenRouter models in config
- `fetch_available_models()`: Query OpenRouter API for models and pricing data
- `validate_models()`: Compare config vs API models for validity
- `remove_invalid_entries()`: Remove invalid model entries
- `save_config()`: Write updated configuration
- `generate_report()`: Summary reporting with model changes, cost updates, and additions

### Cost Update Methods
- `validate_and_update_costs()`: Compare and update model costs with API pricing
- `preview_cost_changes()`: Show cost changes in dry-run mode with percentage differences

### Model Addition Methods
- `generate_model_name()`: Generate appropriate model names from OpenRouter model IDs
- `find_model_in_api()`: Search for specific models in API data with fallback logic
- `add_model_to_config()`: Add new models to configuration with proper cost handling
- `preview_add_model()`: Preview model additions in dry-run mode with duplicate checking

### Enhanced Features
- **Automatic cost synchronization**: Keeps your config costs current with API pricing
- **Easy model addition**: Add new OpenRouter models with automatic cost detection
- **Dual version support**: Automatically adds both free and paid versions with same model name
- **Free model compatibility**: Handles LiteLLM's requirement for non-zero costs
- **Duplicate prevention**: Prevents adding models that already exist
- **Smart naming**: Generates appropriate model names with conflict resolution
- **Percentage-based logging**: Clear visibility into cost impact
- **Triple functionality**: Model validation, cost updates, and model addition in a single tool

## License

This script is provided as-is for LiteLLM configuration management.