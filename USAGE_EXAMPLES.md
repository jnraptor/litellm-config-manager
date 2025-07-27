# Usage Examples

This document provides practical examples of using the OpenRouter Model Cleanup, Cost Update, and Model Addition Script.

## Quick Start

### 1. First Time Usage (Recommended)

Always start with a dry-run to see what would be changed:

```bash
# Preview all changes (model validation + cost updates) with detailed output
python cleanup_openrouter_models.py --dry-run --verbose
```

### 2. Apply Changes

Once you're satisfied with the preview:

```bash
# Actually remove invalid models and update costs
python cleanup_openrouter_models.py
```

## Common Scenarios

### Scenario 1: Adding New Models

```bash
# Add a new model with preview first
python cleanup_openrouter_models.py --add-model "anthropic/claude-3-5-sonnet-20241022" --dry-run
# Review the output, then add it:
python cleanup_openrouter_models.py --add-model "anthropic/claude-3-5-sonnet-20241022"

# Add multiple models (run separately)
python cleanup_openrouter_models.py --add-model "qwen/qwen-2.5-72b-instruct"
python cleanup_openrouter_models.py --add-model "meta-llama/llama-3.2-1b-instruct"
```

### Scenario 2: Regular Maintenance

```bash
# Weekly cleanup and cost sync routine
python cleanup_openrouter_models.py --dry-run
# Review the output (model removals + cost updates), then:
python cleanup_openrouter_models.py
```

### Scenario 3: Custom Config File

```bash
# For non-standard config file locations
python cleanup_openrouter_models.py --config /path/to/my-litellm-config.yaml --dry-run
python cleanup_openrouter_models.py --config /path/to/my-litellm-config.yaml

# Add models to custom config
python cleanup_openrouter_models.py --config /path/to/my-litellm-config.yaml --add-model "anthropic/claude-3-5-haiku" --dry-run
```

### Scenario 4: Debugging Issues

```bash
# Get detailed information about model validation and cost comparisons
python cleanup_openrouter_models.py --verbose --dry-run

# Debug model addition issues
python cleanup_openrouter_models.py --add-model "some/model-id" --verbose --dry-run
```

### Scenario 5: Cost-Only Monitoring

```bash
# Check for cost changes without worrying about invalid models
python cleanup_openrouter_models.py --dry-run --verbose | grep -E "(Cost update|Input cost|Output cost)"
```

### Scenario 6: Automated Scripts

```bash
#!/bin/bash
# Example automation script for model validation, cost updates, and model addition

echo "Checking for invalid OpenRouter models and cost updates..."
python cleanup_openrouter_models.py --dry-run --verbose

echo "Press Enter to continue with cleanup and cost updates, or Ctrl+C to cancel"
read

echo "Applying model cleanup and cost updates..."
python cleanup_openrouter_models.py --verbose

# Optional: Add new models
echo "Would you like to add any new models? (y/n)"
read add_models

if [ "$add_models" = "y" ]; then
    echo "Enter model ID to add (e.g., anthropic/claude-3-5-haiku):"
    read model_id
    
    echo "Previewing model addition..."
    python cleanup_openrouter_models.py --add-model "$model_id" --dry-run
    
    echo "Press Enter to add the model, or Ctrl+C to cancel"
    read
    
    echo "Adding model..."
    python cleanup_openrouter_models.py --add-model "$model_id"
fi

echo "Cleanup, cost sync, and model management complete!"
```

## Expected Output Scenarios

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

### Cost Updates Found (Dry-Run)
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

### Cost Updates Applied (Actual Run)
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

### Invalid Models Found (Dry-Run)
```
2025-07-27 20:46:33 - INFO - Loading configuration from config.yaml
2025-07-27 20:46:33 - INFO - Found 29 OpenRouter models in configuration
2025-07-27 20:46:33 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 20:46:33 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 20:46:33 - INFO - Identified 3 invalid OpenRouter models
2025-07-27 20:46:33 - INFO - [DRY-RUN] Would remove the following invalid OpenRouter models:
2025-07-27 20:46:33 - INFO -   - Model: openrouter/qwen/qwen3-32b:free (model_name: or-qwen3-32b)
2025-07-27 20:46:33 - INFO -   - Model: openrouter/qwen/qwen3-235b-a22b-07-25:free (model_name: or-qwen3-235b-a22b-07-25)
2025-07-27 20:46:33 - INFO -   - Model: openrouter/qwen/qwen3-235b-a22b-07-25 (model_name: or-qwen3-235b-a22b-07-25)
2025-07-27 20:46:33 - INFO - [DRY-RUN] 3 model entries would be removed from config.yaml
2025-07-27 20:46:33 - INFO - [DRY-RUN] No changes made to file. Use without --dry-run to apply changes.
```

### Adding New Models (Dry-Run)
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

### Adding New Models (Actual Run)
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

### Duplicate Model Detection
```
2025-07-27 21:34:49 - INFO - Loading configuration from config.yaml
2025-07-27 21:34:49 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:34:49 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:34:49 - INFO - Loading configuration from config.yaml
2025-07-27 21:34:49 - INFO - Found 29 OpenRouter models in configuration
2025-07-27 21:34:49 - WARNING - [DRY-RUN] Model 'meta-llama/llama-3.2-1b-instruct' already exists in configuration
```

### Model Not Found in API
```
2025-07-27 21:33:30 - INFO - Loading configuration from config.yaml
2025-07-27 21:33:30 - INFO - Fetching available models with pricing from OpenRouter API...
2025-07-27 21:33:30 - INFO - Fetched 319 available models from OpenRouter API
2025-07-27 21:33:30 - ERROR - [DRY-RUN] Model 'anthropic/claude-3-5-sonnet-20241022' not found in OpenRouter API
```

## Integration Examples

### With Git Workflow

```bash
# Before running the script
git status
git add config.yaml
git commit -m "Before OpenRouter cleanup"

# Run the cleanup
python cleanup_openrouter_models.py --dry-run
python cleanup_openrouter_models.py

# Review and commit changes
git diff config.yaml
git add config.yaml
git commit -m "Remove invalid OpenRouter models"
```

### With CI/CD Pipeline

```yaml
# Example GitHub Actions workflow
name: Cleanup OpenRouter Models, Update Costs, and Manage Models
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
  workflow_dispatch:
    inputs:
      add_model:
        description: 'Model ID to add (e.g., anthropic/claude-3-5-haiku)'
        required: false
        type: string

jobs:
  cleanup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Check for invalid models and cost updates
        run: python cleanup_openrouter_models.py --dry-run --verbose
      - name: Clean up invalid models and update costs
        run: python cleanup_openrouter_models.py --verbose
      - name: Add new model (if specified)
        if: ${{ github.event.inputs.add_model != '' }}
        run: |
          echo "Adding model: ${{ github.event.inputs.add_model }}"
          python cleanup_openrouter_models.py --add-model "${{ github.event.inputs.add_model }}" --dry-run
          python cleanup_openrouter_models.py --add-model "${{ github.event.inputs.add_model }}"
      - name: Commit changes
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add config.yaml
          git diff --staged --quiet || git commit -m "Auto-cleanup invalid OpenRouter models, update costs, and manage models"
          git push
```

## Troubleshooting Examples

### Network Issues
```bash
# If you get network errors, try with verbose logging
python cleanup_openrouter_models.py --verbose --dry-run

# Example error output:
# 2025-07-27 20:46:33 - ERROR - Error fetching models from OpenRouter API: HTTPSConnectionPool(host='openrouter.ai', port=443): Max retries exceeded
```

### File Permission Issues
```bash
# If you get permission errors
chmod +w config.yaml
python cleanup_openrouter_models.py
```

### YAML Syntax Issues
```bash
# If your config has syntax errors, you'll see:
# 2025-07-27 20:46:33 - ERROR - YAML parsing error: while parsing a block mapping...

# Validate your YAML first:
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

## Best Practices

1. **Always dry-run first**: `--dry-run` is your friend for model validation, cost updates, and model additions
2. **Use version control**: Commit before running the script to track changes
3. **Regular maintenance**: Run weekly or monthly to keep costs current and remove invalid models
4. **Preview model additions**: Always use `--dry-run` when adding new models to see what will be added
5. **Monitor cost changes**: Pay attention to percentage changes in costs - significant reductions can save money
6. **Verbose debugging**: Use `--verbose` when troubleshooting to see detailed information
7. **Review cost updates**: Large cost changes (>50%) should be reviewed before applying
8. **Free model awareness**: The script preserves `1e-09` costs for free models to maintain LiteLLM compatibility
9. **Check for dual versions**: When adding models, the script automatically adds both free and paid versions with the same name
10. **Verify model IDs**: Ensure model IDs are correct before adding (check OpenRouter documentation)
11. **Batch additions**: Add models one at a time rather than trying to add multiple simultaneously

## Performance Notes

- The script typically processes 20-30 OpenRouter models in under 5 seconds
- API calls are cached during a single run and include both model validation and pricing data
- Network latency affects total runtime (usually 1-3 seconds for API call)
- YAML parsing and writing is very fast for typical config sizes
- Cost comparison adds minimal overhead to the existing model validation process

## Cost Update Benefits

- **Automatic cost synchronization**: No manual tracking of OpenRouter price changes
- **Significant savings**: Recent tests showed cost reductions of 70-90% for some models
- **Transparency**: Clear percentage-based logging shows exactly what changed
- **Safety**: Free models are handled correctly to maintain LiteLLM compatibility