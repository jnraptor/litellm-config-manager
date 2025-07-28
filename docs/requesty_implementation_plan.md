# Requesty Model Management Implementation Plan

## Overview
This document outlines the plan to create a Requesty model management script similar to the existing OpenRouter cleanup script. The key difference is that we'll maintain existing model names to allow LiteLLM to route requests across different providers for the same model.

## Analysis of Existing Implementation

### OpenRouter Script Features
1. **Model Fetching**: Fetches available models from `https://openrouter.ai/api/v1/models`
2. **Model Validation**: Compares config models against API models
3. **Cost Updates**: Updates input/output costs when they differ from API
4. **Model Addition**: Can add new models with appropriate naming
5. **Sorting**: Sorts model list alphabetically

### Requesty Models in Current Config
Looking at the config.yaml, Requesty models follow this pattern:
- Model format: `openai/<category>/<model-name>` (e.g., `openai/coding/gemini-2.5-flash`)
- API base: `https://router.requesty.ai/v1`
- API key: `os.environ/REQUESTY_API_KEY`
- Model names are shared across providers (e.g., `gemini-2.5-flash` is used for both Requesty and OpenRouter)

Current Requesty models in config:
1. `openai/coding/gemini-2.5-flash` → model_name: `gemini-2.5-flash`
2. `openai/google/gemini-2.5-flash-lite-preview-06-17` → model_name: `gemini-2.5-flash-lite`
3. `openai/coding/gemini-2.5-pro` → model_name: `gemini-2.5-pro`
4. `openai/smart/task` → model_name: `req-smart-task`

## Key Differences from OpenRouter Implementation

1. **Model Naming**: Keep existing model names to allow load balancing across providers
2. **Model ID Format**: Requesty uses `openai/` prefix instead of `openrouter/`
3. **API Endpoint**: `https://router.requesty.ai/v1/models`

## Implementation Steps

### 1. Create RequestyModelCleaner Class
- Inherit similar structure from OpenRouterModelCleaner
- Modify model extraction to look for `api_base: https://router.requesty.ai/v1`
- Adjust model ID parsing for Requesty format

### 2. Model Extraction Logic
```python
def extract_requesty_models(self, config):
    # Look for models with api_base containing router.requesty.ai
    # Extract models that use REQUESTY_API_KEY
```

### 3. API Integration
- Fetch models from `https://router.requesty.ai/v1/models`
- Parse response format (may differ from OpenRouter)
- Extract pricing information

### 4. Model Validation
- Compare config models against API models
- Handle the `openai/` prefix appropriately

### 5. Cost Updates
- Update costs while preserving model names
- Handle free tier models (1e-09 cost)

### 6. Adding New Models
- When adding new models, check if model_name already exists
- If it exists, keep the same name for load balancing
- If new, generate appropriate name

## Command Line Interface
Maintain same CLI as OpenRouter script:
```bash
python cleanup_requesty_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_requesty_models.py --add-model "category/model-name" [--dry-run]
```

## Testing Plan
1. Test API endpoint connectivity
2. Validate model extraction from config
3. Test cost update functionality
4. Test adding new models with existing names
5. Verify sorting functionality

## Next Steps
1. Switch to code mode to implement the script
2. Test with dry-run mode first
3. Update documentation