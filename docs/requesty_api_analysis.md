# Requesty API Analysis

## API Response Format

The Requesty models endpoint (`https://router.requesty.ai/v1/models`) returns a JSON response with the following structure:

```json
{
  "object": "list",
  "data": [
    {
      "id": "coding/gemini-2.5-flash",
      "object": "model",
      "created": 1747765524,
      "owned_by": "system",
      "input_price": 3e-7,
      "caching_price": 5.5e-7,
      "cached_price": 7.5e-8,
      "output_price": 0.0000025,
      "max_output_tokens": 65535,
      "context_window": 1048576,
      "supports_caching": true,
      "supports_vision": true,
      "supports_computer_use": false,
      "supports_reasoning": true,
      "description": "..."
    }
  ]
}
```

## Model ID Mapping

The Requesty API returns model IDs in the format: `category/model-name`
- Example: `coding/gemini-2.5-flash`

In the config.yaml, these are prefixed with `openai/`:
- Config format: `openai/category/model-name`
- Example: `openai/coding/gemini-2.5-flash`

This is a simple prefix mapping, making implementation straightforward.

## Key Differences from OpenRouter

1. **Price Fields**: 
   - Requesty uses `input_price` and `output_price` (not `prompt` and `completion` like OpenRouter)
   - Additional fields: `caching_price`, `cached_price`

2. **Model ID Format**:
   - OpenRouter: `provider/model-name` → config: `openrouter/provider/model-name`
   - Requesty: `category/model-name` → config: `openai/category/model-name`

3. **Additional Metadata**:
   - `supports_caching`, `supports_vision`, `supports_computer_use`, `supports_reasoning`
   - `context_window`, `max_output_tokens`
   - `description`

## Implementation Notes

1. **Model Extraction**: Look for models with `api_base: https://router.requesty.ai/v1`
2. **Model Validation**: Strip `openai/` prefix and compare with API model IDs
3. **Cost Mapping**:
   - `input_price` → `input_cost_per_token`
   - `output_price` → `output_cost_per_token`
4. **Model Names**: Keep existing model names for load balancing across providers

## Current Requesty Models in Config

- `openai/coding/gemini-2.5-flash` → model_name: `gemini-2.5-flash`
- `openai/google/gemini-2.5-flash-lite-preview-06-17` → model_name: `gemini-2.5-flash-lite`
- `openai/coding/gemini-2.5-pro` → model_name: `gemini-2.5-pro`
- `openai/smart/task` → model_name: `req-smart-task`