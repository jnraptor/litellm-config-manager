## openrouter
### input
```json
{
  "id": "qwen/qwen3.7-max",
  "canonical_slug": "qwen/qwen3.7-max-20260520",
  "hugging_face_id": null,
  "name": "Qwen: Qwen3.7 Max",
  "created": 1779376861,
  "description": "Qwen3.7-Max is the flagship model in Alibaba's Qwen3.7 series. It supports text input and output and is designed for agent-centric workloads, with particular strengths in coding, office and productivity tasks,...",
  "context_length": 1000000,
  "architecture": {
    "modality": "text->text",
    "input_modalities": [
      "text"
    ],
    "output_modalities": [
      "text"
    ],
    "tokenizer": "Qwen",
    "instruct_type": null
  },
  "pricing": {
    "prompt": "0.00000125",
    "completion": "0.00000375",
    "input_cache_read": "0.00000025",
    "input_cache_write": "0.0000015625"
  },
  "top_provider": {
    "context_length": 1000000,
    "max_completion_tokens": 65536,
    "is_moderated": false
  },
  "per_request_limits": null,
  "supported_parameters": [
    "include_reasoning",
    "logprobs",
    "max_tokens",
    "presence_penalty",
    "reasoning",
    "response_format",
    "seed",
    "structured_outputs",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p"
  ],
  "default_parameters": {
    "temperature": null,
    "top_p": null,
    "top_k": null,
    "frequency_penalty": null,
    "presence_penalty": null,
    "repetition_penalty": null
  },
  "supported_voices": null,
  "knowledge_cutoff": null,
  "expiration_date": null,
  "links": {
    "details": "/api/v1/models/qwen/qwen3.7-max-20260520/endpoints"
  },
  "benchmarks": {
    "design_arena": [
      {
        "arena": "agents",
        "category": "androidnative",
        "elo": 1175,
        "win_rate": 45.4,
        "rank": 16
      },
      {
        "arena": "agents",
        "category": "fullstack",
        "elo": 1224,
        "win_rate": 48.3,
        "rank": 10
      },
      {
        "arena": "agents",
        "category": "mobileapps",
        "elo": 1221,
        "win_rate": 48.3,
        "rank": 11
      },
      {
        "arena": "agents",
        "category": "webapps",
        "elo": 1260,
        "win_rate": 50.6,
        "rank": 6
      },
      {
        "arena": "models",
        "category": "3d",
        "elo": 1326,
        "win_rate": 60,
        "rank": 10
      },
      {
        "arena": "models",
        "category": "asciiart",
        "elo": 1262,
        "win_rate": 54.3,
        "rank": 9
      },
      {
        "arena": "models",
        "category": "codecategories",
        "elo": 1311,
        "win_rate": 57.4,
        "rank": 10
      },
      {
        "arena": "models",
        "category": "dataviz",
        "elo": 1289,
        "win_rate": 56.2,
        "rank": 13
      },
      {
        "arena": "models",
        "category": "gamedev",
        "elo": 1324,
        "win_rate": 58.7,
        "rank": 10
      },
      {
        "arena": "models",
        "category": "svg",
        "elo": 1278,
        "win_rate": 60.5,
        "rank": 8
      },
      {
        "arena": "models",
        "category": "uicomponent",
        "elo": 1327,
        "win_rate": 59.6,
        "rank": 6
      },
      {
        "arena": "models",
        "category": "website",
        "elo": 1304,
        "win_rate": 56.5,
        "rank": 13
      }
    ],
    "artificial_analysis": {
      "intelligence_index": 46,
      "coding_index": 66,
      "agentic_index": 30.6
    }
  },
  "reasoning": {
    "mandatory": false,
    "default_enabled": true
  }
}
```
### output
```yaml
- model_name: qwen3.7-max
  litellm_params:
    model: openrouter/qwen/qwen3.7-max
    order: 5
    input_cost_per_token: 1.25e-06
    output_cost_per_token: 3.75e-06
    cache_creation_input_token_cost: 1.5625e-06
    cache_read_input_token_cost: 2.5e-07
```

## vercel
### input
```json
{
  "id": "alibaba/qwen3.7-max",
  "object": "model",
  "created": 1755815280,
  "released": 1779321600,
  "owned_by": "alibaba",
  "name": "Qwen 3.7 Max",
  "description": "Qwen3.7 is a next‑generation flagship model designed for the agent‑centric era, with its core strengths lying in the breadth and depth of its agent‑level capabilities: it excels at programming, office and productivity tasks, and long‑term autonomous execution.",
  "context_window": 991000,
  "max_tokens": 64000,
  "type": "language",
  "tags": [
    "implicit-caching",
    "reasoning",
    "tool-use"
  ],
  "pricing": {
    "input": "0.00000125",
    "output": "0.00000375",
    "input_cache_read": "0.00000025",
    "input_cache_write": "0.0000015625"
  }
}
```
### output
```yaml
- model_name: qwen3.7-max
  litellm_params:
    model: vercel_ai_gateway/alibaba/qwen3.7-max
    order: 5
    input_cost_per_token: 1.25e-06
    output_cost_per_token: 3.75e-06
    cache_creation_input_token_cost: 1.5625e-06
    cache_read_input_token_cost: 2.5e-07
```

## poe
### input
```json
{
  "id": "qwen3.7-max",
  "object": "model",
  "created": 1779377367791,
  "description": "Qwen3.7 Max is a next‑generation flagship model designed for the agent‑centric era, with its core strengths lying in the breadth and depth of its agent‑level capabilities: it excels at programming, office and productivity tasks, and long‑term autonomous execution.\nThis model is served by Alibaba Cloud Int. from Singapore.\n\nNotes:\n- Context Window: 1,000,000\n- Text only input\n\nThis bot supports optional parameters for additional customization.",
  "owned_by": "EmpirioLabs AI",
  "root": "qwen3.7-max",
  "architecture": {
    "input_modalities": [
      "text"
    ],
    "output_modalities": [
      "text"
    ],
    "modality": "text->text"
  },
  "supported_features": [
    "tools"
  ],
  "supported_endpoints": [],
  "pricing": {
    "prompt": "0.0000025253",
    "completion": "0.0000075758",
    "image": null,
    "request": null,
    "input_cache_read": null,
    "input_cache_write": null
  },
  "context_window": null,
  "context_length": null,
  "metadata": {
    "display_name": "Qwen3.7-Max",
    "image": {
      "url": "https://qph.cf2.poecdn.net/main-thumb-pb-6998003-200-wogoiajhsikxcwozjxarwrpuvidslpdx.jpeg",
      "alt": "Qwen3.7-Max model icon",
      "width": 200,
      "height": 200
    },
    "url": "https://poe.com/qwen3.7-max"
  },
  "reasoning": null,
  "parameters": [
    {
      "name": "enable_thinking",
      "schema": {
        "type": "boolean"
      },
      "default_value": true,
      "description": "Let the model reason step-by-step before answering"
    },
    {
      "name": "tool_web_search",
      "schema": {
        "type": "boolean"
      },
      "default_value": false,
      "description": "Search the web for real-time information"
    },
    {
      "name": "tool_web_extractor",
      "schema": {
        "type": "boolean"
      },
      "default_value": false,
      "description": "Extract and read content from URLs (requires Web Search and Thinking)"
    },
    {
      "name": "tool_code_interpreter",
      "schema": {
        "type": "boolean"
      },
      "default_value": false,
      "description": "Run Python code in a sandbox (requires thinking)"
    }
  ]
}
```
### output
```yaml
- model_name: qwen3.7-max
  litellm_params:
    model: poe/qwen3.7-max
    order: 4
    api_key: os.environ/POE_API_KEY
    input_cost_per_token: 2.5253e-06
    output_cost_per_token: 7.5758e-06
```

## nvidia
### input
```json
{
  "id": "z-ai/glm5",
  "object": "model",
  "created": 735790403,
  "owned_by": "z-ai"
}
```
### output
```yaml
- model_name: zai-glm-5
  litellm_params:
    model: nvidia_nim/z-ai/glm5
    order: 3
    input_cost_per_token: 1.0e-09
    output_cost_per_token: 1.0e-09
```

## kilo
### input
```json
{
  "id": "qwen/qwen3.7-max",
  "name": "Qwen: Qwen3.7 Max (50% off)",
  "created": 1779376861,
  "description": "Qwen3.7-Max is the flagship model in Alibaba's Qwen3.7 series. It supports text input and output and is designed for agent-centric workloads, with particular strengths in coding, office and productivity tasks,...",
  "architecture": {
    "modality": "text->text",
    "input_modalities": [
      "text"
    ],
    "output_modalities": [
      "text"
    ],
    "tokenizer": "Qwen"
  },
  "top_provider": {
    "is_moderated": false,
    "context_length": 1000000,
    "max_completion_tokens": 65536
  },
  "pricing": {
    "prompt": "0.000001250000",
    "completion": "0.000003750000",
    "input_cache_read": "0.000000125000",
    "input_cache_write": "0.000001562500"
  },
  "context_length": 1000000,
  "per_request_limits": null,
  "supported_parameters": [
    "include_reasoning",
    "logprobs",
    "max_tokens",
    "presence_penalty",
    "reasoning",
    "response_format",
    "seed",
    "structured_outputs",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p"
  ],
  "expiration_date": null,
  "terminalBench": {
    "overallScore": 0.54606742,
    "avgAttemptCostUsd": 20.6499772
  },
  "isFree": false,
  "mayTrainOnYourPrompts": false,
  "opencode": {
    "variants": {
      "instant": {
        "reasoning": {
          "enabled": false,
          "effort": "none"
        }
      },
      "thinking": {
        "reasoning": {
          "enabled": true,
          "effort": "high"
        }
      }
    }
  }
}
```
### output
```yaml
- model_name: qwen3.7-max
  litellm_params:
    model: openai/qwen/qwen3.7-max
    order: 4
    api_base: https://api.kilo.ai/api/gateway
    api_key: os.environ/KILO_API_KEY
    input_cost_per_token: 1.25e-06
    output_cost_per_token: 3.75e-06
    cache_creation_input_token_cost: 1.5625e-06
    cache_read_input_token_cost: 1.25e-07
```

## ollama
### input
```json
{
  "name": "nemotron-3-super",
  "model": "nemotron-3-super",
  "modified_at": "2026-03-11T00:00:00Z",
  "size": 230500000000,
  "digest": "da11955bb451",
  "details": {
    "parent_model": "",
    "format": "",
    "family": "",
    "families": null,
    "parameter_size": "",
    "quantization_level": ""
  }
}
```
### output
```yaml
- model_name: nvidia-nemotron-3-super
  litellm_params:
    model: ollama_chat/nemotron-3-super
    order: 3
    api_base: https://ollama.com
    api_key: os.environ/OLLAMA_API_KEY
    input_cost_per_token: 1.0e-09
    output_cost_per_token: 1.0e-09
```

## fireworks
### input
```json
{
  "id": "accounts/fireworks/models/glm-5",
  "object": "model",
  "owned_by": "fireworks",
  "created": 1770826344,
  "kind": "HF_BASE_MODEL",
  "supports_chat": true,
  "supports_image_input": false,
  "supports_tools": true,
  "context_length": 202752
}
```
### output
```yaml
- model_name: zai-glm-5
  litellm_params:
    model: fireworks_ai/accounts/fireworks/models/glm-5
    order: 4
    api_key: os.environ/FIREWORKS_AI_API_KEY
```

## opencode-zen
### input
```json
{
  "id": "glm-5.1",
  "object": "model",
  "created": 1775653844,
  "owned_by": "opencode"
}
```
### output
```yaml
- model_name: zai-glm-5.1
  litellm_params:
    model: openai/glm-5.1
    api_base: https://opencode.ai/zen/v1
    api_key: os.environ/OPENCODE_API_KEY
    order: 5
```

## opencode-go
### input
```json
{
  "id": "glm-5",
  "object": "model",
  "created": 1775653844,
  "owned_by": "opencode"
}
```
### output
```yaml
- model_name: glm-5
  litellm_params:
    model: openai/glm-5
    api_base: https://opencode.ai/zen/go/v1
    api_key: os.environ/OPENCODE_API_KEY
    order: 1
```

## opencode-go-anthropic
### input
```json
{
  "id": "minimax-m2.5",
  "object": "model",
  "created": 1775653844,
  "owned_by": "opencode"
}
```
### output
```yaml
- model_name: minimax-m2.5
  litellm_params:
    model: anthropic/minimax-m2.5
    api_base: https://opencode.ai/zen/go
    api_key: os.environ/OPENCODE_API_KEY
    order: 1
```

## requesty
### input
```json
{
  "api": "chat",
  "id": "alibaba/qwen3.7-max",
  "object": "model",
  "created": 1779791470,
  "owned_by": "system",
  "input_price": 0.0000025,
  "caching_price": 0.000003125,
  "cached_price": 2.5e-7,
  "output_price": 0.0000075,
  "max_output_tokens": 65536,
  "context_window": 1048576,
  "supports_caching": true,
  "supports_vision": false,
  "supports_computer_use": false,
  "supports_reasoning": false,
  "supports_image_generation": false,
  "supports_tool_calling": true,
  "supports_role_developer": false,
  "supports_web_search": false,
  "supports_output_json_object": true,
  "supports_output_json_schema": true,
  "description": "Qwen3.7-Max is the flagship model in Alibaba's Qwen3.7 series. It supports text input and output and is designed for agent-centric workloads, with particular strengths in coding, office and productivity tasks, and long-horizon autonomous execution. The model offers notable gains in coding and agentic performance over prior Qwen generations and supports explicit prompt caching for efficient repeated context use.",
  "privacy_comments": "N/A",
  "geolocation": "global"
}
```
### output
```yaml
- litellm_params:
    model: openai/alibaba/qwen3.7-max
    api_base: https://router.requesty.ai/v1
    api_key: os.environ/REQUESTY_API_KEY
    input_cost_per_token: 2.5e-06
    output_cost_per_token: 7.5e-06
    cache_creation_input_token_cost: 3.125e-06
    cache_read_input_token_cost: 2.5e-07
    order: 5
  model_name: qwen3.7-max
```
