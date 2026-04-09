## openrouter
### input
```json
{
  "id": "z-ai/glm-5",
  "canonical_slug": "z-ai/glm-5-20260211",
  "hugging_face_id": "zai-org/GLM-5",
  "name": "Z.ai: GLM 5",
  "created": 1770829182,
  "description": "GLM-5 is Z.ai’s flagship open-source foundation model engineered for complex systems design and long-horizon agent workflows. Built for expert developers, it delivers production-grade performance on large-scale programming tasks, rivaling leading closed-source models. With advanced agentic planning, deep backend reasoning, and iterative self-correction, GLM-5 moves beyond code generation to full-system construction and autonomous execution.",
  "context_length": 202752,
  "architecture": {
    "modality": "text->text",
    "input_modalities": [
      "text"
    ],
    "output_modalities": [
      "text"
    ],
    "tokenizer": "Other",
    "instruct_type": null
  },
  "pricing": {
    "prompt": "0.0000008",
    "completion": "0.00000256",
    "input_cache_read": "0.00000016"
  },
  "top_provider": {
    "context_length": 202752,
    "max_completion_tokens": null,
    "is_moderated": false
  },
  "per_request_limits": null,
  "supported_parameters": [
    "frequency_penalty",
    "include_reasoning",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "min_p",
    "presence_penalty",
    "reasoning",
    "repetition_penalty",
    "response_format",
    "seed",
    "stop",
    "structured_outputs",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p"
  ],
  "default_parameters": {
    "temperature": 1,
    "top_p": 0.95,
    "frequency_penalty": null
  },
  "expiration_date": null
}
```
### output
```yaml
- model_name: zai-glm-5
  litellm_params:
    model: openrouter/z-ai/glm-5
    order: 5
    input_cost_per_token: 8.0e-07
    output_cost_per_token: 2.56e-06
```

## vercel
### input
```json
{
  "id": "zai/glm-5",
  "object": "model",
  "created": 1755815280,
  "released": 1770854400,
  "owned_by": "zai",
  "name": "GLM-5",
  "description": "GLM-5 is Zai’s new-generation flagship foundation model, designed for Agentic Engineering, capable of providing reliable productivity in complex system engineering and long-range Agent tasks. In terms of Coding and Agent capabilities, GLM-5 has achieved state-of-the-art (SOTA) performance in open source, with its usability in real programming scenarios approaching that of Claude Opus 4.5.",
  "context_window": 202800,
  "max_tokens": 131072,
  "type": "language",
  "tags": [
    "reasoning",
    "implicit-caching",
    "tool-use"
  ],
  "pricing": {
    "input": "0.000001",
    "output": "0.0000032",
    "input_cache_read": "0.0000002"
  }
}
```
### output
```yaml
- model_name: zai-glm-5
  litellm_params:
    model: vercel_ai_gateway/zai/glm-5
    order: 5
    input_cost_per_token: 1.0e-06
    output_cost_per_token: 3.2e-06
```

## poe
### input
```json
{
  "id": "glm-5",
  "object": "model",
  "created": 1770830055326,
  "description": "GLM-5 is an open-source foundation model engineered for complex system engineering and long-horizon Agent tasks, delivering reliable productivity for top-tier programmers. Transcending the boundary from \"writing code\" to \"building systems,\" it moves beyond traditional snippet generation to offer senior-architect-level planning and execution capabilities. By rejecting the \"frontend-heavy, logic-light\" approach, GLM-5 demonstrates exceptional reasoning and self-healing abilities in backend refactoring, complex algorithm implementation, and deep debugging—autonomously analyzing logs and iteratively fixing persistent bugs until the system runs. As the first open-source model featuring Opus-class style and system engineering depth, GLM-5 provides extreme logic density alongside the freedom of local deployment and high cost-effectiveness, making it the ideal choice for large-scale backend development and automated Agent construction. Context window: 205k tokens\n\nThis bot supports optional parameters for additional customization.",
  "owned_by": "Novita AI",
  "root": "glm-5",
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
  "pricing": {
    "prompt": "0.0000010",
    "completion": "0.0000032",
    "image": null,
    "request": null,
    "input_cache_read": "0.00000020",
    "input_cache_write": null
  },
  "context_window": null,
  "context_length": null,
  "metadata": {
    "display_name": "GLM-5",
    "image": {
      "url": "https://qph.cf2.poecdn.net/main-thumb-pb-6802092-200-sqrsdkzksaocmqhyynlwkagfcrkajwyw.jpeg",
      "alt": "GLM-5 model icon",
      "width": 200,
      "height": 200
    },
    "url": "https://poe.com/glm-5"
  },
  "reasoning": null,
  "parameters": [
    {
      "name": "enable_thinking",
      "schema": {
        "type": "boolean"
      },
      "default_value": false,
      "description": "This will cause the model to think..."
    },
    {
      "name": "temperature",
      "schema": {
        "type": "number",
        "minimum": 0,
        "maximum": 2
      },
      "default_value": 0.7,
      "description": "Controls randomness in the response. Lower values make the output more focused and deterministic."
    },
    {
      "name": "max_output_tokens",
      "schema": {
        "type": "number",
        "minimum": 1,
        "maximum": 131072
      },
      "default_value": 131072,
      "description": "Maximum number of tokens to generate in the response."
    }
  ]
}
```
### output
```yaml
- model_name: zai-glm-5
  litellm_params:
    model: poe/glm-5
    order: 5
    api_key: os.environ/POE_API_KEY
    input_cost_per_token: 1.0e-06
    output_cost_per_token: 3.2e-06
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
    order: 5
    input_cost_per_token: 1.0e-09
    output_cost_per_token: 1.0e-09
```

## kilo
### input
```json
{
  "id": "z-ai/glm-5",
  "name": "Z.ai: GLM 5",
  "created": 1770829182,
  "description": "GLM-5 is Z.ai’s flagship open-source foundation model engineered for complex systems design and long-horizon agent workflows. Built for expert developers, it delivers production-grade performance on large-scale programming tasks, rivaling leading closed-source models. With advanced agentic planning, deep backend reasoning, and iterative self-correction, GLM-5 moves beyond code generation to full-system construction and autonomous execution.",
  "architecture": {
    "input_modalities": [
      "text"
    ],
    "output_modalities": [
      "text"
    ],
    "tokenizer": "Other"
  },
  "top_provider": {
    "is_moderated": false,
    "context_length": 202752,
    "max_completion_tokens": null
  },
  "pricing": {
    "prompt": "0.0000008",
    "completion": "0.00000256",
    "input_cache_read": "0.00000016"
  },
  "context_length": 202752,
  "per_request_limits": null,
  "supported_parameters": [
    "frequency_penalty",
    "include_reasoning",
    "logit_bias",
    "logprobs",
    "max_tokens",
    "min_p",
    "presence_penalty",
    "reasoning",
    "repetition_penalty",
    "response_format",
    "seed",
    "stop",
    "structured_outputs",
    "temperature",
    "tool_choice",
    "tools",
    "top_k",
    "top_logprobs",
    "top_p"
  ],
  "preferredIndex": 10,
  "versioned_settings": {
    "4.146.0": {
      "included_tools": [
        "write_file",
        "edit_file"
      ],
      "excluded_tools": [
        "apply_diff"
      ]
    }
  },
  "opencode": {
    "variants": {
      "instant": {
        "reasoning": {
          "enabled": false
        }
      },
      "thinking": {
        "reasoning": {
          "enabled": true
        }
      }
    }
  }
}
```
### output
```yaml
- model_name: zai-glm-5
  litellm_params:
    model: openai/z-ai/glm-5
    order: 5
    api_base: https://api.kilo.ai/api/gateway
    api_key: os.environ/KILO_API_KEY
    input_cost_per_token: 8.0e-07
    output_cost_per_token: 2.56e-06
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
    order: 5
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
    order: 5
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
