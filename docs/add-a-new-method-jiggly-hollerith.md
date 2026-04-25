# Add config.yaml Validation Method to Cleanup Scripts

## Context

Users manually edit `config.yaml` to add/modify model entries. Typos, missing fields, wrong prefixes, and duplicate entries can slip in and only surface at runtime in LiteLLM. A new `--validate` flag will catch these issues offline — no API calls needed — so users can verify their config before deploying.

## Implementation

### 1. Add data classes to `cleanup_base.py` (around line 140, before `APIClient`)

```python
from dataclasses import dataclass, field
from enum import Enum

class ValidationSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class ValidationIssue:
    severity: ValidationSeverity
    category: str
    entry_index: int      # 0-based index in model_list, or -1 for file-level
    model_name: str       # from entry, or "" if N/A
    model_id: str         # litellm_params.model, or "" if N/A
    message: str
    suggestion: str = ""

@dataclass
class ValidationReport:
    issues: list = field(default_factory=list)
    total_entries: int = 0
    valid_entries: int = 0

    @property
    def has_errors(self) -> bool:
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)
```

Also add `VALID_MODEL_MODES = frozenset({"chat", "completion", "embedding", "image_generation", "rerank", "audio_transcription", "image"})` and `FALLBACK_KNOWN_PREFIXES` (for providers not in providers.yaml like `azure/`, `azure_ai/`, `openai/`, `dashscope/`, `jina_ai/`, `ollama/`, `ollama_chat/`, `anthropic/`, `vercel_ai_gateway/`).

### 2. Add `validate_config()` method to `BaseModelCleaner` (around line 340, after `save_config`)

Method signature:
```python
def validate_config(self, config: Optional[Dict[str, Any]] = None) -> ValidationReport
```

Checks per entry in `model_list`:

| # | Check | Severity | Detail |
|---|-------|----------|--------|
| 1 | Entry is a dict | ERROR | If not dict, skip rest of checks for this entry |
| 2 | `model_name` present & string | ERROR | Missing or non-string model_name |
| 3 | `litellm_params` present & dict | ERROR | Missing or non-dict litellm_params |
| 4 | `litellm_params.model` present & non-empty | ERROR | Required provider-prefixed model ID |
| 5 | Duplicate `(model_name, model)` | ERROR | Same display name + same model ID = exact dup |
| 6 | `input_cost_per_token` is numeric | ERROR | String or other non-numeric type |
| 7 | `output_cost_per_token` is numeric | ERROR | Same as above |
| 8 | Cost >= 0 | ERROR | Negative costs are invalid |
| 9 | Cost > 0.01 | WARNING | Suspiciously high (> $10 per 1000 tokens) |
| 10 | `order` is positive int | ERROR | Non-positive or non-integer order |
| 11 | `model_info.mode` in VALID_MODEL_MODES | ERROR | Invalid mode value when present |
| 12 | Missing `api_key` for providers that need it | WARNING | Provider requires api_key_env but entry has none |
| 13 | Unknown provider prefix | WARNING | Model ID prefix matches no known provider |

Steps 12-13 require `ProviderConfigLoader` to read providers.yaml. For api_key detection, iterate all providers and check if the entry matches the provider's model_detection (prefix or api_base), then verify api_key is set.

### 3. Add `validate_config()` to `UnifiedModelCleaner` in `cleanup_models.py`

```python
def validate_config(self, config: Optional[Dict[str, Any]] = None) -> ValidationReport:
    # 1. Load config if not provided (use self.load_config)
    # 2. Call BaseModelCleaner.validate_config() on any one cleaner instance
    # 3. Add provider-specific api_key checks across all cleaners
    # 4. Return aggregated ValidationReport
```

Since `BaseModelCleaner.validate_config()` validates all entries structurally (provider-agnostic), UnifiedModelCleaner just needs to pick any cleaner to run it, then aggregate provider-specific checks from each cleaner for their recognized entries.

### 4. Add `--validate` CLI flag to `cleanup_models.py`

In `main()`, add:
```python
parser.add_argument("--validate", action="store_true",
    help="Validate config.yaml structure without API calls (offline)")
```

Add handling branch before provider processing:
```python
if args.validate:
    cleaner = UnifiedModelCleaner(args.config, provider_names, dry_run=False, verbose=args.verbose)
    report = cleaner.validate_config()
    _print_validation_report(report)
    return 1 if report.has_errors else 0
```

### 5. Add `_print_validation_report()` helper to `cleanup_models.py`

Formats output grouped by severity:
```
ERRORS (2):
  Entry #4 (model_name='gpt-5.4') [model='openai/gpt-5.4']
    - Duplicate entry: same model_name and model ID first seen at index #1
  Entry #12 (model_name='bad-entry')
    - Missing required field 'litellm_params'

WARNINGS (1):
  Entry #8 (model_name='minimax-m2.5') [model='openai/minimax-m2.5']
    - Model ID has unknown provider prefix

Summary: 87 entries checked, 1 errors, 1 warnings
```

### 6. Update `create_provider_main()` factory in `cleanup_base.py`

Add `--validate` flag so all provider scripts also support it. Before cleanup runs:
```python
if args.validate:
    cleaner = cleaner_class(config_path=args.config, dry_run=False, verbose=args.verbose)
    report = cleaner.validate_config()
    _print_report(report)  # inline print logic
    return 1 if report.has_errors else 0
```

### Files to modify

1. **`cleanup_base.py`** — data classes, `validate_config()` on BaseModelCleaner, update `create_provider_main()`, update `__all__`
2. **`cleanup_models.py`** — `validate_config()` on UnifiedModelCleaner, `--validate` flag, `_print_validation_report()` helper

### Verification

```bash
# Run validation on current config (should pass or show any issues)
python cleanup_models.py --provider all --validate

# Run with verbose to see per-provider details
python cleanup_models.py --provider all --validate --verbose

# Test per-provider validation
python cleanup_models.py --provider openrouter --validate

# Dry-run validation (same as --validate, no writes)
python cleanup_models.py --provider all --dry-run --validate
```

### Tests
Add required tests after adding the method.
