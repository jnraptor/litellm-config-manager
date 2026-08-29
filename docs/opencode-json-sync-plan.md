# Plan & Implementation Record: Sync opencode.json from LiteLLM Config

**Date:** 2026-08-29 — **Status:** ✅ Implemented and verified (304 tests passing)

## TL;DR

Every time a cleanup script writes `config.yaml`, `opencode.json` is now regenerated
so the opencode harness stays aware of all models available through LiteLLM. To make
the output accurate, the pipeline was extended to capture **modalities** from models.dev
and store them in `config.yaml` as LiteLLM-compatible boolean flags under `model_info`
(`supports_vision`, `supports_pdf_input`, `supports_audio_input`,
`supports_audio_output`), with per-provider / global `modalities_default` fallbacks in
`providers.yaml`. The full opencode `modalities` dict is derived from those flags and
written only to `opencode.json` (LiteLLM does not understand the opencode modalities
dict, so it is never written to `config.yaml`).

## Agreed Decisions (from planning session)

| Topic | Decision |
|---|---|
| Sync strategy | Full regen of `provider.litellm.models` on every save; wrapper (`$schema`, `npm`, `name`, `options.baseURL`, other providers) preserved. Hand-added entries are dropped. |
| Modalities | Stored in `config.yaml` as `model_info.supports_*` boolean flags (`supports_vision`, `supports_pdf_input`, `supports_audio_input`, `supports_audio_output`); source = models.dev only + `modalities_default` fallback in `providers.yaml` (no provider-API field parsing). The full opencode `modalities` dict is derived from these flags and written only to `opencode.json`. Video is intentionally ignored (LiteLLM does not support it). |
| Standalone scripts | Both unified (`cleanup_models.py`) and per-provider scripts sync opencode.json. |
| CLI flags | Always on, no flag. `--dry-run` respected everywhere (preview log, no writes). |
| Key naming | opencode.json keyed by `model_name` as-is (sorted alphabetically for stable diffs). |
| Skip modes | `embedding`, `rerank`, and `image_generation` models are excluded from opencode.json (constant `OPENCODE_SKIP_MODES`, easily extended). A `model_name` is skipped only if ALL its entries carry a skipped mode. |

---

## Plan (as designed)

### Phase A — Modality capture (`cleanup_base.py`, `providers.yaml`)

- **A1.** `ModelsDevClient`: new `get_model_modalities(provider_id, model_id)` reading
  models.dev `modalities.input/output`; normalize via `_parse_modalities()`; also expose
  `"modalities"` key in `get_provider_models()` dicts.
- **A2.** `ConfigDrivenModelCleaner.parse_api_model()`: add flat `modalities` key; fill
  from models.dev when `pricing.models_dev_id` is set (limits-style fallback, not gated
  on pricing); else fall back to provider-level or top-level `modalities_default`.
- **A3.** `create_model_entry()`: convert the modalities dict to `supports_*` flags
  (`modalities_to_supports()`) and write those into `model_info` when non-empty.
- **A4.** `validate_and_update_costs()`: sync `supports_*` flags with **additive**
  semantics — update when source has data, never delete when it stops reporting
  (protects providers whose models.dev entry lacks modality data).
- **A5.** `validate_config()`: new warning-level check — each `model_info.supports_*`
  flag (`supports_vision`, `supports_pdf_input`, `supports_audio_input`,
  `supports_audio_output`) must be a boolean.
- **A6.** `providers.yaml`: top-level `modalities_default: {input: [text], output: [text]}`
  (per-provider key overrides it).

### Phase B — opencode.json regeneration (`cleanup_base.py`, `cleanup_models.py`)

- **B1.** Module-level `build_opencode_models(config)`: group `model_list` by
  `model_name` (load-balanced duplicates collapse; max limits win); map
  `model_info.max_input_tokens` → `limit.context`, `max_output_tokens` → `limit.output`
  (omit when unknown); `modalities` derived from `model_info.supports_*` flags via
  `supports_to_modalities()` (text/text default when no flags); skip names in
  `OPENCODE_SKIP_MODES`; sorted keys.
- **B2.** Module-level `regenerate_opencode_json(config, path, logger, dry_run)`:
  no-op if file missing; require `provider.litellm` section else warn+skip; replace ONLY
  `provider.litellm.models`; backup to `opencode.json.backup`; `json.dump(indent=2)` +
  trailing newline; restore-on-error; unchanged content → no write; dry-run previews.
- **B3.** Hooks: `BaseModelCleaner.save_config()` (standalone scripts) and
  `UnifiedModelCleaner.save_config()` (all CLI paths: cleanup, add-model, mapped-add,
  delete-model, delete-provider). Dry-run branch logs a `[DRY-RUN] Would update ...`
  preview. Sync failures log warnings but never fail the cleanup run.

### Phase C — Tests

- `tests/test_models_dev.py`: sample data extended with `modalities`; new
  `TestGetModelModalities` class (present/partial/missing/invalid/not-loaded);
  `get_provider_models` modalities test; `parse_api_model` fallback + default + none
  tests; `create_model_entry` merge/omit tests.
- `tests/test_cleanup_coverage.py`: additive-sync tests — `supports_*` flags added,
  updated, and **not removed** when the source goes silent.
- `tests/test_validation.py`: valid `supports_*` flags, non-boolean flag (warning),
  malformed flag (warning).
- `tests/test_input_outputs.py`: mock `get_model_modalities` → None and disable
  `_modalities_default` so fixtures in `input-and-outputs.md` stay stable.
- New `tests/test_opencode_sync.py` (21 tests): build/regen unit tests plus integration
  through `UnifiedModelCleaner.save_config` (real save, dry-run untouched, missing file
  no-op, backup created, wrapper preserved, stale hand-written models replaced).

### Phase D — CI + docs

- All 6 workflows (`.github/workflows/cleanup-*.yml`): change detection
  `git diff --quiet config.yaml opencode.json` and commit
  `git add config.yaml opencode.json`.
- `CLAUDE.md` / `AGENTS.md`: documented opencode.json sync behavior,
  `model_info.supports_*` flags, `modalities_default`, `OPENCODE_SKIP_MODES`, new test file.

---

## Completed Steps (implementation record)

1. ✅ **A1** — `ModelsDevClient._parse_modalities()` + `get_model_modalities()`;
   `"modalities"` added to `get_provider_models()` output.
2. ✅ **A2** — `parse_api_model()` resolves modalities: models.dev first, then
   `self._modalities_default` (provider-level, falling back to top-level via new
   `ProviderConfigLoader.get_global()`).
3. ✅ **A3** — `create_model_entry()` converts modalities to `supports_*` flags via
   `modalities_to_supports()` and writes those into `model_info` (never the raw dict).
4. ✅ **A4** — Additive `supports_*` flag sync block in `validate_and_update_costs()`
   (reuses the `limit_changed` change-tracking channel so cost-change reports include it).
5. ✅ **A5** — Check 11b in `validate_config()` validates the `supports_*` flags are
   booleans.
6. ✅ **A6** — Top-level `modalities_default` added to `providers.yaml`. Verified live:
   all 12 models.dev provider ids report modalities for 100% of their catalog models,
   so the default is a last-resort fallback only.
7. ✅ **B1/B2** — `build_opencode_models()` (derives `modalities` from `supports_*`
   flags via `supports_to_modalities()`) and `regenerate_opencode_json()` added to
   `cleanup_base.py` (exported in `__all__`); `OPENCODE_SKIP_MODES` constant defined
   next to `VALID_MODEL_MODES`.
8. ✅ **B3** — Save hooks wired in both `cleanup_base.py` and `cleanup_models.py`
   (including the dry-run preview path).
9. ✅ **C** — All new tests written; one regression fixed along the way (Mock auto-attr
   leaking into expected YAML in `test_input_outputs.py` — resolved with explicit
   `return_value=None` and disabling the default in the test double).
10. ✅ **D1** — All 6 CI workflows updated to detect + commit `opencode.json`.
11. ✅ **D2** — `CLAUDE.md` and `AGENTS.md` updated.

## Verification Results

- `pytest tests/ -q` → **306 passed** (coverage ~80%).
- `python cleanup_models.py --provider all --dry-run` → opencode.json untouched,
  `[DRY-RUN] Would update ... opencode models` logged.
- Real run in temp workspace: 42 chat models generated, keys sorted, wrapper preserved
  (`@ai-sdk/openai-compatible`, baseURL intact), multimodal models show `image`/`pdf`/
  `audio` input derived from `supports_*` flags, exactly the 8 embedding/rerank/
  image-generation names excluded, `opencode.json.backup` created.
- Standalone `cleanup_openrouter_models.py` run also syncs correctly.

## Notes / Caveats

- The two original hand-written sample models in `opencode.json` (`gpt-4`,
  `claude-3-5-sonnet-20241022`) do not exist in `config.yaml` and will be replaced by
  the generated catalog on the first real save.
- `config.yaml` carries only the LiteLLM-compatible `supports_*` boolean flags; the
  opencode `modalities` dict is derived from them and lives only in `opencode.json`.
  Because the flags capture only image/pdf/audio (video is ignored), the derived
  opencode modalities never include `video`.
- Out of scope (deliberately): parsing provider-API modality fields (OpenRouter
  `architecture.*`, Requesty `supports_vision`), per-model static override maps, and
  model-name de-prefixing.
