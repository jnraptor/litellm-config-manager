#!/usr/bin/env python3
"""
Tests for the config validation functionality in cleanup_base.py and cleanup_models.py.
"""

import pytest
import logging
from pathlib import Path
from unittest.mock import Mock, patch
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup_base import (
    BaseModelCleaner,
    ValidationSeverity,
    ValidationIssue,
    ValidationReport,
    VALID_MODEL_MODES,
    FALLBACK_KNOWN_PREFIXES,
    _print_validation_report,
    create_provider_main,
)
from cleanup_models import UnifiedModelCleaner


class MockModelCleaner(BaseModelCleaner):
    """Concrete implementation of BaseModelCleaner for testing."""

    def __init__(
        self,
        config_path: str = "test_config.yaml",
        dry_run: bool = False,
        verbose: bool = False,
    ):
        super().__init__(config_path, dry_run, verbose)
        self.PROVIDER_NAME = "TestProvider"
        self.SPECIAL_MODELS = {"special-model"}
        self.defaults = {}

    def extract_provider_models(self, config):
        models = []
        for i, entry in enumerate(config.get("model_list", [])):
            model_id = entry.get("litellm_params", {}).get("model", "")
            model_name = entry.get("model_name", "")
            if model_id.startswith("test/"):
                models.append((i, model_id, model_name))
        return models

    def fetch_available_models(self):
        return {
            "model1": {"id": "model1", "input_cost": 1e-06, "output_cost": 2e-06},
        }

    def get_api_model_id(self, model_id):
        if model_id.startswith("test/"):
            return model_id[5:]
        return model_id

    def create_model_entry(self, model_id, api_model_info, model_name):
        return {
            "model_name": model_name,
            "litellm_params": {
                "model": f"test/{model_id}",
                "input_cost_per_token": api_model_info.get("input_cost"),
                "output_cost_per_token": api_model_info.get("output_cost"),
            },
        }


class TestValidationDataClasses:
    """Tests for ValidationSeverity, ValidationIssue, and ValidationReport."""

    def test_validation_severity_values(self):
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.INFO.value == "info"

    def test_validation_issue_defaults(self):
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            category="test",
            entry_index=0,
            model_name="m",
            model_id="id",
            message="msg",
        )
        assert issue.suggestion == ""

    def test_validation_report_has_errors_true(self):
        report = ValidationReport()
        report.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="test",
                entry_index=0,
                model_name="m",
                model_id="id",
                message="msg",
            )
        )
        assert report.has_errors is True

    def test_validation_report_has_errors_false(self):
        report = ValidationReport()
        report.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="test",
                entry_index=0,
                model_name="m",
                model_id="id",
                message="msg",
            )
        )
        assert report.has_errors is False

    def test_validation_report_defaults(self):
        report = ValidationReport()
        assert report.issues == []
        assert report.total_entries == 0
        assert report.valid_entries == 0


class TestValidationConstants:
    """Tests for VALID_MODEL_MODES and FALLBACK_KNOWN_PREFIXES."""

    def test_valid_model_modes(self):
        expected = {
            "chat",
            "completion",
            "embedding",
            "image_generation",
            "rerank",
            "audio_transcription",
            "image",
            "ocr",
        }
        assert VALID_MODEL_MODES == expected

    def test_fallback_known_prefixes(self):
        expected = {
            "azure/",
            "azure_ai/",
            "openai/",
            "dashscope/",
            "jina_ai/",
            "ollama/",
            "ollama_chat/",
            "anthropic/",
            "vercel_ai_gateway/",
            "auto_router/",
            "gemini/",
            "vertex_ai/",
        }
        assert FALLBACK_KNOWN_PREFIXES == expected


class TestValidateConfig:
    """Tests for BaseModelCleaner.validate_config()."""

    @pytest.fixture
    def cleaner(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            import yaml

            yaml.dump({"model_list": []}, f)
        return MockModelCleaner(config_path=str(config_file))

    def test_empty_model_list(self, cleaner):
        config = {"model_list": []}
        report = cleaner.validate_config(config)
        assert report.total_entries == 0
        assert report.valid_entries == 0
        assert not report.has_errors

    def test_valid_entry(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "input_cost_per_token": 1e-06,
                        "output_cost_per_token": 2e-06,
                        "order": 1,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.total_entries == 1
        assert report.valid_entries == 1
        assert not report.has_errors

    def test_entry_not_dict(self, cleaner):
        config = {"model_list": ["not-a-dict"]}
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "type" for i in report.issues)

    def test_missing_model_name(self, cleaner):
        config = {
            "model_list": [
                {
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                    }
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "model_name" for i in report.issues)

    def test_non_string_model_name(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": 123,
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "model_name" for i in report.issues)

    def test_missing_litellm_params(self, cleaner):
        config = {"model_list": [{"model_name": "model-1"}]}
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "litellm_params" for i in report.issues)

    def test_non_dict_litellm_params(self, cleaner):
        config = {
            "model_list": [{"model_name": "model-1", "litellm_params": "not-a-dict"}]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "litellm_params" for i in report.issues)

    def test_missing_model_id(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "order": 1,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "model_id" for i in report.issues)

    def test_duplicate_entry(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {"model": "openrouter/gpt-4"},
                },
                {
                    "model_name": "model-1",
                    "litellm_params": {"model": "openrouter/gpt-4"},
                },
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "duplicate" for i in report.issues)

    def test_duplicate_different_api_base(self, cleaner):
        """Same model_name+model but different api_base should NOT be flagged as duplicate."""
        config = {
            "model_list": [
                {
                    "model_name": "claude-fable-5",
                    "litellm_params": {
                        "model": "openai/anthropic/claude-fable-5",
                        "api_base": "https://api.kilo.ai/api/gateway",
                    },
                },
                {
                    "model_name": "claude-fable-5",
                    "litellm_params": {
                        "model": "openai/anthropic/claude-fable-5",
                        "api_base": "https://router.requesty.ai/v1",
                    },
                },
            ]
        }
        report = cleaner.validate_config(config)
        assert not any(i.category == "duplicate" for i in report.issues)

    def test_non_numeric_input_cost(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "input_cost_per_token": "expensive",
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(
            i.category == "cost" and "input_cost_per_token is non-numeric" in i.message
            for i in report.issues
        )

    def test_non_numeric_output_cost(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "output_cost_per_token": "expensive",
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(
            i.category == "cost" and "output_cost_per_token is non-numeric" in i.message
            for i in report.issues
        )

    def test_input_cost_scientific_notation_string(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "input_cost_per_token": "4e-06",
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert not report.has_errors
        assert not any(i.category == "cost" for i in report.issues)

    def test_output_cost_scientific_notation_string(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "output_cost_per_token": "1.5e-05",
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert not report.has_errors
        assert not any(i.category == "cost" for i in report.issues)

    def test_input_cost_scientific_notation_float(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "input_cost_per_token": 4e-06,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert not report.has_errors
        assert not any(i.category == "cost" for i in report.issues)

    def test_negative_input_cost(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "input_cost_per_token": -1e-06,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(
            i.category == "cost" and "negative" in i.message for i in report.issues
        )

    def test_negative_output_cost(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "output_cost_per_token": -1e-06,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(
            i.category == "cost" and "negative" in i.message for i in report.issues
        )

    def test_high_input_cost_warning(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "input_cost_per_token": 0.02,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert not report.has_errors
        assert any(
            i.severity == ValidationSeverity.WARNING
            and "suspiciously high" in i.message
            for i in report.issues
        )

    def test_high_output_cost_warning(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "output_cost_per_token": 0.05,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert not report.has_errors
        assert any(
            i.severity == ValidationSeverity.WARNING
            and "suspiciously high" in i.message
            for i in report.issues
        )

    def test_invalid_order_zero(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "order": 0,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "order" for i in report.issues)

    def test_invalid_order_negative(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "order": -1,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "order" for i in report.issues)

    def test_invalid_order_bool(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                        "order": True,
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "order" for i in report.issues)

    def test_invalid_mode(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                    },
                    "model_info": {"mode": "invalid_mode"},
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert report.has_errors
        assert any(i.category == "mode" for i in report.issues)

    def test_valid_mode(self, cleaner):
        for mode in VALID_MODEL_MODES:
            config = {
                "model_list": [
                    {
                        "model_name": "model-1",
                        "litellm_params": {
                            "model": "openrouter/gpt-4",
                        },
                        "model_info": {"mode": mode},
                    }
                ]
            }
            report = cleaner.validate_config(config)
            assert not any(i.category == "mode" for i in report.issues), (
                f"Mode {mode} should be valid"
            )

    def test_unknown_provider_prefix_warning(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "unknown_provider/gpt-4",
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert not report.has_errors
        assert any(
            i.severity == ValidationSeverity.WARNING and i.category == "provider_prefix"
            for i in report.issues
        )

    def test_known_provider_prefix_no_warning(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "model-1",
                    "litellm_params": {
                        "model": "openrouter/gpt-4",
                    },
                }
            ]
        }
        report = cleaner.validate_config(config)
        assert not any(i.category == "provider_prefix" for i in report.issues)

    def test_valid_entry_counts(self, cleaner):
        config = {
            "model_list": [
                {
                    "model_name": "good",
                    "litellm_params": {"model": "openrouter/gpt-4", "order": 1},
                },
                "bad-entry",
                {
                    "model_name": "also-good",
                    "litellm_params": {"model": "openrouter/gpt-4o", "order": 2},
                },
            ]
        }
        report = cleaner.validate_config(config)
        assert report.total_entries == 3
        # 2 valid entries, 1 error entry
        assert report.valid_entries == 2


class TestUnifiedModelCleanerValidation:
    """Tests for UnifiedModelCleaner.validate_config()."""

    @pytest.fixture(autouse=True)
    def reset_provider_singleton(self, monkeypatch):
        """Reset ProviderConfigLoader singleton before each test."""
        from cleanup_base import ProviderConfigLoader

        # Reset the singleton instance
        ProviderConfigLoader._instance = None
        ProviderConfigLoader._config = {}
        yield
        # Reset again after test
        ProviderConfigLoader._instance = None
        ProviderConfigLoader._config = {}

    def test_unified_validate_config_delegates(self, tmp_path, monkeypatch):
        # Create minimal providers.yaml
        providers_content = {
            "providers": {
                "test_provider": {
                    "name": "TestProvider",
                    "description": "Test provider",
                    "api_url": "https://api.test.com/models",
                    "model_prefix": "test/",
                    "model_detection": {"type": "prefix"},
                    "pricing": {
                        "input_field": "pricing.prompt",
                        "output_field": "pricing.completion",
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                }
            }
        }
        providers_file = tmp_path / "providers.yaml"
        with open(providers_file, "w") as f:
            import yaml

            yaml.dump(providers_content, f)

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"model_list": []}, f)

        # Monkeypatch ProviderConfigLoader to use our temp file
        from cleanup_base import ProviderConfigLoader

        original_new = ProviderConfigLoader.__new__

        def mock_new(cls, config_path="providers.yaml"):
            instance = original_new.__get__(None, cls)(cls)
            instance._config_path = providers_file
            instance._load_config()
            return instance

        monkeypatch.setattr(ProviderConfigLoader, "__new__", mock_new)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["test_provider"],
            dry_run=False,
            verbose=False,
        )

        config = {"model_list": []}
        report = cleaner.validate_config(config)
        assert report.total_entries == 0

    def test_unified_validate_config_loads_config(self, tmp_path, monkeypatch):
        providers_content = {
            "providers": {
                "test_provider": {
                    "name": "TestProvider",
                    "description": "Test provider",
                    "api_url": "https://api.test.com/models",
                    "model_prefix": "test/",
                    "model_detection": {"type": "prefix"},
                    "pricing": {
                        "input_field": "pricing.prompt",
                        "output_field": "pricing.completion",
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                }
            }
        }
        providers_file = tmp_path / "providers.yaml"
        with open(providers_file, "w") as f:
            import yaml

            yaml.dump(providers_content, f)

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(
                {
                    "model_list": [
                        {
                            "model_name": "m",
                            "litellm_params": {"model": "test/x", "order": 1},
                        }
                    ]
                },
                f,
            )

        from cleanup_base import ProviderConfigLoader

        original_new = ProviderConfigLoader.__new__

        def mock_new(cls, config_path="providers.yaml"):
            instance = original_new.__get__(None, cls)(cls)
            instance._config_path = providers_file
            instance._load_config()
            return instance

        monkeypatch.setattr(ProviderConfigLoader, "__new__", mock_new)

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=["test_provider"],
            dry_run=False,
            verbose=False,
        )
        report = cleaner.validate_config()
        assert report.total_entries == 1

    def test_unified_validate_config_no_cleaners(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            import yaml

            yaml.dump(
                {
                    "model_list": [
                        {"model_name": "m", "litellm_params": {"model": "test/x"}}
                    ]
                },
                f,
            )

        cleaner = UnifiedModelCleaner(
            config_path=str(config_file),
            provider_names=[],
            dry_run=False,
            verbose=False,
        )
        report = cleaner.validate_config()
        assert report.total_entries == 1


class TestPrintValidationReport:
    """Tests for _print_validation_report()."""

    def test_print_empty_report(self, capsys):
        report = ValidationReport()
        report.total_entries = 0
        _print_validation_report(report)
        captured = capsys.readouterr()
        assert "0 entries checked" in captured.out

    def test_print_with_errors(self, capsys):
        report = ValidationReport()
        report.total_entries = 2
        report.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="model_name",
                entry_index=0,
                model_name="",
                model_id="",
                message="Missing model_name",
            )
        )
        _print_validation_report(report)
        captured = capsys.readouterr()
        assert "ERRORS (1):" in captured.out
        assert "Missing model_name" in captured.out
        assert "1 errors" in captured.out

    def test_print_with_warnings(self, capsys):
        report = ValidationReport()
        report.total_entries = 1
        report.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                category="provider_prefix",
                entry_index=0,
                model_name="m",
                model_id="unknown/x",
                message="Unknown prefix",
            )
        )
        _print_validation_report(report)
        captured = capsys.readouterr()
        assert "WARNINGS (1):" in captured.out
        assert "Unknown prefix" in captured.out
        assert "0 errors, 1 warnings" in captured.out

    def test_print_with_model_info(self, capsys):
        report = ValidationReport()
        report.total_entries = 1
        report.issues.append(
            ValidationIssue(
                severity=ValidationSeverity.ERROR,
                category="duplicate",
                entry_index=1,
                model_name="gpt-4",
                model_id="openrouter/gpt-4",
                message="Duplicate entry",
            )
        )
        _print_validation_report(report)
        captured = capsys.readouterr()
        assert "model_name='gpt-4'" in captured.out
        assert "[model='openrouter/gpt-4']" in captured.out


class TestCreateProviderMainValidate:
    """Tests for create_provider_main with --validate flag."""

    def test_validate_flag_exits_zero(self, tmp_path, capsys, monkeypatch):
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            import yaml

            yaml.dump({"model_list": []}, f)

        main_fn = create_provider_main(MockModelCleaner, "Test cleaner")
        monkeypatch.setattr(
            sys, "argv", ["test_script", "--validate", "--config", str(config_file)]
        )
        exit_code = main_fn()
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "0 entries checked" in captured.out

    def test_validate_flag_exits_one_on_errors(self, tmp_path, capsys, monkeypatch):
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            import yaml

            yaml.dump({"model_list": ["not-a-dict"]}, f)

        main_fn = create_provider_main(MockModelCleaner, "Test cleaner")
        monkeypatch.setattr(
            sys, "argv", ["test_script", "--validate", "--config", str(config_file)]
        )
        exit_code = main_fn()
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "ERRORS" in captured.out


class TestCleanupModelsMainValidate:
    """Tests for cleanup_models.py main() with --validate."""

    def test_main_validate_flag(self, tmp_path, monkeypatch, capsys):
        # Need providers.yaml for UnifiedModelCleaner init
        providers_content = {
            "providers": {
                "test_provider": {
                    "name": "TestProvider",
                    "description": "Test provider",
                    "api_url": "https://api.test.com/models",
                    "model_prefix": "test/",
                    "model_detection": {"type": "prefix"},
                    "pricing": {
                        "input_field": "pricing.prompt",
                        "output_field": "pricing.completion",
                    },
                    "model_name_prefix": "",
                    "model_name_cleanup": [],
                    "special_models": [],
                }
            }
        }
        providers_file = tmp_path / "providers.yaml"
        with open(providers_file, "w") as f:
            import yaml

            yaml.dump(providers_content, f)

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"model_list": []}, f)

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cleanup_models",
                "--provider",
                "test_provider",
                "--validate",
                "--config",
                str(config_file),
            ],
        )

        from cleanup_models import main

        # The ProviderConfigLoader singleton may have been loaded before,
        # so we need to reset it or patch it to use our temp providers.yaml
        # Since ProviderConfigLoader is a singleton, patch its _load_config behavior
        from cleanup_base import ProviderConfigLoader

        original_new = ProviderConfigLoader.__new__

        def mock_new(cls, config_path="providers.yaml"):
            instance = original_new.__get__(None, cls)(cls)
            instance._config_path = providers_file
            instance._load_config()
            return instance

        monkeypatch.setattr(ProviderConfigLoader, "__new__", mock_new)

        exit_code = main()
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "0 entries checked" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
