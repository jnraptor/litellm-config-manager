#!/usr/bin/env python3
"""
Test script that validates cleanup scripts produce correct outputs from input-and-outputs.md.

This script:
1. Parses tests/input-and-outputs.md to extract test cases
2. For each provider, tests that parse_api_model() produces correct output
3. Tests that create_model_entry() produces the expected config entry
4. Validates cost calculations and field mappings

Usage:
    # Run as standalone script
    python tests/test_input_outputs.py

    # Run with pytest
    pytest tests/test_input_outputs.py -v

Adding New Test Cases:
    To add a new test case, edit tests/input-and-outputs.md and add a new section:

    ## provider_name
    ### input
    ```json
    { ... raw API response JSON ... }
    ```
    ### output
    ```yaml
    - model_name: expected-model-name
      litellm_params:
        model: provider/model-id
        order: 5
        input_cost_per_token: 1.0e-06
        output_cost_per_token: 3.0e-06
        # ... other expected fields
    ```

    The provider_name must match a provider defined in providers.yaml.
    The test will automatically pick up new sections from the markdown file.

Test Coverage:
    - openrouter: Tests pricing extraction from pricing.prompt/pricing.completion
    - nano_gpt: Tests per-million token pricing conversion
    - vercel: Tests pricing.input/pricing.output field extraction
    - poe: Tests standard prompt/completion pricing
    - nvidia: Tests default cost handling for free models
    - kilo: Tests api_base and api_key configuration
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml

# Add parent directory to path to import cleanup_base
sys.path.insert(0, str(Path(__file__).parent.parent))

from cleanup_base import ConfigDrivenModelCleaner, ProviderConfigLoader

# Try to import pytest for type hints, but don't require it
try:
    import pytest

    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False
    pytest = None


class InputOutputTestCase:
    """Represents a single test case from input-and-outputs.md."""

    def __init__(self, provider: str, input_data: Dict, expected_output: Dict):
        self.provider = provider
        self.input_data = input_data
        self.expected_output = expected_output


def parse_markdown_file(filepath: str) -> List[InputOutputTestCase]:
    """
    Parse the input-and-outputs.md file to extract test cases.

    The file format is:
    ## provider_name
    ### input
    ```json
    {...}
    ```
    ### output
    ```yaml
    ...
    ```
    """
    with open(filepath, "r") as f:
        content = f.read()

    test_cases = []

    # Split by ## provider headers and process each section
    # First section might contain text before first ##, so skip if needed
    sections = re.split(r"\n##\s+", content)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Extract provider name (first line)
        lines = section.split("\n", 1)
        if len(lines) < 2:
            continue

        provider_name = lines[0].strip().lower()
        # Remove leading ## if present
        provider_name = provider_name.lstrip("#").strip()
        section_content = lines[1]

        # Skip if this doesn't look like a provider section
        if not provider_name or provider_name in ["input", "output"]:
            continue

        # Extract input JSON - look for ```json block after ### input
        input_match = re.search(
            r"###\s*input\s*\n\s*```json\s*\n(.*?)\n\s*```", section_content, re.DOTALL
        )
        if not input_match:
            print(f"Warning: No input found for provider {provider_name}")
            continue

        input_json = input_match.group(1).strip()
        try:
            input_data = json.loads(input_json)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {provider_name}: {e}")
            continue

        # Extract output YAML - look for ```yaml block after ### output
        output_match = re.search(
            r"###\s*output\s*\n\s*```yaml\s*\n(.*?)\n\s*```", section_content, re.DOTALL
        )
        if not output_match:
            print(f"Warning: No output found for provider {provider_name}")
            continue

        output_yaml = output_match.group(1).strip()
        try:
            # Parse YAML - it should be a single model entry
            expected_output = yaml.safe_load(output_yaml)
            # If it's a list, take the first item
            if isinstance(expected_output, list):
                expected_output = expected_output[0]
        except yaml.YAMLError as e:
            print(f"Error parsing YAML for {provider_name}: {e}")
            continue

        test_cases.append(
            InputOutputTestCase(provider_name, input_data, expected_output)
        )

    return test_cases


def costs_match(actual: float, expected: float, rel_tol: float = 1e-9) -> bool:
    """Check if two costs are equal within tolerance."""
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False
    if actual == expected:
        return True
    if actual == 0.0 or expected == 0.0:
        return abs(actual - expected) < 1e-15
    return abs(actual - expected) <= rel_tol * max(abs(actual), abs(expected))


class MockCleaner(ConfigDrivenModelCleaner):
    """Mock cleaner that doesn't require actual config files for testing."""

    def __init__(self, provider_name: str):
        # Initialize without calling parent's __init__ to avoid file loading
        self.logger = type(
            "obj",
            (object,),
            {
                "debug": lambda *args: None,
                "info": lambda *args: None,
                "warning": lambda *args: None,
                "error": lambda *args: None,
            },
        )()
        self.dry_run = True
        self.verbose = False

        # Load provider configuration directly from YAML
        providers_path = Path(__file__).parent.parent / "providers.yaml"
        with open(providers_path, "r") as f:
            providers_data = yaml.safe_load(f)

        if (
            "providers" not in providers_data
            or provider_name not in providers_data["providers"]
        ):
            available = list(providers_data.get("providers", {}).keys())
            raise ValueError(
                f"Provider '{provider_name}' not found. Available: {', '.join(available)}"
            )

        self.provider_config = providers_data["providers"][provider_name]

        # Set class attributes from configuration
        self.PROVIDER_NAME = self.provider_config.get("name", provider_name)
        self.API_URL = self.provider_config["api_url"]
        self.MODEL_PREFIX = self.provider_config["model_prefix"]
        self.SPECIAL_MODELS = set(self.provider_config.get("special_models", []))
        self.PROVIDER_ORDER = self.provider_config.get("order", 2)

        # Store additional config for use in methods
        self._model_detection = self.provider_config.get("model_detection", {})
        self._pricing_config = self.provider_config.get("pricing", {})
        self._model_name_prefix = self.provider_config.get("model_name_prefix", "")
        self._model_name_cleanup = self.provider_config.get("model_name_cleanup", [])
        self._api_base_config = self.provider_config.get("api_base_config")
        self._api_key_env = self.provider_config.get("api_key_env")
        self._embeddings_api_url = self.provider_config.get("embeddings_api_url")
        self._free_variant_suffix = self.provider_config.get("free_variant_suffix")


def run_test_case(test_case: InputOutputTestCase) -> Tuple[bool, List[str]]:
    """
    Run a single test case.

    Returns:
        Tuple of (success: bool, errors: List[str])
    """
    errors = []

    try:
        cleaner = MockCleaner(test_case.provider)
    except Exception as e:
        return False, [f"Failed to create MockCleaner: {e}"]

    # Step 1: Parse the API model
    try:
        parsed_model = cleaner.parse_api_model(test_case.input_data)
    except Exception as e:
        return False, [f"parse_api_model failed: {e}"]

    # Step 2: Create the model entry
    model_id = test_case.input_data.get("id", "")

    # For kilo provider, the model_prefix is 'openai/', so we need the full model_id
    if test_case.provider == "kilo":
        # Kilo uses the full model_id as is (z-ai/glm-5)
        pass

    expected_model_name = test_case.expected_output.get("model_name", "")

    try:
        model_entry = cleaner.create_model_entry(
            model_id=model_id,
            api_model_info=parsed_model,
            model_name=expected_model_name,
        )
    except Exception as e:
        return False, [f"create_model_entry failed: {e}"]

    # Step 3: Validate the entry
    expected_litellm = test_case.expected_output.get("litellm_params", {})
    actual_litellm = model_entry.get("litellm_params", {})

    # Check model field
    if actual_litellm.get("model") != expected_litellm.get("model"):
        errors.append(
            f"litellm_params.model mismatch: "
            f"expected '{expected_litellm.get('model')}', "
            f"got '{actual_litellm.get('model')}'"
        )

    # Check order field
    if actual_litellm.get("order") != expected_litellm.get("order"):
        errors.append(
            f"litellm_params.order mismatch: "
            f"expected {expected_litellm.get('order')}, "
            f"got {actual_litellm.get('order')}"
        )

    # Check api_base field (if expected)
    if "api_base" in expected_litellm:
        if actual_litellm.get("api_base") != expected_litellm.get("api_base"):
            errors.append(
                f"litellm_params.api_base mismatch: "
                f"expected '{expected_litellm.get('api_base')}', "
                f"got '{actual_litellm.get('api_base')}'"
            )

    # Check api_key field (if expected)
    if "api_key" in expected_litellm:
        if actual_litellm.get("api_key") != expected_litellm.get("api_key"):
            errors.append(
                f"litellm_params.api_key mismatch: "
                f"expected '{expected_litellm.get('api_key')}', "
                f"got '{actual_litellm.get('api_key')}'"
            )

    # Check input_cost_per_token
    expected_input_cost = expected_litellm.get("input_cost_per_token")
    actual_input_cost = actual_litellm.get("input_cost_per_token")
    if not costs_match(actual_input_cost, expected_input_cost):
        errors.append(
            f"input_cost_per_token mismatch: "
            f"expected {expected_input_cost}, got {actual_input_cost}"
        )

    # Check output_cost_per_token
    expected_output_cost = expected_litellm.get("output_cost_per_token")
    actual_output_cost = actual_litellm.get("output_cost_per_token")
    if not costs_match(actual_output_cost, expected_output_cost):
        errors.append(
            f"output_cost_per_token mismatch: "
            f"expected {expected_output_cost}, got {actual_output_cost}"
        )

    # Check model_info (if expected)
    if "model_info" in test_case.expected_output:
        expected_model_info = test_case.expected_output.get("model_info", {})
        actual_model_info = model_entry.get("model_info", {})
        if expected_model_info != actual_model_info:
            errors.append(
                f"model_info mismatch: "
                f"expected {expected_model_info}, got {actual_model_info}"
            )

    return len(errors) == 0, errors


def get_test_cases() -> List[InputOutputTestCase]:
    """Get all test cases from the markdown file."""
    test_dir = Path(__file__).parent
    markdown_file = test_dir / "input-and-outputs.md"

    if not markdown_file.exists():
        raise FileNotFoundError(f"Test file not found: {markdown_file}")

    return parse_markdown_file(str(markdown_file))


def main():
    """Main test runner for standalone execution."""
    # Find the input-and-outputs.md file
    test_dir = Path(__file__).parent
    markdown_file = test_dir / "input-and-outputs.md"

    if not markdown_file.exists():
        print(f"Error: {markdown_file} not found")
        sys.exit(1)

    print(f"Loading test cases from {markdown_file}...")
    test_cases = parse_markdown_file(str(markdown_file))
    print(f"Found {len(test_cases)} test cases\n")

    # Track results
    passed = 0
    failed = 0

    for test_case in test_cases:
        print(f"Testing {test_case.provider}...", end=" ")

        success, errors = run_test_case(test_case)

        if success:
            print("✅ PASS")
            passed += 1
        else:
            print("❌ FAIL")
            for error in errors:
                print(f"    - {error}")
            failed += 1

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(test_cases)} total")

    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed! ✅")
        sys.exit(0)


# Pytest-compatible test functions
if HAVE_PYTEST:

    class TestInputOutputs:
        """Pytest test class for input/output validation."""

        @pytest.fixture(scope="class")
        def test_cases(self):
            """Fixture that loads all test cases."""
            return get_test_cases()

        @pytest.mark.parametrize(
            "provider", ["openrouter", "nano_gpt", "vercel", "poe", "nvidia", "kilo"]
        )
        def test_provider_output(self, provider):
            """Test that each provider produces correct output."""
            # Find the test case for this provider
            all_cases = get_test_cases()
            test_case = None
            for case in all_cases:
                if case.provider == provider:
                    test_case = case
                    break

            if test_case is None:
                pytest.skip(f"No test case found for provider: {provider}")

            success, errors = run_test_case(test_case)

            if not success:
                error_msg = "\n".join([f"  - {e}" for e in errors])
                pytest.fail(f"Test failed for {provider}:\n{error_msg}")


if __name__ == "__main__":
    main()
