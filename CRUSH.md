# Development Commands

## Core Scripts
```bash
# Validate models, update costs, and remove invalid entries
python cleanup_openrouter_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_requesty_models.py [--config config.yaml] [--dry-run] [--verbose]
python cleanup_novita_models.py [--config config.yaml] [--dry-run] [--verbose]

# Add new models with automatic cost detection
python cleanup_openrouter_models.py --add-model "model/name" [--dry-run]
python cleanup_requesty_models.py --add-model "provider/model" [--dry-run]
python cleanup_novita_models.py --add-model "provider/model" [--dry-run]

# Multiple model addition (space-separated)
python cleanup_openrouter_models.py --add-model model1 model2 model3 [--dry-run]
```

## Dependencies
```bash
pip install -r requirements.txt

# Create virtual environment (if needed)
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

## Testing Workflow
Always run with `--dry-run --verbose` first to preview changes before applying them.

# Code Style Guidelines

## Python Style
- Use Python 3.7+ with type hints (`from typing import Dict, List, Tuple, Any, Optional`)
- Follow PEP 8 for code formatting
- Use descriptive variable and method names (snake_case)
- Classes use PascalCase (e.g., `OpenRouterModelCleaner`)

## Error Handling
- Use comprehensive try-except blocks with specific exception types
- Log errors with detailed messages using self.logger
- Raise exceptions with clear error messages
- Handle YAML parsing errors, network errors, and file operations

## Logging
- Use structured logging with timestamps: `'%(asctime)s - %(levelname)s - %(message)s'`
- Set log level based on `--verbose` flag (DEBUG for verbose, INFO for normal)
- Use appropriate log levels: DEBUG, INFO, WARNING, ERROR
- Always log key operations and state changes

## API Integration
- Use requests library with timeout (30 seconds default)
- Validate API response structure before processing
- Handle network failures gracefully with proper error messages
- Use raise_for_status() for HTTP error handling

## Configuration Management
- Use yaml.safe_load() for secure YAML parsing
- Validate configuration structure (check for required keys)
- Use Path objects for file path operations
- Preserve YAML formatting when writing files

## Constants and Patterns
- Define API URLs and default values as module-level constants
- Use class-based architecture with clear separation of concerns
- Implement dry-run mode for all destructive operations
- Use percentage-based logging for cost changes
