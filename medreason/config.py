"""Configuration management for Veridicus."""

from __future__ import annotations

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env file
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value and key not in os.environ:
                os.environ[key] = value
RESULTS_DIR = PROJECT_ROOT / "results"
BENCHMARK_DIR = PROJECT_ROOT / "benchmark"

RESULTS_DIR.mkdir(exist_ok=True)
BENCHMARK_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://veridicus:veridicus@localhost:5432/veridicus"
)
SQLITE_PATH = PROJECT_ROOT / "veridicus.db"

# API keys
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# Embedding
EMBEDDING_DIM = 1536
