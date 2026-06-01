"""
Select delirium case-classification (Agent 2 interpretation) prompt by version.

Environment:
  DELIRIUM_PROMPT_VERSION=v1|v2  (default: v1)

Prompt files (do not overwrite agent_interpretation.txt — legacy reference):
  prompts/delirium_case_classification_v1.txt
  prompts/delirium_case_classification_v2.txt
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.pipeline.paths import PROJECT_ROOT
from src.pipeline.prompt_run_paths import (
    ALLOWED_PROMPT_VERSIONS,
    get_prompt_version_from_env,
    normalize_prompt_version,
)

LOGGER = logging.getLogger(__name__)

PROMPTS_DIR = PROJECT_ROOT / "prompts"
LEGACY_INTERPRETATION_PROMPT_PATH = PROMPTS_DIR / "agent_interpretation.txt"


def interpretation_prompt_filename(version: str) -> str:
    v = normalize_prompt_version(version)
    return f"delirium_case_classification_{v}.txt"


def resolve_interpretation_prompt_path(version: str | None = None) -> Path:
    v = normalize_prompt_version(version) if version is not None else get_prompt_version_from_env()
    path = PROMPTS_DIR / interpretation_prompt_filename(v)
    if not path.exists():
        raise FileNotFoundError(
            f"Delirium classification prompt missing for version={v}: {path}"
        )
    return path


def load_interpretation_prompt(version: str | None = None) -> str:
    path = resolve_interpretation_prompt_path(version)
    text = path.read_text(encoding="utf-8")
    LOGGER.debug("Loaded interpretation prompt version=%s path=%s", version or "env", path)
    return text
