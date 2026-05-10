"""
Central keyword list for delirium-related text hints.

Used by report reduction (coverage counting) and the LLM runtime prefilter.
"""

from __future__ import annotations

DELIRIUM_HINT_KEYWORDS = (
    "delir",
    "delirium",
    "delirant",
    "delirös",
    "verwirrt",
    "verwirrtheit",
    "desorientiert",
    "desorientierung",
    "agitiert",
    "agitation",
    "unruhig",
    "vigilanz",
    "vigilanzminderung",
    "somnolent",
    "soporös",
    "bewusstseinsstörung",
    "bewusstseinstrübung",
)


def haystack_contains_delirium_hint(haystack: str) -> bool:
    """Return True if any DELIRIUM_HINT_KEYWORDS substring appears (case-insensitive)."""
    if not haystack or not str(haystack).strip():
        return False
    lower = str(haystack).lower()
    for kw in DELIRIUM_HINT_KEYWORDS:
        if kw.lower() in lower:
            return True
    return False
