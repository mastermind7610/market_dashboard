"""
Build / deploy identity for the Streamlit footer.

Resolves a short git SHA when possible so a stale Community Cloud
instance is obvious after a merge that did not auto-redeploy.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 — fixed argv: git rev-parse only
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_VERSION = "0.6.0"


@lru_cache(maxsize=1)
def git_sha(short: bool = True) -> str | None:
    """Best-effort commit SHA. Never raises; returns None when unavailable."""
    env_keys = (
        "GIT_COMMIT",
        "GITHUB_SHA",
        "COMMIT_SHA",
        "STREAMLIT_COMMIT_HASH",
        "HEROKU_SLUG_COMMIT",
    )
    for key in env_keys:
        value = (os.environ.get(key) or "").strip()
        if value:
            return value[:7] if short else value
    try:
        output = subprocess.check_output(  # nosec B603 B607 — constant argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            timeout=2,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    if not output:
        return None
    return output[:7] if short else output


def build_label() -> str:
    """Human-readable stamp: ``v0.6.0 · abc1234`` or version alone."""
    sha = git_sha()
    if sha:
        return f"v{PACKAGE_VERSION} · {sha}"
    return f"v{PACKAGE_VERSION}"


if __name__ == "__main__":
    print(build_label())
