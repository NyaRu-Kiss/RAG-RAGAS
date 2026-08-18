"""Deterministic, non-sensitive helpers for evaluation artifacts."""

import hashlib
import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


_EVAL_TABLE_NAME = re.compile(r"eval_[a-z0-9_]+\Z")
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "access_token",
    "refresh_token",
)


def canonical_json_sha256(payload: object) -> str:
    """Return a stable SHA-256 digest for JSON-serializable data."""
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def source_sample_ids_sha256(source_sample_ids: Sequence[str]) -> str:
    """Hash source sample IDs in their supplied order."""
    return canonical_json_sha256(list(source_sample_ids))


def get_git_sha(repo_path: Path | None = None) -> str | None:
    """Return the current commit SHA, or None when git metadata is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def safe_config_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy configuration while dropping recursively nested sensitive keys."""

    def sanitize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): sanitize(item)
                for key, item in value.items()
                if not _is_sensitive_key(str(key))
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        return value

    return sanitize(config)


def validate_eval_table_name(table_name: str) -> str:
    """Validate the only PostgreSQL table namespace evaluation may modify."""
    if not _EVAL_TABLE_NAME.fullmatch(table_name):
        raise ValueError("Evaluation table name must match eval_[a-z0-9_]+")
    return table_name


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)
