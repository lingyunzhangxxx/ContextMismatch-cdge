#!/usr/bin/env python3
"""Reject credentials, private paths, and model-weight artifacts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_SUFFIXES = {".pem", ".key", ".p12", ".pt", ".pth", ".safetensors"}
TEXT_PATTERNS = {
    "private macOS path": re.compile(r"/Users/[A-Za-z0-9._-]+/"),
    "private home path": re.compile(r"/home/[A-Za-z0-9._-]+/"),
    "root home path": re.compile(r"/root/"),
    "private cluster path": re.compile(r"/WORK/[A-Za-z0-9._-]+/"),
    "provider token": re.compile(r"\b(?:gh[pousr]_|github_pat_|hf_|sk-)[A-Za-z0-9_-]{20,}"),
    "private network address": re.compile(r"\b(?:10\.[0-9]{1,3}|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}\b"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "assigned credential": re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|password|passwd|secret)"
        r"\s*[:=]\s*['\"][^'\"\n]{8,}['\"]"
    ),
    "bearer token": re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._-]{8,}"),
}


def files_to_scan() -> list[Path]:
    if shutil.which("git") and (ROOT / ".git").exists():
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            text=True,
        )
        return [ROOT / item for item in output.splitlines() if item]
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache"}
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not excluded.intersection(path.relative_to(ROOT).parts)
    ]


def main() -> None:
    findings: list[str] = []
    for path in files_to_scan():
        relative = path.relative_to(ROOT).as_posix()
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            findings.append(f"environment credential file: {relative}")
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append(f"blocked artifact type: {relative}")
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            findings.append(f"file exceeds 20 MiB: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if relative == "scripts/privacy_audit.py":
            continue
        for label, pattern in TEXT_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{label}: {relative}")
    if findings:
        raise SystemExit("Privacy audit failed:\n  - " + "\n  - ".join(sorted(set(findings))))
    print(f"Privacy audit passed: {len(files_to_scan())} publishable files checked")


if __name__ == "__main__":
    main()
