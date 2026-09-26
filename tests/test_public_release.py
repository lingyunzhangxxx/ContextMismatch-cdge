"""Checks that publication cannot silently omit or alter evidence/dependencies."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.public_demo import check_finite, reject_nonfinite, verify_manifest

ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        payload = root / "example.json"
        payload.write_text('{}\n')
        (root / "MANIFEST.sha256").write_text(f"{hashlib.sha256(payload.read_bytes()).hexdigest()}  example.json\n")
        return root

    def test_manifest_rejects_unlisted_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / "unlisted.json").write_text('{}\n')
            with self.assertRaisesRegex(ValueError, "coverage"):
                verify_manifest(root)

    def test_manifest_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / "example.json").write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_manifest(root)

    def test_manifest_rejects_escape_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            manifest = root / "MANIFEST.sha256"
            original = manifest.read_text()
            for text in (original + original, original.replace("example.json", "../example.json")):
                manifest.write_text(text)
                with self.assertRaisesRegex(ValueError, "invalid or duplicate"):
                    verify_manifest(root)

    def test_nonfinite_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            json.loads('{"margin":NaN}', parse_constant=reject_nonfinite)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            check_finite(json.loads('{"nested":[{"margin":1e999}]}'))

    def test_all_local_script_imports_are_released(self):
        missing = []
        for path in (ROOT / "scripts").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
                modules = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                for name in modules:
                    if name.startswith("scripts."):
                        target = ROOT.joinpath(*name.split("."))
                        if not target.with_suffix(".py").is_file() and not target.is_dir():
                            missing.append((str(path.relative_to(ROOT)), name))
        self.assertEqual(missing, [], "release is missing imported experiment modules")
