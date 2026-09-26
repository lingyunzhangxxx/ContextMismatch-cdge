import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v15.materialize_native_capture_authorization import materialize


class NativeCaptureAuthorizationTest(unittest.TestCase):
    def _file(self, root: Path, name: str, value=None) -> Path:
        path = root / name
        if value is None:
            path.write_text(name + "\n")
        else:
            atomic_write_text(path, json.dumps(value, sort_keys=True) + "\n")
        return path

    def test_binds_three_native_sites(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "code-v90"; code.mkdir()
            self._file(code, "bundle.sha256")
            protocol = self._file(root, "protocol.json", {
                "method_short_name": "C-DGE-V4.2",
                "status": "frozen_before_qwen3_5_native_discovery_forward",
                "final_test_open": False, "final_test_open_count": 0,
                "production_rollout_approved": False,
            })
            site = self._file(root, "sites.json", {
                "stage": "qwen35_cdge_v4_2_native_site_selection",
                "method": "C-DGE-V4.2", "selection_partition": "subspace_fit",
                "locked": True, "operator_dev_accessed": False,
                "final_test_open": False, "final_test_open_count": 0,
                "production_rollout_approved": False,
                "selected_sites_ordered": [
                    {"layer": 2, "component": "mlp"},
                    {"layer": 7, "component": "self_attn"},
                    {"layer": 19, "component": "mlp"},
                ],
                "audits": {"residual": {"success": True}, "component": {"success": True}},
            })
            paths = {name: self._file(root, name) for name in (
                "selector", "crossover", "manifest", "model", "archive",
                "coordinator", "pull", "design",
            )}
            args = argparse.Namespace(
                stage="governance", code_root=code, selected_node="a06",
                created_utc="2026-07-31T00:00:00Z", protocol=protocol,
                site_manifest=site, selector_snapshot=paths["selector"],
                crossover_contract=paths["crossover"], manifest=paths["manifest"],
                model_contract=paths["model"], archive_helper=paths["archive"],
                archive_coordinator=paths["coordinator"], pull_helper=paths["pull"],
                design_audit=paths["design"], controls=None,
            )
            value = materialize(args)
            self.assertEqual(value["sites"], ["2:mlp", "7:self_attn", "19:mlp"])
            self.assertEqual(value["site_manifest_sha256"], sha256_file(site))
            self.assertEqual(value["expected_rows"], 6144)
            self.assertFalse(value["final_test_open"])


if __name__ == "__main__":
    unittest.main()
