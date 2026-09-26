"""Read non-tensor metadata from a torch ZIP checkpoint without importing torch.

The unpickler accepts only the small set of reconstruction globals emitted by
torch.save. Tensor/storage payloads are replaced with inert sentinels; no model
weights are materialized or executed.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any


class _TensorSentinel:
    pass


def _rebuild_tensor(*_args: Any, **_kwargs: Any) -> _TensorSentinel:
    return _TensorSentinel()


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name.startswith("_rebuild"):
            return _rebuild_tensor
        if module in {"torch", "torch.storage"} and (
            name.endswith("Storage") or name in {"UntypedStorage", "TypedStorage"}
        ):
            return _TensorSentinel
        raise pickle.UnpicklingError(f"refusing checkpoint global: {module}.{name}")

    def persistent_load(self, _pid: Any) -> _TensorSentinel:
        return _TensorSentinel()


def load_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members) != 1:
            raise ValueError("checkpoint must contain exactly one data.pkl")
        value = _MetadataUnpickler(io.BytesIO(archive.read(members[0]))).load()
    if not isinstance(value, dict):
        raise ValueError("checkpoint root is not a dict")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    value = load_metadata(args.checkpoint)
    sites = value.get("sites")
    if not isinstance(sites, list):
        raise SystemExit("checkpoint sites is not a list")
    print(
        json.dumps(
            {
                "base_model_weights_included": value.get("base_model_weights_included"),
                "cap_materialized_without_refit": value.get(
                    "cap_materialized_without_refit"
                ),
                "maximum_relative_corrections": [
                    site.get("maximum_relative_correction")
                    if isinstance(site, dict)
                    else None
                    for site in sites
                ],
                "site_count": len(sites),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
