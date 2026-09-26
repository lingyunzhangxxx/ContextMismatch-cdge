from __future__ import annotations

import ast
import hashlib
import sys
import types
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


REQUIRED_SOURCE_FILES = {
    "caa": "CAA/generate_vectors.py",
    "cast": "activation-steering/activation_steering/steering_vector.py",
    "loreft": "pyreft/pyreft/interventions.py",
    "reps_intervention": "axbench/axbench/models/interventions.py",
    "reps_loss": "axbench/axbench/models/preference_model.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_required_sources(source_root: Path, expected: dict[str, str]) -> dict[str, str]:
    observed = {}
    if set(expected) != set(REQUIRED_SOURCE_FILES):
        raise ValueError("official source checkpoint does not bind every required source")
    for key, relative in REQUIRED_SOURCE_FILES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing fixed official source: {path}")
        observed[key] = sha256_file(path)
        if observed[key] != expected[key]:
            raise ValueError(f"official source SHA mismatch: {key}")
    return observed


class _SourcelessIntervention:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _TrainableIntervention:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class _DistributedRepresentationIntervention(nn.Module):
    def __init__(self, *args, **kwargs):
        self.embed_dim = int(kwargs.pop("embed_dim"))
        kwargs.pop("keep_last_dim", None)
        super().__init__()


class _ConstantSourceIntervention:
    pass


class _CollectIntervention:
    pass


class _SigmoidMaskIntervention:
    pass


InterventionOutput = namedtuple("InterventionOutput", ("output", "latent"))


def _pyvene_shim() -> types.ModuleType:
    module = types.ModuleType("pyvene")
    module.SourcelessIntervention = _SourcelessIntervention
    module.TrainableIntervention = _TrainableIntervention
    module.DistributedRepresentationIntervention = _DistributedRepresentationIntervention
    module.ConstantSourceIntervention = _ConstantSourceIntervention
    module.CollectIntervention = _CollectIntervention
    module.SigmoidMaskIntervention = _SigmoidMaskIntervention
    module.InterventionOutput = InterventionOutput
    return module


def _extract_definition(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text(), filename=str(path))
    matches = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one official definition for {name}")
    return matches[0]


def load_loreft_class(source_root: Path, expected_sha: str):
    path = source_root / REQUIRED_SOURCE_FILES["loreft"]
    if sha256_file(path) != expected_sha:
        raise ValueError("LoReFT official source SHA mismatch")
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in {"LowRankRotateLayer", "LoreftIntervention"}
    ]
    if [node.name for node in selected] != ["LowRankRotateLayer", "LoreftIntervention"]:
        raise ValueError("LoReFT official class definitions are missing or reordered")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {
        "torch": torch,
        "OrderedDict": __import__("collections").OrderedDict,
        "SourcelessIntervention": _SourcelessIntervention,
        "TrainableIntervention": _TrainableIntervention,
        "DistributedRepresentationIntervention": _DistributedRepresentationIntervention,
        "ACT2FN": {"linear": lambda value: value},
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    cls = namespace["LoreftIntervention"]
    cls.__official_source_path__ = str(path)
    cls.__official_source_sha256__ = expected_sha
    return cls


def load_reps_class(source_root: Path, expected_sha: str):
    path = source_root / REQUIRED_SOURCE_FILES["reps_intervention"]
    if sha256_file(path) != expected_sha:
        raise ValueError("RePS official intervention source SHA mismatch")
    node = _extract_definition(path, "PreferenceVectorIntervention")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {
        "torch": torch,
        "SourcelessIntervention": _SourcelessIntervention,
        "TrainableIntervention": _TrainableIntervention,
        "DistributedRepresentationIntervention": _DistributedRepresentationIntervention,
        "InterventionOutput": InterventionOutput,
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    cls = namespace["PreferenceVectorIntervention"]
    cls.__official_source_path__ = str(path)
    cls.__official_source_sha256__ = expected_sha
    return cls


def load_preference_loss(source_root: Path, expected_sha: str):
    path = source_root / REQUIRED_SOURCE_FILES["reps_loss"]
    if sha256_file(path) != expected_sha:
        raise ValueError("RePS official loss source SHA mismatch")
    node = _extract_definition(path, "preference_loss")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch, "F": F, "Tuple": tuple}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    function = namespace["preference_loss"]
    function.__official_source_path__ = str(path)
    function.__official_source_sha256__ = expected_sha
    return function
