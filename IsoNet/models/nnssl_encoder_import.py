"""Extract and strictly load nnSSL ResEncL encoder checkpoints.

This module intentionally has no nnSSL or dynamic-network-architectures import.
Only the canonical ``conv`` and ``norm`` registrations are retained; redundant
or stale ``all_modules`` aliases from the source state dict are discarded.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


FORMAT_VERSION = 1
ARCHITECTURE = "ResEncL"
EXPECTED_ENCODER_PARAMETERS = 90_296_576
SOURCE_PREFIXES = ("encoder.stem.", "encoder.stages.")


def _torch_load(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    path = str(path)
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except (TypeError, RuntimeError):
        return torch.load(path, **kwargs)


def _is_encoder_key(key: str) -> bool:
    return key.startswith(SOURCE_PREFIXES)


def _canonical_alias_key(key: str) -> str | None:
    """Translate a duplicate ``all_modules`` key back to its canonical key."""
    marker = ".all_modules."
    if marker not in key:
        return None
    prefix, suffix = key.split(marker, 1)
    module_index, _, parameter = suffix.partition(".")
    if module_index == "0":
        return f"{prefix}.conv.{parameter}"
    if module_index == "1":
        return f"{prefix}.norm.{parameter}"
    return None


def extract_encoder_state_dict(
    checkpoint: Mapping[str, Any],
    target_state_dict: Mapping[str, torch.Tensor] | None = None,
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, Any]]:
    """Extract canonical dense encoder tensors and perform strict validation."""
    if "encoder_state_dict" in checkpoint:
        source = checkpoint["encoder_state_dict"]
        compact_input = True
    elif "network_weights" in checkpoint:
        source = checkpoint["network_weights"]
        compact_input = False
    else:
        raise KeyError("Checkpoint must contain network_weights or encoder_state_dict")

    canonical: OrderedDict[str, torch.Tensor] = OrderedDict()
    duplicate_keys: list[str] = []
    ignored_non_encoder: list[str] = []
    alias_mismatches: list[str] = []

    for source_key, tensor in source.items():
        if compact_input:
            target_key = source_key.removeprefix("encoder.")
        else:
            if not _is_encoder_key(source_key):
                ignored_non_encoder.append(source_key)
                continue
            alias_key = _canonical_alias_key(source_key)
            if alias_key is not None:
                duplicate_keys.append(source_key)
                canonical_source = source.get(alias_key)
                if canonical_source is None or not torch.equal(tensor, canonical_source):
                    alias_mismatches.append(source_key)
                continue
            target_key = source_key.removeprefix("encoder.")
        if target_key in canonical:
            raise ValueError(f"Duplicate canonical encoder key: {target_key}")
        canonical[target_key] = tensor.detach().cpu()

    missing: list[str] = []
    unexpected: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    if target_state_dict is not None:
        missing = sorted(set(target_state_dict) - set(canonical))
        unexpected = sorted(set(canonical) - set(target_state_dict))
        for key in sorted(set(canonical) & set(target_state_dict)):
            if canonical[key].shape != target_state_dict[key].shape:
                shape_mismatches.append(
                    {
                        "key": key,
                        "source": tuple(canonical[key].shape),
                        "target": tuple(target_state_dict[key].shape),
                    }
                )
        if missing or unexpected or shape_mismatches:
            raise ValueError(
                "Strict encoder validation failed: "
                f"missing={len(missing)}, unexpected={len(unexpected)}, "
                f"shape_mismatches={len(shape_mismatches)}"
            )

    parameter_count = sum(tensor.numel() for tensor in canonical.values())
    if parameter_count != EXPECTED_ENCODER_PARAMETERS:
        raise ValueError(
            f"Expected {EXPECTED_ENCODER_PARAMETERS:,} encoder parameters, got {parameter_count:,}"
        )

    stages = sorted(
        {int(key.split(".")[1]) for key in canonical if key.startswith("stages.")}
    )
    report = {
        "parameter_count": parameter_count,
        "tensor_count": len(canonical),
        "stem_loaded": any(key.startswith("stem.") for key in canonical),
        "stages_loaded": stages,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "discarded_duplicate_count": len(duplicate_keys),
        "discarded_duplicate_keys": duplicate_keys,
        # SparK conversion can replace the canonical .conv/.norm attributes
        # without updating the old Sequential aliases. Canonical tensors are
        # authoritative even when a stale alias no longer compares equal.
        "stale_alias_count": len(alias_mismatches),
        "stale_alias_keys": alias_mismatches,
        "ignored_non_encoder_count": len(ignored_non_encoder),
    }
    if not report["stem_loaded"] or stages != list(range(6)):
        raise ValueError(f"Incomplete ResEncL coverage: stem={report['stem_loaded']}, stages={stages}")
    return canonical, report


def _normalization_from_checkpoint(checkpoint: Mapping[str, Any]) -> str:
    init_args = checkpoint.get("init_args", {})
    plan = init_args.get("plan", {}) if isinstance(init_args, Mapping) else {}
    configuration_name = init_args.get("configuration_name") if isinstance(init_args, Mapping) else None
    configuration = plan.get("configurations", {}).get(configuration_name, {})
    schemes = configuration.get("normalization_schemes", [])
    return "zscore" if any("zscore" in str(item).lower() for item in schemes) else "unknown"


def _spacing_from_checkpoint(checkpoint: Mapping[str, Any]) -> list[float] | None:
    if "pretrain_spacing" in checkpoint:
        spacing = checkpoint["pretrain_spacing"]
    else:
        init_args = checkpoint.get("init_args", {})
        plan = init_args.get("plan", {}) if isinstance(init_args, Mapping) else {}
        spacing = plan.get("original_median_spacing_after_transp")
    if spacing is None:
        return None
    return [float(value) for value in spacing]


def build_compact_checkpoint(
    source_path: str | os.PathLike[str],
    *,
    target_encoder: nn.Module | None = None,
    pretrain_patch_size: tuple[int, int, int] = (160, 160, 160),
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = str(source_path)
    checkpoint = _torch_load(source_path)
    target_state = target_encoder.state_dict() if target_encoder is not None else None
    encoder_state, report = extract_encoder_state_dict(checkpoint, target_state)
    compact = {
        "format_version": FORMAT_VERSION,
        "architecture": ARCHITECTURE,
        "source_trainer": checkpoint.get("source_trainer", checkpoint.get("trainer_name", "unknown")),
        "source_epoch": checkpoint.get(
            "source_epoch", checkpoint.get("current_epoch", checkpoint.get("epoch"))
        ),
        "source_checkpoint": os.path.basename(source_path),
        "normalization": checkpoint.get("normalization", _normalization_from_checkpoint(checkpoint)),
        "pretrain_spacing": _spacing_from_checkpoint(checkpoint),
        "pretrain_patch_size": list(pretrain_patch_size),
        "encoder_state_dict": encoder_state,
    }
    return compact, report


def load_encoder_pretrained(
    encoder: nn.Module, checkpoint_path: str | os.PathLike[str]
) -> dict[str, Any]:
    checkpoint = _torch_load(checkpoint_path)
    state_dict, report = extract_encoder_state_dict(checkpoint, encoder.state_dict())
    incompatible = encoder.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected strict-load result: {incompatible}")
    report.update({
        "source_trainer": checkpoint.get("source_trainer", checkpoint.get("trainer_name", "unknown")),
        "source_epoch": checkpoint.get("source_epoch", checkpoint.get("current_epoch", checkpoint.get("epoch"))),
        "normalization": checkpoint.get("normalization", _normalization_from_checkpoint(checkpoint)),
        "pretrain_spacing": _spacing_from_checkpoint(checkpoint),
        "pretrain_patch_size": checkpoint.get("pretrain_patch_size"),
    })
    return report


def convert_checkpoint(
    source_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    pretrain_patch_size: tuple[int, int, int] = (160, 160, 160),
) -> dict[str, Any]:
    from .resencl import ResEncLEncoder3D

    compact, report = build_compact_checkpoint(
        source_path,
        target_encoder=ResEncLEncoder3D(),
        pretrain_patch_size=pretrain_patch_size,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(compact, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an nnSSL ResEncL checkpoint for IsoNet2")
    parser.add_argument("source", help="nnSSL checkpoint_latest.pth/checkpoint_best.pth")
    parser.add_argument("output", help="Output compact encoder .pt checkpoint")
    parser.add_argument("--pretrain-patch-size", type=int, nargs=3, default=(160, 160, 160))
    args = parser.parse_args()
    report = convert_checkpoint(
        args.source, args.output, pretrain_patch_size=tuple(args.pretrain_patch_size)
    )
    print(
        f"Converted {report['parameter_count']:,} parameters in {report['tensor_count']} tensors; "
        f"stem={report['stem_loaded']}, stages={report['stages_loaded']}, "
        f"discarded aliases={report['discarded_duplicate_count']}"
    )


if __name__ == "__main__":
    main()
