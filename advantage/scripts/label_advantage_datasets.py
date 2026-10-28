#!/usr/bin/env python3
# Modified for HIL-UMI.
"""Label one or more LeRobot datasets with an advantage-estimator checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hil_umi_advantage.training.config_names import canonicalize_config_name  # noqa: E402
from scripts import label_towel_mix_acbc as engine  # noqa: E402

LOGGER = logging.getLogger("advantage_dataset_labeling")
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: Path


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    if not DATASET_NAME_PATTERN.fullmatch(name) or name in {".", ".."}:
        raise argparse.ArgumentTypeError(
            "Dataset NAME must start with an alphanumeric character and only contain alphanumerics, '.', '_', or '-'"
        )
    return name, Path(raw_path)


def resolve_dataset_source(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if (candidate / "meta/info.json").is_file():
        return candidate
    origin = candidate / "origin"
    if (origin / "meta/info.json").is_file():
        return origin.resolve()
    raise FileNotFoundError(f"Not a LeRobot dataset and no origin/ child was found: {candidate}")


def normalize_dataset_specs(values: list[tuple[str, Path]]) -> list[DatasetSpec]:
    names = [name for name, _ in values]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"Dataset output names must be unique: {duplicate_names}")
    specs = [DatasetSpec(name, resolve_dataset_source(path)) for name, path in values]
    sources = [spec.source for spec in specs]
    duplicate_sources = sorted({str(path) for path in sources if sources.count(path) > 1})
    if duplicate_sources:
        raise ValueError(f"Each input dataset may only be specified once: {duplicate_sources}")
    return specs


def normalize_named_paths(values: list[tuple[str, Path]], valid_names: set[str], option: str) -> dict[str, Path]:
    result = {}
    for name, path in values:
        if name not in valid_names:
            raise ValueError(f"{option} references unknown dataset name {name!r}")
        if name in result:
            raise ValueError(f"{option} was specified more than once for {name!r}")
        result[name] = path.expanduser().resolve()
    return result


def validate_generic_paths(specs: list[DatasetSpec], output_root: Path) -> Path:
    output_root = output_root.expanduser().resolve()
    for spec in specs:
        source = spec.source
        if source == output_root or engine._is_within(output_root, source) or engine._is_within(source, output_root):  # noqa: SLF001
            raise ValueError(f"Input and output paths must not overlap: source={source}, output={output_root}")
    return output_root


def validate_advantage_checkpoint_metadata(
    metadata: dict[str, Any],
    checkpoint_dir: Path,
    requested_config_name: str | None,
) -> dict[str, Any]:
    config = metadata.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint metadata does not contain a serialized training config")
    if not config.get("advantage_estimator"):
        raise ValueError("Checkpoint training config is not an advantage estimator")

    global_step = int(metadata.get("global_step", -1))
    if checkpoint_dir.name.isdigit() and global_step != int(checkpoint_dir.name):
        raise ValueError(
            f"Checkpoint step mismatch: directory={checkpoint_dir.name}, metadata global_step={global_step}"
        )
    original_config_name = str(config.get("name", ""))
    if not original_config_name:
        raise ValueError("Checkpoint metadata is missing its training config name")
    config_name = canonicalize_config_name(original_config_name)
    if requested_config_name is not None and canonicalize_config_name(requested_config_name) != config_name:
        raise ValueError(
            f"Checkpoint config mismatch: requested {requested_config_name}, metadata contains {config_name}"
        )

    training_groups = {
        "group_a": [str(Path(path).expanduser().resolve()) for path in config.get("advantage_dataset_group_a", ())],
        "group_b": [str(Path(path).expanduser().resolve()) for path in config.get("advantage_dataset_group_b", ())],
    }
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_step": global_step,
        "config_name": config_name,
        "original_config_name": original_config_name,
        "experiment_name": config.get("exp_name"),
        "training_groups": training_groups,
    }


def build_dataset_jobs(
    specs: list[DatasetSpec],
    output_root: Path,
    whole_dataset_names: set[str],
    annotation_paths: dict[str, Path],
) -> list[engine.DatasetJob]:
    conflicting = sorted(whole_dataset_names.intersection(annotation_paths))
    if conflicting:
        raise ValueError(f"Datasets cannot use both --whole-dataset and --subtask-annotations: {conflicting}")

    jobs = []
    for spec in specs:
        info, schema = engine.validate_dataset(spec.source)
        automatic_annotations = spec.source / "augmentation_metadata.json"
        annotation_path = annotation_paths.get(spec.name)
        if annotation_path is None and spec.name not in whole_dataset_names and automatic_annotations.is_file():
            annotation_path = automatic_annotations

        if annotation_path is not None:
            annotations, group_count = engine.load_subtask_annotations(annotation_path, spec.source, info)
            grouping = "manual_subtasks"
        else:
            annotations = None
            group_count = 1
            grouping = "whole_dataset"
        jobs.append(
            engine.DatasetJob(
                name=spec.name,
                source=spec.source,
                output=output_root / spec.name,
                grouping=grouping,
                group_count=group_count,
                schema=schema,
                info=info,
                annotations=annotations,
                annotation_path=annotation_path,
            )
        )
    return jobs


def find_parity_job(jobs: list[engine.DatasetJob], checkpoint_metadata: dict[str, Any]):
    config = checkpoint_metadata["config"]
    data = config.get("data")
    static_repo_id = data.get("repo_id") if isinstance(data, dict) else None
    if not static_repo_id:
        return None
    expected_source = Path(static_repo_id).expanduser().resolve()
    return next((job for job in jobs if job.source == expected_source), None)


def write_generic_root_summary(
    output_root: Path,
    checkpoint_context: dict[str, Any],
    manifests: dict[str, dict[str, Any]],
) -> None:
    engine.common.atomic_write_json(
        output_root / "advantage_labeling.json",
        {"checkpoint": checkpoint_context, "datasets": manifests},
    )
    lines = ["# Advantage-labeled datasets", ""]
    for name, manifest in manifests.items():
        lines.append(
            f"- `{name}/`: {manifest['total_episodes']} episodes, {manifest['total_frames']} frames, "
            f"grouping={manifest['grouping']}"
        )
    lines.append("")
    temporary_path = output_root / ".STATUS.md.tmp"
    temporary_path.write_text("\n".join(lines))
    os.replace(temporary_path, output_root / "STATUS.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bash scripts/label_advantage_datasets.sh \\
    --checkpoint /checkpoints/base_stage1_mix/10000 \\
    --output-root data/fold_towel/stage1/labeled \\
    --dataset base=data/fold_towel/base \\
    --dataset stage1=data/fold_towel/stage1 \\
    --batch-size 32

  bash scripts/label_advantage_datasets.sh \\
    --checkpoint /checkpoints/base_stage1_stage2_mix/20000 \\
    --output-root data/fold_towel/mix_v2_labeled \\
    --dataset base=data/fold_towel/base \\
    --dataset stage1=data/fold_towel/stage1 \\
    --dataset stage2=data/fold_towel/stage2
""",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=parse_named_path,
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Dataset to label; repeat for any number of datasets. A stage directory automatically resolves to origin/.",
    )
    parser.add_argument(
        "--whole-dataset",
        action="append",
        default=[],
        metavar="NAME",
        help="Force NAME to use one global percentile group, ignoring automatic augmentation_metadata.json detection.",
    )
    parser.add_argument(
        "--subtask-annotations",
        type=parse_named_path,
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Use per-episode subtask boundaries for NAME; otherwise augmentation_metadata.json is auto-detected.",
    )
    parser.add_argument(
        "--config-name", help="Optional safety assertion; defaults to the config recorded in metadata.pt"
    )
    parser.add_argument(
        "--devices", type=engine.common.parse_devices, default=engine.common.parse_devices("cuda:0,cuda:1")
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-chunk-size", type=int, default=64)
    parser.add_argument("--future-frames", type=int, default=50)
    parser.add_argument("--top-percent", type=float, default=30.0)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--verify-training-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", force=True)
    if args.batch_size <= 0 or args.decode_chunk_size <= 0 or args.future_frames <= 0:
        raise ValueError("batch_size, decode_chunk_size, and future_frames must be positive")
    if args.decode_chunk_size < args.batch_size:
        raise ValueError("decode_chunk_size must be greater than or equal to batch_size")
    if not 0.0 < args.top_percent < 100.0:
        raise ValueError("top_percent must be between 0 and 100")

    specs = normalize_dataset_specs(args.dataset)
    names = {spec.name for spec in specs}
    unknown_whole_names = sorted(set(args.whole_dataset) - names)
    if unknown_whole_names:
        raise ValueError(f"--whole-dataset references unknown dataset names: {unknown_whole_names}")
    annotation_paths = normalize_named_paths(args.subtask_annotations, names, "--subtask-annotations")
    output_root = validate_generic_paths(specs, args.output_root)

    checkpoint_dir = engine.common.resolve_latest_complete_checkpoint(args.checkpoint)
    checkpoint_metadata = engine.read_checkpoint_metadata(checkpoint_dir)
    checkpoint_context = validate_advantage_checkpoint_metadata(
        checkpoint_metadata,
        checkpoint_dir,
        args.config_name,
    )
    jobs = build_dataset_jobs(specs, output_root, set(args.whole_dataset), annotation_paths)
    LOGGER.info(
        "Inference settings: devices=%s batch_size=%d decode_chunk_size=%d future_frames=%d top_percent=%g",
        args.devices,
        args.batch_size,
        args.decode_chunk_size,
        args.future_frames,
        args.top_percent,
    )
    for job in jobs:
        engine.validate_existing_output(job, checkpoint_context)
        LOGGER.info(
            "Validated %s: schema=%s grouping=%s groups=%d episodes=%s frames=%s output=%s",
            job.name,
            job.schema.name,
            job.grouping,
            job.group_count,
            job.info["total_episodes"],
            job.info["total_frames"],
            job.output,
        )
    LOGGER.info("Validated advantage checkpoint: %s", json.dumps(checkpoint_context, sort_keys=True))
    if args.validate_only:
        LOGGER.info("Validation-only mode complete; no output was written")
        return

    engine.ensure_devices_available(args.devices)
    parity_job = find_parity_job(jobs, checkpoint_metadata) if args.verify_training_parity else None
    if args.verify_training_parity and parity_job is None:
        LOGGER.warning("No target dataset matches the config's static repo_id; skipping DataLoader parity check")

    manifests = {}
    for job in jobs:
        manifests[job.name] = engine.label_job(
            job,
            checkpoint_dir,
            checkpoint_context,
            checkpoint_context["config_name"],
            args.devices,
            args.batch_size,
            args.decode_chunk_size,
            args.future_frames,
            args.top_percent,
            verify_training_parity=job is parity_job,
        )
    write_generic_root_summary(output_root, checkpoint_context, manifests)
    LOGGER.info("Advantage labeling complete: %s", output_root)


if __name__ == "__main__":
    main()
