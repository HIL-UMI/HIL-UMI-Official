#!/usr/bin/env python3
# Copyright 2026 HIL-UMI contributors.
# SPDX-License-Identifier: Apache-2.0

"""Export the current Git commit, including documentation media, without history."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def package(root: Path, output: Path) -> None:
    """Archive committed files with Git's standard source export."""
    output = output.resolve()
    # Do not let an output path overwrite source files or Git metadata.
    if output.is_relative_to(root):
        relative = output.relative_to(root)
        tracked = (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "ls-files",
                    "--error-unmatch",
                    "--",
                    relative.as_posix(),
                ],
                capture_output=True,
            ).returncode
            == 0
        )
        if ".git" in relative.parts or tracked:
            raise ValueError("Output must not overwrite source or Git metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "archive",
            "--format=tar.gz",
            "--prefix=HIL-UMI/",
            "--output",
            str(output),
            "HEAD",
        ],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dist/HIL-UMI-source.tar.gz")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        package(root, output)
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        print(f"Cannot export source: {error}")
        return 1
    print(f"Created {output} (committed source and media; no Git history).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
