#!/usr/bin/env python3
"""Create a clean source archive without legacy notebooks, data, or results."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL_FILES = ("README.md", "pyproject.toml", ".gitignore")
SOURCE_DIRECTORIES = ("phase_transitions", "scripts", "tests", "notebooks")
EXCLUDED_DIRECTORY_NAMES = {"__pycache__", ".ipynb_checkpoints"}


def source_files() -> list[Path]:
    files: list[Path] = []
    for name in TOP_LEVEL_FILES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(path)

    for directory_name in SOURCE_DIRECTORIES:
        directory = ROOT / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative_parts = path.relative_to(ROOT).parts
            if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_parts):
                continue
            files.append(path)

    return sorted(set(files))


def create_archive(output: Path) -> tuple[Path, int]:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing archive: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    files = source_files()
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT))
    return output, len(files)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pack only the refactored source code, tests, documentation, "
            "and canonical notebooks."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "phase-transitions-ml-source.zip",
        help="destination archive; existing files are never overwritten",
    )
    args = parser.parse_args()
    output, count = create_archive(args.output)
    print(f"Created {output} ({count} files)")


if __name__ == "__main__":
    main()
