#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
move_oversized_csv_to_large_files.py

Run this script from the repository root, for example:

    cd /d D:\easm_project01
    python move_oversized_csv_to_large_files.py

What it does:
1. Finds CSV files >= SIZE_THRESHOLD_MB.
2. Skips .git, __pycache__, and the destination folder itself.
3. Moves those large CSV files into:
       _large_untracked_files/oversized_csv/<original_relative_path>
4. Preserves the original relative directory structure.
5. Writes a manifest:
       _large_untracked_files/oversized_csv_manifest.csv
6. Adds the destination folder to .gitignore, so these large files will not be uploaded.
7. Does not overwrite existing files; if a destination exists, it adds a suffix.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path


# =========================
# User-editable settings
# =========================

SIZE_THRESHOLD_MB = 100
DEST_DIR_NAME = "_large_untracked_files"
SUBDIR_NAME = "oversized_csv"

# If True, actually move files.
# If False, only report what would be moved.
MOVE_FILES = True

# If True, append the large-file folder to .gitignore if missing.
UPDATE_GITIGNORE = True


# =========================
# Internal helpers
# =========================

ROOT = Path(__file__).resolve().parent
DEST_ROOT = ROOT / DEST_DIR_NAME / SUBDIR_NAME
MANIFEST_PATH = ROOT / DEST_DIR_NAME / "oversized_csv_manifest.csv"
GITIGNORE_PATH = ROOT / ".gitignore"

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
    "ENV",
    DEST_DIR_NAME,
}


def is_inside_skipped_dir(path: Path) -> bool:
    parts = set(path.parts)
    return any(skip in parts for skip in SKIP_DIR_NAMES)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def make_unique_path(path: Path) -> Path:
    """Avoid overwriting existing files."""
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    i = 1
    while True:
        candidate = parent / f"{stem}__moved_duplicate_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def append_gitignore_rule() -> None:
    if not UPDATE_GITIGNORE:
        return

    rule_lines = [
        "",
        "# Large local files moved out of normal Git tracking",
        f"{DEST_DIR_NAME}/",
    ]

    if GITIGNORE_PATH.exists():
        text = GITIGNORE_PATH.read_text(encoding="utf-8", errors="replace")
    else:
        text = ""

    if f"{DEST_DIR_NAME}/" in text:
        print(f"[gitignore] Rule already exists: {DEST_DIR_NAME}/")
        return

    with GITIGNORE_PATH.open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(rule_lines))
        f.write("\n")

    print(f"[gitignore] Added rule to .gitignore: {DEST_DIR_NAME}/")


def collect_large_csv_files() -> list[Path]:
    threshold_bytes = SIZE_THRESHOLD_MB * 1024 * 1024
    files: list[Path] = []

    print(f"[scan] Repository root: {ROOT}")
    print(f"[scan] Threshold: >= {SIZE_THRESHOLD_MB} MB")
    print("[scan] Searching for oversized CSV files...")

    for path in ROOT.rglob("*.csv"):
        if not path.is_file():
            continue
        if is_inside_skipped_dir(path.relative_to(ROOT)):
            continue

        try:
            if path.stat().st_size >= threshold_bytes:
                files.append(path)
        except OSError as exc:
            print(f"[warn] Could not read file size: {path} | {exc}")

    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files


def write_manifest(rows: list[dict[str, str]]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "action",
        "size_mb",
        "original_relative_path",
        "moved_relative_path",
        "original_absolute_path",
        "moved_absolute_path",
    ]

    write_header = not MANIFEST_PATH.exists()

    with MANIFEST_PATH.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"[manifest] Written: {MANIFEST_PATH}")


def main() -> None:
    append_gitignore_rule()

    large_files = collect_large_csv_files()

    if not large_files:
        print("[done] No oversized CSV files found.")
        return

    print(f"[found] Oversized CSV files found: {len(large_files)}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: list[dict[str, str]] = []

    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for i, src in enumerate(large_files, start=1):
        rel = src.relative_to(ROOT)
        size = file_size_mb(src)

        dest = DEST_ROOT / rel
        dest = make_unique_path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(large_files)}] {size:.2f} MB | {rel}")

        if MOVE_FILES:
            shutil.move(str(src), str(dest))
            action = "moved"
            print(f"    -> moved to: {dest.relative_to(ROOT)}")
        else:
            action = "dry_run"
            print(f"    -> would move to: {dest.relative_to(ROOT)}")

        rows.append({
            "timestamp": timestamp,
            "action": action,
            "size_mb": f"{size:.2f}",
            "original_relative_path": str(rel).replace("\\", "/"),
            "moved_relative_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
            "original_absolute_path": str(src),
            "moved_absolute_path": str(dest),
        })

    write_manifest(rows)

    print("")
    print("[done] Oversized CSV handling complete.")
    print("")
    print("Next suggested Git commands:")
    print("    git status --short")
    print("    git add .gitignore")
    print(f"    git add {DEST_DIR_NAME}/oversized_csv_manifest.csv")
    print("    git add -A")
    print('    git commit -m "move oversized csv files out of tracked outputs"')
    print("    git push origin main")
    print("")
    print("Note:")
    print(f"    The moved CSV files are under {DEST_DIR_NAME}/ and should be ignored by Git.")


if __name__ == "__main__":
    main()
