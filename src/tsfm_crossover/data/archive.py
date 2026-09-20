"""Read-only ZIP inspection and guarded extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from tsfm_crossover.tracking.atomic import write_json_atomic


class UnsafeArchiveError(RuntimeError):
    """Raised before extraction when an archive violates safety rules."""


@dataclass(frozen=True)
class ArchiveInspection:
    sha256: str
    archive_bytes: int
    member_count: int
    compressed_bytes: int
    uncompressed_bytes: int
    paths: tuple[str, ...]
    traversal_paths: tuple[str, ...]
    absolute_paths: tuple[str, ...]
    symlinks: tuple[str, ...]
    encrypted: tuple[str, ...]
    duplicate_paths: tuple[str, ...]
    executable_paths: tuple[str, ...]
    corrupt_member: str | None

    @property
    def safe(self) -> bool:
        return not any(
            (
                self.traversal_paths,
                self.absolute_paths,
                self.symlinks,
                self.encrypted,
                self.duplicate_paths,
                self.executable_paths,
                (self.corrupt_member,) if self.corrupt_member else (),
            )
        )

    def as_dict(self) -> dict[str, object]:
        return asdict(self) | {"safe": self.safe}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_zip(path: str | Path) -> ArchiveInspection:
    archive = Path(path)
    with zipfile.ZipFile(archive) as handle:
        members = handle.infolist()
        names = [member.filename for member in members]
        traversal = [name for name in names if ".." in PurePosixPath(name).parts]
        absolute = [
            name for name in names if name.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", name)
        ]
        symlinks = [
            member.filename
            for member in members
            if stat.S_ISLNK((member.external_attr >> 16) & 0xFFFF)
        ]
        encrypted = [member.filename for member in members if member.flag_bits & 1]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        executable_suffixes = {
            ".exe",
            ".dll",
            ".bat",
            ".cmd",
            ".ps1",
            ".sh",
            ".com",
            ".msi",
            ".scr",
            ".jar",
        }
        executables = [
            name for name in names if PurePosixPath(name).suffix.casefold() in executable_suffixes
        ]
        corrupt = handle.testzip()
    return ArchiveInspection(
        file_sha256(archive),
        archive.stat().st_size,
        len(members),
        sum(member.compress_size for member in members),
        sum(member.file_size for member in members),
        tuple(names),
        tuple(traversal),
        tuple(absolute),
        tuple(symlinks),
        tuple(encrypted),
        tuple(duplicates),
        tuple(executables),
        corrupt,
    )


def extract_zip_safely(
    path: str | Path, destination: str | Path, *, max_bytes: int = 10_000_000_000
) -> ArchiveInspection:
    inspection = inspect_zip(path)
    if not inspection.safe:
        raise UnsafeArchiveError("archive failed safety inspection")
    if inspection.uncompressed_bytes > max_bytes:
        raise UnsafeArchiveError("archive exceeds configured uncompressed size limit")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as handle:
        for member in handle.infolist():
            output = target / member.filename
            if output.exists() and not member.is_dir():
                raise FileExistsError(f"refusing to overwrite {output}")
        handle.extractall(target)
    return inspection


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a ZIP without extracting it")
    parser.add_argument("archive")
    parser.add_argument("--output")
    args = parser.parse_args()
    inspection = inspect_zip(args.archive)
    payload = {"schema_version": "1.0", **inspection.as_dict()}
    if args.output:
        write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0 if inspection.safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
