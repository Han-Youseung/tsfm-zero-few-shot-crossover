"""Package verified writing materials; exclude raw datasets, caches and model weights."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from verify_paper_materials import check


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


VERIFIER = '''"""Run from the unpacked kit: python verify_package.py. Standard library only."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
items = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
for name, expected in items.items():
    target = (root / name).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise ValueError("Missing/unsafe member: " + name)
    with target.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise ValueError("Changed member: " + name)
print(f"Verified {len(items)} files; no model execution.")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("zip", type=Path)
    args = parser.parse_args()
    output, destination = args.output.resolve(), args.zip.resolve()
    if destination.exists():
        raise FileExistsError("Use a new delivery filename; never replace an existing ZIP")
    verified = check(output)
    save(output / "qa/independent_readback.json", verified)
    source = output / "source"
    source.mkdir(exist_ok=True)
    for name in (
        "build_paper_workbook.mjs",
        "render_paper_materials.mjs",
        "verify_paper_materials.py",
        "package_paper_materials.py",
    ):
        shutil.copy2(Path("scripts") / name, source / name)
    shutil.copy2(
        "src/tsfm_crossover/analysis/writing_materials.py", source / "writing_materials.py"
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    save(
        output / "source/source_identity.json",
        {
            "repository": "https://github.com/Han-Youseung/tsfm-zero-few-shot-crossover",
            "packaging_commit": head,
            "model_execution_commit": "3e18305032484239966026f14395b903c9abf268",
            "requires_repository_for_python_builder": True,
            "source_sha256": {
                p.name: digest(p) for p in source.iterdir() if p.suffix in {".py", ".mjs"}
            },
        },
    )
    (output / "verify_package.py").write_text(VERIFIER, encoding="utf-8")
    files = [
        p
        for p in output.iterdir()
        if p.is_file()
        and p.suffix in {".html", ".txt", ".json", ".bib", ".xlsx", ".py"}
        and p.name != "SHA256SUMS.json"
    ]
    for name in ("figures", "tables", "evidence", "qa", "source"):
        files.extend(p for p in (output / name).rglob("*") if p.is_file())
    # Only the explicit allowlisted directories above; never follow node_modules junctions.
    bad_suffixes = {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".parquet"}
    for path in files:
        if path.suffix in bad_suffixes or path.stat().st_size > 20_000_000:
            raise ValueError("Unexpected weight/large file: " + str(path.relative_to(output)))
        if path.suffix in {".json", ".txt", ".csv", ".html", ".py", ".mjs"}:
            text = path.read_text(encoding="utf-8-sig")
            if re.search(r"(?:hf_|ghp_|github_pat_)[A-Za-z0-9]{20,}|C:[/\\]+Users[/\\]+", text):
                raise ValueError("Potential secret/local path: " + str(path.relative_to(output)))
    hashes = {p.relative_to(output).as_posix(): digest(p) for p in sorted(files)}
    save(output / "SHA256SUMS.json", hashes)
    files.append(output / "SHA256SUMS.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(destination) as handle:
        assert handle.testzip() is None
        assert len(handle.namelist()) == len(files) == len(set(handle.namelist()))
        for name, expected in hashes.items():
            assert hashlib.sha256(handle.read(name)).hexdigest() == expected, name
    report = {
        "name": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": digest(destination),
        "files": len(files),
        "zip_crc_and_member_sha256": "passed",
        "readback": verified,
        "packaging_commit": head,
        "new_gpu_execution": False,
    }
    save(destination.with_suffix(".verification.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
