"""Execute the notebook's actual cache guard without Colab, weights or CUDA."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks/40_main_study.ipynb"


def sources():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {c["id"]: "".join(c["source"]) for c in notebook["cells"]}


def execute_guard(root, family, env):
    source = sources()["drive"].split("# Local model cache guard:", 1)[1]
    source = "# Local model cache guard:" + source
    source = source.split('print("Drive results/checkpoints:', 1)[0]
    exec(
        compile(source, str(NOTEBOOK), "exec"),
        {
            "ROOT": root,
            "FAMILY": family,
            "Path": Path,
            "os": SimpleNamespace(environ=env),
        },
    )


@pytest.mark.parametrize("family", ["ttm", "moirai1"])
def test_cache_local_and_existing_evidence_preserved(tmp_path, family):
    cache = tmp_path / ".cache"
    cache.mkdir()
    evidence = cache / "preserve.txt"
    evidence.write_text("unchanged")
    env = {"HF_HOME": "/content/drive/old", "HF_XET_CACHE": "/content/drive/old-xet"}
    execute_guard(tmp_path, family, env)
    execute_guard(tmp_path, family, env)
    assert env == {"HF_HOME": str(cache / "hf-home"), "HF_XET_CACHE": str(cache / "xet")}
    assert evidence.read_text() == "unchanged"


@pytest.mark.parametrize("relative", [".cache", ".cache/ttm", ".cache/hf-home", ".cache/xet"])
def test_drive_resolved_cache_is_blocked_without_mutation(tmp_path, monkeypatch, relative):
    original_resolve = Path.resolve
    drive = original_resolve(Path("/content/drive"))
    candidate = tmp_path / relative

    def resolve(path, *args, **kwargs):
        if path == candidate:
            return drive / "old-model-cache"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    env = {"HF_HOME": "previous", "HF_XET_CACHE": "previous-xet"}
    with pytest.raises(AssertionError, match="preserve it"):
        execute_guard(tmp_path, "ttm", env)
    assert env == {"HF_HOME": "previous", "HF_XET_CACHE": "previous-xet"}
    assert not (tmp_path / ".cache").exists()


def test_only_data_cache_is_linked_to_drive_and_notebook_is_clean():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    tree = ast.parse(sources()["drive"])
    data_loop = next(n for n in tree.body if isinstance(n, ast.For))
    assert ast.literal_eval(data_loop.iter) == ("data/source_cache",)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))
