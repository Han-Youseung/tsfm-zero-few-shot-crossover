import random

from tsfm_crossover.tracking.metadata import (
    collect_environment_metadata,
    collect_git_metadata,
)
from tsfm_crossover.utils.reproducibility import SeedSettings, apply_seeds


def test_git_and_environment_metadata_are_collectable():
    git = collect_git_metadata()
    environment = collect_environment_metadata(torch_module=None)
    assert git["commit_sha"]
    assert isinstance(git["dirty"], bool)
    assert environment["python_version"]
    assert environment["pytorch_version"] is None
    assert environment["cuda_available"] is False


def test_seed_interface_without_pytorch():
    settings = SeedSettings(1, 2, 3, 4, 5, 6, deterministic_requested=True)
    first = apply_seeds(settings, torch_module=None)
    value_one = random.random()
    second = apply_seeds(settings, torch_module=None)
    value_two = random.random()
    assert value_one == value_two
    assert first == second
    assert first["applied"]["torch_cpu"] is False
    assert first["deterministic_applied"] is False
    assert first["bitwise_reproducibility_guaranteed"] is False
