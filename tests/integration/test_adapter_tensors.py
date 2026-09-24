"""Optional CPU tensor tests with a toy backend; no vendor packages or downloads."""

import dataclasses
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="optional tensor integration environment")

from tests.unit.test_adapters import ROOT, adapter_config  # noqa: E402

from tsfm_crossover.models.contract import parameter_hash  # noqa: E402
from tsfm_crossover.models.gpu_smoke import equal_tree, rng_state  # noqa: E402
from tsfm_crossover.models.ttm_adapter import TTMAdapter  # noqa: E402
from tsfm_crossover.models.window_batch import WindowBatch  # noqa: E402


class Toy(torch.nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.horizon = horizon
        self.observed_length = None

    def forward(self, past_values, future_values=None):
        self.observed_length = past_values.shape[1]
        point = past_values[:, -1:, :].expand(-1, self.horizon, -1) * self.weight
        point = point + torch.randn_like(point) * 0  # exercise RNG consumption in evaluation
        loss = None if future_values is None else ((point - future_values) ** 2).mean()
        return SimpleNamespace(prediction_outputs=point, loss=loss)


def toy(config=None):
    config = config or adapter_config()
    adapter = TTMAdapter(config, root=ROOT)
    adapter.model = Toy(config.horizon).eval()
    adapter.pretrained_config = {"fixture": True}
    adapter.initial_hash = parameter_hash(adapter.parameter_state())
    return adapter


def batch(adapter, split="validation"):
    return WindowBatch(
        torch.ones(2, 512, 3),
        torch.zeros(2, adapter.config.horizon, 3),
        split,
        adapter.config.channel_names,
        ("id1", "id2"),
        "a" * 64,
        "b" * 64,
    )


def test_zero_shot_rng_shape_no_update_and_training_mode():
    adapter = toy()
    before, state = parameter_hash(adapter.parameter_state()), rng_state()
    out = adapter.zero_shot_predict(batch(adapter))
    assert out.shape == (2, 96, 3) and not out.requires_grad
    assert equal_tree(state, rng_state())
    assert parameter_hash(adapter.parameter_state()) == before
    assert adapter.optimizer is None and adapter.backward_calls == 0
    adapter.configure_finetuning()
    adapter.train_step(batch(adapter, "train"))
    assert parameter_hash(adapter.parameter_state()) != before
    state = rng_state()
    adapter.predict(batch(adapter))
    assert equal_tree(state, rng_state()) and adapter.model.training
    with pytest.raises(ValueError, match="train targets"):
        adapter.train_step(batch(adapter))


def test_checkpoint_roundtrip_and_resume_cannot_warm_start(tmp_path):
    adapter = toy()
    adapter.configure_finetuning()
    adapter.train_step(batch(adapter, "train"))
    checkpoint = tmp_path / "state.pt"
    adapter.save_training_state(checkpoint)
    state, expected = rng_state(), adapter.predict(batch(adapter))
    fresh = toy()
    assert fresh.initial_hash != parameter_hash(adapter.parameter_state())
    fresh.load_training_state(checkpoint)
    assert equal_tree(state, rng_state())
    assert fresh.global_step == 1
    assert equal_tree(adapter.optimizer.state_dict(), fresh.optimizer.state_dict())
    torch.testing.assert_close(expected, fresh.predict(batch(fresh)))
    adapter.train_step(batch(adapter, "train"))
    expected_hash = parameter_hash(adapter.parameter_state())
    # Resume's restored RNG must reproduce the next update as well as prediction.
    fresh.train_step(batch(fresh, "train"))
    assert parameter_hash(fresh.parameter_state()) == expected_hash
    changed = toy(adapter_config(condition_id="different-rate"))
    with pytest.raises(ValueError, match="warm start"):
        changed.load_training_state(checkpoint)


@pytest.mark.parametrize("horizon,length", [(96, 512), (192, 512), (336, 512), (720, 1024)])
def test_target_and_caller_padding(horizon, length):
    adapter = toy(adapter_config(horizon=horizon))
    assert adapter.predict(batch(adapter)).shape == (2, horizon, 3)
    assert adapter.model.observed_length == length
    adapter.configure_finetuning()
    adapter.train_step(batch(adapter, "train"))
    bad = dataclasses.replace(batch(adapter), future=torch.zeros(2, horizon + 1, 3))
    with pytest.raises(ValueError, match="future"):
        adapter.prepare_batch(bad)


def test_channel_order_nonfinite_and_test_blocked():
    adapter = toy()
    for bad in [
        dataclasses.replace(batch(adapter), channel_names=("b", "a", "c")),
        dataclasses.replace(batch(adapter), past=torch.full((2, 512, 3), float("nan"))),
        batch(adapter, "test"),
    ]:
        with pytest.raises(ValueError):
            adapter.predict(bad)
