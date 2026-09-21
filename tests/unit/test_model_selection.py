from __future__ import annotations

import copy

import pytest

from tsfm_crossover.models.selection import (
    CANDIDATE_PRIORITY,
    CandidateDecision,
    DecisionStatus,
    ModelSelectionGate,
    MultivariateEvidence,
    RequiredCapabilities,
    ScalingEvidence,
    TrainingEvidence,
    next_candidate,
    sample_median,
    validate_probe_data,
)


def capabilities(**changes: object) -> RequiredCapabilities:
    payload = {
        "official_salesforce_weights": True,
        "official_uni2ts_load": True,
        "joint_multivariate_input": True,
        "joint_multivariate_output": True,
        "full_parameter_finetuning": True,
        "official_objective": True,
        "context_512": True,
        "direct_horizons": {96: True, 192: True, 336: True, 720: True},
        "no_recursive_forecasting": True,
        "same_pretrained_checkpoint": True,
        "original_unit_point_forecast": True,
        "colab_installable": True,
        "licenses_confirmed": True,
        "revision_pinned": True,
    }
    payload.update(changes)
    return RequiredCapabilities.model_validate(payload)


def decision(**changes: object) -> CandidateDecision:
    payload = {
        "candidate": CANDIDATE_PRIORITY[0],
        "priority": 1,
        "required_capabilities": capabilities().model_dump(),
        "capability_results": {},
        "selected": True,
        "rejection_reasons": [],
        "evidence": {},
        "revision": {
            "huggingface_repository": CANDIDATE_PRIORITY[0],
            "huggingface_commit": "0" * 40,
            "config_sha256": "1" * 64,
            "code_repository": "https://github.com/SalesforceAIResearch/uni2ts",
            "code_ref": "main",
            "release": "2.0.0",
            "release_tag_commit": "3" * 40,
            "code_commit": "2" * 40,
            "code_license": "Apache-2.0",
            "weight_license": "CC-BY-NC-4.0",
        },
        "environment": {},
        "decision_status": "pending_gpu",
        "decision_date": "2026-09-21",
        "gpu_validated": False,
    }
    payload.update(changes)
    return CandidateDecision.model_validate(payload)


def test_candidate_priority_and_stop_after_success() -> None:
    assert CANDIDATE_PRIORITY[0].endswith("1.1-R-small")
    assert next_candidate([]) == CANDIDATE_PRIORITY[0]
    assert next_candidate([decision()]) is None


def test_missing_capability_blocks_selection() -> None:
    missing = capabilities(official_objective=False).model_dump()
    with pytest.raises(ValueError, match="missing required capability"):
        decision(required_capabilities=missing)


def test_recursive_forecasting_blocks_selection() -> None:
    missing = capabilities(no_recursive_forecasting=False).model_dump()
    with pytest.raises(ValueError, match="missing required capability"):
        decision(required_capabilities=missing)


def test_four_horizon_matrix_is_exact_and_direct() -> None:
    with pytest.raises(ValueError, match="direct_horizons"):
        capabilities(direct_horizons={96: True})
    assert capabilities().all_satisfied()


def test_cpu_evidence_cannot_freeze_candidate() -> None:
    with pytest.raises(ValueError, match="CPU evidence alone"):
        decision(decision_status=DecisionStatus.frozen)


def test_multivariate_dimensions_are_not_channelwise_batches() -> None:
    evidence = MultivariateEvidence(
        semantics="joint_target_dimensions",
        batch_size=1,
        target_dim=7,
        input_shape=(1, 512, 7),
        output_shape=(1, 96, 7),
        channel_order_preserved=True,
        permutation_equivariant=True,
        channelwise_loop=False,
    )
    assert evidence.target_dim == 7
    with pytest.raises(ValueError, match="batch and channel"):
        MultivariateEvidence.model_validate({**evidence.model_dump(), "batch_size": 7})


def test_double_scaling_is_blocked() -> None:
    ScalingEvidence(model_native=True, external_scaler=False, original_unit_output=True)
    with pytest.raises(ValueError, match="cannot both"):
        ScalingEvidence(model_native=True, external_scaler=True, original_unit_output=True)


def test_official_full_finetune_contract_blocks_custom_loss_peft_and_test() -> None:
    payload = {
        "wrapper": "uni2ts.model.moirai.MoiraiFinetune",
        "objective": "uni2ts.loss.packed.PackedNLLLoss",
        "finetune_pattern": "full",
        "custom_loss": False,
        "peft": False,
        "optimizer_steps": 1,
        "finite_loss": True,
        "finite_gradients": True,
        "parameter_update": True,
        "checkpoint_round_trip": True,
        "train_split_only": True,
        "test_split_used": False,
    }
    TrainingEvidence.model_validate(payload)
    for forbidden in ("custom_loss", "peft", "test_split_used"):
        with pytest.raises(ValueError):
            TrainingEvidence.model_validate({**payload, forbidden: True})


def test_sample_median_preserves_batch_horizon_channel_shape() -> None:
    samples = [[[[1.0, 10.0], [3.0, 30.0]], [[5.0, 50.0], [7.0, 70.0]]]]
    assert sample_median(samples) == [[[3.0, 30.0], [5.0, 50.0]]]


def test_probabilistic_samples_are_seed_reproducible_by_contract() -> None:
    left = [[[[1.0]]], [[[2.0]]]]
    assert copy.deepcopy(left) == left
    assert sample_median(left) == sample_median(copy.deepcopy(left))


def test_probe_data_blocks_test_and_bundle() -> None:
    validate_probe_data("ETTh1__official_raw", "train")
    validate_probe_data("ETTh1__official_raw", "validation")
    with pytest.raises(ValueError, match="blocked bundle"):
        validate_probe_data("ETTh1__bundle_long", "train")
    with pytest.raises(ValueError, match="test split"):
        validate_probe_data("ETTh1__official_raw", "test")


def test_gate_preserves_moirai2_protocol_status() -> None:
    gate = ModelSelectionGate(
        schema_version=1,
        candidates=[decision()],
        selected_candidate=CANDIDATE_PRIORITY[0],
        moirai2_status="incompatible_with_required_multivariate_protocol",
        production_adapter_allowed=False,
    )
    assert gate.moirai2_status == "incompatible_with_required_multivariate_protocol"


def test_gate_rejects_selected_candidate_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        ModelSelectionGate(
            schema_version=1,
            candidates=[decision()],
            selected_candidate=None,
            moirai2_status="incompatible_with_required_multivariate_protocol",
            production_adapter_allowed=False,
        )


def test_manifest_revision_is_required() -> None:
    payload = decision().model_dump()
    del payload["revision"]["huggingface_commit"]
    with pytest.raises(ValueError):
        CandidateDecision.model_validate(payload)
