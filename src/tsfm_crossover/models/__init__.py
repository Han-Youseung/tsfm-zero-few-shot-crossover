"""Model-independent adapter contracts and compatibility validation."""

from .contract import (
    AdapterContract,
    CapabilityReport,
    ModelMetadata,
    parameter_hash,
    validate_prediction,
)

__all__ = [
    "AdapterContract",
    "CapabilityReport",
    "ModelMetadata",
    "parameter_hash",
    "validate_prediction",
]
