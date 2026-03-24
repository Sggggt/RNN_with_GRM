"""Configuration utilities for the active segment-memory architecture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import torch


_LEGACY_PRESET_KEYS = {
    "memory_key_dim": ("memory_key_dim", "key_size"),
    "memory_initialization_mode": ("memory_initialization_mode", "memory_init_mode"),
    "retrieval_query_source": ("retrieval_query_source", "query_source"),
    "segment_summary_source": ("segment_summary_source", "summary_source"),
    "segment_length": ("segment_length", "segment_size"),
    "memory_capacity_segments": ("memory_capacity_segments", "max_cached_segments"),
    "retrieval_top_k": ("retrieval_top_k", "top_k"),
    "synthetic_sequence_length": ("synthetic_sequence_length", "task_sequence_length"),
    "memory_storage_dtype": ("memory_storage_dtype", "storage_dtype"),
    "recomputation_ratio": ("recomputation_ratio", "recompute_threshold"),
    "enable_activation_checkpointing": ("enable_activation_checkpointing", "use_gradient_checkpoint"),
    "gradient_accumulation_factor": ("gradient_accumulation_factor", "gradient_accumulation_steps"),
    "segment_pooling_mode": ("segment_pooling_mode", "pool_method"),
    "retrieval_fusion_mode": ("retrieval_fusion_mode", "fusion_mode"),
    "gradient_clip_norm": ("gradient_clip_norm", "grad_clip"),
}


def _lookup_preset_value(params: Dict[str, Any], key: str, default: Any = None) -> Any:
    for candidate in _LEGACY_PRESET_KEYS.get(key, (key,)):
        if candidate in params:
            return params[candidate]
    return default


def _parse_context_length(value: Any, default: int = 2048) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return max(1, value)
    if isinstance(value, str):
        cleaned = value.strip().upper()
        multiplier = 1
        if cleaned.endswith("K"):
            cleaned = cleaned[:-1]
            multiplier = 1024
        try:
            return max(1, int(cleaned) * multiplier)
        except ValueError:
            return default
    return default


def _parse_dtype_spec(value: Any) -> torch.dtype:
    dtype_name = str(value or "float16").lower()
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


@dataclass
class MemoryArchitectureConfig:
    """Preset-backed configuration for model construction and memory scaling."""

    input_size: int = 256
    hidden_size: int = 1024
    output_size: int = 256
    num_layers: int = 1
    rnn_type: Literal["linear"] = "linear"
    memory_key_dim: Optional[int] = None
    memory_initialization_mode: Literal["checkpoint", "independent"] = "checkpoint"
    retrieval_query_source: Literal["input"] = "input"
    segment_summary_source: Literal["input_mean"] = "input_mean"
    segment_length: int = 64
    memory_capacity_segments: int = 128
    retrieval_top_k: int = 8
    batch_size: int = 16
    sequence_length: int = 2048
    synthetic_sequence_length: Optional[int] = None
    memory_storage_dtype: torch.dtype = torch.float16
    recomputation_ratio: float = 0.02
    enable_activation_checkpointing: bool = False
    gradient_accumulation_factor: int = 1
    segment_pooling_mode: Literal["mean", "max", "attention", "weighted"] = "mean"
    retrieval_fusion_mode: Literal["residual", "concat", "gate", "linear"] = "residual"
    use_layer_norm: bool = True
    dropout: float = 0.1
    learning_rate: float = 4e-4
    gradient_clip_norm: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        payload = dict(self.__dict__)
        payload["memory_storage_dtype"] = str(self.memory_storage_dtype)
        return payload

    @classmethod
    def load_preset_config(cls, preset_name: str) -> "MemoryArchitectureConfig":
        return load_preset_config(preset_name)


def load_hardware_preset_catalog() -> Dict[str, Any]:
    preset_path = Path(__file__).resolve().parents[2] / "hardware_presets.json"
    if not preset_path.exists():
        return {}
    return json.loads(preset_path.read_text(encoding="utf-8"))


def build_config_from_hardware_preset(preset_name: str) -> MemoryArchitectureConfig:
    preset_catalog = load_hardware_preset_catalog()
    resolved_name = preset_name

    if resolved_name == "auto":
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram_gb < 6:
                resolved_name = "laptop_1k"
            elif total_vram_gb < 10:
                resolved_name = "laptop_2k"
            elif total_vram_gb < 16:
                resolved_name = "desktop_4k"
            else:
                resolved_name = "server_16k"
        else:
            resolved_name = "laptop_2k"

    emergency_key = resolved_name.removeprefix("emergency_")
    emergency_presets = preset_catalog.get("emergency_presets", {})
    if emergency_key in emergency_presets:
        return _build_config_from_preset_dict(emergency_presets[emergency_key])

    for hardware_profile in preset_catalog.get("hardware_profiles", {}).values():
        presets = hardware_profile.get("presets", {})
        if resolved_name in presets:
            return _build_config_from_preset_dict(presets[resolved_name])

    raise ValueError(f"Unknown preset: {preset_name}")


def _build_config_from_preset_dict(params: Dict[str, Any]) -> MemoryArchitectureConfig:
    segment_length = int(_lookup_preset_value(params, "segment_length", 64))
    synthetic_length = _parse_context_length(
        _lookup_preset_value(params, "synthetic_sequence_length"),
        default=max(128, segment_length * 4),
    )

    return MemoryArchitectureConfig(
        hidden_size=int(params.get("hidden_size", 512)),
        num_layers=max(1, int(params.get("num_layers", 1))),
        rnn_type=str(params.get("recurrent_backend", params.get("rnn_type", "linear"))),
        memory_key_dim=_lookup_preset_value(params, "memory_key_dim"),
        memory_initialization_mode=str(
            _lookup_preset_value(params, "memory_initialization_mode", "checkpoint")
        ),
        retrieval_query_source=str(_lookup_preset_value(params, "retrieval_query_source", "input")),
        segment_summary_source=str(_lookup_preset_value(params, "segment_summary_source", "input_mean")),
        segment_length=segment_length,
        memory_capacity_segments=int(_lookup_preset_value(params, "memory_capacity_segments", 64)),
        retrieval_top_k=int(_lookup_preset_value(params, "retrieval_top_k", 8)),
        batch_size=int(params.get("batch_size", 16)),
        sequence_length=_parse_context_length(params.get("sequence_length"), default=2048),
        synthetic_sequence_length=synthetic_length,
        memory_storage_dtype=_parse_dtype_spec(_lookup_preset_value(params, "memory_storage_dtype", "float16")),
        recomputation_ratio=float(_lookup_preset_value(params, "recomputation_ratio", 0.0)),
        enable_activation_checkpointing=bool(
            _lookup_preset_value(params, "enable_activation_checkpointing", False)
        ),
        gradient_accumulation_factor=int(_lookup_preset_value(params, "gradient_accumulation_factor", 1)),
        segment_pooling_mode=str(_lookup_preset_value(params, "segment_pooling_mode", "mean")),
        retrieval_fusion_mode=str(_lookup_preset_value(params, "retrieval_fusion_mode", "residual")),
    )


_FALLBACK_PRESETS = {
    "laptop_1k": {
        "hidden_size": 192,
        "memory_key_dim": 24,
        "segment_length": 32,
        "memory_capacity_segments": 12,
        "retrieval_top_k": 2,
        "batch_size": 8,
        "sequence_length": "1K",
        "recomputation_ratio": 0.01,
        "enable_activation_checkpointing": True,
        "gradient_accumulation_factor": 4,
    },
    "laptop_2k": {
        "hidden_size": 256,
        "memory_key_dim": 32,
        "segment_length": 64,
        "memory_capacity_segments": 16,
        "retrieval_top_k": 2,
        "batch_size": 4,
        "sequence_length": "2K",
        "recomputation_ratio": 0.015,
        "enable_activation_checkpointing": True,
        "gradient_accumulation_factor": 6,
    },
    "laptop_4k": {
        "hidden_size": 256,
        "memory_key_dim": 32,
        "segment_length": 128,
        "memory_capacity_segments": 12,
        "retrieval_top_k": 2,
        "batch_size": 2,
        "sequence_length": "4K",
        "recomputation_ratio": 0.02,
        "enable_activation_checkpointing": True,
        "gradient_accumulation_factor": 4,
    },
    "desktop_2k": {
        "hidden_size": 320,
        "memory_key_dim": 48,
        "segment_length": 64,
        "memory_capacity_segments": 24,
        "retrieval_top_k": 4,
        "recomputation_ratio": 0.02,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 2,
    },
    "desktop_4k": {
        "hidden_size": 384,
        "memory_key_dim": 64,
        "segment_length": 128,
        "memory_capacity_segments": 24,
        "retrieval_top_k": 4,
        "recomputation_ratio": 0.025,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 2,
    },
    "desktop_8k": {
        "hidden_size": 384,
        "memory_key_dim": 64,
        "segment_length": 256,
        "memory_capacity_segments": 16,
        "retrieval_top_k": 4,
        "recomputation_ratio": 0.03,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 4,
    },
    "highend_4k": {
        "hidden_size": 512,
        "memory_key_dim": 64,
        "segment_length": 128,
        "memory_capacity_segments": 32,
        "retrieval_top_k": 4,
        "recomputation_ratio": 0.03,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 1,
    },
    "highend_8k": {
        "hidden_size": 640,
        "memory_key_dim": 96,
        "segment_length": 256,
        "memory_capacity_segments": 24,
        "retrieval_top_k": 4,
        "recomputation_ratio": 0.04,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 2,
    },
    "highend_16k": {
        "hidden_size": 640,
        "memory_key_dim": 96,
        "segment_length": 512,
        "memory_capacity_segments": 16,
        "retrieval_top_k": 4,
        "recomputation_ratio": 0.05,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 4,
    },
    "server_16k": {
        "hidden_size": 768,
        "memory_key_dim": 128,
        "segment_length": 256,
        "memory_capacity_segments": 32,
        "retrieval_top_k": 6,
        "recomputation_ratio": 0.05,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 1,
    },
    "server_32k": {
        "hidden_size": 768,
        "memory_key_dim": 128,
        "segment_length": 512,
        "memory_capacity_segments": 24,
        "retrieval_top_k": 6,
        "recomputation_ratio": 0.05,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 2,
    },
    "emergency_tiny": {
        "hidden_size": 128,
        "memory_key_dim": 16,
        "segment_length": 16,
        "memory_capacity_segments": 8,
        "retrieval_top_k": 2,
        "recomputation_ratio": 0.0,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 8,
    },
    "emergency_nano": {
        "hidden_size": 96,
        "memory_key_dim": 16,
        "segment_length": 16,
        "memory_capacity_segments": 4,
        "retrieval_top_k": 1,
        "recomputation_ratio": 0.0,
        "enable_activation_checkpointing": False,
        "gradient_accumulation_factor": 16,
    },
    "paper_2k": {
        "hidden_size": 256,
        "memory_key_dim": 32,
        "num_layers": 4,
        "segment_length": 64,
        "memory_capacity_segments": 32,
        "retrieval_top_k": 8,
        "batch_size": 2,
        "sequence_length": "2K",
        "synthetic_sequence_length": 256,
        "recomputation_ratio": 0.03,
        "enable_activation_checkpointing": True,
        "gradient_accumulation_factor": 8,
    },
    "paper_4k": {
        "hidden_size": 320,
        "memory_key_dim": 48,
        "num_layers": 4,
        "segment_length": 128,
        "memory_capacity_segments": 32,
        "retrieval_top_k": 8,
        "batch_size": 1,
        "sequence_length": "4K",
        "synthetic_sequence_length": 512,
        "recomputation_ratio": 0.03,
        "enable_activation_checkpointing": True,
        "gradient_accumulation_factor": 16,
    },
    "paper_8k": {
        "hidden_size": 384,
        "memory_key_dim": 64,
        "num_layers": 6,
        "segment_length": 256,
        "memory_capacity_segments": 24,
        "retrieval_top_k": 8,
        "batch_size": 1,
        "sequence_length": "8K",
        "synthetic_sequence_length": 1024,
        "recomputation_ratio": 0.05,
        "enable_activation_checkpointing": True,
        "gradient_accumulation_factor": 32,
    },
}

_PRESET_ALIASES = {
    "minimal": "emergency_tiny",
    "standard": "desktop_2k",
    "long_sequence": "server_16k",
}


def load_preset_config(preset_name: str) -> MemoryArchitectureConfig:
    resolved_name = _PRESET_ALIASES.get(preset_name, preset_name)

    try:
        return build_config_from_hardware_preset(resolved_name)
    except ValueError:
        if resolved_name in _FALLBACK_PRESETS:
            return _build_config_from_preset_dict(_FALLBACK_PRESETS[resolved_name])
        raise


def estimate_memory_footprint(config: MemoryArchitectureConfig, batch_size: int) -> Dict[str, float]:
    parameter_elements = (
        config.input_size * config.hidden_size * 3
        + config.hidden_size * config.input_size
        + config.hidden_size * config.hidden_size
    )
    parameter_memory_mb = parameter_elements * 4 / (1024**2)

    per_segment_elements = (
        batch_size
        * config.hidden_size
        * (config.memory_key_dim or max(16, config.hidden_size // 8))
    )
    dtype_size_bytes = 2 if config.memory_storage_dtype in {torch.float16, torch.bfloat16} else 4
    archive_memory_mb = (
        per_segment_elements
        * config.memory_capacity_segments
        * config.num_layers
        * dtype_size_bytes
        / (1024**2)
    )

    activation_elements = batch_size * config.sequence_length * config.hidden_size
    activation_memory_mb = activation_elements * 4 / (1024**2)
    if config.enable_activation_checkpointing:
        activation_memory_mb *= 0.6

    total_memory_mb = parameter_memory_mb + archive_memory_mb + activation_memory_mb
    return {
        "parameter_memory_mb": round(parameter_memory_mb, 2),
        "archive_memory_mb": round(archive_memory_mb, 2),
        "activation_memory_mb": round(activation_memory_mb, 2),
        "total_mb": round(total_memory_mb, 2),
        "estimated_total_mb": round(total_memory_mb, 2),
        "total_gb": round(total_memory_mb / 1024.0, 4),
        "estimated_total_gb": round(total_memory_mb / 1024.0, 4),
    }


def build_context_length_config(
    sequence_length: int,
    input_size: int,
    hidden_size: Optional[int] = None,
    **overrides: Any,
) -> MemoryArchitectureConfig:
    if sequence_length <= 1024:
        preset_name = "laptop_1k"
    elif sequence_length <= 2048:
        preset_name = "laptop_2k"
    elif sequence_length <= 4096:
        preset_name = "desktop_4k"
    elif sequence_length <= 8192:
        preset_name = "highend_8k"
    else:
        preset_name = "server_16k"

    config = load_preset_config(preset_name)
    config.input_size = input_size
    config.output_size = input_size
    config.sequence_length = sequence_length

    if hidden_size is not None:
        config.hidden_size = hidden_size
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


__all__ = [
    "MemoryArchitectureConfig",
    "build_config_from_hardware_preset",
    "build_context_length_config",
    "estimate_memory_footprint",
    "load_hardware_preset_catalog",
    "load_preset_config",
]
