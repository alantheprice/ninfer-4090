"""Pinned official Qwen3.5-9B frontend resources used by artifact conversion.

Same contract as the qwen3_6 family module, with the 9B checkpoint's own
resource hashes (tokenizer/template differ from the 27B family pins).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

from tools.convert.qwen3_6.common.conversion import ResourcePayload, load_resources
from tools.convert.qwen3_6.common.inventory import ResourceSpec


OFFICIAL_RESOURCE_SHA256 = {
    "frontend/tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "frontend/tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "frontend/chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "frontend/generation_config.json": (
        "98cc62c0ad60faa4067de78f12b909e65e4dc093d099f406ac5bb163ad95a2f4"
    ),
    "frontend/preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "frontend/video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
}


def validate_official_resource_hashes(
    actual_hashes: Mapping[str, str],
) -> None:
    """Require the complete official six-resource profile."""

    expected_names = tuple(OFFICIAL_RESOURCE_SHA256)
    actual_names = tuple(actual_hashes)
    if actual_names != expected_names:
        raise ValueError(
            "Qwen3.5-9B frontend resource set mismatch: "
            f"expected {expected_names!r}, got {actual_names!r}"
        )
    for name, expected in OFFICIAL_RESOURCE_SHA256.items():
        actual = actual_hashes[name]
        if actual != expected:
            filename = name.removeprefix("frontend/")
            raise ValueError(
                f"official Qwen3.5-9B resource hash mismatch for {filename}: "
                f"expected {expected}, got {actual}"
            )


def load_official_resources(
    model_dir: str | Path,
    resource_specs: Sequence[ResourceSpec],
) -> tuple[ResourcePayload, ...]:
    resources = load_resources(model_dir, resource_specs)
    actual_hashes = {
        resource.name: hashlib.sha256(resource.data).hexdigest()
        for resource in resources
    }
    validate_official_resource_hashes(actual_hashes)
    return resources
