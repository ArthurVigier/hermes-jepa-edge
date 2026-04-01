"""Visual embedding sources used by the training and export scripts."""

from src.visual_encoders.factory import (
    DirectVJEPA2Source,
    LeWMProjectionSource,
    build_visual_source,
)

__all__ = [
    "DirectVJEPA2Source",
    "LeWMProjectionSource",
    "build_visual_source",
]
