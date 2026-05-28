from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EncodeRequest(BaseModel):
    frames: list[Any] = Field(
        description="Image tensor payload. Supports NCHW/NHWC and single image [H,W,C] or [C,H,W]."
    )


class EncodeResponse(BaseModel):
    latents: list[Any]
    shape: list[int]


class DecodeRequest(BaseModel):
    latents: list[Any] = Field(description="Latent tensor payload with shape [N, C, H, W].")


class DecodeResponse(BaseModel):
    frames: list[Any]
    shape: list[int]
    dtype: str = "uint8"
    layout: str = "NHWC"
