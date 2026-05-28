from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    encoder_model_name: str = Field(default="KVAE_ENCODER", alias="KVAE_ENCODER_MODEL_NAME")
    decoder_model_name: str = Field(default="KVAE_DECODER", alias="KVAE_DECODER_MODEL_NAME")
    model_stage: str | None = Field(default=None, alias="KVAE_MODEL_STAGE")
    encoder_model_version: str | None = Field(default=None, alias="KVAE_ENCODER_MODEL_VERSION")
    decoder_model_version: str | None = Field(default=None, alias="KVAE_DECODER_MODEL_VERSION")
    max_concurrent_inference: int = Field(
        default=4, alias="KVAE_API_MAX_CONCURRENT_INFERENCE", ge=1
    )
    torch_num_threads: int = Field(default=4, alias="KVAE_API_TORCH_NUM_THREADS", ge=1)
    torch_num_interop_threads: int = Field(
        default=1,
        alias="KVAE_API_TORCH_NUM_INTEROP_THREADS",
        ge=1,
    )


@lru_cache(maxsize=1)
def get_api_settings() -> ApiSettings:
    return ApiSettings()
