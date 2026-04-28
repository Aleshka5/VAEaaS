from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    movies_bucket_name: str = Field(default="movies", alias="MOVIES_BUCKET_NAME")
    mlflow_bucket_name: str = Field(default="mlflow", alias="MLFLOW_BUCKET_NAME")

    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_default_region: str = Field(default="ru-1", alias="AWS_DEFAULT_REGION")
    s3_endpoint_url: str | None = Field(default=None, alias="S3_ENDPOINT_URL")
    mlflow_s3_endpoint_url: str | None = Field(default=None, alias="MLFLOW_S3_ENDPOINT_URL")

    gpu_support: bool = Field(default=True, alias="GPU_SUPPORT")
    datasets_folder: Path | None = Field(default=None, alias="DATASETS_FOLDER")

    default_h: int = Field(default=420, alias="DEFAULT_H")
    default_w: int = Field(default=720, alias="DEFAULT_W")
    extended_h: int = Field(default=420, alias="EXTENDED_H")
    extended_w: int = Field(default=1120, alias="EXTENDED_W")
    inference_pair_batch_size: int = Field(default=512, alias="INFERENCE_PAIR_BATCH_SIZE")

    mlflow_tracking_uri: str | None = Field(default=None, alias="MLFLOW_TRACKING_URI")
    mlflow_registry_uri: str | None = Field(default=None, alias="MLFLOW_REGISTRY_URI")
    mlflow_experiment_name: str = Field(default="VAE", alias="MLFLOW_EXPERIMENT_NAME")

    kvae_pretrained_path: Path = Field(
        default=None,
        alias="KVAE_PRETRAINED_PATH",
    )
    kvae_subfolder: str = Field(default="diffusers", alias="KVAE_SUBFOLDER")
    kvae_batch_size: int = Field(default=32, alias="KVAE_BATCH_SIZE")
    kvae_input_h: int = Field(default=240, alias="KVAE_INPUT_H")
    kvae_input_w: int = Field(default=520, alias="KVAE_INPUT_W")
    kvae_latents_54_key: str = Field(default="latents_54x30", alias="KVAE_LATENTS_54_KEY")
    kvae_latents_16_key: str = Field(default="latents_16x30", alias="KVAE_LATENTS_16_KEY")
    kvae_latents_edge_width: int = Field(default=8, alias="KVAE_LATENTS_EDGE_WIDTH")
    kvae_frames_key: str = Field(default="frames", alias="KVAE_FRAMES_KEY")
    kvae_decode_output_suffix: str = Field(default="_decoded.sft", alias="KVAE_DECODE_OUTPUT_SUFFIX")

    @property
    def default_device(self) -> str:
        return "cuda" if self.gpu_support else "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
