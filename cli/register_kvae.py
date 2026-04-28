from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from src.config import get_settings
    from src.repositories.mlflow import MLFlowRepository

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Split KVAE into encoder/decoder and register both models in MLflow."
    )
    parser.add_argument(
        "--pretrained-path",
        default=str(settings.kvae_pretrained_path),
        help="Путь к локальному snapshot модели KVAE.",
    )
    parser.add_argument("--encoder-model-name", required=True, help="Имя encoder модели в MLflow registry.")
    parser.add_argument("--decoder-model-name", required=True, help="Имя decoder модели в MLflow registry.")
    parser.add_argument(
        "--subfolder",
        default=settings.kvae_subfolder,
        help="Подпапка с diffusers-конфигом и весами.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Загружать модель только из локального кэша (без сети).",
    )
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-registry-uri", default=settings.mlflow_registry_uri)
    parser.add_argument("--run-name", default="kvae-components")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Тип весов при загрузке из pretrained.",
    )
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    repo = MLFlowRepository(
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
    )
    print(
        f"MLflow connection: tracking_uri={repo.tracking_uri}, "
        f"registry_uri={repo.registry_uri}, experiment={repo.experiment_name}"
    )
    result = repo.save_kvae_from_pretrained(
        pretrained_model_path=args.pretrained_path,
        encoder_model_name=args.encoder_model_name,
        decoder_model_name=args.decoder_model_name,
        run_name=args.run_name,
        subfolder=args.subfolder,
        local_files_only=args.local_files_only,
        torch_dtype=dtype_map[args.dtype],
    )
    print(f"Registered encoder='{args.encoder_model_name}' decoder='{args.decoder_model_name}'")
    print(f"MLflow run_id: {result['run_id']}")


if __name__ == "__main__":
    main()
