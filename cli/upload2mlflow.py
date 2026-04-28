from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_local_model(path: str, device: str):
    """
    Поддерживаем 2 формата:
    1) локальная папка MLflow-модели (с файлом MLmodel);
    2) torch-сериализованная модель (torch.save(model, ...)).
    """
    from src.repositories.mlflow import MLFlowRepository

    model_path = Path(path)
    if model_path.is_dir() and (model_path / "MLmodel").exists():
        repo = MLFlowRepository()
        model = repo.mlflow.pytorch.load_model(model_uri=str(model_path), map_location=device)
        return model.eval()

    model = torch.load(model_path, map_location=device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            f"Файл '{path}' не содержит torch.nn.Module. "
            "Ожидается torch.save(model, path) или папка MLflow-модели."
        )
    return model.eval()


def main() -> None:
    from src.config import get_settings
    from src.repositories.mlflow import MLFlowRepository

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Upload local encoder/decoder models to MLflow run and optionally register them."
    )
    parser.add_argument("--encoder-path", required=True, help="Путь к локальной encoder модели.")
    parser.add_argument("--decoder-path", required=True, help="Путь к локальной decoder модели.")
    parser.add_argument(
        "--encoder-model-name",
        required=True,
        help="Имя encoder модели в MLflow registry.",
    )
    parser.add_argument(
        "--decoder-model-name",
        required=True,
        help="Имя decoder модели в MLflow registry.",
    )
    parser.add_argument("--run-name", default="kvae-upload")
    parser.add_argument("--device", default=settings.default_device if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-registry-uri", default=settings.mlflow_registry_uri)
    args = parser.parse_args()

    encoder = _load_local_model(args.encoder_path, device=args.device)
    decoder = _load_local_model(args.decoder_path, device=args.device)

    repo = MLFlowRepository(
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
    )
    result = repo.save_encoder_decoder(
        encoder=encoder,
        decoder=decoder,
        encoder_model_name=args.encoder_model_name,
        decoder_model_name=args.decoder_model_name,
        run_name=args.run_name,
    )

    print("Encoder/decoder uploaded to MLflow.")
    print(f"run_id={result['run_id']}")
    print(f"encoder_model_name={args.encoder_model_name}")
    print(f"decoder_model_name={args.decoder_model_name}")


if __name__ == "__main__":
    main()
