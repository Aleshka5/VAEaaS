from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _print_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}"
    )


def main() -> None:
    from src.config import get_settings
    from src.repositories.mlflow import MLFlowRepository

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Download encoder/decoder from MLflow Registry and run a sanity forward pass."
    )
    parser.add_argument("--encoder-model-name", required=True, help="Имя encoder модели в MLflow registry.")
    parser.add_argument("--decoder-model-name", required=True, help="Имя decoder модели в MLflow registry.")
    parser.add_argument("--encoder-stage", default=None)
    parser.add_argument("--encoder-version", default=None)
    parser.add_argument("--decoder-stage", default=None)
    parser.add_argument("--decoder-version", default=None)
    parser.add_argument("--device", default=settings.default_device if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=settings.kvae_input_h)
    parser.add_argument("--image-width", type=int, default=settings.kvae_input_w)
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-registry-uri", default=settings.mlflow_registry_uri)
    args = parser.parse_args()

    repo = MLFlowRepository(
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
    )
    encoder = repo.load_encoder(
        model_name=args.encoder_model_name,
        stage=args.encoder_stage,
        version=args.encoder_version,
        device=args.device,
    ).eval()
    decoder = repo.load_decoder(
        model_name=args.decoder_model_name,
        stage=args.decoder_stage,
        version=args.decoder_version,
        device=args.device,
    ).eval()

    x = torch.randn(
        args.batch_size,
        3,
        args.image_height,
        args.image_width,
        device=args.device,
        dtype=torch.float32,
    ).clamp(-1.0, 1.0)
    _print_tensor_stats("input", x)

    with torch.no_grad():
        z = encoder(x)
        x_hat = decoder(z)

    _print_tensor_stats("latents", z)
    _print_tensor_stats("reconstruction", x_hat)
    print("Sanity check passed: encoder and decoder loaded and executed successfully.")


if __name__ == "__main__":
    main()
