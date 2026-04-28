from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _to_uint8_nhwc(frames_nchw: torch.Tensor) -> torch.Tensor:
    # Выход декодера обычно в диапазоне [-1, 1]
    x = (frames_nchw.clamp(-1.0, 1.0) + 1.0) / 2.0
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous()


def _reconstruct_latents_70x30(
    latents_54x30: torch.Tensor,
    latents_16x30: torch.Tensor,
    edge_width: int,
) -> torch.Tensor:
    if latents_54x30.ndim != 4 or latents_16x30.ndim != 4:
        raise ValueError("Ожидаются 4D латенты [N, C, H, W].")
    if latents_54x30.shape[:3] != latents_16x30.shape[:3]:
        raise ValueError(
            "latents_54x30 и latents_16x30 должны совпадать по [N, C, H], "
            f"получено {tuple(latents_54x30.shape)} и {tuple(latents_16x30.shape)}"
        )
    if latents_54x30.shape[-1] != 54:
        raise ValueError(f"Ожидается ширина 54 у latents_54x30, получено {latents_54x30.shape[-1]}")
    if latents_16x30.shape[-1] != edge_width * 2:
        raise ValueError(
            f"Ожидается ширина {edge_width * 2} у latents_16x30, "
            f"получено {latents_16x30.shape[-1]}"
        )

    left = latents_16x30[:, :, :, :edge_width]
    right = latents_16x30[:, :, :, edge_width:]
    return torch.cat((left, latents_54x30, right), dim=-1)


def _build_output_path(input_sft_path: str, output_dir: str, output_suffix: str) -> Path:
    input_path = Path(input_sft_path)
    out_dir = Path(output_dir)
    return out_dir / f"{input_path.stem}{output_suffix}"


def main() -> None:
    from src.config import get_settings
    from src.repositories.mlflow import MLFlowRepository
    from src.utils.sft_reader import SFTReader

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Decode KVAE latents from .sft into image frames.")
    parser.add_argument(
        "--input-sft",
        required=True,
        nargs="+",
        help="Один или несколько путей к .sft с латентами 54x30 и 16x30.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Папка, куда сохранять восстановленные .sft (1 файл на входной датасет).",
    )
    parser.add_argument("--output-suffix", default=settings.kvae_decode_output_suffix)
    parser.add_argument(
        "--decoder-model-name", default="KVAE_DECODER", help="Имя decoder модели в MLflow registry."
    )
    parser.add_argument("--latents-54-key", default=settings.kvae_latents_54_key)
    parser.add_argument("--latents-16-key", default=settings.kvae_latents_16_key)
    parser.add_argument("--latents-edge-width", type=int, default=settings.kvae_latents_edge_width)
    parser.add_argument(
        "--output-key", default=settings.kvae_frames_key, help="Ключ тензора кадров в выходном .sft."
    )
    parser.add_argument(
        "--device", default=settings.default_device if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=settings.kvae_batch_size)
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-registry-uri", default=settings.mlflow_registry_uri)
    parser.add_argument(
        "--model-stage", default=None, help="Стадия модели в registry, напр. Production."
    )
    parser.add_argument("--model-version", default=None, help="Версия модели в registry.")
    args = parser.parse_args()

    reader = SFTReader(sft_path=args.input_sft, device=args.device)
    tensors = reader.read_sft()
    if args.input_key not in tensors:
        available = ", ".join(tensors.keys())
        raise KeyError(
            f"Ключ '{args.input_key}' не найден во входном sft. Доступные ключи: {available}"
        )
    latents = tensors[args.input_key]
    if latents.ndim != 4:
        raise ValueError(f"Ожидается латент с shape [N, C, H, W], получено {tuple(latents.shape)}")

    repo = MLFlowRepository(
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
    )
    decoder = repo.load_decoder(
        model_name=args.decoder_model_name,
        stage=args.model_stage,
        version=args.model_version,
        device=args.device,
    ).eval()

    saved_paths: list[str] = []
    for dataset_idx, input_sft_path in enumerate(args.input_sft, start=1):
        print(f"\n[{dataset_idx}/{len(args.input_sft)}] Processing dataset: {input_sft_path}")
        reader = SFTReader(sft_path=input_sft_path, device=args.device)
        tensors = reader.read_sft()

        if args.latents_54_key not in tensors or args.latents_16_key not in tensors:
            available = ", ".join(tensors.keys())
            raise KeyError(
                f"Требуемые ключи '{args.latents_54_key}' и '{args.latents_16_key}' "
                f"не найдены во входном sft '{input_sft_path}'. Доступные ключи: {available}"
            )

        latents_54x30 = tensors[args.latents_54_key]
        latents_16x30 = tensors[args.latents_16_key]
        latents = _reconstruct_latents_70x30(
            latents_54x30=latents_54x30,
            latents_16x30=latents_16x30,
            edge_width=args.latents_edge_width,
        )

        frames_batches = []
        with torch.no_grad():
            for batch in latents.split(args.batch_size, dim=0):
                frames_batches.append(decoder(batch))
        frames_nchw = torch.cat(frames_batches, dim=0)
        frames_nhwc = _to_uint8_nhwc(frames_nchw).cpu()

        output_path = _build_output_path(
            input_sft_path=input_sft_path,
            output_dir=args.output_dir,
            output_suffix=args.output_suffix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file({args.output_key: frames_nhwc}, str(output_path))
        saved_paths.append(str(output_path))
        print(f"Saved: {output_path}")

    print("\nDecoded files:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
