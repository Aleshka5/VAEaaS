from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
import time

import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _to_nchw_float01(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4:
        raise ValueError(f"Ожидается 4D тензор кадров, получено shape={tuple(frames.shape)}.")
    if frames.shape[1] in (1, 3):
        # Уже NCHW
        tensor = frames.float()
    elif frames.shape[-1] in (1, 3):
        # NHWC -> NCHW
        tensor = frames.permute(0, 3, 1, 2).contiguous().float()
    else:
        raise ValueError(
            "Не удалось определить формат кадров: ожидается NCHW или NHWC с 1/3 каналами."
        )
    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _normalize_to_vae_input(frames: torch.Tensor) -> torch.Tensor:
    # VAE ожидает диапазон [-1, 1]
    return frames * 2.0 - 1.0


def _split_latents(latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if latents.ndim != 4:
        raise ValueError("Латенты должны иметь shape [N, C, H, W].")
    _, _, _, width = latents.shape
    if width < 54:
        raise ValueError(f"Ширина латента должна быть >=54, получено: {width}")
    margin = (width - 54) // 2
    start = margin
    end = start + 54
    latents_54x30 = latents[:, :, :, start:end]
    left = latents[:, :, :, :start]
    right = latents[:, :, :, end:]
    latents_16x30 = torch.cat((left, right), dim=-1)
    return latents, latents_54x30, latents_16x30


def _save_latent_sft_file(
    output_sft_path: Path,
    latents_70x30: torch.Tensor,
    latents_54x30: torch.Tensor,
    latents_16x30: torch.Tensor,
) -> str:
    output_sft_path.parent.mkdir(parents=True, exist_ok=True)
    latents_70_cpu = latents_70x30.detach()
    latents_54_cpu = latents_54x30.detach()
    latents_16_cpu = latents_16x30.detach()
    if latents_70_cpu.device.type != "cpu":
        latents_70_cpu = latents_70_cpu.cpu()
    if latents_54_cpu.device.type != "cpu":
        latents_54_cpu = latents_54_cpu.cpu()
    if latents_16_cpu.device.type != "cpu":
        latents_16_cpu = latents_16_cpu.cpu()

    save_file(
        {
            "latents_70x30": latents_70_cpu,
            "latents_54x30": latents_54_cpu,
            "latents_16x30": latents_16_cpu,
        },
        str(output_sft_path),
    )
    return str(output_sft_path)


def _build_output_path(input_sft_path: str, output_dir: str, output_suffix: str) -> Path:
    input_path = Path(input_sft_path)
    out_dir = Path(output_dir)
    return out_dir / f"{input_path.stem}{output_suffix}"


def _render_progress(current: int, total: int, *, width: int = 30) -> str:
    if total <= 0:
        return "[------------------------------] 0.0% (0/0)"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}% ({current}/{total})"


def main() -> None:
    from src.config import get_settings
    from src.repositories.mlflow import MLFlowRepository
    from src.utils.sft_reader import SFTReader

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Encode image frames from .sft into KVAE latent tensors."
    )
    parser.add_argument(
        "--input-sft",
        required=True,
        nargs="+",
        help="Один или несколько путей к .sft с исходными кадрами.",
    )
    parser.add_argument(
        "--output-dir",
        default="D:/Films/converted_96/latents",
        help="Папка, куда сохранять .sft с латентами (1 файл на каждый входной датасет).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_latents.sft",
        help="Суффикс имени выходного файла, напр. _latents.sft",
    )
    parser.add_argument(
        "--encoder-model-name", default="KVAE_ENCODER", help="Имя encoder модели в MLflow registry."
    )
    parser.add_argument("--input-key", default="cuts", help="Ключ тензора кадров во входном .sft.")
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
    parser.add_argument(
        "--log-sft-artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Логировать созданные .sft файлы в новый MLflow run.",
    )
    parser.add_argument("--artifacts-run-name", default="encode-dataset-artifacts")
    parser.add_argument("--artifacts-subdir", default="encoded_sft")
    args = parser.parse_args()

    repo = MLFlowRepository(
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
    )
    encoder = repo.load_encoder(
        model_name=args.encoder_model_name,
        stage=args.model_stage,
        version=args.model_version,
        device=args.device,
    ).eval()

    saved_paths: list[str] = []
    for dataset_idx, input_sft_path in enumerate(args.input_sft, start=1):
        print(f"\n[{dataset_idx}/{len(args.input_sft)}] Processing dataset: {input_sft_path}")
        reader = SFTReader(sft_path=input_sft_path, device=args.device)
        tensors = reader.read_sft()
        if args.input_key not in tensors:
            available = ", ".join(tensors.keys())
            raise KeyError(
                f"Ключ '{args.input_key}' не найден во входном sft '{input_sft_path}'. "
                f"Доступные ключи: {available}"
            )

        frames = _normalize_to_vae_input(_to_nchw_float01(tensors[args.input_key]))
        # Больше не держим словарь всех тензоров после извлечения нужного ключа.
        del tensors
        total_frames = int(frames.shape[0])
        print(f"Frames found: {total_frames}")
        latents_70_cpu_batches: list[torch.Tensor] = []
        latents_54_cpu_batches: list[torch.Tensor] = []
        latents_16_cpu_batches: list[torch.Tensor] = []
        processed_frames = 0
        started_at = time.time()
        with torch.no_grad():
            for batch in frames.split(args.batch_size, dim=0):
                batch_latents = encoder(batch)
                batch_70, batch_54, batch_16 = _split_latents(batch_latents)
                latents_70_cpu_batches.append(batch_70.detach().cpu())
                latents_54_cpu_batches.append(batch_54.detach().cpu())
                latents_16_cpu_batches.append(batch_16.detach().cpu())
                processed_frames += int(batch.shape[0])
                progress = _render_progress(processed_frames, total_frames)
                print(f"\rEncoding frames {progress}", end="", flush=True)
                del batch_latents, batch_70, batch_54, batch_16, batch
        elapsed = time.time() - started_at
        print(f"\rEncoding frames {_render_progress(total_frames, total_frames)} | {elapsed:.1f}s")
        latents_70x30 = torch.cat(latents_70_cpu_batches, dim=0)
        latents_54x30 = torch.cat(latents_54_cpu_batches, dim=0)
        latents_16x30 = torch.cat(latents_16_cpu_batches, dim=0)

        output_sft_path = _build_output_path(
            input_sft_path=input_sft_path,
            output_dir=args.output_dir,
            output_suffix=args.output_suffix,
        )
        saved_path = _save_latent_sft_file(
            output_sft_path=output_sft_path,
            latents_70x30=latents_70x30,
            latents_54x30=latents_54x30,
            latents_16x30=latents_16x30,
        )
        saved_paths.append(saved_path)
        print(f"Saved: {saved_path}")

        if args.log_sft_artifacts:
            result = repo.log_artifacts(
                artifact_paths=[saved_path],
                run_name=args.artifacts_run_name,
                artifact_subdir=args.artifacts_subdir,
                tags={
                    "pipeline": "encode_dataset",
                    "encoder_model_name": args.encoder_model_name,
                    "input_sft_count": str(len(args.input_sft)),
                    "input_sft_path": str(input_sft_path),
                },
            )
            print(f"MLflow artifact run_id for dataset: {result['run_id']}")

        # Явно освобождаем крупные объекты после каждого датасета.
        del frames
        del latents_70_cpu_batches, latents_54_cpu_batches, latents_16_cpu_batches
        del latents_70x30, latents_54x30, latents_16x30
        gc.collect()
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    print("Latents saved:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
    """
     uv run --env-file .env -m cli.encode_dataset --input-sft "D:\Films\converted_96\sfts\GOT1_cuts.sft" "D:\Films\converted_96\sfts\GOT2_cuts.sft" "D:\Films\converted_96\sfts\GOT3_cuts.sft" "D:\Films\converted_96\sfts\GOT4_cuts.sft" "D:\Films\converted_96\sfts\GOT5_cuts.sft" "D:\Films\converted_96\sfts\GOT6_cuts.sft" "D:\Films\converted_96\sfts\GOT7_cuts.sft" "D:\Films\converted_96\sfts\GOT8_cuts.sft" "D:\Films\converted_96\sfts\GOT9_cuts.sft" "D:\Films\converted_96\sfts\GOT10_cuts.sft"
    """
