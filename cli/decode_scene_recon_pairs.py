from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _collect_input_paths(input_sft: list[str] | None, input_dir: str | None) -> list[Path]:
    paths: list[Path] = []
    if input_sft:
        paths.extend(Path(path) for path in input_sft)
    if input_dir:
        paths.extend(sorted(Path(input_dir).glob("*.sft")))
    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        raise ValueError("Не найдены входные .sft файлы. Укажите --input-sft или --input-dir.")
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Некоторые входные .sft не найдены: {missing}")
    return paths


def _to_uint8_nhwc(frames_nchw: torch.Tensor) -> torch.Tensor:
    x = frames_nchw.clamp(0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous()


def _decode_latents(
    decoder: torch.nn.Module,
    latents: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"Ожидается тензор латентов [N, C, H, W], получено {tuple(latents.shape)}")

    frames_batches: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in latents.split(batch_size, dim=0):
            decoded = decoder(batch)
            decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
            frames_batches.append(decoded.detach())
    return torch.cat(frames_batches, dim=0)


def _select_indices(total: int, max_samples: int, seed: int) -> list[int]:
    if total <= 0:
        return []
    k = min(max_samples, total)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(total, generator=generator)
    return order[:k].tolist()


def _save_visualization_samples(
    *,
    output_dir: Path,
    stem: str,
    sample_indices: list[int],
    predicted_sides_rgb_uint8: torch.Tensor,
    reconstructed_rgb_uint8: torch.Tensor,
) -> list[Path]:
    saved_paths: list[Path] = []
    for idx in sample_indices:
        pred = predicted_sides_rgb_uint8[idx].cpu()
        recon = reconstructed_rgb_uint8[idx].cpu()
        if pred.shape[0] != recon.shape[0]:
            raise ValueError(
                "Высота predicted_sides и reconstructed должна совпадать для визуализации, "
                f"получено {pred.shape[0]} и {recon.shape[0]}"
            )
        gap = torch.zeros((pred.shape[0], 12, 3), dtype=torch.uint8)
        canvas = torch.cat((pred, gap, recon), dim=1).numpy()
        image = Image.fromarray(canvas, mode="RGB")
        output_path = output_dir / f"{stem}_sample_{idx:06d}.png"
        image.save(output_path)
        saved_paths.append(output_path)
    return saved_paths


def _build_reconstructed_frame(
    source_rgb: torch.Tensor,
    center_rgb: torch.Tensor,
    predicted_sides_rgb: torch.Tensor,
) -> torch.Tensor:
    if source_rgb.ndim != 4 or center_rgb.ndim != 4 or predicted_sides_rgb.ndim != 4:
        raise ValueError("Ожидаются 4D RGB тензоры [N, 3, H, W] после decode.")
    if (
        source_rgb.shape[0] != center_rgb.shape[0]
        or source_rgb.shape[0] != predicted_sides_rgb.shape[0]
    ):
        raise ValueError("source/center/predicted_sides должны совпадать по размеру батча N.")
    if source_rgb.shape[1] != 3 or center_rgb.shape[1] != 3 or predicted_sides_rgb.shape[1] != 3:
        raise ValueError("После decode у всех тензоров должно быть 3 канала RGB.")
    if (
        source_rgb.shape[2] != center_rgb.shape[2]
        or source_rgb.shape[2] != predicted_sides_rgb.shape[2]
    ):
        raise ValueError("source/center/predicted_sides должны совпадать по высоте H.")

    sides_width = int(predicted_sides_rgb.shape[-1])
    if sides_width % 2 != 0:
        raise ValueError(
            "Ширина predicted_sides должна быть четной (left+right одинаковой ширины), "
            f"получено {sides_width}"
        )
    side_width = sides_width // 2
    if source_rgb.shape[-1] < side_width * 2:
        raise ValueError(
            "source_frame слишком узкий для вычитания левой/правой частей: "
            f"source_w={source_rgb.shape[-1]}, side_width={side_width}"
        )

    pred_left = predicted_sides_rgb[:, :, :, :side_width]
    pred_right = predicted_sides_rgb[:, :, :, side_width:]
    src_left = source_rgb[:, :, :, :side_width]
    src_right = source_rgb[:, :, :, -side_width:]

    left_diff = (pred_left - src_left).clamp(0.0, 1.0)
    right_diff = (pred_right - src_right).clamp(0.0, 1.0)
    return torch.cat((left_diff, center_rgb, right_diff), dim=-1)


def main() -> None:
    from src.config import get_settings
    from src.repositories.mlflow import MLFlowRepository
    from src.utils.sft_reader import SFTReader

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Декодирует source/center/predicted_sides из .sft в RGB и сохраняет: "
            "1) исходный decoded predicted_sides, "
            "2) сборку left_diff + center + right_diff, где diff = predicted_side - source_side."
        )
    )
    parser.add_argument("--input-sft", nargs="+", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help="Сколько случайных сэмплов визуализировать из каждого входного .sft.",
    )
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--source-frame-key", default="source_frame")
    parser.add_argument("--next-frame-center-key", default="next_frame_center")
    parser.add_argument("--predicted-sides-key", default="pred_diff_map_sides")
    parser.add_argument(
        "--device", default=settings.default_device if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decoder-model-name", default="KVAE_DECODER")
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-registry-uri", default=settings.mlflow_registry_uri)
    parser.add_argument("--model-stage", default=None)
    parser.add_argument("--model-version", default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size должен быть > 0.")
    if args.max_samples <= 0:
        raise ValueError("--max-samples должен быть > 0.")
    if not math.isfinite(args.seed):
        raise ValueError("--seed должен быть конечным числом.")

    input_paths = _collect_input_paths(args.input_sft, args.input_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    for dataset_idx, input_sft_path in enumerate(input_paths, start=1):
        print(f"\n[{dataset_idx}/{len(input_paths)}] Processing: {input_sft_path}")
        tensors = SFTReader(sft_path=str(input_sft_path), device=args.device).read_sft()

        required_keys = {
            args.source_frame_key,
            args.next_frame_center_key,
            args.predicted_sides_key,
        }
        missing = [key for key in required_keys if key not in tensors]
        if missing:
            available = ", ".join(tensors.keys())
            raise KeyError(
                f"Во входном .sft отсутствуют ключи: {missing}. Доступные ключи: {available}"
            )

        source_rgb = _decode_latents(
            decoder=decoder,
            latents=tensors[args.source_frame_key],
            batch_size=args.batch_size,
        )
        center_rgb = _decode_latents(
            decoder=decoder,
            latents=tensors[args.next_frame_center_key],
            batch_size=args.batch_size,
        )
        predicted_sides_rgb = _decode_latents(
            decoder=decoder,
            latents=tensors[args.predicted_sides_key],
            batch_size=args.batch_size,
        )

        reconstructed_rgb = _build_reconstructed_frame(
            source_rgb=source_rgb,
            center_rgb=center_rgb,
            predicted_sides_rgb=predicted_sides_rgb,
        )

        predicted_nhwc = _to_uint8_nhwc(predicted_sides_rgb)
        reconstructed_nhwc = _to_uint8_nhwc(reconstructed_rgb)
        indices = _select_indices(
            total=int(predicted_nhwc.shape[0]),
            max_samples=args.max_samples,
            seed=args.seed + dataset_idx,
        )
        rendered = _save_visualization_samples(
            output_dir=output_dir,
            stem=input_sft_path.stem,
            sample_indices=indices,
            predicted_sides_rgb_uint8=predicted_nhwc,
            reconstructed_rgb_uint8=reconstructed_nhwc,
        )
        saved_paths.extend(str(path) for path in rendered)
        print(f"Saved visualizations: {len(rendered)} files")

    print("\nRendered visualization files:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
