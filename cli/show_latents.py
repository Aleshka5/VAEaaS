from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from safetensors.torch import load_file


def _to_2d_map(latent: torch.Tensor) -> torch.Tensor:
    """
    Приводит латент [C, H, W] к 2D-карте [H, W] для отображения.
    Используем среднее по каналам, чтобы видеть общую структуру.
    """
    if latent.ndim != 3:
        raise ValueError(f"Ожидается латент [C, H, W], получено shape={tuple(latent.shape)}")
    return latent.float().mean(dim=0)


def _select_indices(total: int, sample_size: int, seed: int | None) -> list[int]:
    if total <= 0:
        raise ValueError("Входной тензор пустой: нет сэмплов для визуализации.")
    k = min(sample_size, total)
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), k=k))


def _validate_latents(
    latents_70x30: torch.Tensor, latents_54x30: torch.Tensor, latents_16x30: torch.Tensor
) -> None:
    if latents_70x30.ndim != 4 or latents_54x30.ndim != 4 or latents_16x30.ndim != 4:
        raise ValueError("Все латенты должны иметь shape [N, C, H, W].")
    if (
        latents_70x30.shape[:3] != latents_54x30.shape[:3]
        or latents_70x30.shape[:3] != latents_16x30.shape[:3]
    ):
        raise ValueError(
            "latents_70x30, latents_54x30 и latents_16x30 должны совпадать по [N, C, H]. "
            f"Получено: {tuple(latents_70x30.shape)}, {tuple(latents_54x30.shape)}, {tuple(latents_16x30.shape)}"
        )
    if latents_70x30.shape[-1] != 70:
        raise ValueError(f"Ожидается ширина 70 у latents_70x30, получено {latents_70x30.shape[-1]}")
    if latents_54x30.shape[-1] != 54:
        raise ValueError(f"Ожидается ширина 54 у latents_54x30, получено {latents_54x30.shape[-1]}")
    if latents_16x30.shape[-1] != 16:
        raise ValueError(f"Ожидается ширина 16 у latents_16x30, получено {latents_16x30.shape[-1]}")


def _plot_samples(
    latents_70x30: torch.Tensor,
    latents_54x30: torch.Tensor,
    latents_16x30: torch.Tensor,
    decoder: torch.nn.Module | None,
    device: str,
    sample_indices: list[int],
    cmap: str,
    figsize_scale: float,
) -> plt.Figure:
    n_rows = len(sample_indices)
    row_factor = 2 if decoder is not None else 1
    # Сохраняем пропорции исходных ширин: 8, 54, 8, 70
    width_ratios = (8, 54, 8, 70)
    fig_w = max(12.0, 20.0 * figsize_scale)
    fig_h = max(2.2, (4.0 if decoder is not None else 2.2) * n_rows * figsize_scale)
    fig, axes = plt.subplots(
        n_rows * row_factor,
        4,
        figsize=(fig_w, fig_h),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
        constrained_layout=True,
    )

    for row_idx, sample_idx in enumerate(sample_indices):
        latent70_chw = latents_70x30[sample_idx]
        latent54_chw = latents_54x30[sample_idx]
        latent16_chw = latents_16x30[sample_idx]

        # 16x30 разбиваем по оси X в исходном CxHxW формате.
        left8_chw = latent16_chw[:, :, :8]
        right8_chw = latent16_chw[:, :, 8:]

        parts_chw = (left8_chw, latent54_chw, right8_chw, latent70_chw)
        parts_2d = tuple(_to_2d_map(part) for part in parts_chw)
        titles = ("left 8x30", "center 54x30", "right 8x30", "full 70x30")

        decoded_images = None
        if decoder is not None:
            decoded_images = [
                _decode_to_rgb01(decoder=decoder, latent_chw=part, device=device) for part in parts_chw
            ]

        latent_row = row_idx * row_factor
        for col_idx, (part_2d, title) in enumerate(zip(parts_2d, titles, strict=True)):
            ax = axes[latent_row, col_idx]
            ax.imshow(
                part_2d.cpu().numpy(),
                cmap=cmap,
                aspect="equal",
                interpolation="nearest",
            )
            if row_idx == 0:
                ax.set_title(title)
            ax.set_ylabel(f"sample #{sample_idx}" if col_idx == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])

        if decoded_images is not None:
            decoded_row = latent_row + 1
            for col_idx, decoded in enumerate(decoded_images):
                ax = axes[decoded_row, col_idx]
                ax.imshow(decoded, aspect="equal", interpolation="nearest")
                ax.set_ylabel("decoded" if col_idx == 0 else "")
                ax.set_xticks([])
                ax.set_yticks([])

    title = "Latent slices: left + center + right + full"
    if decoder is not None:
        title += " (+ decoded)"
    fig.suptitle(title, fontsize=12)
    return fig


def _decode_to_rgb01(
    decoder: torch.nn.Module, latent_chw: torch.Tensor, device: str
) -> torch.Tensor:
    latent = latent_chw.unsqueeze(0).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        decoded = decoder(latent)
    if decoded.ndim != 4:
        raise ValueError(f"Декодер вернул тензор некорректной размерности: {tuple(decoded.shape)}")
    decoded = decoded[0].detach().float().cpu()
    decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
    if decoded.shape[0] == 1:
        decoded = decoded.repeat(3, 1, 1)
    if decoded.shape[0] < 3:
        raise ValueError(
            f"Ожидается хотя бы 1 или 3 канала у декодированного изображения, получено {decoded.shape[0]}"
        )
    return decoded[:3].permute(1, 2, 0).contiguous()


def main() -> None:
    from src.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Показывает случайные латенты из .sft в формате: "
            "[left 8x30] [center 54x30] [right 8x30] [full 70x30]"
        )
    )
    parser.add_argument("--input-sft", required=True, help="Путь к .sft с латентами.")
    parser.add_argument("--latents-70-key", default="latents_70x30")
    parser.add_argument("--latents-54-key", default="latents_54x30")
    parser.add_argument("--latents-16-key", default="latents_16x30")
    parser.add_argument("--samples", type=int, default=8, help="Сколько случайных строк показать.")
    parser.add_argument("--seed", type=int, default=42, help="Seed для воспроизводимой выборки.")
    parser.add_argument(
        "--device", default=settings.default_device if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--decoder-model-name",
        default="KVAE_DECODER",
        help="Имя decoder модели в MLflow registry.",
    )
    parser.add_argument(
        "--model-stage", default=None, help="Стадия decoder модели, например Production."
    )
    parser.add_argument("--model-version", default=None, help="Версия decoder модели.")
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-registry-uri", default=settings.mlflow_registry_uri)
    parser.add_argument(
        "--skip-mlflow-decoder-load",
        action="store_true",
        help="Не пытаться загружать decoder из MLflow; показать только latent-карты.",
    )
    parser.add_argument(
        "--cmap", default="viridis", help="Colormap matplotlib, например: viridis/magma/gray."
    )
    parser.add_argument("--figscale", type=float, default=1.0, help="Масштаб размера фигуры.")
    parser.add_argument(
        "--output",
        default=None,
        help="Если указан путь, сохранить картинку туда. Иначе показать интерактивно.",
    )
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples должен быть > 0.")
    if not math.isfinite(args.figscale) or args.figscale <= 0:
        raise ValueError("--figscale должен быть положительным числом.")

    tensors = load_file(args.input_sft, device=args.device)
    required_keys = (args.latents_70_key, args.latents_54_key, args.latents_16_key)
    missing = [key for key in required_keys if key not in tensors]
    if missing:
        available = ", ".join(tensors.keys())
        raise KeyError(
            f"Во входном .sft отсутствуют ключи: {missing}. Доступные ключи: {available}"
        )

    latents_70x30 = tensors[args.latents_70_key]
    latents_54x30 = tensors[args.latents_54_key]
    latents_16x30 = tensors[args.latents_16_key]
    _validate_latents(latents_70x30, latents_54x30, latents_16x30)

    decoder: torch.nn.Module | None = None
    if not args.skip_mlflow_decoder_load:
        from src.repositories.mlflow import MLFlowRepository

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
        print("Decoder loaded via MLFlowRepository.load_decoder.")
    else:
        print("Skipping MLflow decoder load (--skip-mlflow-decoder-load).")

    indices = _select_indices(
        total=int(latents_70x30.shape[0]), sample_size=args.samples, seed=args.seed
    )
    fig = _plot_samples(
        latents_70x30=latents_70x30,
        latents_54x30=latents_54x30,
        latents_16x30=latents_16x30,
        decoder=decoder,
        device=args.device,
        sample_indices=indices,
        cmap=args.cmap,
        figsize_scale=args.figscale,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved visualization: {output_path}")
        plt.close(fig)
        return

    plt.show()


if __name__ == "__main__":
    main()
