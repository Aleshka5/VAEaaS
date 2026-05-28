from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import save_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class MapSpec:
    key: str
    channels: int


# Все тензорные карты scene dataset, которые кодируются в KVAE (по одному encode на ключ).
MAP_SPECS: tuple[MapSpec, ...] = (
    MapSpec("source_frame", 3),
    MapSpec("opti_map_1", 2),
    MapSpec("opti_map_2", 2),
    MapSpec("depth_map", 1),
    MapSpec("diff_map_center", 3),
    MapSpec("diff_map_sides", 3),
    MapSpec("next_frame_center", 3),
)


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


def _to_nchw_float(tensor: torch.Tensor, key: str, channels: int) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"Ожидается 4D тензор для '{key}', получено shape={tuple(tensor.shape)}")
    if tensor.shape[1] == channels:
        x = tensor.float()
    elif tensor.shape[-1] == channels:
        x = tensor.permute(0, 3, 1, 2).contiguous().float()
    else:
        raise ValueError(
            f"Ключ '{key}' должен иметь {channels} каналов, но shape={tuple(tensor.shape)}."
        )
    return x


def _to_n3_float01(tensor: torch.Tensor, key: str) -> torch.Tensor:
    x = _to_nchw_float(tensor=tensor, key=key, channels=3)
    if float(x.max().item()) > 1.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def _to_n1_float01(tensor: torch.Tensor, key: str) -> torch.Tensor:
    x = _to_nchw_float(tensor=tensor, key=key, channels=1)
    if float(x.max().item()) > 1.0:
        x = x / 255.0
    return x.clamp(0.0, 1.0)


def _to_n2_flow(tensor: torch.Tensor, key: str) -> torch.Tensor:
    x = _to_nchw_float(tensor=tensor, key=key, channels=2).float()
    if not torch.isfinite(x).all():
        raise ValueError(f"Ключ '{key}' содержит NaN/Inf в optical flow.")
    max_abs = float(x.abs().max().item())
    if max_abs > 4.0:
        _, _, h, w = x.shape
        x = x.clone()
        x[:, 0] = x[:, 0] / max(float(w), 1.0)
        x[:, 1] = x[:, 1] / max(float(h), 1.0)
    return x.clamp(-1.0, 1.0)


def _expand_to_vae_3ch(native: torch.Tensor, channels: int) -> torch.Tensor:
    if channels == 3:
        x = native * 2.0 - 1.0
    elif channels == 2:
        zeros = torch.zeros(
            (native.shape[0], 1, native.shape[2], native.shape[3]),
            dtype=native.dtype,
            device=native.device,
        )
        x = torch.cat((native, zeros), dim=1)
    elif channels == 1:
        x = native.repeat(1, 3, 1, 1)
    else:
        raise ValueError(f"Неподдерживаемое число каналов: {channels}")
    return x.clamp(-1.0, 1.0)


def _prepare_vae_input(tensor: torch.Tensor, key: str, channels: int) -> torch.Tensor:
    if channels == 3:
        native = _to_n3_float01(tensor, key=key)
    elif channels == 2:
        native = _to_n2_flow(tensor, key=key)
    elif channels == 1:
        native = _to_n1_float01(tensor, key=key) * 2.0 - 1.0
    else:
        raise ValueError(f"Неподдерживаемое число каналов: {channels}")
    return _expand_to_vae_3ch(native, channels)


def _to_cpu_detached(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor


def _encode_map(
    encoder: torch.nn.Module,
    vae_input: torch.Tensor,
    batch_size: int,
    map_key: str,
) -> torch.Tensor:
    latent_batches: list[torch.Tensor] = []
    total_frames = int(vae_input.shape[0])

    with torch.no_grad():
        processed = 0
        for batch in vae_input.split(batch_size, dim=0):
            latents = encoder(batch)
            latent_batches.append(_to_cpu_detached(latents))
            processed += int(batch.shape[0])
            print(
                f"\r  [{map_key}] encode {_render_progress(processed, total_frames)}",
                end="",
                flush=True,
            )
    print(f"\r  [{map_key}] encode {_render_progress(total_frames, total_frames)}")
    return torch.cat(latent_batches, dim=0)


def _render_progress(current: int, total: int, width: int = 30) -> str:
    if total <= 0:
        return "[------------------------------] 0.0% (0/0)"
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:5.1f}% ({current}/{total})"


def _build_output_path(input_sft_path: Path, output_dir: Path, output_suffix: str) -> Path:
    return output_dir / f"{input_sft_path.stem}{output_suffix}"


def _resolve_map_specs(args: argparse.Namespace) -> tuple[MapSpec, ...]:
    return (
        MapSpec(args.source_key, 3),
        MapSpec(args.flow1_key, 2),
        MapSpec(args.flow2_key, 2),
        MapSpec(args.depth_key, 1),
        MapSpec(args.diff_center_key, 3),
        MapSpec(args.diff_sides_key, 3),
        MapSpec(args.next_frame_key, 3),
    )


def _collect_pass_through_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    encoded_input_keys: set[str],
    scene_id_key: str,
    frame_id_key: str,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for key, value in tensors.items():
        if key in encoded_input_keys:
            continue
        output[key] = _to_cpu_detached(value)
    if scene_id_key not in output:
        raise KeyError(f"В .sft отсутствует обязательный ключ '{scene_id_key}'.")
    if frame_id_key not in output:
        raise KeyError(f"В .sft отсутствует обязательный ключ '{frame_id_key}'.")
    return output


def main() -> None:
    from src.config import get_settings
    from src.repositories.mlflow import MLFlowRepository
    from src.utils.sft_reader import SFTReader

    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Кодирует scene dataset (.sft) в латенты KVAE под теми же ключами. "
            "Flow-карты дополняются нулевым 3-м каналом перед encode."
        )
    )
    parser.add_argument("--input-sft", nargs="+", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--output-suffix",
        default="_latents.sft",
        help="Суффикс выходного файла, напр. scene_dataset_part_00001_latents.sft",
    )
    parser.add_argument("--encoder-model-name", default="KVAE_ENCODER")
    parser.add_argument(
        "--device", default=settings.default_device if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=settings.kvae_batch_size)
    parser.add_argument("--scene-id-key", default="scene_logical_id")
    parser.add_argument("--frame-id-key", default="frame_id")
    parser.add_argument("--source-key", default="source_frame")
    parser.add_argument("--flow1-key", default="opti_map_1")
    parser.add_argument("--flow2-key", default="opti_map_2")
    parser.add_argument("--depth-key", default="depth_map")
    parser.add_argument("--diff-center-key", default="diff_map_center")
    parser.add_argument("--diff-sides-key", default="diff_map_sides")
    parser.add_argument("--next-frame-key", default="next_frame_center")
    parser.add_argument("--mlflow-tracking-uri", default=settings.mlflow_tracking_uri)
    parser.add_argument("--mlflow-registry-uri", default=settings.mlflow_registry_uri)
    parser.add_argument("--model-stage", default=None)
    parser.add_argument("--encoder-version", default=None)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size должен быть > 0.")

    map_specs = _resolve_map_specs(args)
    input_paths = _collect_input_paths(args.input_sft, args.input_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = MLFlowRepository(
        tracking_uri=args.mlflow_tracking_uri,
        registry_uri=args.mlflow_registry_uri,
    )
    encoder = repo.load_encoder(
        model_name=args.encoder_model_name,
        stage=args.model_stage,
        version=args.encoder_version,
        device=args.device,
    ).eval()

    saved_paths: list[str] = []
    for file_idx, sft_path in enumerate(input_paths, start=1):
        print(f"\n[{file_idx}/{len(input_paths)}] Processing: {sft_path}")
        started_at = time.time()
        tensors = SFTReader(sft_path=str(sft_path), device=args.device).read_sft()

        required_keys = {spec.key for spec in map_specs}
        missing = [key for key in required_keys if key not in tensors]
        if missing:
            raise KeyError(
                f"Во входном .sft отсутствуют ключи: {missing}. Доступные: {list(tensors.keys())}"
            )

        encoded_input_keys = {spec.key for spec in map_specs}
        output_tensors = _collect_pass_through_tensors(
            tensors,
            encoded_input_keys=encoded_input_keys,
            scene_id_key=args.scene_id_key,
            frame_id_key=args.frame_id_key,
        )

        for spec in map_specs:
            vae_input = _prepare_vae_input(
                tensors[spec.key],
                key=spec.key,
                channels=spec.channels,
            )
            encoded = _encode_map(
                encoder=encoder,
                vae_input=vae_input,
                batch_size=args.batch_size,
                map_key=spec.key,
            )
            output_tensors[spec.key] = encoded
            del vae_input, encoded
            gc.collect()
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

        output_path = _build_output_path(sft_path, output_dir, args.output_suffix)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(output_tensors, str(output_path))
        saved_paths.append(str(output_path))

        elapsed = time.time() - started_at
        print(f"Saved: {output_path} ({elapsed:.1f}s)")
        del tensors, output_tensors
        gc.collect()

    print("\nLatent datasets saved:")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()
