from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LatentDistribution:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self) -> torch.Tensor:
        std = torch.exp(0.5 * self.logvar)
        noise = torch.randn_like(std)
        return self.mean + noise * std

    def mode(self) -> torch.Tensor:
        return self.mean


class KVAEEncoder(torch.nn.Module):
    """
    Класс для кодирования кадров в латентное пространство.
    Отделяется от общей KVAE (AutoencoderKL) и может логироваться в MLflow как отдельная модель.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        quant_conv: Optional[torch.nn.Module],
        scaling_factor: float,
        shift_factor: Optional[float],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.quant_conv = quant_conv
        self.scaling_factor = float(scaling_factor)
        self.shift_factor = shift_factor

    @classmethod
    def from_autoencoder(cls, vae) -> "KVAEEncoder":
        return cls(
            encoder=vae.encoder,
            quant_conv=getattr(vae, "quant_conv", None),
            scaling_factor=getattr(vae.config, "scaling_factor", 1.0),
            shift_factor=getattr(vae.config, "shift_factor", None),
        )

    @staticmethod
    def _load_vae(
        pretrained_model_path: str,
        *,
        subfolder: str,
        local_files_only: bool,
        torch_dtype: Optional[torch.dtype],
    ):
        import diffusers

        candidate_classes = []
        if hasattr(diffusers, "AutoencoderKLKVAE"):
            candidate_classes.append(getattr(diffusers, "AutoencoderKLKVAE"))
        if hasattr(diffusers, "AutoencoderKL"):
            candidate_classes.append(getattr(diffusers, "AutoencoderKL"))

        last_error = None
        for vae_cls in candidate_classes:
            try:
                return vae_cls.from_pretrained(
                    pretrained_model_path,
                    subfolder=subfolder,
                    local_files_only=local_files_only,
                    torch_dtype=torch_dtype,
                )
            except Exception as error:  # noqa: BLE001
                last_error = error
        raise RuntimeError("Не удалось загрузить KVAE модель из diffusers.") from last_error

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_path: str,
        *,
        subfolder: str = "diffusers",
        local_files_only: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "KVAEEncoder":
        vae = cls._load_vae(
            pretrained_model_path=pretrained_model_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        return cls.from_autoencoder(vae)

    def _get_latent_dist(self, x: torch.Tensor) -> LatentDistribution:
        h = self.encoder(x)
        if self.quant_conv is not None:
            h = self.quant_conv(h)
        mean, logvar = torch.chunk(h, 2, dim=1)
        return LatentDistribution(mean=mean, logvar=logvar)

    def encode(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        latent_dist = self._get_latent_dist(x)
        z = latent_dist.sample() if sample else latent_dist.mode()
        if self.shift_factor is not None:
            z = z - self.shift_factor
        z = z * self.scaling_factor
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x, sample=False)
