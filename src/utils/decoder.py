from __future__ import annotations

from typing import Optional

import torch


class KVAEDecoder(torch.nn.Module):
    """
    Класс для декодирования кадров из латентного пространства в изображения.
    Отделяется от общей KVAE (AutoencoderKL) и может логироваться в MLflow как отдельная модель.
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        post_quant_conv: Optional[torch.nn.Module],
        scaling_factor: float,
        shift_factor: Optional[float],
    ) -> None:
        super().__init__()
        self.decoder = decoder
        self.post_quant_conv = post_quant_conv
        self.scaling_factor = float(scaling_factor)
        self.shift_factor = shift_factor

    @classmethod
    def from_autoencoder(cls, vae) -> "KVAEDecoder":
        return cls(
            decoder=vae.decoder,
            post_quant_conv=getattr(vae, "post_quant_conv", None),
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
    ) -> "KVAEDecoder":
        vae = cls._load_vae(
            pretrained_model_path=pretrained_model_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        return cls.from_autoencoder(vae)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        z = latents / self.scaling_factor
        if self.shift_factor is not None:
            z = z + self.shift_factor
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decode(latents)
