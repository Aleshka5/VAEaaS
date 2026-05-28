from __future__ import annotations

import json
from typing import Any

import torch
try:
    from ts.torch_handler.base_handler import BaseHandler  # pyright: ignore[reportMissingImports]
except Exception:  # noqa: BLE001
    class BaseHandler:  # type: ignore[no-redef]
        pass


class EncoderHandler(BaseHandler):
    """
    TorchServe handler for KVAE encoder.
    Expects JSON payload with one of the keys:
    - "frames"
    - "instances"
    - "input"
    Or direct tensor-like payload.
    """

    def initialize(self, context) -> None:
        super().initialize(context)
        self.device = self.device if hasattr(self, "device") else torch.device("cpu")

    @staticmethod
    def _extract_payload(item: Any) -> Any:
        if isinstance(item, dict):
            payload = item.get("data")
            if payload is None:
                payload = item.get("body", item)
        else:
            payload = item

        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            payload = json.loads(payload)
        return payload

    @staticmethod
    def _to_nchw_float01(frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape={tuple(frames.shape)}.")

        if frames.shape[1] in (1, 3):
            x = frames.float()
        elif frames.shape[-1] in (1, 3):
            x = frames.permute(0, 3, 1, 2).contiguous().float()
        else:
            raise ValueError("Expected NCHW or NHWC tensor with 1 or 3 channels.")

        if x.max() > 1.0:
            x = x / 255.0
        return x

    def preprocess(self, data):
        tensors: list[torch.Tensor] = []
        for item in data:
            payload = self._extract_payload(item)
            if isinstance(payload, dict):
                frames = payload.get("frames")
                if frames is None:
                    frames = payload.get("instances")
                if frames is None:
                    frames = payload.get("input")
                if frames is None:
                    raise ValueError("Payload dict must contain 'frames', 'instances', or 'input'.")
            else:
                frames = payload

            tensor = torch.as_tensor(frames, dtype=torch.float32, device=self.device)
            tensor = self._to_nchw_float01(tensor)
            tensor = tensor * 2.0 - 1.0
            tensors.append(tensor)

        return torch.cat(tensors, dim=0)

    def inference(self, model_input):
        with torch.no_grad():
            return self.model(model_input)

    def postprocess(self, model_output):
        output = model_output.detach().float().cpu()
        return [{"latents": output.tolist(), "shape": list(output.shape)}]
