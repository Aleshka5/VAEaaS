from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Response, status

from api.config import get_api_settings
from api.models import DecodeRequest, DecodeResponse, EncodeRequest, EncodeResponse
from src.config import get_settings
from src.repositories.mlflow import MLFlowRepository


def _to_nchw_float01(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got shape={tuple(frames.shape)}.")

    if frames.shape[1] in (1, 3):
        tensor = frames.float()
    elif frames.shape[-1] in (1, 3):
        tensor = frames.permute(0, 3, 1, 2).contiguous().float()
    else:
        raise ValueError("Expected NCHW or NHWC tensor with 1 or 3 channels.")

    if tensor.max() > 1.0:
        tensor = tensor / 255.0
    return tensor


def _to_uint8_nhwc(frames_nchw: torch.Tensor) -> torch.Tensor:
    x = (frames_nchw.clamp(-1.0, 1.0) + 1.0) / 2.0
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous()


@dataclass
class ModelState:
    ready: bool = False
    loading: bool = True
    error: str | None = None
    encoder: torch.nn.Module | None = None
    decoder: torch.nn.Module | None = None
    device: str = "cpu"
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


state = ModelState()
api_settings = get_api_settings()
inference_semaphore = asyncio.Semaphore(api_settings.max_concurrent_inference)


def _configure_torch_threading() -> None:
    if api_settings.torch_num_threads is not None:
        torch.set_num_threads(api_settings.torch_num_threads)
    if api_settings.torch_num_interop_threads is not None:
        torch.set_num_interop_threads(api_settings.torch_num_interop_threads)


def _load_models_sync() -> tuple[torch.nn.Module, torch.nn.Module, str]:
    settings = get_settings()
    device = settings.default_device if torch.cuda.is_available() else "cpu"
    _configure_torch_threading()
    repo = MLFlowRepository(
        tracking_uri=settings.mlflow_tracking_uri,
        registry_uri=settings.mlflow_registry_uri,
    )
    encoder = repo.load_encoder(
        model_name=api_settings.encoder_model_name,
        stage=api_settings.model_stage,
        version=api_settings.encoder_model_version,
        device=device,
    ).eval()
    decoder = repo.load_decoder(
        model_name=api_settings.decoder_model_name,
        stage=api_settings.model_stage,
        version=api_settings.decoder_model_version,
        device=device,
    ).eval()
    return encoder, decoder, device


async def _initialize_models() -> None:
    async with state.lock:
        state.loading = True
        state.ready = False
        state.error = None
    try:
        encoder, decoder, device = await asyncio.to_thread(_load_models_sync)
    except Exception as error:  # noqa: BLE001
        async with state.lock:
            state.error = str(error)
            state.loading = False
            state.ready = False
        return

    async with state.lock:
        state.encoder = encoder
        state.decoder = decoder
        state.device = device
        state.ready = True
        state.loading = False
        state.error = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_task = asyncio.create_task(_initialize_models())
    try:
        yield
    finally:
        if not init_task.done():
            init_task.cancel()
            with suppress(asyncio.CancelledError):
                await init_task


app = FastAPI(title="KVAE API", version="0.1.0", lifespan=lifespan)


async def _ensure_ready() -> tuple[torch.nn.Module, torch.nn.Module, str]:
    async with state.lock:
        if state.ready and state.encoder is not None and state.decoder is not None:
            return state.encoder, state.decoder, state.device
        if state.loading:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service is initializing models from MLflow.",
            )
        if state.error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model initialization failed: {state.error}",
            )
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Service is not ready.",
    )


@app.get("/readiness")
async def readiness(response: Response) -> dict[str, Any]:
    async with state.lock:
        if state.ready:
            response.status_code = status.HTTP_200_OK
            return {"ready": True}
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"ready": False, "loading": state.loading, "error": state.error}


@app.post("/encode", response_model=EncodeResponse)
async def encode(payload: EncodeRequest) -> EncodeResponse:
    encoder, _, device = await _ensure_ready()
    try:
        async with inference_semaphore:
            latents = await asyncio.to_thread(_run_encode_inference, payload.frames, encoder, device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid frames tensor payload: {error}",
        ) from error
    return EncodeResponse(latents=latents.tolist(), shape=list(latents.shape))


@app.post("/decode", response_model=DecodeResponse)
async def decode(payload: DecodeRequest) -> DecodeResponse:
    _, decoder, device = await _ensure_ready()
    try:
        async with inference_semaphore:
            frames = await asyncio.to_thread(_run_decode_inference, payload.latents, decoder, device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid latents tensor payload: {error}",
        ) from error
    return DecodeResponse(frames=frames.tolist(), shape=list(frames.shape))


def _run_encode_inference(
    raw_frames: list[Any], encoder: torch.nn.Module, device: str
) -> torch.Tensor:
    frames = torch.as_tensor(raw_frames, dtype=torch.float32, device=device)
    frames = _to_nchw_float01(frames)
    frames = frames * 2.0 - 1.0
    with torch.no_grad():
        return encoder(frames).detach().float().cpu()


def _run_decode_inference(
    raw_latents: list[Any], decoder: torch.nn.Module, device: str
) -> torch.Tensor:
    latents = torch.as_tensor(raw_latents, dtype=torch.float32, device=device)
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4:
        raise ValueError(f"Expected [N, C, H, W], got shape={tuple(latents.shape)}.")
    with torch.no_grad():
        frames_nchw = decoder(latents)
    return _to_uint8_nhwc(frames_nchw).cpu()
