from __future__ import annotations

import argparse
import json
from typing import Any

import requests


def _build_demo_frame(height: int, width: int) -> list[list[list[int]]]:
    frame: list[list[list[int]]] = []
    for y in range(height):
        row: list[list[int]] = []
        for x in range(width):
            r = int((x / max(width - 1, 1)) * 255)
            g = int((y / max(height - 1, 1)) * 255)
            b = 127
            row.append([r, g, b])
        frame.append(row)
    return frame


def _fallback_latents() -> list[Any]:
    # Typical KVAE latent shape for 240x520 inputs is [1, 16, 30, 70].
    return [[[[0.0 for _ in range(70)] for _ in range(30)] for _ in range(16)]]


def _print_response(label: str, response: requests.Response) -> None:
    print(f"\n[{label}] status={response.status_code}")
    try:
        print(json.dumps(response.json(), ensure_ascii=False)[:1000])
    except Exception:  # noqa: BLE001
        print(response.text[:1000])


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple example: call /readiness, /encode, /decode.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--h", type=int, default=240, help="Input frame height for /encode demo.")
    parser.add_argument("--w", type=int, default=520, help="Input frame width for /encode demo.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    readiness_resp = requests.get(f"{base_url}/readiness", timeout=args.timeout)
    _print_response("readiness", readiness_resp)

    encode_payload = {"frames": [_build_demo_frame(args.h, args.w)]}
    encode_resp = requests.post(
        f"{base_url}/encode",
        json=encode_payload,
        timeout=args.timeout,
    )
    _print_response("encode", encode_resp)

    latents = _fallback_latents()
    if encode_resp.ok:
        body = encode_resp.json()
        latents = body.get("latents", latents)

    decode_resp = requests.post(
        f"{base_url}/decode",
        json={"latents": latents},
        timeout=args.timeout,
    )
    _print_response("decode", decode_resp)


if __name__ == "__main__":
    main()
