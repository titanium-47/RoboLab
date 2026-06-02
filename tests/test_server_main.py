"""
Integration test that pings every server command exposed by EnvServer.

Usage (server must already be running):
    python tests/test_server_main.py [--host localhost] [--port 50051]

For each command:
  - Text responses  → printed to stdout.
  - Image responses → saved as a grid PNG under output/tests/ and the path is printed.
"""

import argparse
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

# ── project root on sys.path so we can import robolab ──────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robolab.lamer.protocol import EnvRequest, EnvResponse, send_message, recv_message
import socket


# ── output directory ────────────────────────────────────────────────────────
_OUTPUT_DIR = _ROOT / "output" / "tests"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_request(method: str, *args, **kwargs) -> EnvRequest:
    return EnvRequest(request_id=str(uuid.uuid4()), method=method, args=args, kwargs=kwargs)


def _send(sock: socket.socket, method: str, *args, **kwargs) -> Any:
    req = _make_request(method, *args, **kwargs)
    t0 = time.monotonic()
    send_message(sock, req)
    resp: EnvResponse = recv_message(sock)
    elapsed = time.monotonic() - t0
    print(f"  [timing] {method}: {elapsed:.3f}s")
    if resp.status != "ok":
        raise RuntimeError(f"[{method}] Server error:\n{resp.error_message}")
    return resp.result


def _is_image(obj: Any) -> bool:
    """Return True if obj looks like one or more numpy image arrays."""
    if isinstance(obj, np.ndarray) and obj.ndim in (2, 3):
        return True
    if isinstance(obj, (list, tuple)):
        return any(_is_image(item) for item in obj if item is not None)
    return False


def _collect_images(obj: Any) -> list[np.ndarray]:
    """Flatten a nested structure into a list of HxWxC uint8 arrays."""
    images: list[np.ndarray] = []
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3 and obj.shape[2] in (1, 3, 4):
            images.append(obj)
        elif obj.ndim == 4:
            # batch of images
            for i in range(obj.shape[0]):
                images.append(obj[i])
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if item is not None:
                images.extend(_collect_images(item))
    return images


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Normalise an array to uint8 HxWx3."""
    if img.dtype != np.uint8:
        lo, hi = img.min(), img.max()
        if hi > lo:
            img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            img = np.zeros_like(img, dtype=np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _make_grid(images: list[np.ndarray], pad: int = 4) -> np.ndarray:
    """Tile images into a single grid (fills rows left-to-right)."""
    imgs = [_to_uint8_rgb(im) for im in images]
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    # uniform cell size = max of all images
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)

    canvas = np.zeros((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), dtype=np.uint8)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        y = r * (h + pad) + pad
        x = c * (w + pad) + pad
        # centre smaller images in their cell
        dy = (h - im.shape[0]) // 2
        dx = (w - im.shape[1]) // 2
        canvas[y + dy : y + dy + im.shape[0], x + dx : x + dx + im.shape[1]] = im
    return canvas


def _save_images(result: Any, name: str) -> str:
    """Save image(s) to output/tests/<name>.png; return the saved path."""
    from PIL import Image

    images = _collect_images(result)
    if not images:
        raise ValueError("No valid image arrays found in result.")

    grid = _make_grid(images)
    path = _OUTPUT_DIR / f"{name}.png"
    Image.fromarray(grid).save(path)
    return str(path)


def _print_result(method: str, result: Any) -> None:
    print(f"\n{'='*60}")
    print(f"  Command: {method}")
    print(f"{'='*60}")

    if result is None:
        print("  (no result / None)")
        return

    if _is_image(result):
        path = _save_images(result, method)
        print(f"  [image] saved grid → {path}")
        return

    # observations dict (text + optional image)
    if isinstance(result, dict):
        for k, v in result.items():
            if _is_image(v):
                path = _save_images(v, f"{method}_{k}")
                print(f"  [{k}] [image] saved grid → {path}")
            else:
                print(f"  [{k}] {v}")
        return

    # (observations, infos) or (observations, rewards, dones, infos)
    if isinstance(result, (list, tuple)):
        for i, item in enumerate(result):
            if isinstance(item, dict):
                for k, v in item.items():
                    if _is_image(v):
                        path = _save_images(v, f"{method}_item{i}_{k}")
                        print(f"  [item{i}][{k}] [image] saved grid → {path}")
                    else:
                        print(f"  [item{i}][{k}] {v}")
            elif _is_image(item):
                path = _save_images(item, f"{method}_item{i}")
                print(f"  [item{i}] [image] saved grid → {path}")
            else:
                print(f"  [item{i}] {item}")
        return

    print(f"  {result}")


# ── per-command helpers ──────────────────────────────────────────────────────

def _print_obs_tuple(method: str, result: tuple) -> None:
    """Pretty-print (observations_dict, ...) tuples from reset/step/restart/reflect."""
    obs = result[0] if isinstance(result, (list, tuple)) else result

    print(f"\n{'='*60}")
    print(f"  Command: {method}")
    print(f"{'='*60}")

    if isinstance(obs, dict):
        for k, v in obs.items():
            if k == "image" and _is_image(v):
                path = _save_images(v, f"{method}_image")
                print(f"  obs[image] → saved grid → {path}")
            else:
                print(f"  obs[{k}] = {v}")

    if isinstance(result, (list, tuple)) and len(result) > 1:
        for i, extra in enumerate(result[1:], 1):
            print(f"  result[{i}] = {extra}")


# ── main test sequence ───────────────────────────────────────────────────────

def run_tests(host: str, port: int, close: bool = False) -> None:
    print(f"Connecting to env server at {host}:{port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.settimeout(120)
    print("Connected.\n")

    try:
        # 1. get_properties — metadata dict
        result = _send(sock, "get_properties")
        _print_result("get_properties", result)

        # 2. reset — (observations, infos)
        result = _send(sock, "reset")
        _print_obs_tuple("reset", result)

        # Grab instruction for later use
        obs = result[0] if isinstance(result, (list, tuple)) else result
        instruction = ""
        if isinstance(obs, dict):
            texts = obs.get("text", [""])
            if texts:
                instruction = texts[0] or ""

        # 3. build_text_obs — list of strings
        result = _send(sock, "build_text_obs", "play")
        print(f"\n{'='*60}")
        print(f"  Command: build_text_obs")
        print(f"{'='*60}")
        print(f"  {result}")

        # 4. render — list of numpy image arrays
        result = _send(sock, "render")
        print(f"\n{'='*60}")
        print(f"  Command: render")
        print(f"{'='*60}")
        if result is not None and _is_image(result):
            path = _save_images(result, "render")
            print(f"  [image] saved grid → {path}")
        else:
            print(f"  {result}")

        # 5. step — (observations, rewards, dones, infos)
        num_envs = 1
        # try to get num_processes from get_properties
        props_result = _send(sock, "get_properties")
        if isinstance(props_result, dict):
            num_envs = props_result.get("num_processes", 1) or 1
        text_actions = [instruction] * num_envs
        result = _send(sock, "step", text_actions, "play")
        _print_obs_tuple("step", result)

        # 6. reflect — (observations, infos)
        result = _send(sock, "reflect")
        _print_obs_tuple("reflect", result)

        # 7. restart — (observations, infos)
        result = _send(sock, "restart")
        _print_obs_tuple("restart", result)

        # 8. close (optional) — destroys the env; server won't be reusable after this
        if close:
            print(f"\n{'='*60}")
            print(f"  Command: close")
            print(f"{'='*60}")
            result = _send(sock, "close")
            print(f"  Server acknowledged close. result={result}")
        else:
            print("\n  (skipping close — server remains alive for subsequent runs)")

    finally:
        sock.close()

    print(f"\nAll commands tested. Images saved to: {_OUTPUT_DIR}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ping all EnvServer commands.")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=50051, help="Server port (default: 50051)")
    parser.add_argument("--close", action="store_true",
                        help="Send close at the end, shutting down the server env (default: off)")
    args = parser.parse_args()

    run_tests(args.host, args.port, close=args.close)
