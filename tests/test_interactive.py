"""
Interactive console test for EnvServer.

Connects to a running EnvServer and lets you drive the environment step-by-step
by typing a language instruction at the prompt for each outer-loop step.

Usage:
    python tests/test_interactive.py [--host localhost] [--port 50051] [--fps 10]

Keyboard shortcuts at the instruction prompt:
    <enter> with empty input  → reuse the last instruction
    quit / exit / q           → send "close" and exit

A video of the rollout (one frame captured via render() after every step) is
saved to output/tests/<timestamp>_rollout.mp4 when the session ends.
"""

import argparse
import os
import socket
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image

# ── project root on sys.path ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from robolab.lamer.protocol import EnvRequest, EnvResponse, send_message, recv_message
from robolab.core.utils.video_utils import VideoWriter

# ── output directory ────────────────────────────────────────────────────────
_OUTPUT_DIR = _ROOT / "output" / "tests"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── protocol helpers ─────────────────────────────────────────────────────────

def _send(sock: socket.socket, method: str, *args, **kwargs) -> Any:
    req = EnvRequest(request_id=str(uuid.uuid4()), method=method, args=args, kwargs=kwargs)
    t0 = time.monotonic()
    send_message(sock, req)
    resp: EnvResponse = recv_message(sock)
    elapsed = time.monotonic() - t0
    print(f"  [timing] {method}: {elapsed:.3f}s")
    if resp.status != "ok":
        raise RuntimeError(f"[{method}] Server error:\n{resp.error_message}")
    return resp.result


# ── image helpers ────────────────────────────────────────────────────────────

def _collect_images(obj: Any) -> list[np.ndarray]:
    """Flatten nested structure into a list of HxWxC arrays."""
    images: list[np.ndarray] = []
    if isinstance(obj, np.ndarray):
        if obj.ndim == 3 and obj.shape[2] in (1, 3, 4):
            images.append(obj)
        elif obj.ndim == 4:
            for i in range(obj.shape[0]):
                images.append(obj[i])
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            if item is not None:
                images.extend(_collect_images(item))
    return images


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        lo, hi = float(img.min()), float(img.max())
        img = ((img - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _make_grid(images: list[np.ndarray], pad: int = 4) -> np.ndarray:
    """Tile images into a grid (square-ish layout)."""
    imgs = [_to_uint8_rgb(im) for im in images]
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    canvas = np.zeros((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), dtype=np.uint8)
    for idx, im in enumerate(imgs):
        r, c = divmod(idx, cols)
        y = r * (h + pad) + pad
        x = c * (w + pad) + pad
        dy = (h - im.shape[0]) // 2
        dx = (w - im.shape[1]) // 2
        canvas[y + dy : y + dy + im.shape[0], x + dx : x + dx + im.shape[1]] = im
    return canvas


def _save_obs_images(obs: Any, label: str, out_dir: Path) -> None:
    """Save all image arrays from an observation dict to out_dir."""
    if not isinstance(obs, dict):
        return
    raw = obs.get("image")
    if raw is None:
        return
    images = _collect_images(raw)
    for i, img in enumerate(images):
        img_u8 = _to_uint8_rgb(img)
        fname = out_dir / f"{label}_cam{i}.png"
        Image.fromarray(img_u8).save(str(fname))
    if images:
        print(f"  [obs] saved {len(images)} image(s) → {out_dir / (label + '_cam*.png')}")


def _render_frame(sock: socket.socket) -> Optional[np.ndarray]:
    """Fetch a rendered frame from the server; returns a grid or None."""
    try:
        result = _send(sock, "render")
    except Exception as exc:
        print(f"  [render] warning: {exc}")
        return None
    images = _collect_images(result)
    if not images:
        return None
    return _make_grid(images)


def _obs_summary(obs: Any) -> str:
    """One-line summary of an observation dict."""
    if not isinstance(obs, dict):
        return str(obs)
    parts = []
    for k, v in obs.items():
        if k == "image":
            imgs = _collect_images(v) if v is not None else []
            parts.append(f"image: {len(imgs)} frame(s)")
        else:
            parts.append(f"{k}: {v!r}")
    return "  " + " | ".join(parts)


# ── interactive loop ──────────────────────────────────────────────────────────

def run_interactive(host: str, port: int, fps: int, close: bool = False) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = str(_OUTPUT_DIR / f"{timestamp}_rollout.mp4")
    video = VideoWriter(video_path, fps=fps)
    img_dir = _OUTPUT_DIR / "interactive" / timestamp
    img_dir.mkdir(parents=True, exist_ok=True)
    print(f"Observation images will be saved to: {img_dir}")

    print(f"Connecting to env server at {host}:{port} …")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    sock.settimeout(300)
    print("Connected.\n")

    try:
        # Fetch server properties
        props = _send(sock, "get_properties")
        num_envs: int = (props or {}).get("num_processes", 1) or 1
        print(f"Server properties: {props}")
        print(f"num_envs = {num_envs}\n")

        # Reset
        print("Resetting environment …")
        reset_result = _send(sock, "reset")
        obs = reset_result[0] if isinstance(reset_result, (list, tuple)) else reset_result

        # Grab initial instruction from env as default
        default_instruction = ""
        if isinstance(obs, dict):
            texts = obs.get("text") or []
            default_instruction = texts[0] if texts else ""

        print(f"Initial observation:\n{_obs_summary(obs)}")
        _save_obs_images(obs, "reset", img_dir)
        if default_instruction:
            print(f"Default instruction from env: \"{default_instruction}\"")

        # Capture first frame
        frame = _render_frame(sock)
        if frame is not None:
            video.write(frame)

        step_idx = 0
        last_instruction = default_instruction

        print("\nEnter a language instruction for each step.")
        print("  <enter>        → reuse last instruction")
        print("  quit/exit/q    → end session\n")

        while True:
            prompt = f"Step {step_idx} instruction"
            if last_instruction:
                prompt += f" [{last_instruction!r}]"
            prompt += ": "

            try:
                user_input = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nInterrupted.")
                break

            if user_input.lower() in ("quit", "exit", "q"):
                break

            instruction = user_input if user_input else last_instruction
            if not instruction:
                print("  (no instruction provided — please type one)")
                continue

            last_instruction = instruction
            text_actions = [instruction] * num_envs

            print(f"  → stepping with: {instruction!r}")
            try:
                step_result = _send(sock, "step", text_actions, "play")
            except RuntimeError as exc:
                print(f"  [step] error: {exc}")
                break

            # Unpack: (observations, rewards, dones, infos)
            if isinstance(step_result, (list, tuple)) and len(step_result) >= 4:
                obs, rewards, dones, infos = step_result[0], step_result[1], step_result[2], step_result[3]
                print(f"  rewards={rewards}  dones={dones}")
                for i, info in enumerate(infos if isinstance(infos, list) else [infos]):
                    won = info.get("won", "?")
                    total_reward = info.get("total_reward", "?")
                    print(f"  env[{i}] won={won}  total_reward={total_reward}")
            else:
                obs = step_result
                print(f"  result: {step_result}")

            print(f"  obs: {_obs_summary(obs)}")
            _save_obs_images(obs, f"step{step_idx:04d}", img_dir)

            frame = _render_frame(sock)
            if frame is not None:
                video.write(frame)

            step_idx += 1

            # Check if all envs are done
            if isinstance(dones, np.ndarray) and dones.all():
                print("\n  All environments terminated.")
                break

        # Close connection (only if requested — close() destroys the env)
        if close:
            print("\nSending close …")
            try:
                _send(sock, "close")
            except Exception:
                pass
        else:
            print("\n(skipping close — server remains alive for subsequent runs)")

    finally:
        sock.close()
        video.release()
        print(f"\nRollout video saved to: {video_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive console driver for EnvServer.")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=50051, help="Server port (default: 50051)")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS (default: 10)")
    parser.add_argument("--close", action="store_true",
                        help="Send close at the end, shutting down the server env (default: off)")
    args = parser.parse_args()

    run_interactive(args.host, args.port, args.fps, close=args.close)
