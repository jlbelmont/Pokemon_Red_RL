"""
Headless video utilities for the slim agent.

Implements MP4 + GIF export plus PNG frame dumps per SLIM_AGENT_VIDEO_SPEC.md.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

import imageio
import numpy as np

# Canonical Game Boy resolution
GAME_WIDTH = 160
GAME_HEIGHT = 144


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_game_size(frame: np.ndarray, target_hw: tuple[int, int] = (144, 160)) -> np.ndarray:
    """
    Ensure frames are saved at the native Game Boy resolution (160x144) without interpolation.
    Only crop (no padding/upsampling). If larger than native, take the top-left native view.
    """
    target_h, target_w = GAME_HEIGHT, GAME_WIDTH
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return arr
    h, w = arr.shape[:2]
    # If width/height are swapped (e.g., 160x144 instead of 144x160), transpose to HxW.
    if h == target_w and w == target_h:
        arr = np.transpose(arr, (1, 0, 2)) if arr.ndim == 3 else arr.T
        h, w = arr.shape[:2]
    # If larger than target, center-crop to native size (no downsampling to preserve pixel fidelity).
    if h > target_h or w > target_w:
        start_h = max(0, (h - target_h) // 2)
        start_w = max(0, (w - target_w) // 2)
        arr = (
            arr[start_h : start_h + target_h, start_w : start_w + target_w, ...]
            if arr.ndim == 3
            else arr[start_h : start_h + target_h, start_w : start_w + target_w]
        )
        h, w = arr.shape[:2]
    # fast path when already exact
    if h == target_h and w == target_w:
        return arr
    # If smaller, leave as-is (no padding/upsampling) to avoid altering native content.
    return arr


def save_mp4(frames: Iterable[np.ndarray], path: str | Path, fps: int = 30) -> dict:
    start = time.time()
    path = Path(path)
    _ensure_dir(path)
    frames_list = [_to_game_size(f) for f in frames]
    imageio.mimsave(path, frames_list, fps=fps, codec="libx264", macro_block_size=1)
    duration = time.time() - start
    return {
        "path": str(path),
        "frames": len(frames_list),
        "fps": fps,
        "seconds": duration,
        "type": "mp4",
        "bytes": Path(path).stat().st_size,
    }


def save_gif(frames: Iterable[np.ndarray], path: str | Path, fps: int = 15) -> dict:
    start = time.time()
    path = Path(path)
    _ensure_dir(path)
    frames_list = [_to_game_size(f) for f in frames]
    imageio.mimsave(path, frames_list, fps=fps)
    duration = time.time() - start
    return {
        "path": str(path),
        "frames": len(frames_list),
        "fps": fps,
        "seconds": duration,
        "type": "gif",
        "bytes": Path(path).stat().st_size,
    }


def save_frames_as_png(frames: Iterable[np.ndarray], output_dir: str | Path) -> List[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for idx, frame in enumerate(frames):
        fname = output_dir / f"frame_{idx:05d}.png"
        imageio.imwrite(fname, frame)
        written.append(str(fname))
    return written


def maybe_save_video(
    frames: List[np.ndarray],
    video_type: str,
    output_path: str | Path,
    fps: int = 30,
) -> list[dict]:
    """
    Helper to save MP4/GIF/Both depending on CLI flag.
    """
    artifacts: list[dict] = []
    if video_type in {"mp4", "both"}:
        artifacts.append(save_mp4(frames, output_path, fps=fps))
    if video_type in {"gif", "both"}:
        gif_path = Path(output_path).with_suffix(".gif")
        artifacts.append(save_gif(frames, gif_path, fps=max(1, fps // 2)))
    return artifacts
