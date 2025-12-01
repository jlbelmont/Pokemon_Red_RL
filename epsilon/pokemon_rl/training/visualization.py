from __future__ import annotations

from typing import List, Optional

import json
import math
import os
import time
import numpy as np

try:  # Matplotlib is optional at runtime.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled gracefully at runtime
    plt = None

try:  # Optional dependency for resizing background overlays.
    import cv2
except Exception:  # pragma: no cover - gracefully degrade
    cv2 = None

try:  # Optional dependency for resizing backgrounds
    from PIL import Image
except Exception:  # pragma: no cover - optional import
    Image = None


class KantoAggregateProjector:
    'Project per-map tile coordinates onto a stitched Kanto map.'

    def __init__(
        self,
        layout_path: str,
        *,
        background_path: Optional[str] = None,
        tile_scale: int = 16,
        pad_tiles: int = 12,
        overlay_alpha: float = 0.72,
    ) -> None:
        if plt is None:
            raise RuntimeError("Matplotlib is required for aggregate map overlays.")
        self.tile_scale = max(1, int(tile_scale))
        self.pad = max(0, int(pad_tiles))
        self.regions: dict[int, dict[str, int]] = {}
        self._load_layout(layout_path)
        self.overlay_alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
        self.background: Optional[np.ndarray] = None
        if background_path:
            try:
                self.background = self._ensure_rgb(plt.imread(background_path))
            except Exception as exc:
                print(f"[visualization] Failed to load aggregate background {background_path}: {exc}")
                self.background = None
        self.counts = np.zeros((self.height_tiles, self.width_tiles), dtype=np.float32)
        self.pixel_width = self.width_tiles * self.tile_scale
        self.pixel_height = self.height_tiles * self.tile_scale
        self.extent = (0, self.pixel_width, 0, self.pixel_height)

    def _load_layout(self, layout_path: str) -> None:
        with open(layout_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        regions = data.get("regions", data)
        max_x = max_y = 0
        for entry in regions:
            try:
                map_id = int(entry["id"])
                coords = entry.get("coordinates") or [0, 0]
                tile_size = entry.get("tileSize") or [16, 16]
            except Exception:
                continue
            map_x = int(coords[0]) + self.pad
            map_y = int(coords[1]) + self.pad
            width = max(1, int(tile_size[0]))
            height = max(1, int(tile_size[1]))
            self.regions[map_id] = {"x": map_x, "y": map_y, "w": width, "h": height}
            max_x = max(max_x, map_x + width)
            max_y = max(max_y, map_y + height)
        self.width_tiles = max(max_x, 256 + self.pad)
        self.height_tiles = max(max_y, 256 + self.pad)

    def accumulate(self, map_id: int, coords: Optional[tuple[int, int]]) -> None:
        region = self.regions.get(int(map_id))
        if region is None or not coords:
            return
        gx = region["x"] + int(coords[0])
        gy = region["y"] + int(coords[1])
        if 0 <= gx < self.width_tiles and 0 <= gy < self.height_tiles:
            self.counts[gy, gx] += 1.0

    def accumulate_from_info(self, info: dict) -> None:
        map_id = info.get("map_id")
        coords = info.get("agent_coords")
        if map_id is None or coords is None:
            return
        try:
            self.accumulate(int(map_id), (int(coords[0]), int(coords[1])))
        except (TypeError, ValueError):
            return

    def render(self) -> np.ndarray:
        if self.counts.max() > 0:
            normalized = self.counts / self.counts.max()
        else:
            normalized = self.counts
        cmap = plt.get_cmap("magma")
        overlay = cmap(normalized)[:, :, :3]
        overlay = np.kron(
            overlay,
            np.ones((self.tile_scale, self.tile_scale, 1), dtype=np.float32),
        )
        return overlay

    def composite(self) -> np.ndarray:
        overlay = self.render()
        if self.background is None:
            return overlay
        base = self._resize_background(self.background, overlay.shape[1], overlay.shape[0])
        alpha = self.overlay_alpha
        composite = (1.0 - alpha) * base + alpha * overlay
        return np.clip(composite, 0.0, 1.0)

    def reset(self) -> None:
        self.counts.fill(0.0)

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        return array.astype(np.float32)

    def _resize_background(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        if image.shape[0] == height and image.shape[1] == width:
            return image
        if cv2 is not None:
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
            return np.clip(resized, 0.0, 1.0)
        if Image is not None:
            pil_img = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
            pil_resized = pil_img.resize((width, height), Image.BILINEAR)
            return np.asarray(pil_resized, dtype=np.float32) / 255.0
        padded = np.zeros((height, width, image.shape[-1]), dtype=np.float32)
        h = min(height, image.shape[0])
        w = min(width, image.shape[1])
        padded[:h, :w] = image[:h, :w]
        return padded


class RouteMapVisualizer:
    """Live heatmap of the agent's position and visitation counts."""

    def __init__(
        self,
        *,
        map_size: int = 256,
        cmap: str = "plasma",
        max_alpha: float = 1.0,
        title: str = "Route 1 Exploration",
        target_fps: float | None = 30.0,
    ) -> None:
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for map visualization. "
                "Install it with `pip install matplotlib`."
            )
        self.map_size = map_size
        self.max_alpha = max_alpha
        self._episode = 0
        self._last_coords: tuple[int, int] | None = None
        self._min_interval = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
        self._last_draw = 0.0

        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(4.5, 4.5))
        self.fig.canvas.manager.set_window_title(title)
        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        self.occupancy = np.zeros((map_size, map_size), dtype=np.float32)
        self.image = self.ax.imshow(
            self.occupancy,
            cmap=cmap,
            origin="lower",
            vmin=0.0,
            vmax=max_alpha,
            interpolation="nearest",
        )
        (self.agent_marker,) = self.ax.plot([], [], "wo", markersize=6, markeredgecolor="black")
        self.status_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="white",
            bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
        )
        self.ax.set_xlim(0, map_size - 1)
        self.ax.set_ylim(0, map_size - 1)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def new_episode(self, index: int) -> None:
        self._episode = index
        self.status_text.set_text(f"Episode {index}")
        self._draw()

    def reset(self) -> None:
        self.occupancy.fill(0.0)
        self._last_coords = None
        self.status_text.set_text(f"Episode {self._episode}")
        self.agent_marker.set_data([], [])
        self._draw()

    def update(self, info: dict, reward: float | None = None, terminal: bool = False) -> None:
        coords = info.get("agent_coords")
        if not coords:
            return
        x, y = (int(coords[0]), int(coords[1]))
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return

        self._last_coords = (x, y)
        self.occupancy[y, x] += 1.0
        vmax = max(1.0, self.occupancy.max())
        self.image.set_data(self.occupancy)
        self.image.set_clim(0.0, vmax)
        self.agent_marker.set_data([x], [y])

        map_name = info.get("map_name", "unknown")
        map_id = info.get("map_id")
        if isinstance(map_id, int):
            status = f"Episode {self._episode}\nMap: {map_name} (0x{map_id:02X})"
        else:
            status = f"Episode {self._episode}\nMap: {map_name}"
        if reward is not None:
            status += f"\nReward: {reward: .2f}"
        if terminal:
            status += "\n★ Episode End"
        self.status_text.set_text(status)
        self._draw()

    def _draw(self) -> None:
        if plt is None:
            return
        now = time.perf_counter()
        if self._min_interval > 0.0 and (now - self._last_draw) < self._min_interval:
            return
        self._last_draw = now
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        if plt is not None:
            plt.close(self.fig)

    def save(self, path: str) -> None:
        if plt is None:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


class MultiRouteMapVisualizer:
    """Visualize multiple environments and an aggregate heatmap in real time."""

    def __init__(
        self,
        num_envs: int,
        *,
        map_size: int = 256,
        cmap: str = "plasma",
        max_alpha: float = 1.0,
        title: str = "Parallel Route Exploration",
        show_env_panels: bool = True,
        target_fps: float | None = 30.0,
        aggregate_projector: Optional[KantoAggregateProjector] = None,
    ) -> None:
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for map visualization. "
                "Install it with `pip install matplotlib`."
            )
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.map_size = map_size
        self.max_alpha = max_alpha
        self.show_env_panels = show_env_panels
        self.aggregate_projector = aggregate_projector
        self._min_interval = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
        self._last_draw = 0.0

        plt.ion()
        if show_env_panels:
            total_panels = num_envs + 1
            cols = int(math.ceil(math.sqrt(total_panels)))
            rows = int(math.ceil(total_panels / cols))
            self.fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
            try:
                self.fig.canvas.manager.set_window_title(title)
            except Exception:  # pragma: no cover
                pass
            axes = np.atleast_1d(axes).reshape(-1)
            self.env_axes: List = list(axes[:num_envs])
            self.aggregate_ax = axes[num_envs]
            for ax in axes[total_panels:]:
                ax.axis("off")
        else:
            self.fig, self.aggregate_ax = plt.subplots(figsize=(5.5, 5.5))
            try:
                self.fig.canvas.manager.set_window_title(title)
            except Exception:  # pragma: no cover
                pass
            self.env_axes = []
        self.aggregate_ax.set_title("Aggregate")
        self.aggregate_ax.set_xlabel("X")
        self.aggregate_ax.set_ylabel("Y")

        self.occupancies = [
            np.zeros((map_size, map_size), dtype=np.float32) for _ in range(num_envs)
        ]
        self.env_images = []
        self.agent_markers = []
        self.status_texts = []

        if self.show_env_panels:
            for idx, ax in enumerate(self.env_axes):
                ax.set_title(f"Env {idx + 1}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                image = ax.imshow(
                    self.occupancies[idx],
                    cmap=cmap,
                    origin="lower",
                    vmin=0.0,
                    vmax=max_alpha,
                    interpolation="nearest",
                )
                self.env_images.append(image)
                (marker,) = ax.plot([], [], "wo", markersize=6, markeredgecolor="black")
                self.agent_markers.append(marker)
                status = ax.text(
                    0.02,
                    0.98,
                    "",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
                )
                self.status_texts.append(status)

        self.aggregate_background_artist = None
        if self.aggregate_projector:
            extent = self.aggregate_projector.extent
            if self.aggregate_projector.background is not None:
                bg = self.aggregate_projector._resize_background(
                    self.aggregate_projector.background,
                    self.aggregate_projector.pixel_width,
                    self.aggregate_projector.pixel_height,
                )
                self.aggregate_background_artist = self.aggregate_ax.imshow(
                    bg,
                    origin="lower",
                    extent=extent,
                    interpolation="bilinear",
                )
            self.aggregate_image = self.aggregate_ax.imshow(
                self.aggregate_projector.render(),
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
            self.aggregate_ax.set_xlim(extent[0], extent[1])
            self.aggregate_ax.set_ylim(extent[2], extent[3])
        else:
            self.aggregate_image = self.aggregate_ax.imshow(
                np.zeros((map_size, map_size), dtype=np.float32),
                cmap=cmap,
                origin="lower",
                vmin=0.0,
                vmax=max_alpha * num_envs,
                interpolation="nearest",
            )
            self.aggregate_ax.set_xlim(0, map_size - 1)
            self.aggregate_ax.set_ylim(0, map_size - 1)
        if self.show_env_panels:
            for ax in self.env_axes:
                ax.set_xlim(0, map_size - 1)
                ax.set_ylim(0, map_size - 1)
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def new_episode(self, env_index: int, episode: int) -> None:
        if self.show_env_panels and 0 <= env_index < self.num_envs:
            self.status_texts[env_index].set_text(f"Env {env_index + 1} | Episode {episode}")
            self._draw()

    def reset(self, env_index: Optional[int] = None) -> None:
        indices = range(self.num_envs) if env_index is None else [env_index]
        for idx in indices:
            self.occupancies[idx].fill(0.0)
            if self.show_env_panels:
                self.agent_markers[idx].set_data([], [])
                self.status_texts[idx].set_text(f"Env {idx + 1}")
        if self.aggregate_projector:
            self.aggregate_projector.reset()
        self._refresh_aggregate()

    def update(
        self,
        env_index: int,
        info: dict,
        *,
        reward: float | None = None,
        terminal: bool = False,
        update_aggregate: bool = True,
    ) -> None:
        if not (0 <= env_index < self.num_envs):
            return
        coords = info.get("agent_coords")
        if not coords:
            return
        x, y = (int(coords[0]), int(coords[1]))
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return

        occ = self.occupancies[env_index]
        occ[y, x] += 1.0
        vmax = max(1.0, occ.max())
        if self.show_env_panels:
            self.env_images[env_index].set_data(occ)
            self.env_images[env_index].set_clim(0.0, vmax)
            self.agent_markers[env_index].set_data([x], [y])

        map_name = info.get("map_name", "unknown")
        map_id = info.get("map_id")
        if self.show_env_panels:
            if isinstance(map_id, int):
                status = f"Env {env_index + 1} | Map: {map_name} (0x{map_id:02X})"
            else:
                status = f"Env {env_index + 1} | Map: {map_name}"
            if reward is not None:
                status += f"\nReward: {reward: .2f}"
            if terminal:
                status += "\n★ Episode End"
            self.status_texts[env_index].set_text(status)
        if update_aggregate:
            if self.aggregate_projector:
                self.aggregate_projector.accumulate_from_info(info)
            self._refresh_aggregate()

    def _refresh_aggregate(self) -> None:
        if self.aggregate_projector:
            overlay = self.aggregate_projector.composite()
            if self.aggregate_image is None:
                self.aggregate_image = self.aggregate_ax.imshow(
                    overlay,
                    origin="lower",
                    extent=self.aggregate_projector.extent,
                    interpolation="nearest",
                )
            else:
                self.aggregate_image.set_data(overlay)
            self._draw()
            return
        aggregate = np.sum(self.occupancies, axis=0)
        self.aggregate_image.set_data(aggregate)
        self.aggregate_image.set_clim(0.0, max(1.0, aggregate.max()))
        self._draw()

    def _draw(self) -> None:
        if plt is None:
            return
        now = time.perf_counter()
        if self._min_interval > 0.0 and (now - self._last_draw) < self._min_interval:
            return
        self._last_draw = now
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        if plt is not None:
            plt.close(self.fig)

    def save(self, path: str) -> None:
        if plt is None:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")


class GameplayGridVisualizer:
    """Display raw gameplay frames for multiple environments in a tiled grid."""

    def __init__(
        self,
        num_envs: int,
        *,
        frame_shape: tuple[int, ...] = (72, 80),
        cmap: str = "gray",
        title: str = "Parallel Gameplay",
        vmin: float = 0.0,
        vmax: float = 255.0,
        target_fps: float | None = 24.0,
    ) -> None:
        if plt is None:
            raise RuntimeError(
                "matplotlib is required for gameplay visualization. "
                "Install it with `pip install matplotlib`."
            )
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.cmap = cmap
        self.frame_shape = tuple(int(dim) for dim in frame_shape)
        self.frame_shapes: List[tuple[int, ...]] = [self.frame_shape for _ in range(num_envs)]
        self.vmin = vmin
        self.vmax = vmax
        self._min_interval = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
        self._last_draw = 0.0

        plt.ion()
        cols = int(math.ceil(math.sqrt(num_envs)))
        rows = int(math.ceil(num_envs / cols))
        self.fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:  # pragma: no cover - backend specific
            pass
        axes_flat = np.atleast_1d(axes).reshape(-1)
        self.axes: List = list(axes_flat[:num_envs])
        for ax in axes_flat[num_envs:]:
            ax.axis("off")

        blank = np.zeros(self.frame_shape, dtype=np.float32)
        self.images = []
        self.status_texts = []
        for idx, ax in enumerate(self.axes):
            ax.set_title(f"Env {idx + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            if blank.ndim == 2:
                image = ax.imshow(
                    blank,
                    cmap=self.cmap,
                    origin="upper",
                    vmin=self.vmin,
                    vmax=self.vmax,
                    interpolation="nearest",
                )
            else:
                image = ax.imshow(blank.astype(np.uint8), origin="upper", interpolation="nearest")
            self.images.append(image)
            status = ax.text(
                0.02,
                0.98,
                "",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="white",
                bbox=dict(facecolor="black", alpha=0.6, boxstyle="round,pad=0.2"),
            )
            self.status_texts.append(status)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def new_episode(self, env_index: int, episode: int) -> None:
        if 0 <= env_index < self.num_envs:
            self.status_texts[env_index].set_text(f"Env {env_index + 1} | Episode {episode}")
            self._draw()

    def reset(self, env_index: Optional[int] = None) -> None:
        indices = range(self.num_envs) if env_index is None else [env_index]
        for idx in indices:
            shape = self.frame_shapes[idx]
            blank = np.zeros(shape, dtype=np.float32)
            artist = self.images[idx]
            if blank.ndim == 2:
                artist.set_data(blank)
                artist.set_clim(self.vmin, self.vmax)
            else:
                artist.set_data(blank.astype(np.uint8))
            self.status_texts[idx].set_text(f"Env {idx + 1}")
        self._draw()

    def update(
        self,
        env_index: int,
        frame: np.ndarray,
        *,
        info: Optional[dict] = None,
        reward: float | None = None,
        terminal: bool = False,
    ) -> None:
        if not (0 <= env_index < self.num_envs):
            return
        if frame is None:
            return
        array = np.asarray(frame)
        if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        array = np.squeeze(array)
        artist = self.images[env_index]
        artist.set_data(array)
        if array.ndim == 2:
            artist.set_clim(self.vmin, self.vmax)
        self.frame_shapes[env_index] = array.shape

        map_name = info.get("map_name", "unknown") if info else "unknown"
        map_id = info.get("map_id") if info else None
        if isinstance(map_id, int):
            status = f"Env {env_index + 1} | Map: {map_name} (0x{map_id:02X})"
        else:
            status = f"Env {env_index + 1} | Map: {map_name}"
        if reward is not None:
            status += f"\nReward: {reward: .2f}"
        if terminal:
            status += "\n★ Episode End"
        self.status_texts[env_index].set_text(status)
        self._draw()

    def _draw(self) -> None:
        if plt is None:
            return
        now = time.perf_counter()
        if self._min_interval > 0.0 and (now - self._last_draw) < self._min_interval:
            return
        self._last_draw = now
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        if plt is not None:
            plt.close(self.fig)

    def save(self, path: str) -> None:
        if plt is None:
            return
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
