from __future__ import annotations

from typing import List, Optional

import math
import time
import numpy as np

try:  # Matplotlib is optional at runtime.
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - handled gracefully at runtime
    plt = None


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
            self._refresh_aggregate()

    def _refresh_aggregate(self) -> None:
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
