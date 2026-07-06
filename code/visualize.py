"""Visualization helpers for the rescue robot environment.

The text helpers are intentionally dependency-light. The plotting and animation
helpers import matplotlib lazily, so students can still run non-visual tests in
minimal environments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rescue_env import RescueEnv
from rescue_types import ARROWS, Action, State


CELL_COLORS = {
    "#": "#4b5563",
    ".": "#f8fafc",
    "S": "#dbeafe",
    "P": "#fef3c7",
    "R": "#dcfce7",
    "E": "#bbf7d0",
    "X": "#fecaca",
}

HAZARD_EDGE_COLORS = {
    "normal": "#cbd5e1",
    "smoky": "#64748b",
    "slippery": "#38bdf8",
    "damaged": "#f97316",
    "wall": "#4b5563",
}


def format_state(state: State) -> str:
    row, col, carrying, battery, time_remaining = state
    carrying_mark = "C" if carrying else "-"
    return f"({row},{col},{carrying_mark},b={battery},t={time_remaining})"


def print_map(env: RescueEnv) -> None:
    for row in env.grid:
        print(" ".join(f"{cell:>2}" for cell in row))


def print_hazards(env: RescueEnv) -> None:
    short = {
        "#": "##",
        "normal": "N",
        "smoky": "Sm",
        "slippery": "Sl",
        "damaged": "D",
    }
    for row in env.hazards:
        print(" ".join(f"{short.get(cell, cell):>2}" for cell in row))


def print_policy_slice(
    env: RescueEnv,
    policy: dict[Any, Any],
    time_remaining: int,
    carrying: bool = False,
    battery: int | None = None,
) -> None:
    """Print a grid of actions for one time/battery/carrying slice.

    Walls are printed as ##. Terminal cells keep their map symbol.
    """

    battery_level = env.max_battery if battery is None else battery
    for r, row in enumerate(env.grid):
        rendered = []
        for c, cell in enumerate(row):
            if cell == "#":
                rendered.append("##")
                continue
            state: State = (r, c, carrying, battery_level, time_remaining)
            if env.is_terminal(state):
                rendered.append(f"{cell:>2}")
                continue
            action = _lookup_policy(policy, state, time_remaining)
            if action is None:
                rendered.append("  ")
                continue
            rendered.append(f"{ARROWS.get(action, '?'):>2}")
        print(" ".join(rendered))


def plot_map(env: RescueEnv, ax: Any | None = None, show_hazards: bool = True) -> tuple[Any, Any]:
    """Draw the rescue map and return ``(fig, ax)``.

    Walls, start, pickup, recharge, exit, and danger cells are shown with
    distinct colors. When ``show_hazards`` is true, each traversable cell gets a
    hazard-colored border.
    """

    plt, patches, _animation = _load_matplotlib()
    fig, ax = _prepare_axes(env, ax)
    for r, row in enumerate(env.grid):
        for c, cell in enumerate(row):
            face = CELL_COLORS.get(cell, CELL_COLORS["."])
            edge = HAZARD_EDGE_COLORS.get(env.hazard(r, c), "#cbd5e1") if show_hazards else "#cbd5e1"
            if cell == "#":
                edge = CELL_COLORS["#"]
            rect = patches.Rectangle((c, r), 1, 1, facecolor=face, edgecolor=edge, linewidth=2)
            ax.add_patch(rect)
            if cell not in {".", "#"}:
                ax.text(c + 0.5, r + 0.5, cell, ha="center", va="center", fontsize=12, weight="bold")
    return fig, ax


def plot_policy_slice(
    env: RescueEnv,
    policy: dict[Any, Any],
    time_remaining: int,
    carrying: bool = False,
    battery: int | None = None,
    ax: Any | None = None,
    title: str | None = None,
) -> tuple[Any, Any]:
    """Draw a policy grid for a fixed time/carrying/battery slice."""

    plt, _patches, _animation = _load_matplotlib()
    fig, ax = plot_map(env, ax=ax)
    battery_level = env.max_battery if battery is None else battery
    for r, row in enumerate(env.grid):
        for c, cell in enumerate(row):
            if cell == "#":
                continue
            state: State = (r, c, carrying, battery_level, time_remaining)
            if env.is_terminal(state):
                continue
            action = _lookup_policy(policy, state, time_remaining)
            if action is None:
                continue
            dx, dy = _arrow_delta(action)
            marker_x = c + 0.5
            marker_y = r + 0.5
            marker_scale = 0.28
            if cell not in {".", "#"}:
                marker_x = c + 0.72
                marker_y = r + 0.28
                marker_scale = 0.18
            if action == "WAIT":
                ax.text(marker_x, marker_y, "o", ha="center", va="center", fontsize=14, color="#111827")
            else:
                ax.arrow(
                    marker_x,
                    marker_y,
                    dx * marker_scale,
                    dy * marker_scale,
                    head_width=0.16,
                    head_length=0.12,
                    length_includes_head=True,
                    color="#111827",
                    linewidth=1.5,
                )
    if title is None:
        title = f"Policy slice: t={time_remaining}, carrying={carrying}, battery={battery_level}"
    ax.set_title(title)
    return fig, ax


def save_policy_slice(
    env: RescueEnv,
    policy: dict[Any, Any],
    output_path: str | Path,
    time_remaining: int,
    carrying: bool = False,
    battery: int | None = None,
) -> None:
    """Save a policy-slice visualization to an image file."""

    plt, _patches, _animation = _load_matplotlib()
    fig, _ax = plot_policy_slice(env, policy, time_remaining, carrying, battery)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def trajectory_to_path(trajectory: list[tuple[State, Action, float, State, bool]]) -> list[tuple[int, int]]:
    """Convert an environment trajectory into a row/column path."""

    if not trajectory:
        return []
    path = [(trajectory[0][0][0], trajectory[0][0][1])]
    for _state, _action, _reward, next_state, _done in trajectory:
        path.append((next_state[0], next_state[1]))
    return path


class RescueAnimation:
    """Animate one or more rescue-robot paths on the map.

    Paths are lists of ``(row, col)`` locations. This class is adapted from a
    generic grid animation pattern and specialized for the single-map rescue
    environment used in this homework.
    """

    def __init__(
        self,
        env: RescueEnv,
        paths: list[list[tuple[int, int]]],
        labels: list[str] | None = None,
        interval: int = 150,
    ) -> None:
        plt, patches, animation = _load_matplotlib()
        self.plt = plt
        self.patches_module = patches
        self.paths = paths
        self.labels = labels if labels is not None else [str(i) for i in range(len(paths))]

        self.fig, self.ax = plot_map(env)
        self.agents = []
        self.agent_labels = []
        colors = ["#2563eb", "#16a34a", "#f97316", "#a855f7"]
        for i, path in enumerate(paths):
            if not path:
                continue
            row, col = path[0]
            agent = patches.Circle((col + 0.5, row + 0.5), 0.28, facecolor=colors[i % len(colors)], edgecolor="black")
            self.ax.add_patch(agent)
            self.agents.append(agent)
            text = self.ax.text(col + 0.5, row + 0.18, self.labels[i], ha="center", va="center", fontsize=8, color="white")
            self.agent_labels.append(text)

        self.max_steps = max((len(path) for path in paths), default=1)
        self.animation = animation.FuncAnimation(
            self.fig,
            self._animate,
            frames=max(1, (self.max_steps - 1) * 10 + 1),
            interval=interval,
            blit=True,
        )

    def save(self, output_path: str | Path, fps: int = 10) -> None:
        """Save the animation. Use ``.gif`` for Pillow or ``.mp4`` for ffmpeg."""

        self.animation.save(output_path, fps=fps, dpi=160, savefig_kwargs={"bbox_inches": "tight"})

    def show(self) -> None:
        self.plt.show()

    def _animate(self, frame: int) -> list[Any]:
        t = frame / 10
        artists: list[Any] = []
        for i, path in enumerate(self.paths):
            if not path or i >= len(self.agents):
                continue
            row, col = _interpolated_position(t, path)
            self.agents[i].center = (col + 0.5, row + 0.5)
            self.agent_labels[i].set_position((col + 0.5, row + 0.18))
            artists.extend([self.agents[i], self.agent_labels[i]])
        return artists


def animate_trajectory(
    env: RescueEnv,
    trajectory: list[tuple[State, Action, float, State, bool]],
    output_path: str | Path | None = None,
    interval: int = 150,
) -> RescueAnimation:
    """Create an animation from one simulated trajectory.

    If ``output_path`` is provided, the animation is saved before being returned.
    """

    anim = RescueAnimation(env, [trajectory_to_path(trajectory)], labels=["robot"], interval=interval)
    if output_path is not None:
        anim.save(output_path)
    return anim


def _lookup_policy(policy: dict[Any, Any], state: State, time_remaining: int) -> Action | None:
    if time_remaining in policy:
        return policy[time_remaining].get(state)
    return policy.get(state)


def _load_matplotlib() -> tuple[Any, Any, Any]:
    import matplotlib.pyplot as plt
    from matplotlib import animation, patches

    return plt, patches, animation


def _prepare_axes(env: RescueEnv, ax: Any | None) -> tuple[Any, Any]:
    plt, _patches, _animation = _load_matplotlib()
    if ax is None:
        fig, ax = plt.subplots(figsize=(env.width, env.height))
    else:
        fig = ax.figure
    ax.set_xlim(0, env.width)
    ax.set_ylim(env.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(env.width + 1))
    ax.set_yticks(range(env.height + 1))
    ax.grid(color="#94a3b8", linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return fig, ax


def _arrow_delta(action: Action) -> tuple[float, float]:
    if action == "UP":
        return (0.0, -1.0)
    if action == "DOWN":
        return (0.0, 1.0)
    if action == "LEFT":
        return (-1.0, 0.0)
    if action == "RIGHT":
        return (1.0, 0.0)
    return (0.0, 0.0)


def _interpolated_position(t: float, path: list[tuple[int, int]]) -> tuple[float, float]:
    if not path:
        return (0.0, 0.0)
    if t <= 0:
        row, col = path[0]
        return (float(row), float(col))
    if t >= len(path) - 1:
        row, col = path[-1]
        return (float(row), float(col))
    i = int(t)
    frac = t - i
    row0, col0 = path[i]
    row1, col1 = path[i + 1]
    return (row0 + (row1 - row0) * frac, col0 + (col1 - col0) * frac)
