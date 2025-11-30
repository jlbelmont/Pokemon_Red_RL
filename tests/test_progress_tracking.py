import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from epsilon.pokemon_rl.progress_tracking import (
    BADGE_NAMES,
    BayesProgressTracker,
    ProgressEvent,
)


def _blank_badge_info() -> dict:
    return {
        "badge_bits": 0,
        "badges": {name: False for name in BADGE_NAMES},
        "champion_flag_raw": False,
        "champion_defeated": False,
    }


def test_badge_success_requires_bit_flip(tmp_path: Path) -> None:
    event = ProgressEvent(name="Boulder", event_type="badge", badge_index=1, step_limit=100)
    tracker = BayesProgressTracker([event], num_envs=1, metrics_path=str(tmp_path / "metrics.json"))

    info = _blank_badge_info()
    tracker.begin_episode(0, info)
    tracker.observe(0, info, 0)

    post = _blank_badge_info()
    post["badge_bits"] = 0b00000001
    post["badges"]["boulder"] = True
    tracker.observe(0, post, 5)
    tracker.finish_episode(0)

    summary = tracker.summarise()[0]
    assert summary["successes"] == 1

    tracker.begin_episode(0, post)
    tracker.observe(0, post, 0)
    tracker.finish_episode(0)
    summary = tracker.summarise()[0]
    assert summary["successes"] == 1  # unchanged; baseline already had badge


def test_champion_success_requires_flag_flip(tmp_path: Path) -> None:
    event = ProgressEvent(name="Champion", event_type="champion", step_limit=100)
    tracker = BayesProgressTracker([event], num_envs=1, metrics_path=str(tmp_path / "metrics.json"))

    info = _blank_badge_info()
    tracker.begin_episode(0, info)
    tracker.observe(0, info, 0)

    post = _blank_badge_info()
    post["champion_flag_raw"] = True
    post["champion_defeated"] = True
    tracker.observe(0, post, 10)
    tracker.finish_episode(0)

    summary = tracker.summarise()[0]
    assert summary["successes"] == 1

    tracker.begin_episode(0, post)
    tracker.observe(0, post, 0)
    tracker.finish_episode(0)
    summary = tracker.summarise()[0]
    assert summary["successes"] == 1  # no new success when baseline already set


def test_sync_baseline_prevents_false_badge(tmp_path: Path) -> None:
    event = ProgressEvent(name="Boulder", event_type="badge", badge_index=1, step_limit=100)
    tracker = BayesProgressTracker([event], num_envs=1, metrics_path=str(tmp_path / "metrics.json"))

    info = _blank_badge_info()
    tracker.begin_episode(0, info)

    mid = _blank_badge_info()
    mid["badge_bits"] = 0b0001
    mid["badges"]["boulder"] = True
    tracker.sync_baseline(0, mid)
    tracker.observe(0, mid, 10)
    tracker.finish_episode(0)
    summary = tracker.summarise()[0]
    assert summary["successes"] == 0
