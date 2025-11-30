"""Data loading helpers for posterior analysis."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_progress_json(path: Path) -> pd.DataFrame:
    """Load a single progress_metrics.json file into a normalized DataFrame."""
    payload = json.loads(path.read_text())
    records = payload.get("progress_events")
    if not records:
        records = payload.get("events") or []
    frame = pd.json_normalize(records)
    frame["source_file"] = path.name
    frame["run_id"] = path.parent.name
    return frame


def load_all_progress(root: Path) -> pd.DataFrame:
    """Concatenate every progress JSON underneath root/run_id directories."""
    frames = []
    for manifest in root.glob("*/progress_metrics.json"):
        frames.append(load_progress_json(manifest))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_summary_csv(path: Path) -> pd.DataFrame:
    """Read the training summary CSV (if it exists)."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
