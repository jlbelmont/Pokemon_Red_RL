from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import torch

BADGE_NAMES = (
    "boulder",
    "cascade",
    "thunder",
    "rainbow",
    "soul",
    "marsh",
    "volcano",
    "earth",
)


@dataclass
class ProgressEvent:
    name: str
    event_type: str  # "story_flag", "badge", "elite"
    flag_key: Optional[str] = None
    badge_index: Optional[int] = None
    counter_key: Optional[str] = None
    count_threshold: Optional[float] = None
    absolute_counter: bool = False
    step_limit: int = 500000
    decision_threshold: float = 0.5
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def triggered(self, info: Dict, baseline: Optional[float] = None) -> bool:
        if self.event_type == "story_flag":
            story_flags = info.get("story_flags") or {}
            return bool(story_flags.get(self.flag_key, False))
        if self.event_type == "badge":
            badge_idx = int(self.badge_index or 0)
            return int(info.get("badge_count") or 0) >= badge_idx
        if self.event_type in {"elite", "champion"}:
            return bool(info.get("champion_defeated"))
        if self.event_type in {"counter", "pokedex_count", "pokemon_caught", "pokemon_defeated"}:
            key = self.counter_key
            if self.event_type == "pokedex_count":
                key = key or "pokedex_owned_count"
            elif self.event_type == "pokemon_caught":
                key = key or "pokemon_caught_total"
            elif self.event_type == "pokemon_defeated":
                key = key or "pokemon_defeated_total"
            threshold = self.count_threshold
            if key is None or threshold is None:
                return False
            try:
                value = float(info.get(key) or 0.0)
            except (TypeError, ValueError):
                return False
            base = 0.0 if self.absolute_counter else (baseline or 0.0)
            return (value - base) >= threshold
        return False


class BayesProgressTracker:
    def __init__(
        self,
        events: List[ProgressEvent],
        num_envs: int,
        metrics_path: str,
        step_margin: int = 0,
    ) -> None:
        self.events = events
        self.num_envs = num_envs
        self.metrics_path = metrics_path
        self.step_margin = step_margin
        self._event_state: List[Dict[str, bool]] = [
            {event.name: False for event in events} for _ in range(num_envs)
        ]
        self._episode_active = [False for _ in range(num_envs)]
        self._stats: Dict[str, Dict[str, float]] = {
            event.name: {"success": 0.0, "total": 0.0} for event in events
        }
        self._baseline_badge_bits = [0 for _ in range(num_envs)]
        self._baseline_badge_sets: List[set[str]] = [set() for _ in range(num_envs)]
        self._baseline_champion = [False for _ in range(num_envs)]
        self._baseline_counter_values: List[Dict[str, float]] = [
            {event.name: 0.0 for event in events} for _ in range(num_envs)
        ]

    def begin_episode(self, env_idx: int, info: Optional[Dict] = None) -> None:
        if not self.events:
            return
        self._episode_active[env_idx] = True
        for event in self.events:
            self._event_state[env_idx][event.name] = False
        self._update_baseline(env_idx, info, update_counters=True)

    def sync_baseline(self, env_idx: int, info: Optional[Dict]) -> None:
        if not self.events:
            return
        self._update_baseline(env_idx, info, update_counters=False)

    def observe(self, env_idx: int, info: Dict, step_count: int) -> None:
        if not self.events or not self._episode_active[env_idx]:
            return
        for event in self.events:
            if self._event_state[env_idx][event.name]:
                continue
            if step_count > (event.step_limit + self.step_margin):
                continue
            if step_count == 0:
                continue
            triggered = False
            baseline_counter = None
            if event.counter_key:
                baseline_counter = self._baseline_counter_values[env_idx].get(event.name, 0.0)
            if event.event_type == "badge":
                badge_idx = max(1, int(event.badge_index or 0))
                bit = 1 << (badge_idx - 1)
                badge_name = BADGE_NAMES[badge_idx - 1] if badge_idx <= len(BADGE_NAMES) else None
                baseline_mask = self._baseline_badge_bits[env_idx]
                baseline_set = self._baseline_badge_sets[env_idx]
                current_mask = int(info.get("badge_bits") or 0)
                badge_dict = info.get("badges") or {}
                current_has = badge_dict.get(badge_name, False) if badge_name else False
                baseline_has = badge_name in baseline_set if badge_name else False
                if not (baseline_mask & bit) and (current_mask & bit):
                    if badge_name is None or (current_has and not baseline_has):
                        triggered = True
            elif event.event_type in {"elite", "champion"}:
                baseline = self._baseline_champion[env_idx]
                if not baseline and bool(info.get("champion_defeated")):
                    triggered = True
            elif event.event_type in {"counter", "pokedex_count", "pokemon_caught", "pokemon_defeated"}:
                triggered = event.triggered(info, baseline_counter)
            else:
                triggered = event.triggered(info, baseline_counter)
            if triggered and step_count <= event.step_limit:
                self._event_state[env_idx][event.name] = True

    def finish_episode(self, env_idx: int) -> Optional[Dict[str, bool]]:
        if not self.events or not self._episode_active[env_idx]:
            return None
        self._episode_active[env_idx] = False
        snapshot: Dict[str, bool] = {}
        for event in self.events:
            state = self._event_state[env_idx][event.name]
            self._stats[event.name]["total"] += 1.0
            if state:
                self._stats[event.name]["success"] += 1.0
            snapshot[event.name] = state
        return snapshot

    def summarise(self) -> List[Dict]:
        summary: List[Dict] = []
        for event in self.events:
            data = self._stats[event.name]
            total = data["total"]
            success = data["success"]
            alpha = event.prior_alpha + success
            beta = event.prior_beta + max(0.0, total - success)
            posterior_mean = alpha / (alpha + beta) if (alpha + beta) > 0 else 0.0
            lower, upper = self._credible_interval(alpha, beta)
            decision = "pursue"
            if posterior_mean < event.decision_threshold:
                decision = "defer"
            summary.append(
                {
                    "name": event.name,
                    "event_type": event.event_type,
                    "step_limit": event.step_limit,
                    "trials": total,
                    "successes": success,
                    "posterior_alpha": alpha,
                    "posterior_beta": beta,
                    "posterior_mean": posterior_mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "decision_threshold": event.decision_threshold,
                    "decision": decision,
                }
            )
        return summary

    def save(self) -> Optional[Dict]:
        if not self.events or not self.metrics_path:
            return None
        directory = os.path.dirname(self.metrics_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "events": self.summarise(),
            "notes": "Posterior parameters assume Beta(prior_alpha, prior_beta) priors.",
        }
        with open(self.metrics_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return payload

    @staticmethod
    def load_metrics(path: str) -> Optional[Dict]:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None

    @staticmethod
    def format_metrics(metrics: Dict) -> str:
        if not metrics:
            return ""
        lines = ["Progress probabilities (Bayesian posterior):"]
        events = metrics.get("events") or []
        for event in events:
            name = event.get("name")
            mean = event.get("posterior_mean", 0.0)
            lower = event.get("ci_lower", 0.0)
            upper = event.get("ci_upper", 0.0)
            threshold = event.get("decision_threshold", 0.5)
            decision = event.get("decision", "pursue")
            trials = int(event.get("trials") or 0)
            successes = int(event.get("successes") or 0)
            lines.append(
                f"  - {name}: mean={mean:.3f} (95% CI [{lower:.3f}, {upper:.3f}]) "
                f"| successes {successes}/{trials} | decision≥{threshold:.2f}: {decision}"
            )
        return "\n".join(lines)

    @staticmethod
    def _credible_interval(alpha: float, beta: float, cred: float = 0.95) -> tuple[float, float]:
        if alpha <= 0 or beta <= 0:
            return (0.0, 1.0)
        # Compute on CPU to avoid backend gaps (e.g., MPS missing Beta.icdf).
        lower_q = (1.0 - cred) / 2.0
        upper_q = 1.0 - lower_q
        dist = torch.distributions.Beta(
            torch.tensor([alpha], dtype=torch.float64, device="cpu"),
            torch.tensor([beta], dtype=torch.float64, device="cpu"),
        )
        try:
            lower = float(dist.icdf(torch.tensor([lower_q], dtype=torch.float64)))
            upper = float(dist.icdf(torch.tensor([upper_q], dtype=torch.float64)))
        except Exception:
            # Fallback: analytic Beta quantile via numpy to avoid long sampling.
            import numpy as _np
            lower = float(_np.quantile(_np.random.beta(alpha, beta, size=10000), lower_q))
            upper = float(_np.quantile(_np.random.beta(alpha, beta, size=10000), upper_q))
        if math.isnan(lower) or math.isnan(upper):
            return (0.0, 1.0)
        return (lower, upper)

    def _update_baseline(self, env_idx: int, info: Optional[Dict], update_counters: bool) -> None:
        data = info or {}
        badge_bits = int(data.get("badge_bits") or 0)
        badge_set = {
            name for name, unlocked in (data.get("badges") or {}).items() if unlocked
        }
        champion = bool(data.get("champion_defeated") or data.get("champion_flag_raw"))
        self._baseline_badge_bits[env_idx] = badge_bits
        self._baseline_badge_sets[env_idx] = badge_set
        self._baseline_champion[env_idx] = champion
        if update_counters:
            counters = self._baseline_counter_values[env_idx]
            for event in self.events:
                if not event.counter_key:
                    continue
                try:
                    counters[event.name] = float(data.get(event.counter_key) or 0.0)
                except (TypeError, ValueError):
                    counters[event.name] = 0.0


def parse_progress_events(raw_events: Optional[List[Dict]]) -> List[ProgressEvent]:
    events: List[ProgressEvent] = []
    if not raw_events:
        return events
    for entry in raw_events:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        event_type = entry.get("type")
        if not name or not event_type:
            continue
        step_limit = max(1, int(entry.get("step_limit", 500000)))
        decision_threshold = float(entry.get("decision_threshold", 0.5))
        prior_alpha = float(entry.get("prior_alpha", 1.0))
        prior_beta = float(entry.get("prior_beta", 1.0))
        flag_key = entry.get("flag_key") or entry.get("flag")
        badge_index = entry.get("badge_index")
        if badge_index is not None:
            try:
                badge_index = int(badge_index)
            except (TypeError, ValueError):
                badge_index = None
        counter_key = entry.get("counter_key") or entry.get("info_key")
        count_threshold = entry.get("count_threshold")
        if count_threshold is None:
            count_threshold = entry.get("threshold")
        if count_threshold is not None:
            try:
                count_threshold = float(count_threshold)
            except (TypeError, ValueError):
                count_threshold = None
        if event_type == "pokedex_count" and not counter_key:
            counter_key = "pokedex_owned_count"
        absolute_counter = bool(entry.get("absolute_counter", False))
        events.append(
            ProgressEvent(
                name=name,
                event_type=event_type,
                flag_key=flag_key,
                badge_index=badge_index,
                counter_key=counter_key,
                count_threshold=count_threshold,
                absolute_counter=absolute_counter,
                step_limit=step_limit,
                decision_threshold=decision_threshold,
                prior_alpha=prior_alpha,
                prior_beta=prior_beta,
            )
        )
    return events
