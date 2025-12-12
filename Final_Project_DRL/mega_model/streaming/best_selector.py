"""
Best agent selection for streaming.

Ranks agents by:
1. Badge count + story flag count (primary)
2. Episode reward (tie-breaker)
3. Timestamp (most recent wins ties)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def progress_score(info: dict, episode_reward: float) -> Tuple[int, float]:
    """
    Compute composite progress score for an agent.
    
    Returns:
        (score, tie_breaker) where:
        - score = badge_count + number of completed story flags
        - tie_breaker = episode_reward
    """
    badges = int(info.get("badge_count", 0) or info.get("badges", 0) or 0)
    
    # Handle story flags - could be dict or int
    story_flags = info.get("story_flags") or info.get("events") or {}
    story_count = 0
    if isinstance(story_flags, dict):
        story_count = sum(1 for v in story_flags.values() if v)
    elif isinstance(story_flags, (int, float)):
        story_count = int(story_flags)
    
    return badges + story_count, float(episode_reward)


@dataclass
class AgentCandidate:
    """Represents an agent's streaming candidacy."""
    agent_id: str
    score: int
    tie_breaker: float
    timestamp: float
    env_idx: int = 0
    job_rank: int = 0


class BestAgentSelector:
    """
    Coordinates best-agent selection across multiple environments/jobs.
    
    Uses a shared JSON file to communicate between jobs (since they may be
    on different nodes). Each job updates its agents' scores, and all jobs
    read the file to determine if they're in the top-K.
    """
    
    def __init__(
        self,
        candidates_path: str,
        top_k: int = 2,
        stale_threshold_s: float = 60.0,
    ):
        """
        Args:
            candidates_path: Path to shared JSON file for candidate tracking
            top_k: Number of top agents to stream
            stale_threshold_s: Remove entries older than this (seconds)
        """
        self.candidates_path = Path(candidates_path)
        self.candidates_path.parent.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.stale_threshold_s = stale_threshold_s
        self._cache: Dict[str, AgentCandidate] = {}
    
    def update(
        self,
        agent_id: str,
        info: dict,
        episode_reward: float,
        env_idx: int = 0,
        job_rank: int = 0,
    ) -> bool:
        """
        Update an agent's score and check if it should stream.
        
        Args:
            agent_id: Unique identifier for this agent (e.g., "job0_env3")
            info: Environment info dict with badge_count, story_flags, etc.
            episode_reward: Cumulative episode reward so far
            env_idx: Environment index within this job
            job_rank: DDP rank / job index
            
        Returns:
            True if this agent is in the top-K and should stream frames
        """
        score, tie = progress_score(info, episode_reward)
        now = time.time()
        
        candidate = AgentCandidate(
            agent_id=agent_id,
            score=score,
            tie_breaker=tie,
            timestamp=now,
            env_idx=env_idx,
            job_rank=job_rank,
        )
        
        # Load existing candidates
        candidates = self._load_candidates()
        
        # Update our entry
        candidates[agent_id] = candidate
        
        # Remove stale entries
        cutoff = now - self.stale_threshold_s
        candidates = {
            k: v for k, v in candidates.items()
            if v.timestamp > cutoff
        }
        
        # Save updated candidates
        self._save_candidates(candidates)
        
        # Determine top-K
        sorted_candidates = sorted(
            candidates.values(),
            key=lambda c: (-c.score, -c.tie_breaker, -c.timestamp)
        )
        top_ids = {c.agent_id for c in sorted_candidates[:self.top_k]}
        
        return agent_id in top_ids
    
    def get_top_agents(self) -> List[AgentCandidate]:
        """Get the current top-K agents."""
        candidates = self._load_candidates()
        now = time.time()
        cutoff = now - self.stale_threshold_s
        
        active = [c for c in candidates.values() if c.timestamp > cutoff]
        sorted_candidates = sorted(
            active,
            key=lambda c: (-c.score, -c.tie_breaker, -c.timestamp)
        )
        return sorted_candidates[:self.top_k]
    
    def _load_candidates(self) -> Dict[str, AgentCandidate]:
        """Load candidates from disk."""
        if not self.candidates_path.exists():
            return {}
        
        try:
            with open(self.candidates_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            candidates = {}
            for agent_id, entry in data.items():
                if isinstance(entry, dict):
                    candidates[agent_id] = AgentCandidate(
                        agent_id=agent_id,
                        score=int(entry.get("score", 0)),
                        tie_breaker=float(entry.get("tie", 0.0)),
                        timestamp=float(entry.get("ts", 0.0)),
                        env_idx=int(entry.get("env_idx", 0)),
                        job_rank=int(entry.get("job_rank", 0)),
                    )
            return candidates
        except Exception:
            return {}
    
    def _save_candidates(self, candidates: Dict[str, AgentCandidate]) -> None:
        """Save candidates to disk atomically."""
        data = {
            agent_id: {
                "score": c.score,
                "tie": c.tie_breaker,
                "ts": c.timestamp,
                "env_idx": c.env_idx,
                "job_rank": c.job_rank,
            }
            for agent_id, c in candidates.items()
        }
        
        tmp_path = self.candidates_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.candidates_path)
        except Exception:
            pass

