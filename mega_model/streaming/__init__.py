"""
SSH tunnel streaming for viewing best agents in real-time.

Architecture:
- Laptop runs tunnel_server.py which opens a TCP socket and displays frames
- User creates reverse SSH tunnel: ssh -R 9999:localhost:9999 user@cluster
- Cluster jobs run tunnel_client.py which connects to localhost:9999 (the tunnel)
- Only top-K agents (by badge+story+reward) stream frames

Usage:
    # On laptop: start the viewer server
    python -m mega_model.streaming.tunnel_server --port 9999

    # Create reverse tunnel (in another terminal)
    ssh -R 9999:localhost:9999 user@cluster

    # The cluster training jobs will auto-connect and stream top-2 agents
"""

from .tunnel_client import StreamClient
from .best_selector import BestAgentSelector, progress_score

__all__ = ["StreamClient", "BestAgentSelector", "progress_score"]
