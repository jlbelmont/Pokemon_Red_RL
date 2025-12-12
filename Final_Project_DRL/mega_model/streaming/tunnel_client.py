"""
Tunnel client for streaming frames from cluster to laptop.

Connects to localhost:PORT which is reverse-tunneled to the laptop.
Sends frames as JSON-encoded messages with base64 frame data.
"""

from __future__ import annotations

import base64
import json
import socket
import struct
import time
from typing import Any, Dict, Optional

import numpy as np


class StreamClient:
    """
    Client for streaming frames to the tunnel server.
    
    Connects to localhost:port which should be reverse-tunneled to the laptop.
    Only sends frames at configured intervals to avoid overwhelming the connection.
    """
    
    def __init__(
        self,
        port: int = 9999,
        host: str = "localhost",
        metadata: Optional[Dict[str, Any]] = None,
        interval_steps: int = 60,
        connect_timeout: float = 2.0,
        send_timeout: float = 1.0,
    ):
        """
        Args:
            port: Port to connect to (reverse tunnel endpoint)
            host: Host to connect to (usually localhost for tunnel)
            metadata: Extra metadata to include with frames
            interval_steps: Only send every N steps
            connect_timeout: Connection timeout in seconds
            send_timeout: Send timeout in seconds
        """
        self.port = port
        self.host = host
        self.metadata = metadata or {}
        self.interval_steps = max(1, int(interval_steps))
        self.connect_timeout = connect_timeout
        self.send_timeout = send_timeout
        
        self._socket: Optional[socket.socket] = None
        self._last_sent_step = 0
        self._last_connect_attempt = 0.0
        self._connect_backoff = 5.0  # Seconds between reconnect attempts
        self._connected = False
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 string."""
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        return base64.b64encode(frame.tobytes()).decode("ascii")
    
    def _ensure_connected(self) -> bool:
        """Ensure we have a connection, with backoff on failures."""
        if self._socket is not None and self._connected:
            return True
        
        # Backoff check
        now = time.time()
        if now - self._last_connect_attempt < self._connect_backoff:
            return False
        
        self._last_connect_attempt = now
        
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self.connect_timeout)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(self.send_timeout)
            self._connected = True
            return True
        except Exception:
            self._close()
            return False
    
    def _close(self) -> None:
        """Close the socket."""
        if self._socket is not None:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False
    
    def _send_message(self, payload: Dict[str, Any]) -> bool:
        """Send a JSON message with length prefix."""
        if self._socket is None:
            return False
        
        try:
            data = json.dumps(payload).encode("utf-8")
            # Send 4-byte length prefix followed by data
            length_prefix = struct.pack(">I", len(data))
            self._socket.sendall(length_prefix + data)
            return True
        except Exception:
            self._close()
            return False
    
    def maybe_send(
        self,
        frame: Optional[np.ndarray],
        step: int,
        agent_id: str = "unknown",
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Maybe send a frame if enough steps have passed.
        
        Args:
            frame: RGB frame as numpy array (H, W, C) or (H, W)
            step: Current training step
            agent_id: Identifier for this agent
            extra_meta: Additional metadata for this frame
            
        Returns:
            True if frame was sent successfully
        """
        if frame is None:
            return False
        
        # Check interval
        if (step - self._last_sent_step) < self.interval_steps:
            return False
        
        # Try to connect
        if not self._ensure_connected():
            return False
        
        self._last_sent_step = step
        
        # Build payload
        payload = {
            "type": "frame",
            "agent_id": agent_id,
            "step": int(step),
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "frame_b64": self._encode_frame(frame),
            "timestamp": time.time(),
            "meta": {**self.metadata, **(extra_meta or {})},
        }
        
        return self._send_message(payload)
    
    def send_status(
        self,
        agent_id: str,
        score: int,
        episode_reward: float,
        step: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Send a status update (no frame).
        
        Args:
            agent_id: Agent identifier
            score: Progress score (badges + story flags)
            episode_reward: Current episode reward
            step: Training step
            extra: Additional info
            
        Returns:
            True if sent successfully
        """
        if not self._ensure_connected():
            return False
        
        payload = {
            "type": "status",
            "agent_id": agent_id,
            "score": score,
            "episode_reward": episode_reward,
            "step": step,
            "timestamp": time.time(),
            "extra": extra or {},
        }
        
        return self._send_message(payload)
    
    def close(self) -> None:
        """Close the connection."""
        self._close()
    
    def __del__(self):
        self.close()


def create_stream_client(
    config: Dict[str, Any],
    agent_id: str,
    job_rank: int = 0,
) -> Optional[StreamClient]:
    """
    Create a stream client from config if streaming is enabled.
    
    Args:
        config: Config dict with streaming section
        agent_id: Agent identifier
        job_rank: DDP job rank
        
    Returns:
        StreamClient if enabled, None otherwise
    """
    streaming_cfg = config.get("streaming", {})
    
    if not streaming_cfg.get("enabled", False):
        return None
    
    return StreamClient(
        port=streaming_cfg.get("tunnel_port", 9999),
        host="localhost",  # Always localhost (reverse tunnel)
        metadata={
            "agent_id": agent_id,
            "job_rank": job_rank,
        },
        interval_steps=streaming_cfg.get("interval_steps", 60),
    )

