"""
Tunnel server for receiving and displaying frames from the cluster.

Runs on the laptop and listens for connections from the reverse SSH tunnel.
Displays frames in a pygame window, showing the top-K best agents.

Usage:
    # Start the server
    python -m mega_model.streaming.tunnel_server --port 9999

    # In another terminal, create the reverse tunnel
    ssh -R 9999:localhost:9999 user@cluster
"""

from __future__ import annotations

import argparse
import base64
import json
import socket
import struct
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import numpy as np

# Try to import pygame for display
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


@dataclass
class AgentState:
    """State for a streaming agent."""
    agent_id: str
    score: int = 0
    episode_reward: float = 0.0
    step: int = 0
    last_frame: Optional[np.ndarray] = None
    last_update: float = field(default_factory=time.time)
    frame_count: int = 0


class FrameReceiver:
    """
    Receives frames from the tunnel and manages agent states.
    
    Runs a background thread to accept connections and receive data.
    """
    
    def __init__(self, port: int, host: str = "0.0.0.0", top_k: int = 2):
        self.port = port
        self.host = host
        self.top_k = top_k
        
        self.agents: Dict[str, AgentState] = {}
        self.frame_queue: Queue = Queue(maxsize=100)
        self._lock = threading.Lock()
        self._running = False
        self._server_socket: Optional[socket.socket] = None
        self._threads: List[threading.Thread] = []
    
    def start(self) -> None:
        """Start the receiver server."""
        self._running = True
        
        # Start server socket
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(10)
        self._server_socket.settimeout(1.0)
        
        # Start accept thread
        accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        accept_thread.start()
        self._threads.append(accept_thread)
        
        print(f"[server] Listening on {self.host}:{self.port}")
        print(f"[server] Create reverse tunnel: ssh -R {self.port}:localhost:{self.port} user@cluster")
    
    def stop(self) -> None:
        """Stop the receiver server."""
        self._running = False
        
        if self._server_socket:
            try:
                self._server_socket.close()
            except Exception:
                pass
        
        for t in self._threads:
            t.join(timeout=2.0)
    
    def _accept_loop(self) -> None:
        """Accept incoming connections."""
        while self._running:
            try:
                if self._server_socket is None:
                    break
                client_socket, addr = self._server_socket.accept()
                print(f"[server] Connection from {addr}")
                
                # Start handler thread for this client
                handler = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, addr),
                    daemon=True
                )
                handler.start()
                self._threads.append(handler)
            except socket.timeout:
                continue
            except Exception as e:
                if self._running:
                    print(f"[server] Accept error: {e}")
                break
    
    def _handle_client(self, client_socket: socket.socket, addr: tuple) -> None:
        """Handle a single client connection."""
        client_socket.settimeout(5.0)
        buffer = b""
        
        try:
            while self._running:
                try:
                    # Read length-prefixed messages
                    while len(buffer) < 4:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            return
                        buffer += chunk
                    
                    msg_len = struct.unpack(">I", buffer[:4])[0]
                    
                    while len(buffer) < 4 + msg_len:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            return
                        buffer += chunk
                    
                    msg_data = buffer[4:4 + msg_len]
                    buffer = buffer[4 + msg_len:]
                    
                    self._process_message(msg_data)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[server] Error from {addr}: {e}")
                    break
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            print(f"[server] Disconnected: {addr}")
    
    def _process_message(self, data: bytes) -> None:
        """Process a received message."""
        try:
            payload = json.loads(data.decode("utf-8"))
            msg_type = payload.get("type", "frame")
            agent_id = payload.get("agent_id", "unknown")
            
            with self._lock:
                if agent_id not in self.agents:
                    self.agents[agent_id] = AgentState(agent_id=agent_id)
                
                agent = self.agents[agent_id]
                agent.last_update = time.time()
                agent.step = payload.get("step", agent.step)
                
                if msg_type == "frame":
                    # Decode frame
                    shape = tuple(payload.get("shape", [144, 160, 3]))
                    dtype = np.dtype(payload.get("dtype", "uint8"))
                    frame_b64 = payload.get("frame_b64", "")
                    
                    if frame_b64:
                        frame_bytes = base64.b64decode(frame_b64)
                        frame = np.frombuffer(frame_bytes, dtype=dtype).reshape(shape)
                        agent.last_frame = frame
                        agent.frame_count += 1
                        
                        # Also get score from metadata
                        meta = payload.get("meta", {})
                        agent.score = int(meta.get("score", agent.score))
                        agent.episode_reward = float(meta.get("episode_reward", agent.episode_reward))
                        
                        # Put in queue for display
                        try:
                            self.frame_queue.put_nowait((agent_id, frame, agent.score))
                        except:
                            pass
                
                elif msg_type == "status":
                    agent.score = int(payload.get("score", agent.score))
                    agent.episode_reward = float(payload.get("episode_reward", agent.episode_reward))
        
        except Exception as e:
            print(f"[server] Error processing message: {e}")
    
    def get_top_agents(self) -> List[AgentState]:
        """Get the top-K agents by score."""
        with self._lock:
            # Remove stale agents (no update in 60 seconds)
            cutoff = time.time() - 60.0
            active = [a for a in self.agents.values() if a.last_update > cutoff]
            
            # Sort by score, then reward
            sorted_agents = sorted(
                active,
                key=lambda a: (-a.score, -a.episode_reward)
            )
            return sorted_agents[:self.top_k]


class FrameViewer:
    """
    Displays frames using pygame.
    
    Shows top-K agents side by side with their scores.
    """
    
    def __init__(
        self,
        receiver: FrameReceiver,
        window_width: int = 800,
        window_height: int = 600,
        scale: int = 2,
    ):
        self.receiver = receiver
        self.window_width = window_width
        self.window_height = window_height
        self.scale = scale
        self.running = False
    
    def run(self) -> None:
        """Run the viewer main loop."""
        if not PYGAME_AVAILABLE:
            print("[viewer] pygame not available, running in headless mode")
            self._run_headless()
            return
        
        pygame.init()
        screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Pokemon Red RL - Best Agents Stream")
        clock = pygame.time.Clock()
        font = pygame.font.Font(None, 24)
        
        self.running = True
        
        try:
            while self.running:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            self.running = False
                
                # Clear screen
                screen.fill((20, 20, 30))
                
                # Get top agents
                top_agents = self.receiver.get_top_agents()
                
                if not top_agents:
                    # No agents connected
                    text = font.render("Waiting for agents to connect...", True, (200, 200, 200))
                    text_rect = text.get_rect(center=(self.window_width // 2, self.window_height // 2))
                    screen.blit(text, text_rect)
                else:
                    # Display each agent
                    num_agents = len(top_agents)
                    panel_width = self.window_width // max(1, num_agents)
                    
                    for i, agent in enumerate(top_agents):
                        x_offset = i * panel_width
                        
                        # Draw frame if available
                        if agent.last_frame is not None:
                            frame = agent.last_frame
                            
                            # Handle grayscale
                            if frame.ndim == 2:
                                frame = np.stack([frame] * 3, axis=-1)
                            elif frame.shape[-1] == 1:
                                frame = np.repeat(frame, 3, axis=-1)
                            
                            # Scale frame
                            h, w = frame.shape[:2]
                            scaled_w = min(panel_width - 20, w * self.scale)
                            scaled_h = int(h * scaled_w / w)
                            
                            # Convert to pygame surface
                            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                            surface = pygame.transform.scale(surface, (scaled_w, scaled_h))
                            
                            # Center in panel
                            frame_x = x_offset + (panel_width - scaled_w) // 2
                            frame_y = 60
                            screen.blit(surface, (frame_x, frame_y))
                        
                        # Draw agent info
                        title = font.render(f"Agent: {agent.agent_id}", True, (255, 255, 255))
                        screen.blit(title, (x_offset + 10, 10))
                        
                        score_text = font.render(
                            f"Score: {agent.score} | Reward: {agent.episode_reward:.1f}",
                            True,
                            (150, 255, 150)
                        )
                        screen.blit(score_text, (x_offset + 10, 35))
                        
                        # Frame count
                        frames_text = font.render(
                            f"Frames: {agent.frame_count} | Step: {agent.step}",
                            True,
                            (150, 150, 255)
                        )
                        screen.blit(frames_text, (x_offset + 10, self.window_height - 30))
                
                pygame.display.flip()
                clock.tick(30)
        
        finally:
            pygame.quit()
    
    def _run_headless(self) -> None:
        """Run without display, just print stats."""
        self.running = True
        last_print = 0
        
        try:
            while self.running:
                time.sleep(0.5)
                
                now = time.time()
                if now - last_print > 2.0:
                    last_print = now
                    top_agents = self.receiver.get_top_agents()
                    
                    if top_agents:
                        print("\n--- Top Agents ---")
                        for agent in top_agents:
                            print(f"  {agent.agent_id}: score={agent.score}, "
                                  f"reward={agent.episode_reward:.1f}, "
                                  f"frames={agent.frame_count}")
                    else:
                        print("Waiting for agents...")
        
        except KeyboardInterrupt:
            self.running = False


def main():
    parser = argparse.ArgumentParser(description="Stream viewer server for Pokemon Red RL")
    parser.add_argument("--port", type=int, default=9999, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--top-k", type=int, default=2, help="Number of best agents to show")
    parser.add_argument("--width", type=int, default=800, help="Window width")
    parser.add_argument("--height", type=int, default=600, help="Window height")
    parser.add_argument("--scale", type=int, default=2, help="Frame scale factor")
    parser.add_argument("--headless", action="store_true", help="Run without display")
    args = parser.parse_args()
    
    receiver = FrameReceiver(port=args.port, host=args.host, top_k=args.top_k)
    receiver.start()
    
    try:
        if args.headless or not PYGAME_AVAILABLE:
            viewer = FrameViewer(receiver, args.width, args.height, args.scale)
            viewer._run_headless()
        else:
            viewer = FrameViewer(receiver, args.width, args.height, args.scale)
            viewer.run()
    except KeyboardInterrupt:
        print("\n[server] Shutting down...")
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
