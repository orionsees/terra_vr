#!/usr/bin/env python3
"""Lightweight IPC bus replacing ROS2 pub/sub.

All communication is JSON newlines over TCP (stdlib only).

Port map:
  5555 – VR data        (tcp_wireless.py publishes; teleop/ik_control subscribe)
  5556 – Left arm cmds  (left bridge listens; teleop/zero_arms send)
  5557 – Right arm cmds (right bridge listens; teleop/zero_arms send)
  5558 – Left arm feedback  (left bridge publishes; consumers subscribe)
  5559 – Right arm feedback (right bridge publishes; consumers subscribe)

Classes:
  Publisher      – TCP server, fans each message to all connected subscribers
  Subscriber     – TCP client, connects and yields decoded messages
  CommandServer  – TCP server, queues commands from any number of senders
  CommandClient  – TCP client, sends commands to a CommandServer
"""

import json
import socket
import threading
import time
from queue import Empty, Queue
from typing import Iterator, Optional

HOST = "127.0.0.1"

VR_DATA_PORT            = 5555
LEFT_ARM_CMD_PORT       = 5556
RIGHT_ARM_CMD_PORT      = 5557
LEFT_ARM_FEEDBACK_PORT  = 5558
RIGHT_ARM_FEEDBACK_PORT = 5559


class Publisher:
    """TCP server — fans each published message to all connected subscribers."""

    def __init__(self, port: int, host: str = HOST) -> None:
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._running = True

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(16)
        srv.settimeout(1.0)
        self._srv = srv

        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with self._lock:
                    self._clients.append(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def publish(self, msg: dict) -> None:
        frame = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        dead: list[socket.socket] = []
        with self._lock:
            for c in list(self._clients):
                try:
                    c.sendall(frame)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)

    def close(self) -> None:
        self._running = False
        self._srv.close()


class Subscriber:
    """TCP client — connects to a Publisher and yields decoded messages."""

    def __init__(self, port: int, host: str = HOST) -> None:
        self._host, self._port = host, port
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self._connect()

    def _connect(self) -> None:
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.connect((self._host, self._port))
                self._sock, self._buf = s, b""
                return
            except OSError:
                time.sleep(0.25)

    def iter_messages(self) -> Iterator[dict]:
        while True:
            try:
                chunk = self._sock.recv(8192)
            except OSError:
                self._connect()
                continue
            if not chunk:
                self._connect()
                continue
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass

    def close(self) -> None:
        if self._sock:
            self._sock.close()


class CommandServer:
    """TCP server — queues JSON commands from any number of connected senders."""

    def __init__(self, port: int, host: str = HOST) -> None:
        self._q: Queue[dict] = Queue(maxsize=128)
        self._running = True

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((host, port))
        srv.listen(8)
        srv.settimeout(1.0)
        self._srv = srv

        threading.Thread(target=self._accept_loop, daemon=True).start()

    def _accept_loop(self) -> None:
        while self._running:
            try:
                conn, _ = self._srv.accept()
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                threading.Thread(
                    target=self._read_conn, args=(conn,), daemon=True
                ).start()
            except socket.timeout:
                continue
            except OSError:
                break

    def _read_conn(self, conn: socket.socket) -> None:
        buf = b""
        try:
            while self._running:
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                    try:
                        if not self._q.full():
                            self._q.put_nowait(json.loads(line))
                    except (json.JSONDecodeError, Exception):
                        pass
        finally:
            conn.close()

    def iter_commands(self) -> Iterator[dict]:
        """Blocking iterator — yields commands as they arrive."""
        while self._running:
            try:
                yield self._q.get(timeout=0.1)
            except Empty:
                pass

    def close(self) -> None:
        self._running = False
        self._srv.close()


class CommandClient:
    """TCP client — sends JSON commands to a CommandServer (thread-safe)."""

    def __init__(self, port: int, host: str = HOST, max_tries: int = 40) -> None:
        self._host, self._port = host, port
        self._max_tries = max_tries
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        for _ in range(self._max_tries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(2.0)
                s.connect((self._host, self._port))
                s.settimeout(None)
                self._sock = s
                return
            except OSError:
                time.sleep(0.25)
        raise ConnectionError(
            f"Cannot connect to {self._host}:{self._port} after {self._max_tries} tries"
        )

    def send(self, msg: dict) -> None:
        frame = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._lock:
            if self._sock is None:
                self._connect()
            try:
                self._sock.sendall(frame)
            except OSError:
                self._sock = None
                self._connect()
                self._sock.sendall(frame)

    def close(self) -> None:
        with self._lock:
            if self._sock:
                self._sock.close()
                self._sock = None
