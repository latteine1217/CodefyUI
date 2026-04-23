"""In-memory LRU store for captured per-run node outputs.

Scope:
  Outputs live only in server memory; they are lost on restart. Indexed
  by (run_id, node_id, port). When the number of tracked runs exceeds
  ``max_runs``, the oldest run is evicted.

Thread safety:
  All mutating operations take an ``asyncio.Lock`` so concurrent WebSocket
  runs don't corrupt the internal dict.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any


class RunOutputStore:
    def __init__(self, max_runs: int = 20) -> None:
        self._max_runs = max_runs
        self._store: dict[str, dict[str, dict[str, Any]]] = {}
        self._order: deque[str] = deque()
        self._lock = asyncio.Lock()

    async def put(self, run_id: str, node_id: str, port: str, value: Any) -> None:
        async with self._lock:
            if run_id not in self._store:
                self._store[run_id] = {}
                self._order.append(run_id)
                while len(self._order) > self._max_runs:
                    oldest = self._order.popleft()
                    self._store.pop(oldest, None)
            nodes = self._store[run_id]
            if node_id not in nodes:
                nodes[node_id] = {}
            nodes[node_id][port] = value

    async def get(self, run_id: str, node_id: str, port: str) -> Any | None:
        async with self._lock:
            return self._store.get(run_id, {}).get(node_id, {}).get(port)

    async def has_run(self, run_id: str) -> bool:
        async with self._lock:
            return run_id in self._store

    async def delete_run(self, run_id: str) -> bool:
        async with self._lock:
            if run_id not in self._store:
                return False
            del self._store[run_id]
            try:
                self._order.remove(run_id)
            except ValueError:
                pass
            return True

    async def list_ports(self, run_id: str) -> list[tuple[str, str]] | None:
        """Return (node_id, port) pairs for a run, or None if run not found."""
        async with self._lock:
            run = self._store.get(run_id)
            if run is None:
                return None
            return [
                (node_id, port)
                for node_id, ports in run.items()
                for port in ports.keys()
            ]

    async def list_runs(self) -> list[str]:
        """Return run_ids in LRU order (oldest first)."""
        async with self._lock:
            return list(self._order)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            self._order.clear()
