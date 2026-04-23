"""Tests for RunOutputStore."""

import asyncio

import pytest

from app.core.run_output_store import RunOutputStore


@pytest.mark.asyncio
async def test_put_and_get_roundtrip():
    store = RunOutputStore(max_runs=5)
    await store.put("r1", "n1", "output", [1, 2, 3])
    assert await store.get("r1", "n1", "output") == [1, 2, 3]


@pytest.mark.asyncio
async def test_missing_returns_none():
    store = RunOutputStore()
    assert await store.get("r1", "n1", "output") is None


@pytest.mark.asyncio
async def test_has_run():
    store = RunOutputStore()
    assert await store.has_run("r1") is False
    await store.put("r1", "n1", "p", 1)
    assert await store.has_run("r1") is True


@pytest.mark.asyncio
async def test_multiple_nodes_and_ports():
    store = RunOutputStore()
    await store.put("r1", "n1", "out1", 1)
    await store.put("r1", "n1", "out2", 2)
    await store.put("r1", "n2", "out1", 3)
    assert await store.get("r1", "n1", "out1") == 1
    assert await store.get("r1", "n1", "out2") == 2
    assert await store.get("r1", "n2", "out1") == 3


@pytest.mark.asyncio
async def test_list_ports():
    store = RunOutputStore()
    await store.put("r1", "n1", "a", 1)
    await store.put("r1", "n1", "b", 2)
    await store.put("r1", "n2", "c", 3)
    ports = await store.list_ports("r1")
    assert ports is not None
    assert set(ports) == {("n1", "a"), ("n1", "b"), ("n2", "c")}


@pytest.mark.asyncio
async def test_list_ports_unknown_run():
    store = RunOutputStore()
    assert await store.list_ports("missing") is None


@pytest.mark.asyncio
async def test_delete_run():
    store = RunOutputStore()
    await store.put("r1", "n1", "p", 1)
    assert await store.delete_run("r1") is True
    assert await store.has_run("r1") is False
    assert await store.delete_run("r1") is False


@pytest.mark.asyncio
async def test_lru_eviction_keeps_newest():
    store = RunOutputStore(max_runs=3)
    for i in range(5):
        await store.put(f"r{i}", "n", "p", i)
    runs = await store.list_runs()
    assert runs == ["r2", "r3", "r4"]
    assert await store.has_run("r0") is False
    assert await store.has_run("r1") is False
    assert await store.has_run("r2") is True
    assert await store.has_run("r4") is True


@pytest.mark.asyncio
async def test_same_run_id_does_not_evict_others():
    store = RunOutputStore(max_runs=2)
    await store.put("r1", "n", "p", 1)
    await store.put("r2", "n", "p", 2)
    # Re-putting into r1 should NOT move it in LRU order or evict r2
    await store.put("r1", "n", "q", 11)
    runs = await store.list_runs()
    assert set(runs) == {"r1", "r2"}
    assert await store.get("r1", "n", "q") == 11


@pytest.mark.asyncio
async def test_concurrent_puts():
    store = RunOutputStore(max_runs=100)

    async def worker(run_id: str, count: int):
        for i in range(count):
            await store.put(run_id, f"node{i}", "port", i)

    await asyncio.gather(
        *[worker(f"r{i}", 10) for i in range(10)]
    )
    for i in range(10):
        assert await store.has_run(f"r{i}")
        ports = await store.list_ports(f"r{i}")
        assert ports is not None
        assert len(ports) == 10


@pytest.mark.asyncio
async def test_clear():
    store = RunOutputStore()
    await store.put("r1", "n", "p", 1)
    await store.put("r2", "n", "p", 2)
    await store.clear()
    assert await store.list_runs() == []
    assert await store.has_run("r1") is False
