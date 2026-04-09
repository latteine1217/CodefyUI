# HuggingFace + Kaggle Dataset Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new data-loading nodes (`HuggingFaceDataset`, `KaggleDataset`) to CodefyUI so users can load image-classification datasets from HuggingFace Hub and Kaggle, slotting into the existing `Transform` → `DataLoader` → CNN training flow with no changes to those nodes.

**Architecture:** Two BaseNode subclasses in `backend/app/nodes/data/`, plus a small internal helper module (`_hf_adapter.py`) that exposes `HFTorchImageDataset` — a `torch.utils.data.Dataset` subclass that wraps a HuggingFace `datasets.Dataset` and mirrors the torchvision Dataset convention (public `transform` attribute). Both nodes lazy-import their optional packages inside `execute()`. Credentials live in environment variables only (never on the node) so `graph.json` files stay safe to commit. Auto-discovery picks up the new nodes without registry edits.

**Tech Stack:** Python 3.11, PyTorch, torchvision, HuggingFace `datasets>=2.14`, `kagglehub>=0.3`, FastAPI (existing), pytest (existing). Backend dep manager: `uv`.

**Spec:** [docs/superpowers/specs/2026-04-10-dataset-extension-huggingface-kaggle-design.md](../specs/2026-04-10-dataset-extension-huggingface-kaggle-design.md)

---

## File Structure

| Path | Status | Responsibility |
|---|---|---|
| `backend/pyproject.toml` | modify | Add `datasets>=2.14` and `kagglehub>=0.3` to `[project.optional-dependencies].ml` |
| `backend/app/nodes/data/_hf_adapter.py` | create | Defines `HFTorchImageDataset(torch.utils.data.Dataset)` — wraps a HF dataset, exposes mutable `transform` attribute. Internal helper, not a node. |
| `backend/app/nodes/data/huggingface_dataset_node.py` | create | `HuggingFaceDatasetNode(BaseNode)` — calls `datasets.load_dataset`, wraps in `HFTorchImageDataset`, validates columns, maps auth errors |
| `backend/app/nodes/data/kaggle_dataset_node.py` | create | `KaggleDatasetNode(BaseNode)` — checks Kaggle auth, calls `kagglehub.dataset_download`, loads as `torchvision.datasets.ImageFolder`, validates folder layout |
| `backend/tests/test_dataset_extension.py` | create | All unit tests for both new nodes plus the adapter; uses monkeypatch + tmp_path, no live network |
| `frontend/src/i18n/nodeLocales/zh-TW.ts` | modify | Add Chinese translations for both new nodes' descriptions and params |

The two new node files are independent and have no shared logic (HuggingFace and Kaggle work very differently). The shared `_hf_adapter.py` only serves the HuggingFace path and is reusable for a future `HuggingFaceTextDataset` node.

---

## Task 1: Add optional dependencies

**Files:**
- Modify: `backend/pyproject.toml`

- [ ] **Step 1: Read current `[project.optional-dependencies]` block**

Run: `Read backend/pyproject.toml`
Expected: see the `ml = [...]` list with torch, torchvision, gymnasium, safetensors

- [ ] **Step 2: Add the two new packages**

Edit `backend/pyproject.toml`. Replace:

```toml
ml = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "gymnasium>=0.29.0",
    "safetensors>=0.4.0",
]
```

with:

```toml
ml = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "gymnasium>=0.29.0",
    "safetensors>=0.4.0",
    "datasets>=2.14",
    "kagglehub>=0.3",
]
```

- [ ] **Step 3: Install the new packages into the existing venv**

Run from `backend/`:

```bash
uv sync --extra ml
```

Expected: uv resolves and installs `datasets`, `kagglehub`, and any transitive deps. No errors.

- [ ] **Step 4: Smoke-test the imports**

Run from `backend/`:

```bash
uv run python -c "import datasets, kagglehub; print(datasets.__version__, kagglehub.__version__)"
```

Expected: prints two version strings, no traceback.

- [ ] **Step 5: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "deps: add datasets and kagglehub to ml extras"
```

---

## Task 2: Test scaffolding & shared helpers

This task creates the test file with the helper utilities used by every later test. No production code yet — but the test file must run cleanly under pytest.

**Files:**
- Create: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Create the test file with helpers and a smoke test**

Create `backend/tests/test_dataset_extension.py`:

```python
"""Unit tests for HuggingFaceDataset and KaggleDataset nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class FakeHFDataset:
    """Minimal stand-in for huggingface `datasets.Dataset`.

    Implements the subset of the API our node uses: __len__, __getitem__,
    and a `features` attribute that exposes column names as dict keys.
    """

    def __init__(self, rows: list[dict[str, Any]], features: dict[str, Any]):
        self._rows = rows
        self.features = features

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._rows[idx]


def _make_image_folder_layout(
    root: Path,
    classes: dict[str, int] | None = None,
) -> None:
    """Create a tiny ImageFolder layout under `root`.

    Default layout: two classes ('cats', 'dogs') with one 4x4 PNG each.
    """
    if classes is None:
        classes = {"cats": 1, "dogs": 1}
    for class_name, count in classes.items():
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = Image.new("RGB", (4, 4), (i * 30 % 255, 0, 0))
            img.save(class_dir / f"{class_name}_{i}.png")


def _make_two_row_hf_image_dataset() -> FakeHFDataset:
    return FakeHFDataset(
        rows=[
            {"image": Image.new("RGB", (8, 8), (255, 0, 0)), "label": 0},
            {"image": Image.new("RGB", (8, 8), (0, 255, 0)), "label": 1},
        ],
        features={"image": None, "label": None},
    )


# ---------------------------------------------------------------------------
# Smoke test (proves the file is collected and helpers work)
# ---------------------------------------------------------------------------


def test_helpers_smoke(tmp_path):
    fake = _make_two_row_hf_image_dataset()
    assert len(fake) == 2
    assert fake[0]["label"] == 0

    _make_image_folder_layout(tmp_path)
    assert (tmp_path / "cats").is_dir()
    assert (tmp_path / "dogs").is_dir()
    assert any(p.suffix == ".png" for p in (tmp_path / "cats").iterdir())
```

- [ ] **Step 2: Run the smoke test**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: `1 passed`. The file is collectable, helpers work.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_dataset_extension.py
git commit -m "test: scaffold dataset extension test file with helpers"
```

---

## Task 3: `HFTorchImageDataset` adapter

The adapter is the bridge between HF's `datasets.Dataset` and PyTorch's `DataLoader`. It must (a) implement `__len__` and `__getitem__`, (b) expose a public mutable `transform` attribute so the existing `TransformNode` can swap it after construction.

**Files:**
- Create: `backend/app/nodes/data/_hf_adapter.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append failing tests to the test file**

Add at the bottom of `backend/tests/test_dataset_extension.py`:

```python
# ---------------------------------------------------------------------------
# HFTorchImageDataset adapter
# ---------------------------------------------------------------------------


def test_hf_adapter_returns_pil_when_no_transform():
    from app.nodes.data._hf_adapter import HFTorchImageDataset

    fake = _make_two_row_hf_image_dataset()
    ds = HFTorchImageDataset(fake, image_column="image", label_column="label")

    assert len(ds) == 2
    img, label = ds[0]
    assert isinstance(img, Image.Image)
    assert label == 0
    assert ds[1][1] == 1


def test_hf_adapter_applies_transform_in_constructor():
    import torch
    from torchvision import transforms

    from app.nodes.data._hf_adapter import HFTorchImageDataset

    fake = _make_two_row_hf_image_dataset()
    ds = HFTorchImageDataset(
        fake,
        image_column="image",
        label_column="label",
        transform=transforms.ToTensor(),
    )

    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 8, 8)
    assert label == 0


def test_hf_adapter_transform_is_mutable_after_init():
    """TransformNode interop: setting `dataset.transform` later must take effect."""
    from app.nodes.data._hf_adapter import HFTorchImageDataset

    fake = _make_two_row_hf_image_dataset()
    ds = HFTorchImageDataset(fake, image_column="image", label_column="label")

    sentinel = object()
    ds.transform = lambda _img: sentinel
    img, _ = ds[0]
    assert img is sentinel
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: the three new `test_hf_adapter_*` tests fail with `ModuleNotFoundError: No module named 'app.nodes.data._hf_adapter'`. The smoke test still passes.

- [ ] **Step 3: Create the adapter module**

Create `backend/app/nodes/data/_hf_adapter.py`:

```python
"""Internal helper: wraps a HuggingFace `datasets.Dataset` as a torch Dataset.

This is a private module (leading underscore). It is reusable from any
HuggingFace-backed node — currently `HuggingFaceDatasetNode`, and a future
`HuggingFaceTextDataset` node could share the same shape with a different
column convention.
"""

from __future__ import annotations

from typing import Any, Callable

from torch.utils.data import Dataset


class HFTorchImageDataset(Dataset):
    """Adapt a HuggingFace `datasets.Dataset` to the torchvision Dataset convention.

    The class deliberately mirrors how torchvision's built-in datasets behave:
    `transform` is a public, mutable attribute that downstream nodes
    (e.g. `TransformNode`) may replace at any time.
    """

    def __init__(
        self,
        hf_dataset: Any,
        image_column: str,
        label_column: str,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        self._ds = hf_dataset
        self._image_col = image_column
        self._label_col = label_column
        self.transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        row = self._ds[idx]
        image = row[self._image_col]
        label = row[self._label_col]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
```

- [ ] **Step 4: Run the adapter tests and confirm they pass**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: 4 tests passing (1 smoke + 3 adapter tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/_hf_adapter.py backend/tests/test_dataset_extension.py
git commit -m "feat: add HFTorchImageDataset adapter for HuggingFace datasets"
```

---

## Task 4: `HuggingFaceDatasetNode` — happy path

Build the node with the smallest possible `execute()` that makes the happy-path test pass. Validation and error mapping come in later tasks.

**Files:**
- Create: `backend/app/nodes/data/huggingface_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the happy-path test**

Add to the bottom of `backend/tests/test_dataset_extension.py`:

```python
# ---------------------------------------------------------------------------
# HuggingFaceDatasetNode
# ---------------------------------------------------------------------------


def _hf_node_default_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "dataset_name": "fake/dataset",
        "subset": "",
        "split": "train",
        "image_column": "image",
        "label_column": "label",
        "cache_dir": "",
    }
    base.update(overrides)
    return base


def test_hf_dataset_node_happy_path(monkeypatch):
    import torch
    import datasets as hf_datasets

    fake = _make_two_row_hf_image_dataset()

    def fake_load_dataset(name, subset=None, split=None, cache_dir=None):
        assert name == "fake/dataset"
        assert subset is None
        assert split == "train"
        assert cache_dir is None
        return fake

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    result = node.execute(inputs={}, params=_hf_node_default_params())

    ds = result["dataset"]
    assert len(ds) == 2
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)  # default ToTensor() applied by node
    assert img.shape == (3, 8, 8)
    assert label == 0
```

- [ ] **Step 2: Run the test and confirm it fails**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_hf_dataset_node_happy_path -v
```

Expected: `ModuleNotFoundError: No module named 'app.nodes.data.huggingface_dataset_node'`.

- [ ] **Step 3: Create the node file**

Create `backend/app/nodes/data/huggingface_dataset_node.py`:

```python
"""HuggingFaceDataset node — load image-classification datasets from HF Hub."""

from __future__ import annotations

from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class HuggingFaceDatasetNode(BaseNode):
    NODE_NAME = "HuggingFaceDataset"
    CATEGORY = "Data"
    DESCRIPTION = "Load an image-classification dataset from HuggingFace Hub"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Dataset wrapped as a torch.utils.data.Dataset",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dataset_name",
                param_type=ParamType.STRING,
                default="ylecun/mnist",
                description="HuggingFace Hub repo id (e.g. cifar10, uoft-cs/cifar100)",
            ),
            ParamDefinition(
                name="subset",
                param_type=ParamType.STRING,
                default="",
                description="Config name for multi-config datasets (empty = no subset)",
            ),
            ParamDefinition(
                name="split",
                param_type=ParamType.STRING,
                default="train",
                description="Split: train/test/validation, or HF slice syntax (train[:1000])",
            ),
            ParamDefinition(
                name="image_column",
                param_type=ParamType.STRING,
                default="image",
                description="Column name for the image feature",
            ),
            ParamDefinition(
                name="label_column",
                param_type=ParamType.STRING,
                default="label",
                description="Column name for the integer label",
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override HF cache directory (empty = HF default)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        from datasets import load_dataset
        from torchvision import transforms as T

        from ._hf_adapter import HFTorchImageDataset

        dataset_name = params.get("dataset_name", "ylecun/mnist")
        subset = params.get("subset", "") or None
        split = params.get("split", "train")
        image_column = params.get("image_column", "image")
        label_column = params.get("label_column", "label")
        cache_dir = params.get("cache_dir", "") or None

        ds = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)

        wrapped = HFTorchImageDataset(
            ds,
            image_column=image_column,
            label_column=label_column,
            transform=T.ToTensor(),
        )
        return {"dataset": wrapped}
```

- [ ] **Step 4: Run the test and confirm it passes**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_hf_dataset_node_happy_path -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/huggingface_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: add HuggingFaceDatasetNode (happy path)"
```

---

## Task 5: `HuggingFaceDatasetNode` — column validation

If the user types the wrong column name, give them a list of available columns instead of a cryptic KeyError from deep in the adapter.

**Files:**
- Modify: `backend/app/nodes/data/huggingface_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the failing test**

Add to `backend/tests/test_dataset_extension.py`:

```python
def test_hf_dataset_node_invalid_image_column_lists_available(monkeypatch):
    import datasets as hf_datasets

    fake = FakeHFDataset(
        rows=[{"img": Image.new("RGB", (8, 8)), "label": 0}],
        features={"img": None, "label": None},
    )
    monkeypatch.setattr(hf_datasets, "load_dataset", lambda *a, **k: fake)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(
            inputs={},
            params=_hf_node_default_params(image_column="image"),  # dataset has 'img'
        )

    msg = str(exc_info.value)
    assert "image" in msg
    assert "img" in msg  # available columns are listed
    assert "label" in msg


def test_hf_dataset_node_invalid_label_column_lists_available(monkeypatch):
    import datasets as hf_datasets

    fake = FakeHFDataset(
        rows=[{"image": Image.new("RGB", (8, 8)), "target": 0}],
        features={"image": None, "target": None},
    )
    monkeypatch.setattr(hf_datasets, "load_dataset", lambda *a, **k: fake)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(
            inputs={},
            params=_hf_node_default_params(label_column="label"),  # dataset has 'target'
        )

    msg = str(exc_info.value)
    assert "label" in msg
    assert "target" in msg
```

- [ ] **Step 2: Run the new tests and confirm they fail**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v -k invalid
```

Expected: both new tests fail (probably with `KeyError` from inside the adapter when `__getitem__` runs, OR they might not raise at all if the test doesn't trigger getitem — in that case the tests just fail because no exception is raised).

- [ ] **Step 3: Add column validation to the node**

Edit `backend/app/nodes/data/huggingface_dataset_node.py`. Replace the section starting with `ds = load_dataset(...)` and ending with `return {"dataset": wrapped}` (the body of `execute` after the param parsing) with:

```python
        ds = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)

        available = list(ds.features.keys()) if hasattr(ds, "features") else None
        if available is not None:
            if image_column not in available:
                raise RuntimeError(
                    f"Column '{image_column}' not found in dataset "
                    f"'{dataset_name}'. Available columns: {available}"
                )
            if label_column not in available:
                raise RuntimeError(
                    f"Column '{label_column}' not found in dataset "
                    f"'{dataset_name}'. Available columns: {available}"
                )

        wrapped = HFTorchImageDataset(
            ds,
            image_column=image_column,
            label_column=label_column,
            transform=T.ToTensor(),
        )
        return {"dataset": wrapped}
```

- [ ] **Step 4: Run the tests and confirm everything passes**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: all tests so far pass (smoke + 3 adapter + happy path + 2 invalid-column).

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/huggingface_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: validate column names in HuggingFaceDatasetNode"
```

---

## Task 6: `HuggingFaceDatasetNode` — missing-package error

When `datasets` isn't installed, the user should see "install with: pip install datasets", not a raw `ModuleNotFoundError`.

**Files:**
- Modify: `backend/app/nodes/data/huggingface_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the failing test**

Add to `backend/tests/test_dataset_extension.py`:

```python
def test_hf_dataset_node_missing_package(monkeypatch):
    import builtins
    import sys

    # Force any future `import datasets` to fail.
    monkeypatch.delitem(sys.modules, "datasets", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("No module named 'datasets'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_hf_node_default_params())

    msg = str(exc_info.value)
    assert "datasets" in msg
    assert "pip install" in msg
```

- [ ] **Step 2: Run the test and confirm it fails**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_hf_dataset_node_missing_package -v
```

Expected: `Failed: DID NOT RAISE <class 'RuntimeError'>` OR raises `ImportError` instead of `RuntimeError`.

- [ ] **Step 3: Wrap the import in try/except**

Edit `backend/app/nodes/data/huggingface_dataset_node.py`. In `execute()`, replace:

```python
        from datasets import load_dataset
        from torchvision import transforms as T

        from ._hf_adapter import HFTorchImageDataset
```

with:

```python
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(
                "HuggingFaceDataset requires the 'datasets' package. "
                "Install with: pip install datasets "
                "(or `uv sync --extra ml` from backend/)"
            ) from e

        from torchvision import transforms as T

        from ._hf_adapter import HFTorchImageDataset
```

- [ ] **Step 4: Run the test and confirm it passes**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/huggingface_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: clear error when 'datasets' package is missing in HuggingFaceDatasetNode"
```

---

## Task 7: `HuggingFaceDatasetNode` — auth error mapping

When loading a gated/private repo without `HF_TOKEN`, HuggingFace raises a `GatedRepoError` (or HTTP 401). The node should map this to "set HF_TOKEN" guidance.

We detect the auth class by **type name** rather than importing the real class — this avoids version-pinning fragility (the exception path has moved between huggingface_hub releases).

**Files:**
- Modify: `backend/app/nodes/data/huggingface_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the failing test**

Add to `backend/tests/test_dataset_extension.py`:

```python
def test_hf_dataset_node_gated_repo_maps_to_token_message(monkeypatch):
    import datasets as hf_datasets

    class GatedRepoError(Exception):
        """Stand-in mimicking huggingface_hub.errors.GatedRepoError."""

    def fake_load_dataset(*args, **kwargs):
        raise GatedRepoError("Access to model X is gated")

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(
            inputs={},
            params=_hf_node_default_params(dataset_name="private/gated"),
        )

    msg = str(exc_info.value)
    assert "HF_TOKEN" in msg
    assert "authentication" in msg.lower() or "auth" in msg.lower()


def test_hf_dataset_node_401_error_maps_to_token_message(monkeypatch):
    import datasets as hf_datasets

    def fake_load_dataset(*args, **kwargs):
        raise Exception("401 Client Error: Unauthorized")

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_hf_node_default_params())

    msg = str(exc_info.value)
    assert "HF_TOKEN" in msg


def test_hf_dataset_node_other_errors_pass_through(monkeypatch):
    """Non-auth errors must NOT be silently rewritten as auth errors."""
    import datasets as hf_datasets

    def fake_load_dataset(*args, **kwargs):
        raise ValueError("dataset name is malformed")

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)

    from app.nodes.data.huggingface_dataset_node import HuggingFaceDatasetNode

    node = HuggingFaceDatasetNode()
    with pytest.raises(ValueError, match="malformed"):
        node.execute(inputs={}, params=_hf_node_default_params())
```

- [ ] **Step 2: Run the tests and confirm two fail**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v -k "gated_repo or 401_error or other_errors"
```

Expected: the two auth-mapping tests fail (the original exception passes through). The "other_errors" test already passes (ValueError bubbles up unchanged).

- [ ] **Step 3: Wrap `load_dataset` in an auth-detecting try/except**

Edit `backend/app/nodes/data/huggingface_dataset_node.py`. Replace the line:

```python
        ds = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
```

with:

```python
        try:
            ds = load_dataset(dataset_name, subset, split=split, cache_dir=cache_dir)
        except Exception as e:
            err_name = type(e).__name__
            err_msg = str(e)
            looks_like_auth = (
                "GatedRepoError" in err_name
                or "RepositoryNotFoundError" in err_name
                or "401" in err_msg
                or "unauthorized" in err_msg.lower()
            )
            if looks_like_auth:
                raise RuntimeError(
                    "HuggingFace authentication required to load "
                    f"'{dataset_name}'. Set the HF_TOKEN environment "
                    "variable to a token with read access. "
                    "See https://huggingface.co/docs/hub/security-tokens"
                ) from e
            raise
```

- [ ] **Step 4: Run the full test file**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/huggingface_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: map HuggingFace auth errors to HF_TOKEN guidance"
```

---

## Task 8: `KaggleDatasetNode` — happy path

Build the node with the smallest `execute()` that loads a Kaggle dataset as `ImageFolder`. Auth check, subdir, and error handling come in later tasks.

**Files:**
- Create: `backend/app/nodes/data/kaggle_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the happy-path test**

Add to `backend/tests/test_dataset_extension.py`:

```python
# ---------------------------------------------------------------------------
# KaggleDatasetNode
# ---------------------------------------------------------------------------


def _kaggle_node_default_params(**overrides: Any) -> dict[str, Any]:
    base = {
        "dataset_slug": "fake/dataset",
        "subdir": "",
        "cache_dir": "",
    }
    base.update(overrides)
    return base


def test_kaggle_dataset_node_happy_path(monkeypatch, tmp_path):
    import torch
    import kagglehub

    _make_image_folder_layout(tmp_path)
    monkeypatch.setattr(kagglehub, "dataset_download", lambda slug: str(tmp_path))

    # Auth pre-check requires either env vars or kaggle.json
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    result = node.execute(inputs={}, params=_kaggle_node_default_params())

    ds = result["dataset"]
    assert len(ds) == 2  # cats + dogs, one image each
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert label in (0, 1)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_kaggle_dataset_node_happy_path -v
```

Expected: `ModuleNotFoundError: No module named 'app.nodes.data.kaggle_dataset_node'`.

- [ ] **Step 3: Create the node file**

Create `backend/app/nodes/data/kaggle_dataset_node.py`:

```python
"""KaggleDataset node — download and load a Kaggle dataset as an ImageFolder."""

from __future__ import annotations

import os
from typing import Any

from ...core.node_base import (
    BaseNode,
    DataType,
    ParamDefinition,
    ParamType,
    PortDefinition,
)


class KaggleDatasetNode(BaseNode):
    NODE_NAME = "KaggleDataset"
    CATEGORY = "Data"
    DESCRIPTION = "Download a Kaggle dataset and load it as an ImageFolder"

    @classmethod
    def define_inputs(cls) -> list[PortDefinition]:
        return []

    @classmethod
    def define_outputs(cls) -> list[PortDefinition]:
        return [
            PortDefinition(
                name="dataset",
                data_type=DataType.DATASET,
                description="Dataset loaded as torchvision ImageFolder",
            ),
        ]

    @classmethod
    def define_params(cls) -> list[ParamDefinition]:
        return [
            ParamDefinition(
                name="dataset_slug",
                param_type=ParamType.STRING,
                default="",
                description="Kaggle dataset slug in 'owner/name' form",
            ),
            ParamDefinition(
                name="subdir",
                param_type=ParamType.STRING,
                default="",
                description="Relative path inside the downloaded dataset where ImageFolder structure begins",
            ),
            ParamDefinition(
                name="cache_dir",
                param_type=ParamType.STRING,
                default="",
                description="Override kagglehub cache directory (empty = default)",
            ),
        ]

    def execute(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        import kagglehub
        from torchvision import transforms as T
        from torchvision.datasets import ImageFolder

        dataset_slug = params.get("dataset_slug", "")
        subdir = params.get("subdir", "")
        cache_dir = params.get("cache_dir", "")

        if cache_dir:
            os.environ["KAGGLEHUB_CACHE"] = cache_dir

        download_path = kagglehub.dataset_download(dataset_slug)
        root = os.path.join(download_path, subdir) if subdir else download_path

        return {"dataset": ImageFolder(root, transform=T.ToTensor())}
```

- [ ] **Step 4: Run the test and confirm it passes**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_kaggle_dataset_node_happy_path -v
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/kaggle_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: add KaggleDatasetNode (happy path)"
```

---

## Task 9: `KaggleDatasetNode` — `subdir` parameter

The `subdir` param exists from Task 8 but is untested. This task adds the test that exercises it (and confirms the happy-path code already handles it correctly).

**Files:**
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the subdir test**

Add to `backend/tests/test_dataset_extension.py`:

```python
def test_kaggle_dataset_node_uses_subdir(monkeypatch, tmp_path):
    import torch
    import kagglehub

    # Layout: tmp_path/train/<class>/<image>.png
    _make_image_folder_layout(tmp_path / "train")
    # Distractor at the top level (no class structure)
    (tmp_path / "README.md").write_text("not an image folder")

    monkeypatch.setattr(kagglehub, "dataset_download", lambda slug: str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    result = node.execute(
        inputs={},
        params=_kaggle_node_default_params(subdir="train"),
    )

    ds = result["dataset"]
    assert len(ds) == 2
    img, _label = ds[0]
    assert isinstance(img, torch.Tensor)
```

- [ ] **Step 2: Run the test and confirm it passes**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_kaggle_dataset_node_uses_subdir -v
```

Expected: `1 passed` (the implementation from Task 8 already supports this).

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_dataset_extension.py
git commit -m "test: cover KaggleDatasetNode subdir parameter"
```

---

## Task 10: `KaggleDatasetNode` — missing-package error

**Files:**
- Modify: `backend/app/nodes/data/kaggle_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the failing test**

Add to `backend/tests/test_dataset_extension.py`:

```python
def test_kaggle_dataset_node_missing_package(monkeypatch):
    import builtins
    import sys

    monkeypatch.delitem(sys.modules, "kagglehub", raising=False)

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "kagglehub" or name.startswith("kagglehub."):
            raise ImportError("No module named 'kagglehub'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_kaggle_node_default_params())

    msg = str(exc_info.value)
    assert "kagglehub" in msg
    assert "pip install" in msg
```

- [ ] **Step 2: Run the test and confirm it fails**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_kaggle_dataset_node_missing_package -v
```

Expected: `ImportError` propagates instead of `RuntimeError` — test fails.

- [ ] **Step 3: Wrap the kagglehub import**

Edit `backend/app/nodes/data/kaggle_dataset_node.py`. In `execute()`, replace:

```python
        import kagglehub
        from torchvision import transforms as T
        from torchvision.datasets import ImageFolder
```

with:

```python
        try:
            import kagglehub
        except ImportError as e:
            raise RuntimeError(
                "KaggleDataset requires the 'kagglehub' package. "
                "Install with: pip install kagglehub "
                "(or `uv sync --extra ml` from backend/)"
            ) from e

        from torchvision import transforms as T
        from torchvision.datasets import ImageFolder
```

- [ ] **Step 4: Run the test and confirm it passes**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/kaggle_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: clear error when 'kagglehub' package is missing"
```

---

## Task 11: `KaggleDatasetNode` — missing-auth pre-check

Detect missing credentials BEFORE calling `kagglehub.dataset_download` so the user gets actionable guidance instead of whatever cryptic error kagglehub would surface.

**Files:**
- Modify: `backend/app/nodes/data/kaggle_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the failing test**

Add to `backend/tests/test_dataset_extension.py`:

```python
def test_kaggle_dataset_node_missing_auth(monkeypatch, tmp_path):
    import kagglehub

    # Make sure neither env vars nor ~/.kaggle/kaggle.json exist
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # Even if dataset_download were called, fail loudly so the test catches it
    def should_not_be_called(slug):
        raise AssertionError("dataset_download was called despite missing auth")

    monkeypatch.setattr(kagglehub, "dataset_download", should_not_be_called)

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_kaggle_node_default_params())

    msg = str(exc_info.value)
    assert "Kaggle" in msg
    assert ("KAGGLE_USERNAME" in msg) or ("kaggle.json" in msg)
```

- [ ] **Step 2: Run the test and confirm it fails**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_kaggle_dataset_node_missing_auth -v
```

Expected: the test fails — either with `AssertionError` (because dataset_download was called) or with whatever exception kagglehub raises.

- [ ] **Step 3: Add an auth pre-check**

Edit `backend/app/nodes/data/kaggle_dataset_node.py`. Add `from pathlib import Path` to the imports near the top of the file (after `import os`):

```python
import os
from pathlib import Path
from typing import Any
```

Then in `execute()`, immediately after the kagglehub import block, add the pre-check before `if cache_dir:`:

```python
        if not _kaggle_credentials_present():
            raise RuntimeError(
                "Kaggle authentication required. Set KAGGLE_USERNAME and "
                "KAGGLE_KEY environment variables, or place kaggle.json at "
                "~/.kaggle/kaggle.json. See https://www.kaggle.com/docs/api"
            )
```

Then add a module-level helper function above the class definition (after the imports, before `class KaggleDatasetNode`):

```python
def _kaggle_credentials_present() -> bool:
    """True if either env-var creds or ~/.kaggle/kaggle.json is present."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    if (Path.home() / ".kaggle" / "kaggle.json").exists():
        return True
    return False
```

- [ ] **Step 4: Run the full test file**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: all tests pass. (The earlier kaggle happy-path tests already set the env vars, so they continue to pass the new pre-check.)

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/kaggle_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: pre-check Kaggle credentials with actionable error"
```

---

## Task 12: `KaggleDatasetNode` — invalid ImageFolder structure

If the user picks the wrong `subdir` (or none when one is needed), `ImageFolder` will raise a confusing FileNotFoundError. We catch the case earlier with a directory listing check.

**Files:**
- Modify: `backend/app/nodes/data/kaggle_dataset_node.py`
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the failing test**

Add to `backend/tests/test_dataset_extension.py`:

```python
def test_kaggle_dataset_node_flat_directory_raises_with_subdir_hint(monkeypatch, tmp_path):
    import kagglehub

    # Flat: just files, no class subdirs
    (tmp_path / "image1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "image2.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (tmp_path / "metadata.csv").write_text("id,label\n")

    monkeypatch.setattr(kagglehub, "dataset_download", lambda slug: str(tmp_path))
    monkeypatch.setenv("KAGGLE_USERNAME", "tester")
    monkeypatch.setenv("KAGGLE_KEY", "testkey")

    from app.nodes.data.kaggle_dataset_node import KaggleDatasetNode

    node = KaggleDatasetNode()
    with pytest.raises(RuntimeError) as exc_info:
        node.execute(inputs={}, params=_kaggle_node_default_params())

    msg = str(exc_info.value)
    assert "subdir" in msg.lower()
    # The error should surface what was actually in the directory
    assert "image1.png" in msg or "metadata.csv" in msg
```

- [ ] **Step 2: Run the test and confirm it fails**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_kaggle_dataset_node_flat_directory_raises_with_subdir_hint -v
```

Expected: ImageFolder raises something other than RuntimeError ("Found no valid file...") and the assertions on `subdir` / `image1.png` fail.

- [ ] **Step 3: Add the structure check before constructing ImageFolder**

Edit `backend/app/nodes/data/kaggle_dataset_node.py`. Replace:

```python
        download_path = kagglehub.dataset_download(dataset_slug)
        root = os.path.join(download_path, subdir) if subdir else download_path

        return {"dataset": ImageFolder(root, transform=T.ToTensor())}
```

with:

```python
        download_path = kagglehub.dataset_download(dataset_slug)
        root = os.path.join(download_path, subdir) if subdir else download_path

        class_dirs = [
            entry for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        ]
        if not class_dirs:
            found = sorted(os.listdir(root))[:10]
            raise RuntimeError(
                f"Path '{root}' does not contain class subdirectories. "
                f"Set 'subdir' to point at the folder that holds your "
                f"per-class subfolders. Found at this path: {found}"
            )

        return {"dataset": ImageFolder(root, transform=T.ToTensor())}
```

- [ ] **Step 4: Run the full test file**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/app/nodes/data/kaggle_dataset_node.py backend/tests/test_dataset_extension.py
git commit -m "feat: validate ImageFolder structure in KaggleDatasetNode"
```

---

## Task 13: Verify both nodes are auto-discovered by the registry

This is a defensive integration test. The test/conftest.py fixture already calls `registry.discover()` once per session, so both nodes should appear without code changes.

**Files:**
- Modify: `backend/tests/test_dataset_extension.py`

- [ ] **Step 1: Append the registry test**

Add to `backend/tests/test_dataset_extension.py`:

```python
# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_new_nodes_are_registered(registry_with_nodes):
    """Auto-discovery (in conftest fixture) should pick up both new nodes."""
    nodes = registry_with_nodes.nodes
    assert "HuggingFaceDataset" in nodes
    assert "KaggleDataset" in nodes

    hf_cls = nodes["HuggingFaceDataset"]
    kaggle_cls = nodes["KaggleDataset"]

    assert hf_cls.CATEGORY == "Data"
    assert kaggle_cls.CATEGORY == "Data"

    hf_param_names = {p.name for p in hf_cls.define_params()}
    assert {"dataset_name", "split", "image_column", "label_column"} <= hf_param_names

    kaggle_param_names = {p.name for p in kaggle_cls.define_params()}
    assert {"dataset_slug", "subdir", "cache_dir"} <= kaggle_param_names
```

- [ ] **Step 2: Run the test**

Run from `backend/`:

```bash
uv run pytest tests/test_dataset_extension.py::test_new_nodes_are_registered -v
```

Expected: `1 passed`. If it fails, the most likely cause is a syntax error or unhandled import in one of the new node files — fix and re-run.

- [ ] **Step 3: Run the FULL backend test suite to catch regressions**

Run from `backend/`:

```bash
uv run pytest -v
```

Expected: every existing test still passes, plus all new tests in `test_dataset_extension.py`.

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_dataset_extension.py
git commit -m "test: assert HuggingFaceDataset and KaggleDataset are auto-registered"
```

---

## Task 14: Frontend i18n (Chinese translations)

The frontend palette pulls node names and English descriptions from the backend automatically, so the only frontend change is adding Chinese translations to the existing `nodeLocales/zh-TW.ts` file. There is no `nodeLocales/en.ts` — English comes from the backend `DESCRIPTION` strings.

**Files:**
- Modify: `frontend/src/i18n/nodeLocales/zh-TW.ts`

- [ ] **Step 1: Read the existing file to find a good insertion point**

Run: `Read frontend/src/i18n/nodeLocales/zh-TW.ts`

Find the existing `Dataset:` block (it lives in the Data section). Insertion point: immediately after the `Dataset:` block, before the next section.

- [ ] **Step 2: Add the two new entries**

Insert (immediately after the existing `Dataset: { ... },` block) the following:

```typescript
  HuggingFaceDataset: {
    description: '從 HuggingFace Hub 載入影像分類資料集（透過 datasets 套件）',
    params: {
      dataset_name: 'HuggingFace Hub 上的 repo id（例：cifar10、ylecun/mnist、uoft-cs/cifar100）',
      subset: '多 config 資料集的 config 名稱（空字串=不指定）',
      split: '資料分割：train/test/validation，亦支援切片語法（如 train[:1000]）',
      image_column: '影像欄位名（不同資料集可能是 image、img、pixel_values）',
      label_column: '標籤欄位名',
      cache_dir: '覆寫 HuggingFace 快取位置（空=用 ~/.cache/huggingface）',
    },
  },
  KaggleDataset: {
    description: '從 Kaggle 下載資料集，並以 ImageFolder 結構載入',
    params: {
      dataset_slug: 'Kaggle dataset 的 owner/slug（例：puneet6060/intel-image-classification）',
      subdir: '下載後資料夾內，包含 class 子資料夾的相對路徑',
      cache_dir: '覆寫 kagglehub 快取位置（空=用預設）',
    },
  },
```

- [ ] **Step 3: Verify the file still type-checks (frontend)**

Run from `frontend/`:

```bash
npm run typecheck
```

Or whichever script the project uses (`tsc --noEmit`, `vite build`, etc.). Check `frontend/package.json` for the exact name.

Expected: no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/i18n/nodeLocales/zh-TW.ts
git commit -m "i18n(zh-TW): translate HuggingFaceDataset and KaggleDataset nodes"
```

---

## Task 15: End-to-end smoke verification

A final manual smoke check that the new nodes appear in the running backend's API.

**Files:** none (read-only)

- [ ] **Step 1: Start the backend in the background**

Run from `backend/`:

```bash
uv run uvicorn app.main:app --port 8000 --reload
```

(Use `run_in_background: true` if calling via Bash tool.)

- [ ] **Step 2: Hit the nodes endpoint and check both new nodes appear**

Run:

```bash
curl -s http://localhost:8000/api/nodes | python -m json.tool | grep -E "HuggingFaceDataset|KaggleDataset"
```

Expected: both names appear in the output (each with its full param schema). If either is missing, check the backend logs for an import error in the new node files.

- [ ] **Step 3: Stop the backend**

Send Ctrl+C to the uvicorn process (or kill it via the background-task UI).

- [ ] **Step 4: Run the full backend test suite one last time**

Run from `backend/`:

```bash
uv run pytest -v
```

Expected: every test passes — no regressions in existing test files.

- [ ] **Step 5: Show a final git log to confirm the commit history is clean**

Run:

```bash
git log --oneline -20
```

Expected: a clean sequence of small commits, one per task.

---

## Acceptance criteria (cross-check with the spec)

| Spec section | Where it's covered |
|---|---|
| §1 New file `_hf_adapter.py` | Task 3 |
| §1 New file `huggingface_dataset_node.py` | Tasks 4-7 |
| §1 New file `kaggle_dataset_node.py` | Tasks 8-12 |
| §1 Modified `pyproject.toml` | Task 1 |
| §1 Modified `nodeLocales/zh-TW.ts` | Task 14 |
| §2 HF node param schema | Task 4 |
| §2 Kaggle node param schema | Task 8 |
| §3 HF data flow + adapter interface | Tasks 3, 4 |
| §3 Kaggle data flow | Task 8 |
| §4 Missing-package errors | Tasks 6, 10 |
| §4 HF auth error mapping | Task 7 |
| §4 Kaggle auth pre-check | Task 11 |
| §4 HF column validation | Task 5 |
| §4 Kaggle ImageFolder structure check | Task 12 |
| §4 Auth invariant: no token params on nodes | Tasks 4, 8 (param lists contain no token field) |
| §5 All eight tests | Tasks 3-12 (test added in the same task as the feature) |
| §5 Registry auto-discovery | Task 13 |
| §6 Out-of-scope items | None of them appear in any task — verified by spec coverage check |
