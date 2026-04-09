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
