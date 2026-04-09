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
