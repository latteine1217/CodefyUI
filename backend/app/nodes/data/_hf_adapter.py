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
