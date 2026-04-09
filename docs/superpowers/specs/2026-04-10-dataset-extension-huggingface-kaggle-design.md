# Dataset Extension: HuggingFace + Kaggle

**Date:** 2026-04-10
**Status:** Approved (brainstorming)
**Owner:** treeleaves30760

## Goal

Extend CodefyUI's Dataset / DataLoader pipeline so users can load image-classification
datasets from HuggingFace Hub and Kaggle, in addition to the existing torchvision
built-ins (MNIST / CIFAR10 / FashionMNIST). The architecture must leave room for future
non-image domains (NLP, tabular) without requiring a rewrite.

## Background

Today the data layer has three nodes in `backend/app/nodes/data/`:

- `DatasetNode` — hard-coded to torchvision MNIST / CIFAR10 / FashionMNIST.
- `TransformNode` — applies resize / normalize / to_tensor by mutating `dataset.transform`.
- `DataLoaderNode` — wraps a `torch.utils.data.Dataset` in a `torch.utils.data.DataLoader`.

There is no way to load community datasets. The DataType system already has a generic
`DATASET` value, which we will continue to use as the conduit (the contract is "anything
that quacks like `torch.utils.data.Dataset`").

## Decisions (from brainstorming)

| # | Decision | Rationale |
|---|---|---|
| Q1 | Support HuggingFace and Kaggle as the two new sources | User priority |
| Q2 | Image first, but design should leave room for NLP / tabular later | Existing pipeline is image-centric; avoid scope creep |
| Q3 | Add **dedicated** nodes per source (not a single node with a source dropdown) | Matches existing BaseNode pattern, no need for conditional params, palette stays clear |
| Q4 | Auth via **environment variables only** in v1; settings UI deferred | Tokens must never land in graph.json; safest default |
| Q5 | HF `datasets.Dataset` is wrapped **inside the node** as a torch Dataset adapter; no separate adapter node | One node = one drag for users; matches torchvision UX |

Architectural approach: **Route 1** — two new node files plus one shared helper module.
No abstraction layer (Route 2) and no refactor of existing `DatasetNode` (Route 3).

## Components

### New files (all under `backend/app/nodes/data/`)

| File | Role |
|---|---|
| `_hf_adapter.py` | Internal helper. Defines `HFTorchImageDataset`, a `torch.utils.data.Dataset` subclass that wraps a HuggingFace `datasets.Dataset`. Has a `transform` attribute for `TransformNode` interop. Reusable when a future `HuggingFaceTextDataset` node is added. |
| `huggingface_dataset_node.py` | `HuggingFaceDatasetNode(BaseNode)` — loads an image-classification dataset from HuggingFace Hub via `datasets.load_dataset`, wraps the result in `HFTorchImageDataset`. |
| `kaggle_dataset_node.py` | `KaggleDatasetNode(BaseNode)` — downloads a Kaggle dataset via `kagglehub.dataset_download`, then loads it as `torchvision.datasets.ImageFolder`. |

### Modified files

- `backend/pyproject.toml` — add `datasets>=2.14` and `kagglehub>=0.3` to
  `[project.optional-dependencies].ml`.
- `frontend/src/i18n/nodeLocales/zh-TW.ts` — add Chinese descriptions and per-param
  translations for the two new nodes. (English descriptions come from each node's
  backend `DESCRIPTION` string; there is no `nodeLocales/en.ts`.)

### Untouched (verified during brainstorming)

- `BaseNode`, `node_registry` — auto-discovery picks up the new files on startup.
- `Transform`, `DataLoader`, existing `Dataset` nodes — no behaviour change.
- Frontend palette / canvas / `ParamField` — palette is built from the backend node list,
  so the new nodes appear automatically.
- `graph.json` schema — unchanged.

## Node specifications

### `HuggingFaceDatasetNode`

- `NODE_NAME = "HuggingFaceDataset"`
- `CATEGORY = "Data"`
- `DESCRIPTION = "Load an image-classification dataset from HuggingFace Hub"`
- Inputs: none
- Outputs: `dataset: DATASET`

| Param | Type | Default | Notes |
|---|---|---|---|
| `dataset_name` | STRING | `"ylecun/mnist"` | HF Hub repo id (`cifar10`, `uoft-cs/cifar100`, etc.) |
| `subset` | STRING | `""` | Config name for multi-config datasets; empty string = no subset |
| `split` | STRING | `"train"` | `train` / `test` / `validation`, or HF slice syntax (`train[:1000]`) |
| `image_column` | STRING | `"image"` | Column name for the image feature |
| `label_column` | STRING | `"label"` | Column name for the integer label |
| `cache_dir` | STRING | `""` | Override HF cache; empty = HF default (`~/.cache/huggingface`) |

### `KaggleDatasetNode`

- `NODE_NAME = "KaggleDataset"`
- `CATEGORY = "Data"`
- `DESCRIPTION = "Download a Kaggle dataset and load it as an ImageFolder"`
- Inputs: none
- Outputs: `dataset: DATASET`

| Param | Type | Default | Notes |
|---|---|---|---|
| `dataset_slug` | STRING | `""` | `owner/slug`, e.g. `puneet6060/intel-image-classification` |
| `subdir` | STRING | `""` | Relative path inside the downloaded dataset where the ImageFolder layout begins |
| `cache_dir` | STRING | `""` | Override; empty = `kagglehub` default. Applied via `KAGGLEHUB_CACHE` env var before download |

## Data flow

### HuggingFace path

```
HuggingFaceDatasetNode.execute()
  ├─ lazy import datasets               (failure → RuntimeError with install hint)
  ├─ load_dataset(dataset_name, subset or None,
  │               split=split, cache_dir=cache_dir or None)
  ├─ validate image_column / label_column exist in features
  ├─ wrap = HFTorchImageDataset(ds, image_column, label_column,
  │                              transform=ToTensor())
  └─ return {"dataset": wrap}
       │
       ▼
TransformNode.execute()         sets wrap.transform = compose(...)
       │
       ▼
DataLoaderNode.execute()        torch DataLoader(wrap, batch_size=...)
       │
       ▼
CNN / MLP training nodes
```

### Kaggle path

```
KaggleDatasetNode.execute()
  ├─ lazy import kagglehub, torchvision.datasets.ImageFolder
  ├─ check KAGGLE_USERNAME + KAGGLE_KEY (or ~/.kaggle/kaggle.json)
  │     missing → RuntimeError with setup instructions
  ├─ if cache_dir: os.environ["KAGGLEHUB_CACHE"] = cache_dir
  ├─ path = kagglehub.dataset_download(dataset_slug)
  ├─ root = os.path.join(path, subdir) if subdir else path
  ├─ verify root contains class subdirectories
  │     missing → RuntimeError telling user to set subdir
  ├─ ds = ImageFolder(root, transform=ToTensor())
  └─ return {"dataset": ds}
```

`ImageFolder` is already a `torch.utils.data.Dataset` subclass, so it slots into the
existing `Transform` / `DataLoader` flow with no extra work.

### `HFTorchImageDataset` interface

```python
class HFTorchImageDataset(Dataset):
    def __init__(self, hf_dataset, image_column, label_column, transform=None):
        self._ds = hf_dataset
        self._image_col = image_column
        self._label_col = label_column
        self.transform = transform        # public, mirrors torchvision Datasets

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx):
        row = self._ds[idx]
        image = row[self._image_col]      # PIL.Image (HF Image feature auto-decodes)
        label = row[self._label_col]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
```

The default `transform=ToTensor()` is set in the node, not in the adapter, so callers
who want a raw PIL stream can construct the adapter directly.

## Error handling

All errors raised from `execute()` are `RuntimeError` with actionable messages.
Tokens are never logged or echoed.

| Situation | Message (paraphrased) |
|---|---|
| `datasets` not installed | `"HuggingFaceDataset requires the 'datasets' package. Install with: pip install datasets (or pip install -e .[ml])"` |
| `kagglehub` not installed | Same shape, suggesting `kagglehub` |
| HF private repo, no `HF_TOKEN` | Catch `GatedRepoError` / 401 → `"HuggingFace authentication required. Set HF_TOKEN environment variable. See https://huggingface.co/docs/hub/security-tokens"` |
| Kaggle credentials missing | Pre-check before download → `"Kaggle authentication required. Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or place kaggle.json at ~/.kaggle/. See https://www.kaggle.com/docs/api"` |
| Wrong HF column name | `f"Column '{image_column}' not found in dataset. Available columns: {list(features.keys())}"` |
| Kaggle path is not an ImageFolder | `f"Path '{root}' does not contain class subdirectories. Try setting 'subdir' to point at the folder containing class folders. Found: {os.listdir(root)[:10]}"` |
| Network / download failure | Not caught — bubbles up to the existing graph engine error display |

**Auth invariant:** no token / username / key parameters appear on either node.
Credentials live in environment variables only, so `graph.json` files are safe to
commit and share.

## Testing

New file: `backend/tests/test_dataset_extension.py`. All tests are unit tests using
`pytest` + `monkeypatch`; no live network calls. Existing `conftest.py` fixtures apply.

| Test | Setup | Asserts |
|---|---|---|
| `test_hf_dataset_node_basic` | Monkeypatch `datasets.load_dataset` to return a fake list-of-dicts dataset | Returns torch Dataset, `len()` matches, `__getitem__` returns `(Tensor, int)` |
| `test_hf_dataset_node_invalid_column` | Same fake dataset, wrong `image_column` | Raises `RuntimeError` whose message lists the available columns |
| `test_hf_dataset_node_missing_package` | Monkeypatch `sys.modules` so `import datasets` fails | Raises `RuntimeError` with install instructions |
| `test_hf_dataset_node_transform_applied` | Set `dataset.transform = lambda x: marker` and call `__getitem__` | Marker is returned, proving `TransformNode` interop |
| `test_kaggle_dataset_node_basic` | `tmp_path` ImageFolder layout (two classes, dummy 1×1 PNG each); monkeypatch `kagglehub.dataset_download` to return `tmp_path` | Returns `ImageFolder`, len = 2, items are `(Tensor, int)` |
| `test_kaggle_dataset_node_subdir` | Same but files under `tmp_path/train/<class>/`, `subdir="train"` | Successfully descends into subdir |
| `test_kaggle_dataset_node_missing_auth` | Clear `KAGGLE_USERNAME` / `KAGGLE_KEY`; monkeypatch `Path.home()` to a temp dir without `.kaggle/` | Raises `RuntimeError` with setup instructions |
| `test_kaggle_dataset_node_bad_structure` | `tmp_path` with no class subdirs | Raises `RuntimeError` mentioning `subdir` |

**Not in scope:** integration tests that hit real HF / Kaggle (slow, require tokens,
flaky in CI). Can be added later behind `@pytest.mark.integration` and an opt-in env
flag.

## Out of scope (deferred to backlog)

These were considered and explicitly deferred during brainstorming:

- HF `streaming=True` (returns `IterableDataset`; existing `DataLoaderNode` would also
  need changes)
- Kaggle Competitions (`kagglehub.competition_download`)
- HuggingFace text / NLP datasets (would need a tokenizer node and a different adapter)
- Tabular datasets (CSV / Parquet → tensors)
- Local `ImageFolder` node (doable but outside this request)
- Settings UI for credentials (Q4 chose env-var-only for v1)
- Dataset browser / autocomplete in the frontend (v1 = type the slug)
- Loading multiple splits in one node (matches existing `DatasetNode` behaviour)

## Risks & non-risks

**Non-risks**

- Auto-discovery picks up new node files with no registry edits.
- DataType system already supports a generic `DATASET` channel; no type changes needed.
- The frontend palette is data-driven from the backend, so new nodes appear without
  any React work.

**Risks worth tracking**

- HF `Image` feature decoding requires Pillow (already a backend dependency) and the
  `datasets` package's optional vision extras. The minimum-version pin (`>=2.14`)
  should cover this; verify during implementation.
- `kagglehub` writes to `~/.cache/kagglehub` by default; on Windows this is a long
  path that occasionally collides with antivirus tooling. The `cache_dir` parameter
  is the user's escape hatch.
- HuggingFace dataset column conventions vary (`image` vs `img` vs `pixel_values`).
  The `image_column` / `label_column` params plus a clear error listing the available
  columns is the v1 mitigation.
