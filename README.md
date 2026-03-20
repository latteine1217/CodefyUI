# CodefyUI

[![zh-TW](https://img.shields.io/badge/語言-繁體中文-blue)](./README_zh-TW.md)

A visual, node-based deep learning pipeline builder. Design CNN, RNN, Transformer, and RL architectures by dragging nodes onto a canvas, connecting them into a DAG, and executing the pipeline — all from the browser.

![CodefyUI Screenshot](Assets/UI.png)

## Features

- **Visual Graph Editor** — Drag-and-drop nodes, connect ports with type-safe edges, real-time validation
- **33 Built-in Nodes** across 9 categories (CNN, RNN, Transformer, RL, Data, Training, IO, Control, Utility)
- **Preset System** — Pre-built model templates (Simple CNN, LSTM Sequence, Training Pipeline) for quick start; export your own subgraphs as reusable presets
- **Multi-Tab Workspace** — Multiple independent canvases, each with its own execution context; switching tabs won't interrupt running pipelines
- **WebSocket Execution** — Real-time per-node progress, Print node output displayed in the Execution Log panel
- **i18n** — English and 繁體中文, with responsive `rem`-based font sizing
- **Persistence** — Auto-saves all tabs to `localStorage`; import/export graph JSON files
- **Custom Nodes** — Drop a Python file, hot-reload, instant UI
- **Dark Theme** — Fully styled dark UI with color-coded categories

## Quick Start

```bash
# Backend
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"       # Core + test deps
pip install -e ".[ml]"        # PyTorch, torchvision, gymnasium (needed for execution)
uvicorn app.main:app --reload

# Frontend (separate terminal)
cd frontend
pnpm install
pnpm dev
```

Open [http://localhost:5173](http://localhost:5173). The frontend proxies API/WS requests to the backend at `:8000`.

## Architecture

```
frontend/   React 19 · TypeScript · React Flow 12 · Zustand 5 · Vite 6
backend/    Python 3.10+ · FastAPI · PyTorch
```

| Principle | Detail |
|-----------|--------|
| **Backend-authoritative** | `GET /api/nodes` returns all node definitions. Adding a backend node auto-appears in the UI. |
| **Single BaseNode component** | One React component renders all node types, parameterized by backend definitions. |
| **WebSocket execution** | `ws://host/ws/execution` streams per-node status. REST handles graph CRUD. |
| **Topological execution** | Kahn's algorithm for DAG sort + cycle detection. |

## Built-in Nodes

| Category | Nodes |
|----------|-------|
| **CNN** | Conv2d, MaxPool2d, BatchNorm2d, Dropout, Activation |
| **RNN** | LSTM, GRU |
| **Transformer** | MultiHeadAttention, TransformerEncoder, TransformerDecoder |
| **RL** | DQN, PPO, EnvWrapper |
| **Data** | Dataset, DataLoader, Transform |
| **Training** | Optimizer, Loss, TrainingLoop |
| **IO** | ImageReader, ImageWriter, ImageBatchReader, FileReader |
| **Control** | If, ForLoop, Compare |
| **Utility** | Print, Reshape, Concat, Flatten, Linear, SequentialModel, Visualize |

## Presets

Pre-built model templates available out of the box:

- **Simple CNN Classifier** — Conv2d → ReLU → MaxPool → Flatten → Linear
- **LSTM Sequence** — LSTM followed by an activation layer
- **Training Pipeline** — Complete Dataset → DataLoader → Optimizer → Loss → TrainingLoop

Export your own canvas as a reusable preset via the **Export Subgraph** button.

## Custom Nodes

Drop a `.py` file in `backend/app/custom_nodes/` extending `BaseNode`:

```python
from app.core.node_base import BaseNode, DataType, PortDefinition

class MyNode(BaseNode):
    NODE_NAME = "MyNode"
    CATEGORY = "Custom"
    DESCRIPTION = "Does something"

    @classmethod
    def define_inputs(cls):
        return [PortDefinition(name="input", data_type=DataType.TENSOR)]

    @classmethod
    def define_outputs(cls):
        return [PortDefinition(name="output", data_type=DataType.TENSOR)]

    def execute(self, inputs, params):
        return {"output": inputs["input"]}
```

Hot-reload via `POST /api/nodes/reload` or the **Reload Nodes** button in the toolbar.

## Key Bindings

| Action | Key |
|--------|-----|
| Delete node | `Delete` |
| Multi-select | `Shift` + click |
| Rename node | Right-click → Rename |
| Duplicate node | Right-click → Duplicate |

## Tests

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
```

## License

MIT
