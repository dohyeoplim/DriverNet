### DriverNet

Distracted Driver Detection using Deep Learning.

[Dataset](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data) is sourced from Kaggle.

#### 0. Prerequisites
- Python 3.10+
- `uv`: [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

#### 1. Create Venv & Install Dependencies

```bash
uv venv
uv sync
```

#### 2. Download Dataset

```bash
uv run main.py --download-dataset
```

- Note: Download requires Kaggle API credentials. Refer to [Kaggle API Documentation](https://www.kaggle.com/docs/api) for more details.
- Place `kaggle.json` in `~/.kaggle/` directory (in the home directory).

> Checking Dataset & Data Module
> ```bash
> uv run scripts/test_datamodule.py
> ```
