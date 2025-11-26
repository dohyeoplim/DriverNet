### DriverNet

Distracted Driver Detection Using Deep Learning Through Depth-Grouped Average Pooling and Parallel Patch-Level Visual Reasoning

---

#### 0. Prerequisites
- Python 3.10+
- `uv`: [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- GPU(CUDA) available environment

#### 1. Create Venv & Install Dependencies

```bash
uv venv
uv sync
```

#### 2. Download Dataset
[Original Dataset](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data) was sourced from Kaggle.

Download preprocessed images from [dohyeoplim/drivernet-images-depth-v2](https://huggingface.co/datasets/dohyeoplim/drivernet-images-depth-v2/tree/main).

#### 3. Configure Pipeline

Set data directories and model hyperparameters in [config.yaml](configs/config.yaml).

#### 4. Run Training & Create Submission

```bash
uv run main.py --train-and-submit
```
