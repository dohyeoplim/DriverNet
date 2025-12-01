### DriverNet

Distracted Driver Detection Using Deep Learning Through Depth-Grouped Pooling and Parallel Multi-Aspect Patching

---

#### 0. Prerequisites
- Python 3.10+
- `uv`: [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- GPU(CUDA) available environment

#### 1. Create Venv & Install Dependencies

```bash
uv venv
source .venv/bin/activate
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

---

**Method 1**
### Depth Grouped Pooling (DGP)
<img width="1920" height="1080" alt="5" src="https://github.com/user-attachments/assets/8e320429-7f52-42f1-850a-72ec7f6bc6b6" />
<img width="1920" height="1080" alt="13" src="https://github.com/user-attachments/assets/f80ace34-f85c-41a4-9254-4e7df6fc203d" />
<img width="1920" height="1080" alt="14" src="https://github.com/user-attachments/assets/05515209-153d-446c-a8f2-3f1f1b75ac59" />

<br />

---

**Method 2**
### Parallel Multi-Aspect Patching (P-MAP)
<img width="1920" height="1080" alt="15" src="https://github.com/user-attachments/assets/9bbc4fa5-9c9a-4c1e-92bb-5d31b47dc822" />
<img width="1920" height="1080" alt="19" src="https://github.com/user-attachments/assets/fe3c20f8-dc11-4745-b737-4eafbfd3663e" />

<br />

---

**Method 3**
### Strong Data Augmentations
<img width="1920" height="1080" alt="21" src="https://github.com/user-attachments/assets/a8a345e6-44d3-49f6-ac33-ed1803b2ec92" />

<br />

---

**Method 4**
### Post-Processing for Stability
<img width="1920" height="1080" alt="24" src="https://github.com/user-attachments/assets/91f0eee5-02c3-4178-916d-15278e151f57" />
<img width="1920" height="1080" alt="25" src="https://github.com/user-attachments/assets/6e4db8a9-6e64-4db2-b182-267929a049ba" />

<br />

---
