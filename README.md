# ViTPose vs HRNet: Pose Estimation Comparison

This repository provides a benchmarking suite to compare **ViTPose** (Transformer-based) and **HRNet** (CNN-based) models for 2D human pose estimation. It evaluates performance across different model sizes and resolutions using the MMPose framework.

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python installed (Conda is recommended). This project requires `mmpose`, `mmengine`, and `mmpretrain`.

```bash
# Example environment setup
conda create -n pose python=3.8 -y
conda activate pose
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.0.0" mmpretrain
```

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/AhmedAbdelbary0/vitpose-hrnet-comparison.git
cd vitpose-hrnet-comparison
pip install -r requirements.txt
```

---

## 📊 Running the Experiment

You can run each model individually or perform a full automated evaluation.

### Individual Model Inference
Each script runs inference on all images in the `data/` directory, computes the average time, and saves visualized skeletons in `results/`.

| Model | Resolution | Command |
| :--- | :--- | :--- |
| **ViTPose-Small** | 256x192 | `python code/run_vitpose_small.py` |
| **ViTPose-Base** | 256x192 | `python code/run_vitpose.py` |
| **HRNet-W32** | 256x192 | `python code/run_hrnet.py` |
| **HRNet-W48** | 384x288 | `python code/run_hrnet_w48.py` |

### Full Comparison Table
To run all models sequentially and generate a comparison table of parameters and inference times:
```bash
python code/evaluate.py
```

---

## 📈 Benchmarking Results (CPU)

Typical results obtained on a standard CPU environment:

| Model Name | Resolution | Params (M) | Avg Time (s) |
| :--- | :--- | :--- | :--- |
| **ViTPose-small** | 256x192 | 24.29 | ~0.12s |
| **HRNet-w32** | 256x192 | 28.54 | ~0.19s |
| **ViTPose-base** | 256x192 | 89.99 | ~0.28s |
| **HRNet-w48** | 384x288 | 63.60 | ~0.58s |

---

## 📂 Repository Structure

- `code/`: Inference and evaluation scripts.
- `configs/`: Custom model configurations (e.g., HRNet-w48).
- `data/`: Input images for testing.
- `results/`: Output visualizations with connected skeletons.
- `mmpose/`: Submodule/Reference for MMPose configurations.

## 🎨 Visualization
The scripts draw connected skeletons with the following color coding:
- **ViTPose**: Blue/Magenta lines with Red dots.
- **HRNet**: Green lines and dots.

Results are saved to `results/<model_name>/<image_name>.jpg`.