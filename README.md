# ViTPose vs HRNet: Pose Estimation Comparison

This repository provides a benchmarking suite to compare ViTPose (Transformer-based) and HRNet (CNN-based) models for 2D human pose estimation. It evaluates performance across different model sizes and resolutions using the MMPose framework.

## Getting Started

### 1. Prerequisites
Ensure you have Python installed (Conda is recommended). This project requires mmpose, mmengine, and mmpretrain.

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

### 3. Data Preparation
To run the full evaluation, you need to set up the COCO dataset structure in the `data/` directory.

1.  **Download COCO val2017:**
    *   Download the [2017 Val images [1GB]](http://images.cocodataset.org/zips/val2017.zip) and extract it to `data/coco/val2017`.
    *   Download the [2017 Train/Val annotations [241MB]](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) and extract `person_keypoints_val2017.json` to `data/coco/annotations/`.

2.  **Expected Structure:**
    ```text
    data/coco/
    ├── annotations/
    │   └── person_keypoints_val2017.json
    ├── val2017/
    │   ├── 000000000139.jpg
    │   └── ...
    └── person_detection_results/
        └── COCO_val2017_detections_AP_H_56_person.json (Auto-downloaded by script)
    ```

---

## Running the Experiment

You can test models either by running preliminary inference on sample images, or by running a full formal evaluation on the COCO dataset to compute AP scores.

### 1. Preliminary Inference Results (Timing & Visualization)
These scripts run inference on all images in the `data/` directory, compute average inference times, and save the visualized skeletons in `results/`. They are useful for quick visual checks and speed benchmarks.

| Model              | Resolution | Command                                                                          |
|--------------------|------------|----------------------------------------------------------------------------------|
| **ViTPose-Small**  | 256x192    | `python code/run_vitpose_small.py`                                               |
| **ViTPose-Base**   | 256x192    | `python code/run_vitpose_base.py`                                                |
| **ViTPose-Base**   | 384x288    | `python code/run_vitpose_base_384x288.py`                                        |
| **HRNet-W32**      | 256x192    | `python code/run_hrnet_w32.py`                                                   |
| **HRNet-W48**      | 384x288    | `python code/run_hrnet_w48.py`                                                   |

To run all models sequentially and generate a detailed speed comparison table:
```bash
python code/evaluate.py
```

### 2. Full COCO Evaluation (AP Scores)
To obtain formal Average Precision (AP) scores, we use MMPose's testing tools to evaluate against the COCO `val2017` dataset. This requires downloading the COCO pre-computed bounding box detections. 

We have provided a unified script that automatically downloads the required bounding box file (if missing) and evaluates HRNet-w32, ViTPose-small, and ViTPose-base in sequence using the correct checkpoint URLs:

```bash
python code/evaluate_coco_ap.py
```

Alternatively, you can run the evaluation for an individual model manually (e.g., for ViTPose-small):
```bash
python mmpose/tools/test.py \
    mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py \
    https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth \
    --cfg-options data_root=./data/coco test_evaluator.type=CocoMetric
```

**Note on ViTPose-base 384x288:** 
Because MMPose natively assumes square bounding boxes during interpolation, running ViTPose-base at 384x288 requires a custom positional embedding patch. You must first run:
```bash
python code/patch_vitpose_checkpoint.py
```
This generates `checkpoints/vitpose_base_384x288_patched.pth`, which you can then test using our custom config `configs/vitpose_base_384x288.py`.

---

## Benchmarking Results (CPU)

Typical results obtained on a standard CPU environment:

| Model Name | Res | Params(M) | GFLOPs | Time(s) | Std(ms) | FPS | Conf | Mem(MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ViTPose-small** | 256x192 | 24.29 | N/A | 0.126 | 2.3 | 7.9 | 0.802 | 4.2 |
| **ViTPose-base** | 256x192 | 89.99 | N/A | 0.297 | 22.1 | 3.4 | 0.859 | 4.0 |
| **HRNet-w32** | 256x192 | 28.54 | N/A | 0.228 | 6.7 | 4.4 | 0.767 | 4.0 |
| **HRNet-w48** | 384x288 | 63.60 | N/A | 0.579 | 12.0 | 1.7 | 0.816 | 4.2 |
| **ViTPose-base-384**| 384x288 | 90.18 | N/A | 0.625 | 10.4 | 1.6 | 0.990 | 4.2 |

---

## Repository Structure

- `code/`: Inference and evaluation scripts.
- `configs/`: Custom model configurations (e.g., HRNet-w48, ViTPose).
- `data/`: Input images for testing.
- `results/`: Output visualizations with connected skeletons.
- `mmpose/`: Submodule/Reference for MMPose configurations.

## Visualization
The scripts draw connected skeletons with the following color coding:
- **ViTPose**: Blue/Magenta lines with Red dots.
- **HRNet**: Green lines and dots.

Results are saved to `results/<model_name>/<image_name>.jpg`.