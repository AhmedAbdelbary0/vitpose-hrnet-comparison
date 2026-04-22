# ViTPose vs HRNet: Pose Estimation Comparison

## Description
This project compares ViTPose (Transformer-based) and HRNet (CNN-based)
for human pose estimation, focusing on accuracy-efficiency trade-offs.

## Setup
pip install -r requirements.txt

## Models
- HRNet-W32 / W48
- ViTPose-B

## Dataset
COCO val2017 (or subset)

## Run HRNet
python code/run_hrnet.py

## Run ViTPose
python code/run_vitpose.py

## Results
Results are stored in /results/


python demo/image_demo.py demo/demo.jpg `
configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py `
C:\Users\user\.cache\mim\td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth `
--out-file vitpose.jpg