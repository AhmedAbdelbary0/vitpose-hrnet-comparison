import os
import subprocess
import urllib.request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "coco")
BBOX_DIR = os.path.join(DATA_DIR, "person_detection_results")
BBOX_FILE = os.path.join(BBOX_DIR, "COCO_val2017_detections_AP_H_56_person.json")
BBOX_URL = "https://huggingface.co/Prophetetc/cocopose/resolve/main/COCO_val2017_detections_AP_H_56_person.json"

MODELS = [
    {
        "name": "HRNet-w32",
        "config": "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
        "checkpoint": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
    },
    {
        "name": "ViTPose-small",
        "config": "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
        "checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
    },
    {
        "name": "ViTPose-base",
        "config": "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
        "checkpoint": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
    }
]

def download_bbox_file():
    if not os.path.exists(BBOX_FILE):
        print(f"Downloading required bounding box detections file to {BBOX_FILE}...")
        os.makedirs(BBOX_DIR, exist_ok=True)
        urllib.request.urlretrieve(BBOX_URL, BBOX_FILE)
        print("Download complete.\n")
    else:
        print("Bounding box detections file already exists. Skipping download.\n")

def run_evaluation():
    for model in MODELS:
        print("="*60)
        print(f"Evaluating {model['name']}")
        print("="*60)
        
        cmd = [
            "python",
            "mmpose/tools/test.py",
            model["config"],
            model["checkpoint"],
            "--cfg-options",
            "data_root=./data/coco",
            "test_evaluator.type=CocoMetric"
        ]
        
        print(f"Running command: {' '.join(cmd)}\n")
        
        # We use subprocess.call to run the evaluation so output streams to terminal
        try:
            subprocess.run(cmd, cwd=BASE_DIR, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {model['name']}: {e}")
        print("\n")

if __name__ == "__main__":
    print("Starting COCO Validation AP Evaluation...")
    download_bbox_file()
    run_evaluation()
    print("All evaluations finished!")
