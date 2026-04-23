from mmpose.apis import init_model, inference_topdown
from mmengine.structures import InstanceData
import os
import cv2
import time
import glob

BASE = os.path.dirname(os.path.dirname(__file__))

data_dir = os.path.join(BASE, "data")
img_paths = glob.glob(os.path.join(data_dir, "*.jpg"))

config = os.path.join(
    BASE,
    "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
)

checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

out_dir = os.path.join(BASE, "results", "hrnet")
os.makedirs(out_dir, exist_ok=True)

# load model
model = init_model(config, checkpoint, device="cpu")

# Print number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: HRNet-w32")
print(f"Number of parameters: {num_params / 1e6:.2f}M")

times = []

for img_path in img_paths:
    # inference with timing
    start = time.time()
    result = inference_topdown(model, img_path)
    end = time.time()
    
    elapsed = end - start
    times.append(elapsed)
    print(f"Inference on {os.path.basename(img_path)}: {elapsed:.4f}s")

    # visualization
    img = cv2.imread(img_path)
    
    skeleton = [
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
        (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    ]

    for person in result:
        if hasattr(person, 'pred_instances'):
            keypoints = person.pred_instances.keypoints[0]
            scores = person.pred_instances.keypoint_scores[0]
            
            # Draw lines
            for i, j in skeleton:
                if scores[i] > 0.3 and scores[j] > 0.3:
                    pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                    pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)


            # Draw dots
            for idx, (x, y) in enumerate(keypoints):
                if scores[idx] > 0.3:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

    out_path = os.path.join(out_dir, os.path.basename(img_path))
    cv2.imwrite(out_path, img)


if times:
    avg_time = sum(times) / len(times)
    print(f"\nAverage Inference Time: {avg_time:.4f}s")
else:
    print("No images found in data directory.")