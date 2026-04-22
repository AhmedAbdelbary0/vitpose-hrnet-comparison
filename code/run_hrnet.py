from mmpose.apis import init_model, inference_topdown
from mmengine.structures import InstanceData
import os
import cv2

BASE = os.path.dirname(os.path.dirname(__file__))
img_path = os.path.join(BASE, "data", "demo.jpg")

config = os.path.join(
    BASE,
    "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
)

checkpoint = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"

out_path = os.path.join(BASE, "results", "hrnet.jpg")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# load model
model = init_model(config, checkpoint, device="cpu")

# inference
result = inference_topdown(model, img_path)

# --- manual visualization (WORKS IN ALL VERSIONS) ---
img = cv2.imread(img_path)

# draw keypoints manually
for person in result:
    if hasattr(person, 'pred_instances'):
        keypoints = person.pred_instances.keypoints[0]
        for x, y in keypoints:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

cv2.imwrite(out_path, img)

print("Saved:", out_path)