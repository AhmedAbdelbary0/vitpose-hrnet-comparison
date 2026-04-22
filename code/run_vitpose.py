from mmpose.apis import init_model, inference_topdown
import mmpretrain  # 👈 IMPORTANT: triggers registry registration
import os
import cv2

BASE = os.path.dirname(os.path.dirname(__file__))
img_path = os.path.join(BASE, "data", "demo.jpg")

config = os.path.join(
    BASE,
    "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/"
    "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py"
)

checkpoint = r"C:\Users\user\.cache\mim\td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"

out_path = os.path.join(BASE, "results", "vitpose.jpg")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

# load model
model = init_model(config, checkpoint, device="cpu")

# inference
result = inference_topdown(model, img_path)

# --- SAME SIMPLE VISUALIZATION STYLE AS HRNET ---
img = cv2.imread(img_path)

for person in result:
    if hasattr(person, "pred_instances"):
        kpts = person.pred_instances.keypoints[0]
        for x, y in kpts:
            cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)

cv2.imwrite(out_path, img)

print("Saved:", out_path)