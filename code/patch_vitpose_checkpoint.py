import torch
from mmpretrain.models.utils.embed import resize_pos_embed
import urllib.request
import os

url = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
local_path = "checkpoints/vitpose_base_256x192.pth"
patched_path = "checkpoints/vitpose_base_384x288_patched.pth"

if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

if not os.path.exists(local_path):
    print("Downloading original checkpoint...")
    urllib.request.urlretrieve(url, local_path)

print("Loading checkpoint...")
ckpt = torch.load(local_path, map_location='cpu')
state_dict = ckpt['state_dict']

# The original pos_embed in this checkpoint is [1, 193, 768]
pos_embed = state_dict['backbone.pos_embed']

# Strip the cls token (index 0) because ViTPose topdown config sets with_cls_token=False
pos_embed_no_cls = pos_embed[:, 1:, :] # [1, 192, 768]

# Original image was H=256, W=192. Patch size=16. So H_patches=16, W_patches=12.
orig_shape = (16, 12)

# New image is H=384, W=288. H_patches=24, W_patches=18.
new_shape = (24, 18)

print(f"Resizing pos_embed from {orig_shape} to {new_shape}...")
new_pos_embed = resize_pos_embed(
    pos_embed_no_cls, 
    orig_shape, 
    new_shape, 
    mode='bicubic', 
    num_extra_tokens=0
)

state_dict['backbone.pos_embed'] = new_pos_embed
torch.save(ckpt, patched_path)
print(f"Success! Patched checkpoint saved to {patched_path}")
