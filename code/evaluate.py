from mmpose.apis import init_model, inference_topdown
import mmpretrain
import os
import time
import glob
import numpy as np
import torch
import tracemalloc
try:
    from mmengine.analysis import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def run_evaluation(model_name, config_path, checkpoint_path, img_paths, resolution, device='cpu'):
    print(f"\nEvaluating {model_name}...")
    
    # Load model
    model = init_model(config_path, checkpoint_path, device=device)
    num_params = count_parameters(model)
    
    # GFLOPs calculation
    gflops = "N/A"
    if get_model_complexity_info:
        try:
            h, w = map(int, resolution.split('x'))
            # Use dummy input instead of shape for better compatibility
            dummy_input = torch.randn(1, 3, h, w)
            if hasattr(model, 'data_preprocessor'):
                dummy_input = model.data_preprocessor({'inputs': [dummy_input]}, training=False)['inputs']
            
            info = get_model_complexity_info(model, input_data=dummy_input, show_table=False, show_arch=False)
            gflops = info['flops_str']
        except Exception as e:
            # Silently fail or log briefly
            pass
    
    times = []
    all_confidences = []
    
    tracemalloc.start()
    mem_start, _ = tracemalloc.get_traced_memory()

    # Warmup run
    if img_paths:
        inference_topdown(model, img_paths[0])
    
    for img_path in img_paths:
        start = time.time()
        result = inference_topdown(model, img_path)
        end = time.time()
        times.append(end - start)

        # Collect keypoint confidence scores
        for person in result:
            if hasattr(person, 'pred_instances'):
                scores = person.pred_instances.keypoint_scores[0]
                all_confidences.extend(scores.tolist())

    _, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_mem_mb = mem_peak / (1024 * 1024)

    avg_time = sum(times) / len(times) if times else 0
    std_time = np.std(times) * 1000 if times else 0  # ms
    fps = 1.0 / avg_time if avg_time > 0 else 0
    avg_confidence = float(np.mean(all_confidences)) if all_confidences else 0.0

    return {
        "params": num_params / 1e6,
        "gflops": gflops,
        "avg_time": avg_time,
        "std_time": std_time,
        "fps": fps,
        "avg_conf": avg_confidence,
        "peak_mem": peak_mem_mb
    }

def main():
    BASE = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(BASE, "data")
    img_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    
    if not img_paths:
        print("No images found in data directory.")
        return

    models_to_evaluate = [
        {
            "name": "ViTPose-small",
            "resolution": "256x192",
            "config": "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192.py",
            "ckpt": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"
        },
        {
            "name": "ViTPose-base",
            "resolution": "256x192",
            "config": "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
            "ckpt": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
        },
        {
            "name": "HRNet-w32",
            "resolution": "256x192",
            "config": "mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py",
            "ckpt": "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w32_coco_256x192-c78dce93_20200708.pth"
        },
        {
            "name": "HRNet-w48",
            "resolution": "384x288",
            "config": "configs/hrnet/hrnet_w48_coco_384x288.py",
            "ckpt": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth"
        },
        {
            "name": "ViTPose-base-384",
            "resolution": "384x288",
            "config": "configs/vitpose_base_384x288.py",
            "ckpt": "checkpoints/vitpose_base_384x288_patched.pth"
        }
    ]

    results = []
    for m in models_to_evaluate:
        metrics = run_evaluation(m['name'], m['config'], m['ckpt'], img_paths, m['resolution'])
        results.append({
            "name": m['name'],
            "resolution": m['resolution'],
            **metrics
        })

    # Print Comparison Table
    col_w = [18, 11, 10, 10, 10, 10, 8, 12, 10]
    total_w = sum(col_w) + len(col_w) * 3 + 1
    header = (
        f"{'Model Name':<{col_w[0]}} | "
        f"{'Res':<{col_w[1]}} | "
        f"{'Params(M)':<{col_w[2]}} | "
        f"{'GFLOPs':<{col_w[3]}} | "
        f"{'Time(s)':<{col_w[4]}} | "
        f"{'Std(ms)':<{col_w[5]}} | "
        f"{'FPS':<{col_w[6]}} | "
        f"{'Conf':<{col_w[7]}} | "
        f"{'Mem(MB)':<{col_w[8]}}"
    )
    print("\n" + "=" * total_w)
    print(header)
    print("-" * total_w)
    for res in results:
        print(
            f"{res['name']:<{col_w[0]}} | "
            f"{res['resolution']:<{col_w[1]}} | "
            f"{res['params']:<{col_w[2]}.2f} | "
            f"{res['gflops']:<{col_w[3]}} | "
            f"{res['avg_time']:<{col_w[4]}.3f} | "
            f"{res['std_time']:<{col_w[5]}.1f} | "
            f"{res['fps']:<{col_w[6]}.1f} | "
            f"{res['avg_conf']:<{col_w[7]}.3f} | "
            f"{res['peak_mem']:<{col_w[8]}.1f}"
        )
    print("=" * total_w)

if __name__ == "__main__":
    main()
