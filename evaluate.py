import os
import csv
import time
import torch
from src.run_experiments import run_patchcore_demo

# ✅ MVTec AD categories
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile",
    "toothbrush", "transistor", "wood", "zipper"
]

def main():
    data_root = "mvtec_anomaly_detection"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = []
    print(f"Running full evaluation on {device.upper()}...\n")

    for category in CATEGORIES:
        print(f"▶ Running category: {category}")
        start = time.time()
        try:
            metrics = run_patchcore_demo(data_root, category, device=device)
            elapsed = time.time() - start
            results.append({
                "category": category,
                "image_auroc": metrics.get("image_auroc", None),
                "pixel_auroc": metrics.get("pixel_auroc", None),
                "aupro": metrics.get("aupro", None),  # ✅ NEW
                "time_sec": round(elapsed, 2)
            })
            print(f"✅ Done {category}: Image-AUROC={metrics.get('image_auroc', None):.4f}, "
                f"Pixel-AUROC={metrics.get('pixel_auroc', None):.4f}, "
                f"AUPRO={metrics.get('aupro', None):.4f}")
        except Exception as e:
            print(f"❌ Failed {category}: {e}")
            results.append({
                "category": category,
                "image_auroc": None,
                "pixel_auroc": None,
                "time_sec": None
            })

    # Save to CSV
    csv_path = "results_mvtec.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "image_auroc", "pixel_auroc","aupro", "time_sec"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ All results saved to {csv_path}")

if __name__ == "__main__":
    main()