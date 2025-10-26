import os
import numpy as np
from glob import glob
from sklearn.metrics import roc_auc_score
from PIL import Image

def load_mask(mask_path):
    """Load ground truth mask (binary 0/1)."""
    mask = np.array(Image.open(mask_path).convert("L"))
    mask = (mask > 0).astype(np.uint8)
    return mask

def load_pred(pred_path):
    """Load predicted anomaly map (float grayscale or 3-channel heatmap)."""
    pred = np.array(Image.open(pred_path).convert("L")).astype(np.float32)
    pred /= 255.0
    return pred

def compute_auroc(gt_list, pred_list):
    gts = np.concatenate([g.flatten() for g in gt_list])
    preds = np.concatenate([p.flatten() for p in pred_list])
    return roc_auc_score(gts, preds)

def evaluate(dataset_name="bottle", pred_subdir="preds"):
    # Root path: one level up from current script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))  # move up 2 dirs to gaseg
    dataset_dir = os.path.join(root_dir, "mvtec_anomaly_detection", dataset_name)
    pred_dir = os.path.join(dataset_dir, pred_subdir)

    gt_base = os.path.join(dataset_dir, "ground_truth")
    test_base = os.path.join(dataset_dir, "test")

    classes = sorted(os.listdir(gt_base))
    pixel_aurocs = []
    image_aurocs = []

    for defect in classes:
        gt_paths = sorted(glob(os.path.join(gt_base, defect, "*.png")))
        pred_paths = sorted(glob(os.path.join(pred_dir, defect, "*.png")))
        test_paths = sorted(glob(os.path.join(test_base, defect, "*.png")))

        if not gt_paths or not pred_paths:
            print(f"[Warning] Missing data for {defect}, skipping...")
            continue

        gt_masks = [load_mask(p) for p in gt_paths]
        preds = [load_pred(p) for p in pred_paths]

        # Pixel-level AUROC
        pixel_auc = compute_auroc(gt_masks, preds)
        pixel_aurocs.append(pixel_auc)

        # Image-level AUROC (max pixel per image)
        img_labels = np.ones(len(test_paths))
        img_scores = [np.max(p) for p in preds]
        image_auc = roc_auc_score(img_labels, img_scores)
        image_aurocs.append(image_auc)

        print(f"{defect:<15} | pixel-AUROC: {pixel_auc:.4f} | image-AUROC: {image_auc:.4f}")

    # Add good images (all zero mask)
    good_paths = sorted(glob(os.path.join(test_base, "good", "*.png")))
    if good_paths:
        img_labels = np.zeros(len(good_paths))
        img_scores = [np.max(load_pred(os.path.join(pred_dir, "good", os.path.basename(p))))
                      for p in good_paths if os.path.exists(os.path.join(pred_dir, "good", os.path.basename(p)))]
        if img_scores:
            good_image_auc = roc_auc_score(img_labels, img_scores)
            image_aurocs.append(good_image_auc)

    print("\n===== Final Summary =====")
    print(f"Mean Pixel AUROC : {np.mean(pixel_aurocs):.4f}")
    print(f"Mean Image AUROC : {np.mean(image_aurocs):.4f}")

    return np.mean(pixel_aurocs), np.mean(image_aurocs)


if __name__ == "__main__":
    mean_pixel_auroc, mean_image_auroc = evaluate()
    print(f"\n>> Mean Pixel AUROC: {mean_pixel_auroc:.4f}")
    print(f">> Mean Image AUROC: {mean_image_auroc:.4f}")
