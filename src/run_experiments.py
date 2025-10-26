import os
import torch
from torch.utils.data import DataLoader
from src.datasets.mvtec_loader import MVTecDataset
from src.backbones.resnet_features import ResNetBackbone
from src.baselines.patchcore import SimplePatchCore
from src.eval.metrics import image_auc, pixel_auc, compute_aupro    
import numpy as np
from tqdm import tqdm
import cv2
from src.baselines.improved_patchcore import ImprovedPatchCore  # your new class  # updated to return multi-layer maps
from src.eval.metrics import image_auc, pixel_auc  # ensure pixel_auc can accept raw maps

def collate_skip_none(batch):
    """Safely collate batch while skipping None samples."""
    clean_batch = [b for b in batch if b is not None]
    if len(clean_batch) == 0:
        return None
    # unpack tuple batches (img, mask, path)
    imgs, masks, paths = zip(*clean_batch)
    imgs = torch.utils.data.dataloader.default_collate(imgs)
    masks = list(masks)
    paths = list(paths)
    return imgs, masks, paths



def run_patchcore_demo(data_root, category, device='cuda'):
    print(f"Running PatchCore demo on category: {category}")

    train_ds = MVTecDataset(data_root, category, split='train')
    test_ds = MVTecDataset(data_root, category, split='test')

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False, collate_fn=collate_skip_none)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_skip_none)

    backbone = ResNetBackbone(pretrained=True)  # this must return dict of features
    pc = ImprovedPatchCore(backbone=backbone, device=torch.device(device),
                           n_neighbors=1, coreset_ratio=0.01, smoothing_radius=1)

    print("Building memory from training data...")
    mem_shape = pc.build_memory(train_loader, selected_layers=("layer2", "layer3"))
    print("Memory shape (num_patches, dim):", mem_shape)

    image_scores, image_labels = [], []
    pixel_scores, pixel_labels = [], []
    pixel_maps_all = []  # List of 2D anomaly maps (float32)
    mask_maps_all = []

    for batch in tqdm(test_loader, desc="Testing"):
        if batch is None:
            continue
        imgs, masks, paths = batch
        imgs = imgs.to(device)
        raw_maps, viz_maps = pc.score_batch(imgs, selected_layers=("layer2", "layer3"))

        for i, path in enumerate(paths):
            raw_map = raw_maps[i]
            image_score = float(raw_map.max())
            image_scores.append(image_score)

            mask = masks[i]
            label = 1 if (mask is not None and np.sum(mask) > 0) else 0
            image_labels.append(label)

            if mask is not None:
                mask_resized = mask.astype(np.uint8)
                raw_map_resized = np.array(cv2.resize(raw_map, (mask_resized.shape[1], mask_resized.shape[0])))
                pixel_scores.extend(raw_map_resized.flatten())
                pixel_labels.extend(mask_resized.flatten())
                pixel_maps_all.append(raw_map_resized.astype(np.float32))
                mask_maps_all.append(mask_resized.astype(np.uint8))

    img_auroc = image_auc(image_scores, image_labels)
    print("Image-AUROC:", round(img_auroc, 4))

    if pixel_labels:
        px_auc = pixel_auc(pixel_scores, pixel_labels)
        print("Pixel-AUROC:", round(px_auc, 4))
        print("[DEBUG] Num maps:", len(pixel_maps_all), " Num masks:", len(mask_maps_all))
        print("[DEBUG] GT non-zero mask pixels:", [np.sum(m) for m in mask_maps_all])

        aupro = compute_aupro(pixel_maps_all, mask_maps_all)
        print("AUPRO:", round(aupro, 4))
    else:
        px_auc = None
        aupro = None
        print("No pixel-level ground truth available for AUROC/AUPRO.")

    return {
        "category": category,
        "image_auroc": img_auroc,
        "pixel_auroc": px_auc if pixel_labels else None,
        "aupro": aupro if pixel_labels else None,
    }




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)
    parser.add_argument(
        "--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_args()

    print(
        f"Running demo for category: {args.category} from {args.data_root} on device {args.device}"
    )
    run_patchcore_demo(args.data_root, args.category, device=args.device)
