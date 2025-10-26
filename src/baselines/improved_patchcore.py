import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import cv2
import time

def extract_patches_from_fmap(fmap):
    """
    fmap: tensor of shape (B, C, H, W)
    Returns: patches shape (B * H * W, C)
    """
    B, C, H, W = fmap.shape
    # flatten spatial dims
    patches = fmap.detach().cpu().numpy().reshape(B, C, H * W)  # (B, C, H*W)
    patches = patches.transpose(0, 2, 1).reshape(-1, C)  # (B*H*W, C)
    return patches, (H, W)

from tqdm import tqdm
import numpy as np

def greedy_coreset_sampling(features: np.ndarray, num_samples: int):
    """
    features: shape (N, D)
    num_samples: how many to pick
    Returns: indices list of length num_samples
    """
    N, D = features.shape
    if num_samples >= N:
        return np.arange(N)

    selected = []
    idx0 = np.random.randint(0, N)
    selected.append(idx0)
    distances = np.linalg.norm(features - features[idx0:idx0+1], axis=1)

    # We will loop num_samples−1 times to pick next points.
    # Use tqdm to show progress.
    for i in tqdm(range(1, num_samples), desc="Greedy sampling"):  
        farthest = np.argmax(distances)
        selected.append(farthest)
        dist_new = np.linalg.norm(features - features[farthest:farthest+1], axis=1)
        distances = np.minimum(distances, dist_new)

    return np.array(selected, dtype=int)


class ImprovedPatchCore:
    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device = None,
        n_neighbors: int = 1,
        coreset_ratio: float = 0.1,
        smoothing_radius: int = 1,
    ):
        """
        backbone: a feature extractor returning a dict of intermediate feature maps
        e.g. returns {'layer2': fmap2, 'layer3': fmap3, 'layer4': fmap4}
        """
        self.backbone = backbone
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.backbone = self.backbone.to(self.device)
        self.n_neighbors = n_neighbors
        self.coreset_ratio = coreset_ratio
        self.memory_bank = None  # (N, D)
        self.nn = None
        self.smoothing_radius = smoothing_radius

    def build_memory(self, dataloader, selected_layers=("layer2", "layer3")):
        feats = []
        for batch in tqdm(dataloader, desc="Building memory"):
            if batch is None:
                continue
            imgs, *_ = batch
            imgs = imgs.to(self.device)
            with torch.no_grad():
                fmap_dict = self.backbone(imgs)

            # Extract and resize features
            fmap_l2 = fmap_dict[selected_layers[0]]
            fmap_l3 = fmap_dict[selected_layers[1]]

            if fmap_l2.shape[2:] != fmap_l3.shape[2:]:
                fmap_l3 = F.interpolate(fmap_l3, size=fmap_l2.shape[2:], mode='bilinear', align_corners=False)

            fmap = torch.cat([fmap_l2, fmap_l3], dim=1)  # (B, C_l2 + C_l3, H, W)
            patches, _ = extract_patches_from_fmap(fmap)
            feats.append(patches)

        if len(feats) == 0:
            raise RuntimeError("No features extracted for memory")

        feats = np.vstack(feats)
        print(f"[DEBUG] Total patches before sampling: {feats.shape}")

        target_samples = max(int(self.coreset_ratio * feats.shape[0]), 1)
        print(f"[DEBUG] Subsampling to {target_samples} patches")

        t0 = time.time()
        print(f"[DEBUG] Starting greedy_coreset_sampling... this may take a while for large datasets")
        selected_indices = greedy_coreset_sampling(feats, target_samples)
        t1 = time.time()
        print(f"[DEBUG] greedy_coreset_sampling took {t1 - t0:.3f} seconds")

        feats_sub = feats[selected_indices]
        print(f"[DEBUG] Coreset subsampled size: {feats_sub.shape}")

        t2 = time.time()
        self.memory_bank = feats_sub
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(self.memory_bank)
        t3 = time.time()
        print(f"[DEBUG] NearestNeighbors.fit took {t3 - t2:.3f} seconds")

        print("[DEBUG] Memory bank built successfully.")
        return self.memory_bank.shape


    def score_batch(self, img_batch, selected_layers=("layer2", "layer3")):
        self.backbone.eval()
        with torch.no_grad():
            imgs = img_batch.to(self.device)
            fmap_dict = self.backbone(imgs)

        fmap_l2 = fmap_dict[selected_layers[0]]
        fmap_l3 = fmap_dict[selected_layers[1]]

        if fmap_l2.shape[2:] != fmap_l3.shape[2:]:
            fmap_l3 = F.interpolate(fmap_l3, size=fmap_l2.shape[2:], mode='bilinear', align_corners=False)

        fmap = torch.cat([fmap_l2, fmap_l3], dim=1)
        patches, (H, W) = extract_patches_from_fmap(fmap)

        dists, _ = self.nn.kneighbors(patches, return_distance=True)
        min_dists = dists.min(axis=1)  # (B*H*W,)
        batch_size = imgs.shape[0]
        raw_maps = min_dists.reshape(batch_size, H, W)

        all_raw_maps, all_viz_maps = [], []
        for i in range(batch_size):
            raw = raw_maps[i]
            all_raw_maps.append(raw)

            s = raw.copy()
            if self.smoothing_radius > 0:
                s = cv2.blur(s, (self.smoothing_radius * 2 + 1, self.smoothing_radius * 2 + 1))

            s_min, s_max = s.min(), s.max()
            s_norm = (s - s_min) / (s_max - s_min + 1e-8)
            s_up = cv2.resize(s_norm.astype('float32'), (256, 256), interpolation=cv2.INTER_LINEAR)
            all_viz_maps.append(s_up)

        return all_raw_maps, all_viz_maps
