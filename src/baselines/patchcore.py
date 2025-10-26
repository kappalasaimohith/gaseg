import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
import os
import hashlib
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False

def extract_patches_from_fmap(fmap):
    B, C, h, w = fmap.shape
    fmap = fmap.detach().cpu().numpy()
    patches = fmap.reshape(B, C, h * w).transpose(0, 2, 1).reshape(-1, C)
    return patches

class SimplePatchCore:
    def __init__(self, backbone, device='cuda' if torch.cuda.is_available() else 'cpu', n_neighbors=1):
        # Use provided backbone if given, otherwise default ResNet50 features
        if backbone is not None:
            self.backbone = backbone.to(device).eval()
        else:
            self.backbone = nn.Sequential(*list(resnet50(weights="IMAGENET1K_V1").children())[:-2])

        self.device = device
        self.memory_bank = None
        self.nn = None
        self.n_neighbors = n_neighbors
        self.max_patches = None  # optional cap on patch count
        # caching & faiss options
        self.cache_dir = None
        self.use_faiss = False
        self.faiss_index = None

    def _cache_path_for(self, img_path):
        h = hashlib.sha1(img_path.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"feat_{h}.npz")

    def build_memory(self, dataloader, max_patches=None, cache_dir=None, coreset_method='greedy'):
        feats = []
        num_samples = getattr(dataloader, 'dataset', None)
        if num_samples is not None:
            num_samples = len(dataloader.dataset)
        else:
            num_samples = "unknown (generator input)"
        print(f"[DEBUG] Starting feature extraction on {num_samples} samples, device={self.device}")

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc='Building memory')):
                # Expect dataloader to yield (imgs, masks, paths) or (imgs, ...)
                if batch is None or len(batch) == 0:
                    print(f"[DEBUG] Skipping empty batch #{batch_idx}")
                    continue

                # extract images and optional paths
                if isinstance(batch, (list, tuple)) and torch.is_tensor(batch[0]):
                    imgs = batch[0]
                    paths = batch[2] if len(batch) > 2 else [None] * imgs.shape[0]
                elif torch.is_tensor(batch):
                    imgs = batch
                    paths = [None] * imgs.shape[0]
                else:
                    print(f"[DEBUG] Unsupported batch format #{batch_idx}: {type(batch)}")
                    continue

                imgs = imgs.to(self.device)
                B = imgs.shape[0]

                for i in range(B):
                    img_tensor = imgs[i].unsqueeze(0)
                    img_path = paths[i] if isinstance(paths, (list, tuple)) else paths[i]
                    cache_path = None
                    if self.cache_dir is not None and img_path is not None:
                        cache_path = self._cache_path_for(img_path)
                    if cache_path is not None and os.path.exists(cache_path):
                        arr = np.load(cache_path)['f']
                        feats.append(arr)
                        continue

                    try:
                        fmap = self.backbone(img_tensor)
                    except Exception as e:
                        print(f"[DEBUG] Backbone failed on sample #{batch_idx}:{i}: {e}")
                        continue

                    if fmap is None:
                        continue

                    patches = extract_patches_from_fmap(fmap)
                    if patches.size == 0:
                        continue

                    if cache_path is not None:
                        np.savez_compressed(cache_path, f=patches)

                    feats.append(patches)

        if len(feats) == 0:
            raise ValueError("No features extracted. Check if your training data is loaded properly.")

        feats = np.vstack(feats)
        print(f"[DEBUG] Extracted total {feats.shape[0]} patches of dimension {feats.shape[1]}")

        # Optional subsampling / coreset selection
        if max_patches is None:
            max_patches = self.max_patches
        if max_patches is not None and feats.shape[0] > max_patches:
            if coreset_method == 'random':
                idx = np.random.choice(feats.shape[0], size=max_patches, replace=False)
                feats = feats[idx]
            else:
                # greedy k-center selection (farthest point sampling)
                m = feats.shape[0]
                centers = []
                cur = np.random.randint(0, m)
                centers.append(cur)
                dists = np.linalg.norm(feats - feats[cur], axis=1)
                for _ in range(1, max_patches):
                    nxt = int(np.argmax(dists))
                    centers.append(nxt)
                    newd = np.linalg.norm(feats - feats[nxt], axis=1)
                    dists = np.minimum(dists, newd)
                feats = feats[centers]

        self.memory_bank = feats

        # fit index: sklearn or faiss
        if self.use_faiss and _FAISS_AVAILABLE and self.memory_bank.size > 0:
            d = self.memory_bank.shape[1]
            index = faiss.IndexFlatL2(d)
            xb = self.memory_bank.astype('float32')
            index.add(xb)
            self.faiss_index = index
            self.nn = None
        else:
            self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto').fit(self.memory_bank)

        print(f"[DEBUG] Memory bank built with shape {self.memory_bank.shape}")
        return self.memory_bank.shape

    def score_batch(self, img_batch):
        self.backbone.eval()
        with torch.no_grad():
            img = img_batch.to(self.device)
            fmap = self.backbone(img)
            B, C, h, w = fmap.shape
            patches = extract_patches_from_fmap(fmap)

            # query NN distances
            if self.use_faiss and _FAISS_AVAILABLE and self.faiss_index is not None:
                xb = patches.astype('float32')
                D, I = self.faiss_index.search(xb, self.n_neighbors)
                # D are squared L2 distances
                dists = D
            else:
                dists, _ = self.nn.kneighbors(patches, return_distance=True)
            scores = dists.mean(axis=1).reshape(B, h, w)

            out_raw, out_viz = [], []
            for i in range(B):
                s = scores[i]

                # ---- raw scores (for image-level AUROC) ----
                out_raw.append(s)

                # ---- normalized for visualization ----
                s_norm = (s - s.min()) / (s.max() - s.min() + 1e-8)
                import cv2
                s_up = cv2.resize((s_norm * 255).astype('uint8'), (256, 256), interpolation=cv2.INTER_LINEAR)
                s_up = s_up.astype('float32') / 255.0
                out_viz.append(s_up)

            return out_raw, out_viz

