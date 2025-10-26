import os
import glob
from PIL import Image
import numpy as np
from torchvision import transforms
# import torch

class MVTecDataset:
    """MVTec Anomaly Detection Dataset Loader."""

    def __init__(self, root_dir, category, split='test', transform=None):
        self.root_dir = root_dir
        self.category = category
        self.split = split
        self.category_dir = os.path.join(self.root_dir, self.category)
        self.img_dir = os.path.join(self.category_dir, self.split)

        # Collect image paths recursively
        self.paths = sorted(glob.glob(os.path.join(self.img_dir, '**', '*.*'), recursive=True))
        self.paths = [p for p in self.paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {self.img_dir}")

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        print(f"[DEBUG] Loaded {len(self.paths)} images for {self.category}/{self.split} from {self.img_dir}")

    def __len__(self):
        return len(self.paths)

    def _mask_path(self, img_path):
        """
        Derive the ground truth mask path for an image.
        Example:
            test/broken_large/000.png -> ground_truth/broken_large/000_mask.png
        """
        # Normalize paths for Windows compatibility
        img_path = os.path.normpath(img_path)

        rel_path = os.path.relpath(img_path, self.category_dir)
        # Extract defect type and filename
        parts = rel_path.split(os.sep)
        if len(parts) < 3:
            return None

        defect_type = parts[1]  # e.g., "broken_large"
        fname = os.path.splitext(parts[-1])[0]  # e.g., "000"

        # Construct mask path
        mask_path = os.path.join(
            self.category_dir,
            "ground_truth",
            defect_type,
            f"{fname}_mask.png"
        )

        if not os.path.exists(mask_path):
            # Try alternative mask naming (if some datasets omit "_mask")
            alt_path = mask_path.replace("_mask.png", ".png")
            if os.path.exists(alt_path):
                mask_path = alt_path
            else:
                return None

        return mask_path

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        mask = None

        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print(f"[WARNING] Could not load or transform {img_path}: {e}")
            return None

        # Load mask only for test split and anomalous images
        if self.split == 'test':
            mask_path = self._mask_path(img_path)
            if mask_path:
                try:
                    mask = Image.open(mask_path).convert('L')
                    mask = mask.resize((256, 256), resample=Image.NEAREST)
                    mask = np.array(mask) > 0
                except Exception as e:
                    print(f"[WARNING] Could not load mask {mask_path}: {e}")
                    mask = None

        return img, mask, img_path
