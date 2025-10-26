# src/sam_integration/sam_prompt_utils.py
"""
Utility helpers to integrate Segment-Anything (SAM).

This is a placeholder file showing how to integrate SAM.
To use SAM:
pip install git+https://github.com/facebookresearch/segment-anything.git

Example usage (pseudo):
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry['default'](checkpoint='sam_vit_b.pth')
predictor = SamPredictor(sam)
predictor.set_image(np.asarray(pil_image))
masks, scores, logits = predictor.predict(point_coords=..., point_labels=..., box=...)
"""

# We'll leave actual code minimal to avoid forcing heavy dependencies here.

def sample_point_prompts_from_mask(gt_mask, n_points=5):
    """
    Generate few point prompts inside GT mask as (x,y) coords (in pixel space)
    """
    import numpy as np
    ys, xs = np.where(gt_mask > 0)
    if len(xs) == 0:
        return np.zeros((0,2)), np.zeros((0,), dtype=np.int32)
    idx = np.random.choice(len(xs), size=min(n_points, len(xs)), replace=False)
    coords = np.stack([xs[idx], ys[idx]], axis=1)
    labels = np.ones(coords.shape[0], dtype=np.int32)
    return coords, labels
