# src/eval/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from skimage.measure import label, regionprops


def image_auc(scores, labels):
    """
    scores: list or array of image-level anomaly scores (higher = more anomalous)
    labels: binary labels (0 normal, 1 anomaly)
    """
    return roc_auc_score(labels, scores)

def pixel_auc(anomaly_map, gt_mask):
    """
    anomaly_map: 2D array (H,W) or list of 2D arrays normalized 0-1
    gt_mask: binary 2D mask(s), same structure as anomaly_map
    returns pixel-AUROC (global across all pixels)
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # Handle list input (combine all images)
    if isinstance(anomaly_map, list):
        preds = np.concatenate([np.asarray(a).flatten() for a in anomaly_map])
    else:
        preds = np.asarray(anomaly_map).flatten()

    if isinstance(gt_mask, list):
        gts = np.concatenate([np.asarray(g).flatten().astype(int) for g in gt_mask])
    else:
        gts = np.asarray(gt_mask).flatten().astype(int)

    # If all ground truth are same (all 0 or all 1), AUC is undefined
    if len(np.unique(gts)) < 2:
        return np.nan

    try:
        return roc_auc_score(gts, preds)
    except ValueError:
        return np.nan

def iou(pred_mask, gt_mask, thresh=0.5):
    p = (pred_mask > thresh).astype(np.uint8)
    g = gt_mask.astype(np.uint8)
    inter = (p & g).sum()
    union = (p | g).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union

# PRO (Per-Region Overlap) is non-trivial; placeholder:
def pro_score(pred_mask, gt_mask):
    """
    Placeholder for PRO / AUPRO computation.
    Use literature implementation for accurate results.
    """
    # For now return IoU as a proxy
    return iou(pred_mask, gt_mask)


def safe_image_auc(scores, labels):
    """
    Wrapper around roc_auc_score that handles degenerate label sets.
    Returns np.nan if labels contain only a single class.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return roc_auc_score(labels, scores)
    except Exception:
        return np.nan


def evaluate_auroc(results):
    """
    Convenience evaluator expecting `results` to be a list/dict containing
    image scores and labels or tuples (scores, labels). This is minimal and
    kept simple for demo scripts. Returns a dict of computed metrics.
    """
    # Accept either dict with keys or a tuple/list
    if isinstance(results, dict):
        scores = results.get('image_scores', [])
        labels = results.get('image_labels', [])
    elif isinstance(results, (list, tuple)) and len(results) >= 2:
        scores, labels = results[0], results[1]
    else:
        raise ValueError('results must be a dict or (scores, labels) tuple')

    image_auroc = safe_image_auc(scores, labels)
    return {'image_auroc': image_auroc}


def compute_pro(pred, gt, threshold):
    """
    Compute PRO (Per-Region Overlap) at a given threshold.
    - pred: anomaly score map (float32) normalized [0,1]
    - gt: binary ground truth mask (0 or 1)
    """
    # Threshold prediction to binary mask
    bin_pred = (pred >= threshold).astype(np.uint8)
    bin_gt = gt.astype(np.uint8)

    # Label connected components in GT (regions)
    labeled_gt = label(bin_gt, connectivity=1)

    overlaps = []
    for region in regionprops(labeled_gt):
        # Create a binary mask for this region
        minr, minc, maxr, maxc = region.bbox
        region_mask = (labeled_gt[minr:maxr, minc:maxc] == region.label).astype(np.uint8)
        pred_region = bin_pred[minr:maxr, minc:maxc]

        intersection = (region_mask & pred_region).sum()
        region_area = region_mask.sum()

        if region_area > 0:
            overlap = intersection / region_area
            overlaps.append(overlap)

    if len(overlaps) == 0:
        return 0.0
    return np.mean(overlaps)

def compute_aupro(pred_maps, gt_masks, max_fpr=0.3, steps=100):
    """
    Compute AUPRO (Area Under Per-Region Overlap) as described in PatchCore paper.
    - pred_maps: list of anomaly maps (float32), same size as gt_masks
    - gt_masks: list of binary ground truth masks (uint8 or bool)
    - max_fpr: maximum false positive rate threshold (e.g. 0.3)
    Returns: AUPRO score (float)
    """

    all_pros = []
    all_fprs = []

    thresholds = np.linspace(0, 1, steps)
    epsilon = 1e-6

    total_pixels = sum([np.prod(m.shape) for m in gt_masks])
    total_pos = sum([np.sum(m) for m in gt_masks])
    total_neg = total_pixels - total_pos

    for t in thresholds:
        pros = []
        fp = 0

        for pred, gt in zip(pred_maps, gt_masks):
            bin_pred = (pred > t).astype(np.uint8)
            region_size = gt.sum()
            if region_size == 0:
                continue

            intersection = np.logical_and(bin_pred, gt).sum()
            pro = intersection / (region_size + epsilon)
            pros.append(pro)

            false_positive = np.logical_and(bin_pred, ~gt.astype(bool)).sum()
            fp += false_positive

        if not pros:
            continue

        avg_pro = np.mean(pros)
        fpr = fp / (total_neg + epsilon)

        if fpr <= max_fpr:
            all_pros.append(avg_pro)
            all_fprs.append(fpr)

    if len(all_fprs) < 2:
        return np.nan

    return auc(all_fprs, all_pros)

def compute_aupro(pred_maps, gt_masks, max_fpr=0.3, steps=100):
    """
    Compute AUPRO (Area Under Per-Region Overlap) as described in PatchCore paper.
    - pred_maps: list of anomaly maps (float32), same size as gt_masks
    - gt_masks: list of binary ground truth masks (uint8 or bool)
    - max_fpr: maximum false positive rate threshold (e.g. 0.3)
    Returns: AUPRO score (float)
    """
    assert len(pred_maps) == len(gt_masks), "Number of predictions and masks must match."

    all_fprs = []
    all_pros = []

    # Compute thresholds based on all pred_maps
    all_values = np.concatenate([p.flatten() for p in pred_maps])
    min_val, max_val = np.min(all_values), np.max(all_values)
    thresholds = np.linspace(min_val, max_val, steps)

    epsilon = 1e-6

    total_pixels = sum([np.prod(m.shape) for m in gt_masks])
    total_pos = sum([np.sum(m) for m in gt_masks])
    total_neg = total_pixels - total_pos

    for t in thresholds:
        pros = []
        fp_total = 0

        for pred, gt in zip(pred_maps, gt_masks):
            gt = gt.astype(bool)
            pred_bin = (pred > t).astype(np.uint8)

            if gt.sum() == 0:
                continue  # skip images with no anomaly

            inter = np.logical_and(pred_bin, gt).sum()
            pro = inter / (gt.sum() + epsilon)
            pros.append(pro)

            fp = np.logical_and(pred_bin, ~gt).sum()
            fp_total += fp

        if len(pros) == 0:
            continue  # no positive regions found at this threshold

        fpr = fp_total / (total_neg + epsilon)

        if fpr <= max_fpr:
            all_fprs.append(fpr)
            all_pros.append(np.mean(pros))

    if len(all_fprs) < 2:
        print("[WARN] Not enough points to compute AUPRO.")
        return np.nan

    # Sort FPRs and corresponding PROs for AUC
    sorted_idx = np.argsort(all_fprs)
    all_fprs = np.array(all_fprs)[sorted_idx]
    all_pros = np.array(all_pros)[sorted_idx]
    # print("[DEBUG] AUPRO fprs:", all_fprs)
    # print("[DEBUG] AUPRO pros:", all_pros)

    return auc(all_fprs, all_pros)
