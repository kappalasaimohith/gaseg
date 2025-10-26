import cv2
import numpy as np

def otsu_segment_batch(img_batch):
    """
    img_batch: torch tensor Bx3xHxW (0..1) or numpy array
    returns list of binary masks (H x W) per image
    """
    masks = []
    if hasattr(img_batch, 'cpu'):
        imgs = img_batch.detach().cpu().numpy()
    else:
        imgs = np.asarray(img_batch)

    # imgs: B x C x H x W
    for img in imgs:
        if img.shape[0] == 3:
            gray = cv2.cvtColor((img.transpose(1,2,0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img.squeeze() * 255).astype(np.uint8)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append((th > 0).astype(np.uint8))
    return masks
