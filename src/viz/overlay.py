# src/viz/overlay.py
import numpy as np
import cv2
from matplotlib import pyplot as plt

def overlay_mask_on_image(img_tensor, mask, alpha=0.5, cmap='jet'):
    """
    img_tensor: torch tensor 3xHxW (0..1)
    mask: 2D numpy array (H x W) values 0..1
    returns RGB overlay as numpy array (H x W x 3)
    """
    img = (img_tensor.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    mask_u8 = (mask * 255).astype(np.uint8)
    heat = cv2.applyColorMap(mask_u8, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img, 1.0, heat, alpha, 0)
    return overlay

def show_overlay(img_tensor, mask, title=None):
    ov = overlay_mask_on_image(img_tensor, mask)
    plt.figure(figsize=(6,6))
    plt.imshow(ov)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
