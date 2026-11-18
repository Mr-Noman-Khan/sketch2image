import numpy as np
import cv2
from PIL import Image


def load_sketch(path, size=512):
    img = Image.open(path).convert("L")
    # resize so the longest side is `size`, keep the aspect ratio
    w, h = img.size
    scale = size / max(w, h)
    # SD needs dimensions divisible by 8
    new_w = int(w * scale) // 8 * 8
    new_h = int(h * scale) // 8 * 8
    return img.resize((new_w, new_h), Image.LANCZOS)


def clean_sketch(img, thickness=3, block_size=25, c=10, min_area=30):
    arr = np.array(img)

    # if the background is dark (digital drawing), flip it to normal first
    if arr.mean() < 127:
        arr = 255 - arr

    # small blur to remove paper texture
    arr = cv2.GaussianBlur(arr, (5, 5), 0)

    # adaptive threshold handles shadows / uneven light in phone photos
    # INV -> lines become white, background black (what the scribble controlnet wants)
    binary = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, c)

    # remove small specks (connected components smaller than min_area pixels)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False  # label 0 is the background
    binary = np.where(keep[labels], 255, 0).astype(np.uint8)

    # make lines thicker so the controlnet sees them clearly
    if thickness > 1:
        kernel = np.ones((thickness, thickness), np.uint8)
        binary = cv2.dilate(binary, kernel)

    return Image.fromarray(binary).convert("RGB")
