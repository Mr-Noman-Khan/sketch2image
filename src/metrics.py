import numpy as np
import cv2
from skimage.morphology import skeletonize


def sketch_lines(control):
    # thin the white lines to 1px so line thickness doesn't change the score
    arr = np.array(control.convert("L")) > 127
    return skeletonize(arr)


def image_edges(img):
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(gray, 50, 150) > 0


def preservation_scores(control, generated, tol=5):
    if generated.size != control.size:
        generated = generated.resize(control.size)
    lines = sketch_lines(control)
    edges = image_edges(generated)

    # distance from each pixel to the nearest edge in the generated image
    dist = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    d = dist[lines]
    return {
        "edge_recall": float((d <= tol).mean()),
        "chamfer_px": float(d.mean()),
    }
