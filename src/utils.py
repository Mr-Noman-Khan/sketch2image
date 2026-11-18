import json
import matplotlib.pyplot as plt


def show_row(images, titles, save_path=None):
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def show_grid(top, bottom, titles, save_path=None):
    # top row = inputs, bottom row = generated images
    n = len(top)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    for i in range(n):
        axes[0, i].imshow(top[i])
        axes[0, i].set_title(titles[i], fontsize=9)
        axes[1, i].imshow(bottom[i])
        axes[0, i].axis("off")
        axes[1, i].axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()


def save_settings(path, settings):
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
