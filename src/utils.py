import json
import matplotlib.pyplot as plt


def show_row(images, titles, save_path=None):
    # images next to each other, used for sketch vs result comparisons
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


def save_settings(path, settings):
    # keep the exact settings next to each result (for reproducibility)
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
