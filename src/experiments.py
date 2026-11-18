import os
from src.pipeline import generate
from src.utils import show_row


def sweep(pipe, control, prompt, param, values, out_path, **fixed):
    # same sketch + prompt + seed, only `param` changes
    images = []
    for v in values:
        settings = {**fixed, param: v}
        images.append(generate(pipe, control, prompt, **settings)[0])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    titles = ["sketch"] + [f"{param}={v}" for v in values]
    show_row([control] + images, titles, save_path=out_path)
    return images
