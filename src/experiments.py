import os
from src.pipeline import generate
from src.utils import show_row, show_grid


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


def style_sweep(pipe, control, subject, styles, out_path, **fixed):
    # same sketch, different styles in the prompt
    images = []
    for style in styles.values():
        images.append(generate(pipe, control, f"{subject}, {style}", **fixed)[0])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    show_row([control] + images, ["sketch"] + list(styles.keys()), save_path=out_path)
    return images


def compare_inputs(pipe, controls, prompt, out_path, **fixed):
    # same prompt, different versions of the sketch as controlnet input
    images = [generate(pipe, c, prompt, **fixed)[0] for c in controls.values()]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    show_grid(list(controls.values()), images, list(controls.keys()), save_path=out_path)
    return images
