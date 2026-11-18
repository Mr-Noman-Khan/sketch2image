# Sketch to Image with Stable Diffusion + ControlNet

Course project (Generative AI, Session 2): turning hand-drawn sketches into realistic or stylized images while keeping the structure of the sketch.

## How it works

1. **Preprocessing** (`src/preprocess.py`): the sketch is resized (longest side 512 px, dimensions divisible by 8), inverted to white lines on black, cleaned with adaptive thresholding, small noise is removed and the lines are thickened a bit. This is the format the scribble ControlNet was trained on.
2. **Generation** (`src/pipeline.py`): Stable Diffusion 1.5 with a scribble ControlNet. The text prompt goes through the CLIP text encoder, and the UNet denoises a random latent over 25 steps (UniPC scheduler). At each step, the ControlNet (a trainable copy of the UNet encoder) reads the sketch and adds its features to the UNet through zero-convolutions, so the generated image follows the sketch lines while the prompt decides colors, materials and style. The VAE then decodes the latent into the final image.
3. **Evaluation** (`src/metrics.py`): sketch preservation is measured by comparing the skeleton of the sketch with Canny edges of the generated image (edge recall within 5 px, and chamfer distance).

## Repository structure

```
configs/prompts.yaml     prompts + generation settings for each sketch
sketches/original/       input sketches
sketches/processed/      sketches after preprocessing (ControlNet input)
src/pipeline.py          model loading + generation
src/preprocess.py        sketch cleanup (OpenCV) + HED alternative
src/experiments.py       parameter / style / preprocessing experiments
src/metrics.py           sketch preservation metric
src/utils.py             plotting and saving helpers
run.py                   reproduce baseline results from the command line
outputs/                 generated images, comparison figures, metric CSVs
```

## Installation

Tested on a Kaggle notebook with an NVIDIA Tesla T4 (16 GB), Python 3.13, PyTorch 2.11 (CUDA 13.0). Exact versions are in `requirements.txt`.

**Kaggle / Colab** (PyTorch is already installed):
```
pip install diffusers transformers accelerate safetensors controlnet_aux peft gradio
```

**Local machine** (NVIDIA GPU with at least 8 GB VRAM recommended):
```
git clone https://github.com/TODO-username/sketch2image.git
cd sketch2image
pip install -r requirements.txt
```
If needed, install the PyTorch build that matches your CUDA version from https://pytorch.org first.

Models are downloaded automatically from Hugging Face on the first run (about 5 GB).

## Usage

Generate all sketches listed in the config:
```
python run.py
```
Only one sketch:
```
python run.py --only butterfly
```
To add a new sketch, put the image in `sketches/original/` and add an entry with the same name in `configs/prompts.yaml`.

Experiments and figures were made in the notebook (`notebook.ipynb`).

## Main settings

| Setting | Value |
|---|---|
| Base model | Stable Diffusion 1.5 |
| ControlNet | control_v11p_sd15_scribble |
| Scheduler | UniPC |
| Steps | 25 |
| Guidance scale | 7.5 |
| ControlNet conditioning scale | 1.0 |
| Resolution | longest side 512 px |
| Seed | 42 |
| Precision | float16 |
| Negative prompt | blurry, low quality, deformed, distorted, watermark, text |

## Experiments

- **ControlNet scale** (0.0, 0.3, 0.6, 1.0, 1.5): how strongly the sketch is enforced.
- **Guidance scale** (3, 7.5, 12): how strongly the prompt is followed.
- **Styles**: same sketch and seed with photo / watercolor / anime / cyberpunk / pencil drawing prompts.
- **Preprocessing**: raw inverted sketch vs OpenCV cleanup vs HED scribble.
- **Metric**: edge recall and chamfer distance for all baseline results, compared with a "wrong sketch" reference.

Mean edge recall on the baseline: TODO (wrong sketch reference: TODO).

## Models

- Stable Diffusion 1.5: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M license)
- ControlNet scribble: https://huggingface.co/lllyasviel/control_v11p_sd15_scribble
- HED annotator: https://huggingface.co/lllyasviel/Annotators

## Libraries

- diffusers: https://github.com/huggingface/diffusers
- transformers: https://github.com/huggingface/transformers
- accelerate: https://github.com/huggingface/accelerate
- controlnet_aux: https://github.com/huggingface/controlnet_aux
- OpenCV, scikit-image, NumPy, Pillow, pandas, Matplotlib

## Papers

- Rombach et al., *High-Resolution Image Synthesis with Latent Diffusion Models*, CVPR 2022.
- Zhang, Rao, Agrawala, *Adding Conditional Control to Text-to-Image Diffusion Models*, ICCV 2023.
- Xie and Tu, *Holistically-Nested Edge Detection*, ICCV 2015.
- Zhao et al., *UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models*, NeurIPS 2023.
- Canny, *A Computational Approach to Edge Detection*, IEEE TPAMI 1986.

## Sketch sources

| Sketch | Source |
|---|---|
| TODO | drawn by me / link + license |

## AI assistance

I used an AI assistant (Claude) for guidance on the approach and help with parts of the code. I wrote, ran and tested the code myself and can explain all parts of the implementation.
