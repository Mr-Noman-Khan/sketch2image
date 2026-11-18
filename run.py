import argparse
import glob
import os
import yaml
from src.pipeline import load_pipeline, generate
from src.preprocess import load_sketch, clean_sketch
from src.utils import save_settings

parser = argparse.ArgumentParser(description="sketch to image with SD1.5 + ControlNet")
parser.add_argument("--config", default="configs/prompts.yaml")
parser.add_argument("--out", default="outputs/baseline")
parser.add_argument("--only", default=None, help="run a single sketch by name")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
d = cfg["defaults"]
os.makedirs(args.out, exist_ok=True)
os.makedirs("sketches/processed", exist_ok=True)

pipe = load_pipeline(d["control_type"])

for name, item in cfg["sketches"].items():
    if args.only and name != args.only:
        continue
    matches = glob.glob(f"sketches/original/{name}.*")
    if not matches:
        print("missing sketch:", name)
        continue

    control = clean_sketch(load_sketch(matches[0]))
    control.save(f"sketches/processed/{name}.png")
    img = generate(pipe, control, item["prompt"], steps=d["steps"],
                   guidance_scale=d["guidance_scale"],
                   controlnet_scale=d["controlnet_scale"], seed=d["seed"])[0]
    img.save(f"{args.out}/{name}.png")
    save_settings(f"{args.out}/{name}.json", {**d, "prompt": item["prompt"]})
    print("done:", name)
