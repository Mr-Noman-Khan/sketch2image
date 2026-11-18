import gradio as gr
from PIL import Image
from src.pipeline import load_pipeline, generate, DEFAULT_NEGATIVE
from src.preprocess import resize_sketch, clean_sketch, invert_only, hed_scribble

STYLES = {
    "none": "",
    "photo": "photorealistic, natural light, highly detailed",
    "watercolor": "watercolor painting, soft colors, paper texture",
    "anime": "anime style, cel shading, vibrant colors",
    "cyberpunk": "cyberpunk style, neon lights, night, futuristic",
    "pencil drawing": "detailed pencil drawing, graphite shading",
}

METHODS = {
    "clean (opencv)": clean_sketch,
    "hed scribble": hed_scribble,
    "raw (inverted)": invert_only,
}


def to_white_background(img):
    # the drawing canvas is transparent, so put it on white first
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, "white")
        bg.alpha_composite(img)
        img = bg
    return img.convert("L")


def build_demo(pipe):
    def run(upload, drawing, prompt, negative, style, method, cn_scale, guidance, steps, seed):
        img = upload
        if img is None and drawing is not None:
            img = drawing["composite"]
        if img is None:
            raise gr.Error("upload or draw a sketch first")
        if not prompt.strip():
            raise gr.Error("write a prompt")

        sketch = resize_sketch(to_white_background(img))
        control = METHODS[method](sketch)
        full_prompt = f"{prompt}, {STYLES[style]}" if STYLES[style] else prompt

        result = generate(pipe, control, full_prompt, negative_prompt=negative,
                          steps=int(steps), guidance_scale=guidance,
                          controlnet_scale=cn_scale, seed=int(seed))[0]
        return control, result

    with gr.Blocks(title="Sketch to Image") as demo:
        gr.Markdown("# Sketch to Image\nUpload or draw a sketch, describe what it should become, and click Generate.")
        with gr.Row():
            with gr.Column():
                with gr.Tab("Upload"):
                    upload = gr.Image(type="pil", image_mode="RGBA", label="sketch")
                with gr.Tab("Draw"):
                    drawing = gr.Sketchpad(type="pil", label="draw here")
                prompt = gr.Textbox(label="prompt", placeholder="a red sports car on a mountain road")
                style = gr.Dropdown(list(STYLES), value="photo", label="style")
                with gr.Accordion("advanced settings", open=False):
                    negative = gr.Textbox(value=DEFAULT_NEGATIVE, label="negative prompt")
                    method = gr.Radio(list(METHODS), value="clean (opencv)", label="preprocessing")
                    cn_scale = gr.Slider(0, 2, value=1.0, step=0.1, label="controlnet scale (sketch strength)")
                    guidance = gr.Slider(1, 15, value=7.5, step=0.5, label="guidance scale (prompt strength)")
                    steps = gr.Slider(10, 50, value=25, step=1, label="steps")
                    seed = gr.Number(value=42, precision=0, label="seed")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                control_out = gr.Image(label="processed sketch (controlnet input)")
                result_out = gr.Image(label="generated image")

        btn.click(run,
                  inputs=[upload, drawing, prompt, negative, style, method, cn_scale, guidance, steps, seed],
                  outputs=[control_out, result_out])
    return demo


if __name__ == "__main__":
    demo = build_demo(load_pipeline("scribble"))
    demo.launch(share=True)
