import torch
from cog import BasePredictor, Input, Path
from diffusers import StableDiffusionXLPipeline

class Predictor(BasePredictor):
    def setup(self):
        # Load base SDXL model
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")

        # ✅ Apply BonkBoy LoRA weights from Hugging Face
        self.pipe.load_lora_weights("BonkBoyMaxi/bonkboy-lora-fortnite")
        self.pipe.fuse_lora()  # optional: fuses LoRA into base weights

    def predict(
        self,
        prompt: str = Input(description="Input prompt"),
        negative_prompt: str = Input(default="blurry, low quality, distorted, deformed, ugly, watermark"),
        width: int = Input(default=1024),
        height: int = Input(default=1024),
        num_inference_steps: int = Input(default=30),
        guidance_scale: float = Input(default=7.5),
        seed: int = Input(default=42),
    ) -> Path:
        generator = torch.manual_seed(seed)

        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        output_path = "/tmp/output.png"
        image.save(output_path)
        return Path(output_path)
