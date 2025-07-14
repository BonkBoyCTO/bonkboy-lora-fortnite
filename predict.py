import torch
from diffusers import StableDiffusionXLPipeline
from safetensors.torch import load_file

class Predictor:
    def setup(self):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")
        self.pipe.load_lora_weights(".", weight_name="bonkboy-lora.safetensors")

    def predict(self, prompt: str):
        result = self.pipe(prompt=prompt).images[0]
        return result