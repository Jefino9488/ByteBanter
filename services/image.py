import requests
import torch
from IPython.display import Image, display
from bs4 import BeautifulSoup
from diffusers import DiffusionPipeline
import os


class ImageGenerator:
    def __init__(self):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            use_safetensors=True
        )
        self.base.to(self.device)

        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base.text_encoder_2,
            vae=self.base.vae,
            use_safetensors=True,
        )
        self.refiner.to(self.device)

    def fetch_images(self, query, num_images=3):
        url = f"https://www.google.com/search?q={query}&tbm=isch"

        response = requests.get(url)

        soup = BeautifulSoup(response.text, 'html.parser')

        image_urls = [img['src'] for img in soup.find_all('img')[:num_images]]

        for img_url in image_urls:
            display(Image(url=img_url))

        torch.cuda.empty_cache()

    def generate_image(self, prompt, output_path="output_image.png", max_batch_size=2):
        self.base.config.max_batch_size = max_batch_size

        try:
            with torch.no_grad():
                image = self.base(prompt).images[0]

            image.save(output_path)

            display(Image(filename=output_path))

        except Exception as e:
            print(f"Error: {str(e)}")

        finally:
            torch.cuda.empty_cache()

    def refine_image(self, prompt, n_steps=40, high_noise_frac=0.8, output_path="refined_image.png"):
        try:
            with torch.no_grad():
                image = self.base(
                    prompt=prompt,
                    num_inference_steps=n_steps,
                    denoising_end=high_noise_frac,
                    output_type="latent",
                ).images

                image = self.refiner(
                    prompt=prompt,
                    num_inference_steps=n_steps,
                    denoising_start=high_noise_frac,
                    image=image,
                ).images[0]

            image.save(output_path)
            print(f"Image saved to {output_path}")

            display(Image(filename=output_path))

        except Exception as e:
            print(f"Error: {str(e)}")

        finally:
            torch.cuda.empty_cache()

# Example usage:
# generator = ImageGenerator()
# generator.fetch_images('your prompt', num_images=3)
# generator.generate_image('sky god', output_path="generated_image.png", max_batch_size=2)
# generator.refine_image('your prompt', n_steps=40, high_noise_frac=0.8, output_path="refined_image.png")
