from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
import time
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Any, Optional

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") 

client = genai.Client(api_key=api_key)

class ImageToText:
    """
    Generate text description of an image.
    """
    def __init__(self, json_path: str = "data/processed/parsed_data.json"):
        self.json_path = os.path.abspath(json_path)
        self.data_dir = os.path.dirname(self.json_path)


    def _load_image(self, rel_path: str) -> Optional[Image.Image]:
        """
        Loads image relative to the JSON file directory.
        """
        full_path = os.path.join(self.data_dir, rel_path) 
        return Image.open(full_path)
    

    def _generate_description(self, image_path: str) -> str:
        image = self._load_image(image_path)

        prompt = r"""
        Analyze this image from the book 'THE ESSENCE OF CHAOS' by Edward N. Lorenz.
        
        1. If it's a **mathematical formula**, transcribe it strictly into LaTeX format (e.g. $\dot{x} = \sigma(y-x)$). Do not add markdown code blocks.
        2. If it's a **phase portrait, attractor, or graph**, describe the topology, fixed points, and stability (e.g. "Butterfly attractor", "Sensitive dependence on initial conditions").
        3. If it's a **diagram**, explain the bifurcation or system dynamics shown.
        
        Output ONLY the technical description. Keep it concise.
        """

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1 
                )
            )
            return response.text if response.text else "No description generated"

        except Exception as e:
            print(f"API Error on {image_path}: {e}")
            return "Error processing image"


    def process(self):
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        total_images = sum(len(p.get("images", [])) for p in data)

        with tqdm(total=total_images, desc="Description generation") as pbar:
            for page in data:
                if "images" in page and page["images"]:
                    descriptions = []
                    
                    for img_path in page["images"]:
                        desc = self._generate_description(img_path)

                        descriptions.append({
                            "path": img_path,
                            "description": desc
                        })

                        time.sleep(10) 
                        pbar.update(1)

                    page["image_descriptions"] = descriptions

        output_path = self.json_path.replace(".json", "_with_descriptions.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    processor = ImageToText()
    processor.process()

