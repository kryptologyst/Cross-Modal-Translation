#!/usr/bin/env python3
"""
Project 934: Cross-modal Translation - Modern Implementation

This is the original simple implementation that has been refactored into a 
comprehensive, production-ready cross-modal translation system.

For the modern implementation, see:
- scripts/example.py - Simple usage example
- scripts/train.py - Training and evaluation
- demo/app.py - Interactive Streamlit demo
- src/ - Core implementation modules

This file is kept for reference and can be used as a simple entry point.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image


def simple_cross_modal_demo():
    """Simple cross-modal translation demo (original implementation)."""
    print("Cross-Modal Translation Demo")
    print("=" * 30)
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load pre-trained Stable Diffusion model for text-to-image generation
    print("Loading Stable Diffusion model...")
    stable_diff_pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v-1-4-original", 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    stable_diff_pipe.to(device)
    
    # Load pre-trained BLIP model for image captioning
    print("Loading BLIP model...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model.to(device)
    
    # Step 1: Generate an image from the text description using Stable Diffusion
    text_prompt = "A beautiful sunset over the ocean with a sailboat on the horizon."
    print(f"\nInput text: {text_prompt}")
    print("Generating image...")
    
    generated_image = stable_diff_pipe(text_prompt).images[0]
    
    # Step 2: Caption the generated image using BLIP
    print("Generating caption...")
    inputs = blip_processor(images=generated_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    
    # Show the generated image and its caption
    print(f"\nGenerated Image Caption: {caption}")
    
    # Save the image
    generated_image.save("cross_modal_result.png")
    print("Image saved as 'cross_modal_result.png'")
    
    return generated_image, caption


def main():
    """Main function."""
    try:
        image, caption = simple_cross_modal_demo()
        print("\nDemo completed successfully!")
        print("\nFor more advanced features, run:")
        print("  python scripts/example.py")
        print("  streamlit run demo/app.py")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("  pip install -r requirements.txt")


if __name__ == "__main__":
    main()

