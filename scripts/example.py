#!/usr/bin/env python3
"""Simple example script for cross-modal translation."""

import torch
from PIL import Image

from src.models import CrossModalTranslationModel
from src.utils import get_device, get_dtype, set_seed


def main():
    """Main function demonstrating cross-modal translation."""
    print("Cross-Modal Translation Example")
    print("=" * 40)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device and dtype
    device = get_device()
    dtype = get_dtype("fp16")
    
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    
    # Initialize model
    print("\nLoading models...")
    model = CrossModalTranslationModel(
        text_to_image_model="CompVis/stable-diffusion-v-1-4-original",
        image_to_text_model="Salesforce/blip-image-captioning-base",
        device=device,
        dtype=dtype
    )
    print("Models loaded successfully!")
    
    # Example text prompt
    text_prompt = "A beautiful sunset over the ocean with a sailboat on the horizon."
    print(f"\nInput text: {text_prompt}")
    
    # Text-to-Image generation
    print("\nGenerating image from text...")
    generated_image = model.text_to_image(
        text_prompt,
        num_inference_steps=20,  # Reduced for faster demo
        guidance_scale=7.5
    )
    print("Image generated!")
    
    # Image-to-Text generation
    print("\nGenerating caption from image...")
    generated_caption = model.image_to_text(
        generated_image,
        max_length=77,
        num_beams=5
    )
    print(f"Generated caption: {generated_caption}")
    
    # Cross-modal translation (full pipeline)
    print("\nRunning full cross-modal translation...")
    final_image, final_caption = model.cross_modal_translation(
        text_prompt,
        num_inference_steps=20,
        guidance_scale=7.5
    )
    print(f"Final caption: {final_caption}")
    
    # Save results
    print("\nSaving results...")
    generated_image.save("generated_image.png")
    final_image.save("final_image.png")
    print("Images saved as 'generated_image.png' and 'final_image.png'")
    
    # Evaluate CLIP score if available
    if model.clip_available:
        print("\nEvaluating CLIP score...")
        clip_score = model.evaluate_clip_score(final_image, text_prompt)
        print(f"CLIP score: {clip_score:.3f}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
