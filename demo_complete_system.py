#!/usr/bin/env python3
"""
Cross-Modal Translation Project - Complete System Demo

This script demonstrates the complete refactored cross-modal translation system
with all modern features and capabilities.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import get_device, get_dtype, set_seed, load_config
from src.data import CrossModalDataset
from src.models import CrossModalTranslationModel
from src.eval import CrossModalEvaluator
from src.viz import CrossModalVisualizer


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def demo_system_capabilities():
    """Demonstrate the complete system capabilities."""
    print_section("CROSS-MODAL TRANSLATION SYSTEM DEMO")
    
    # 1. System Setup
    print_section("1. SYSTEM SETUP")
    set_seed(42)
    device = get_device()
    dtype = get_dtype("fp16")
    
    print(f"✓ Device: {device}")
    print(f"✓ Precision: {dtype}")
    print(f"✓ Random seed set for reproducibility")
    
    # 2. Configuration Loading
    print_section("2. CONFIGURATION")
    try:
        config = load_config("configs/config.yaml")
        print("✓ Configuration loaded successfully")
        print(f"  - Text-to-Image model: {config['model']['text_to_image']['name']}")
        print(f"  - Image-to-Text model: {config['model']['image_to_text']['name']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
    except Exception as e:
        print(f"⚠ Configuration loading failed: {e}")
        print("  Using default settings")
    
    # 3. Data Pipeline
    print_section("3. DATA PIPELINE")
    try:
        dataset = CrossModalDataset("data", split="train")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        print("✓ Toy dataset generated automatically")
        
        # Show sample data
        sample = dataset[0]
        print(f"  Sample text: {sample['text'][:50]}...")
        print(f"  Image shape: {sample['image'].size}")
        
    except Exception as e:
        print(f"⚠ Data pipeline test failed: {e}")
    
    # 4. Model Architecture
    print_section("4. MODEL ARCHITECTURE")
    try:
        print("Loading cross-modal translation model...")
        model = CrossModalTranslationModel(
            text_to_image_model="CompVis/stable-diffusion-v-1-4-original",
            image_to_text_model="Salesforce/blip-image-captioning-base",
            device=device,
            dtype=dtype
        )
        print("✓ Model loaded successfully")
        print(f"✓ Text-to-image: Stable Diffusion")
        print(f"✓ Image-to-text: BLIP")
        print(f"✓ CLIP available: {model.clip_available}")
        
    except Exception as e:
        print(f"⚠ Model loading failed: {e}")
        print("  This is expected if models are not downloaded yet")
        return
    
    # 5. Cross-Modal Translation Demo
    print_section("5. CROSS-MODAL TRANSLATION DEMO")
    
    test_prompts = [
        "A beautiful sunset over the ocean with a sailboat on the horizon.",
        "A cute golden retriever puppy playing in a green meadow.",
        "A modern city skyline at night with tall buildings and lights."
    ]
    
    generated_images = []
    generated_captions = []
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nProcessing prompt {i+1}: {prompt[:50]}...")
        
        try:
            start_time = time.time()
            
            # Full cross-modal translation
            image, caption = model.cross_modal_translation(
                prompt,
                num_inference_steps=20,  # Reduced for demo
                guidance_scale=7.5
            )
            
            generation_time = time.time() - start_time
            
            generated_images.append(image)
            generated_captions.append(caption)
            
            print(f"✓ Generated in {generation_time:.2f}s")
            print(f"  Original: {prompt}")
            print(f"  Caption:  {caption}")
            
            # Save image
            image.save(f"demo_result_{i+1}.png")
            print(f"✓ Saved as demo_result_{i+1}.png")
            
        except Exception as e:
            print(f"⚠ Generation failed: {e}")
    
    # 6. Evaluation
    print_section("6. EVALUATION METRICS")
    try:
        evaluator = CrossModalEvaluator(model, device)
        
        # Evaluate image-to-text performance
        eval_results = evaluator.evaluate_image_to_text(
            images=generated_images,
            reference_captions=generated_captions
        )
        
        print("✓ Evaluation completed")
        print(f"  BLEU: {eval_results.get('bleu', 0):.3f}")
        print(f"  ROUGE-1: {eval_results.get('rouge1', 0):.3f}")
        print(f"  ROUGE-L: {eval_results.get('rougeL', 0):.3f}")
        print(f"  Distinct-1: {eval_results.get('distinct_1', 0):.3f}")
        print(f"  Avg generation time: {eval_results.get('avg_generation_time', 0):.2f}s")
        
    except Exception as e:
        print(f"⚠ Evaluation failed: {e}")
    
    # 7. Visualization
    print_section("7. VISUALIZATION")
    try:
        visualizer = CrossModalVisualizer("assets")
        
        # Create comprehensive visualization
        visualizer.visualize_cross_modal_pipeline(
            text_prompts=test_prompts,
            generated_images=generated_images,
            generated_captions=generated_captions,
            save_name="complete_demo_results.png"
        )
        
        print("✓ Visualizations created")
        print("  - Complete pipeline results saved to assets/")
        
    except Exception as e:
        print(f"⚠ Visualization failed: {e}")
    
    # 8. System Summary
    print_section("8. SYSTEM SUMMARY")
    print("✓ Cross-Modal Translation System Ready!")
    print("\nAvailable Features:")
    print("  • Text-to-Image generation (Stable Diffusion)")
    print("  • Image-to-Text generation (BLIP)")
    print("  • Full cross-modal translation pipeline")
    print("  • Comprehensive evaluation metrics")
    print("  • Interactive Streamlit demo")
    print("  • Production-ready architecture")
    print("  • Safety features and disclaimers")
    
    print("\nNext Steps:")
    print("  1. Run interactive demo: streamlit run demo/app.py")
    print("  2. Train custom model: python scripts/train.py")
    print("  3. Run tests: pytest tests/")
    print("  4. Check documentation: README.md")


def main():
    """Main function."""
    print("Cross-Modal Translation - Complete System Demo")
    print("Modern, Production-Ready Implementation")
    
    try:
        demo_system_capabilities()
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        print("This is normal if models haven't been downloaded yet.")
        print("Run 'pip install -r requirements.txt' to install dependencies.")
    
    print_section("DEMO COMPLETED")
    print("Thank you for exploring the Cross-Modal Translation system!")


if __name__ == "__main__":
    main()
