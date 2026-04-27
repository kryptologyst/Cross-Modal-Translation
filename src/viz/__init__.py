"""Visualization utilities for cross-modal translation."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


class CrossModalVisualizer:
    """Visualization utilities for cross-modal translation results."""
    
    def __init__(self, save_dir: str = "assets"):
        """Initialize the visualizer.
        
        Args:
            save_dir: Directory to save visualization outputs.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_text_to_image_results(
        self,
        text_prompts: List[str],
        generated_images: List[Image.Image],
        reference_images: Optional[List[Image.Image]] = None,
        clip_scores: Optional[List[float]] = None,
        save_name: str = "text_to_image_results.png",
    ) -> None:
        """Visualize text-to-image generation results.
        
        Args:
            text_prompts: List of input text prompts.
            generated_images: List of generated images.
            reference_images: List of reference images (optional).
            clip_scores: List of CLIP scores (optional).
            save_name: Name of the saved visualization file.
        """
        num_samples = len(text_prompts)
        cols = 3 if reference_images is not None else 2
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            
            # Plot generated image
            axes[row, col].imshow(generated_images[i])
            axes[row, col].set_title(f"Generated: {text_prompts[i][:50]}...")
            axes[row, col].axis('off')
            
            if reference_images is not None and i < len(reference_images):
                col += 1
                axes[row, col].imshow(reference_images[i])
                axes[row, col].set_title("Reference")
                axes[row, col].axis('off')
            
            if clip_scores is not None and i < len(clip_scores):
                col += 1
                axes[row, col].text(0.5, 0.5, f"CLIP Score: {clip_scores[i]:.3f}", 
                                  ha='center', va='center', fontsize=12)
                axes[row, col].set_title("CLIP Score")
                axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_image_to_text_results(
        self,
        images: List[Image.Image],
        generated_captions: List[str],
        reference_captions: List[str],
        metrics: Optional[Dict[str, float]] = None,
        save_name: str = "image_to_text_results.png",
    ) -> None:
        """Visualize image-to-text generation results.
        
        Args:
            images: List of input images.
            generated_captions: List of generated captions.
            reference_captions: List of reference captions.
            metrics: Dictionary of evaluation metrics.
            save_name: Name of the saved visualization file.
        """
        num_samples = len(images)
        cols = 2
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_samples):
            row = i // cols
            col = i % cols
            
            # Plot image
            axes[row, col].imshow(images[i])
            axes[row, col].set_title(f"Input Image {i+1}")
            axes[row, col].axis('off')
            
            # Add caption text
            caption_text = f"Generated: {generated_captions[i]}\nReference: {reference_captions[i]}"
            axes[row, col].text(0.5, -0.1, caption_text, ha='center', va='top', 
                              transform=axes[row, col].transAxes, fontsize=10,
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        # Add metrics text if provided
        if metrics:
            metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            fig.text(0.02, 0.02, f"Metrics:\n{metrics_text}", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def visualize_cross_modal_pipeline(
        self,
        text_prompts: List[str],
        generated_images: List[Image.Image],
        generated_captions: List[str],
        reference_captions: Optional[List[str]] = None,
        metrics: Optional[Dict[str, float]] = None,
        save_name: str = "cross_modal_pipeline.png",
    ) -> None:
        """Visualize the full cross-modal translation pipeline.
        
        Args:
            text_prompts: List of input text prompts.
            generated_images: List of generated images.
            generated_captions: List of generated captions.
            reference_captions: List of reference captions (optional).
            metrics: Dictionary of evaluation metrics.
            save_name: Name of the saved visualization file.
        """
        num_samples = len(text_prompts)
        cols = 3
        rows = num_samples
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Text prompt
            axes[i, 0].text(0.5, 0.5, f"Text Prompt:\n{text_prompts[i]}", 
                           ha='center', va='center', fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
            axes[i, 0].set_title("Input Text")
            axes[i, 0].axis('off')
            
            # Generated image
            axes[i, 1].imshow(generated_images[i])
            axes[i, 1].set_title("Generated Image")
            axes[i, 1].axis('off')
            
            # Generated caption
            caption_text = f"Generated Caption:\n{generated_captions[i]}"
            if reference_captions and i < len(reference_captions):
                caption_text += f"\n\nReference:\n{reference_captions[i]}"
            
            axes[i, 2].text(0.5, 0.5, caption_text, ha='center', va='center', 
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
            axes[i, 2].set_title("Output Caption")
            axes[i, 2].axis('off')
        
        # Add metrics text if provided
        if metrics:
            metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            fig.text(0.02, 0.02, f"Pipeline Metrics:\n{metrics_text}", fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        save_name: str = "metrics_comparison.png",
    ) -> None:
        """Create a comparison visualization of different metrics.
        
        Args:
            metrics_dict: Dictionary mapping model names to their metrics.
            save_name: Name of the saved visualization file.
        """
        # Extract metric names and values
        metric_names = list(next(iter(metrics_dict.values())).keys())
        model_names = list(metrics_dict.keys())
        
        # Create subplots for each metric
        num_metrics = len(metric_names)
        cols = 3
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, metric_name in enumerate(metric_names):
            row = i // cols
            col = i % cols
            
            values = [metrics_dict[model].get(metric_name, 0) for model in model_names]
            
            bars = axes[row, col].bar(model_names, values)
            axes[row, col].set_title(metric_name)
            axes[row, col].set_ylabel("Score")
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[row, col].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                   f"{value:.3f}", ha='center', va='bottom')
        
        # Hide empty subplots
        for i in range(num_metrics, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_attention_visualization(
        self,
        image: Image.Image,
        attention_weights: torch.Tensor,
        text_tokens: List[str],
        save_name: str = "attention_visualization.png",
    ) -> None:
        """Create attention visualization for image-text alignment.
        
        Args:
            image: Input image.
            attention_weights: Attention weights tensor.
            text_tokens: List of text tokens.
            save_name: Name of the saved visualization file.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Show attention-weighted image
        attention_map = attention_weights.mean(dim=0).cpu().numpy()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Resize attention map to match image size
        attention_resized = np.array(Image.fromarray(attention_map).resize(image.size))
        
        ax2.imshow(image)
        ax2.imshow(attention_resized, alpha=0.6, cmap='hot')
        ax2.set_title("Attention Map")
        ax2.axis('off')
        
        # Add text tokens with attention weights
        token_attention = attention_weights.mean(dim=1).cpu().numpy()
        token_text = " ".join([f"{token}\n({weight:.3f})" for token, weight in zip(text_tokens, token_attention)])
        
        fig.text(0.5, 0.02, f"Text Tokens with Attention Weights:\n{token_text}", 
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results_grid(
        self,
        results: Dict[str, Any],
        save_name: str = "results_grid.png",
    ) -> None:
        """Save a comprehensive results grid.
        
        Args:
            results: Dictionary containing all results.
            save_name: Name of the saved visualization file.
        """
        # This is a placeholder for a comprehensive results grid
        # In practice, you would create a detailed visualization showing
        # all aspects of the cross-modal translation results
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.text(0.5, 0.5, "Comprehensive Results Grid\n(Implementation depends on specific results structure)",
               ha='center', va='center', fontsize=16,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        ax.set_title("Cross-Modal Translation Results")
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.close()
