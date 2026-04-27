#!/usr/bin/env python3
"""Training script for cross-modal translation model."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import CrossModalDataset, create_data_loaders
from src.eval import CrossModalEvaluator
from src.models import CrossModalTranslationModel, CrossModalAlignmentLoss
from src.utils import get_device, get_dtype, set_seed, load_config, save_config
from src.viz import CrossModalVisualizer


class Trainer:
    """Trainer class for cross-modal translation model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trainer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = get_device()
        self.dtype = get_dtype(config["device"]["precision"])
        
        # Set random seed
        set_seed(42)
        
        # Initialize model
        self.model = CrossModalTranslationModel(
            text_to_image_model=config["model"]["text_to_image"]["name"],
            image_to_text_model=config["model"]["image_to_text"]["name"],
            device=self.device,
            dtype=self.dtype
        )
        
        # Initialize loss function
        self.criterion = CrossModalAlignmentLoss(
            contrastive_weight=config["training"]["contrastive_weight"],
            reconstruction_weight=config["training"]["reconstruction_weight"],
            temperature=config["training"]["temperature"]
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
        
        # Initialize data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_dir=config["data"]["data_dir"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            image_size=config["data"]["image_size"],
            max_length=config["data"]["max_text_length"]
        )
        
        # Initialize evaluator
        self.evaluator = CrossModalEvaluator(self.model, self.device)
        
        # Initialize visualizer
        self.visualizer = CrossModalVisualizer(config["paths"]["assets_dir"])
        
        # Initialize logging
        self.log_dir = Path(config["paths"]["logs_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if config["logging"]["tensorboard"]:
            self.writer = SummaryWriter(self.log_dir / "tensorboard")
        else:
            self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # Create output directories
        for path_key in ["output_dir", "checkpoints_dir"]:
            Path(config["paths"][path_key]).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "contrastive_loss": 0.0,
            "reconstruction_loss": 0.0,
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            text_input_ids = batch["text_input_ids"].to(self.device)
            text_attention_mask = batch["text_attention_mask"].to(self.device)
            images = batch["images"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # For now, we'll use a simplified training approach
            # In practice, you would implement proper feature extraction and alignment
            loss = self._compute_loss(text_input_ids, text_attention_mask, images)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config["training"]["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["max_grad_norm"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics["loss"] += loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "step": self.global_step
            })
            
            # Log to tensorboard
            if self.writer and self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss", loss.item(), self.global_step)
            
            # Save checkpoint
            if self.global_step % self.config["training"]["save_every"] == 0:
                self.save_checkpoint()
        
        # Average metrics
        num_batches = len(self.train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        return epoch_metrics
    
    def _compute_loss(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss.
        
        Args:
            text_input_ids: Text input IDs.
            text_attention_mask: Text attention mask.
            images: Image tensors.
            
        Returns:
            Computed loss.
        """
        # This is a simplified loss computation
        # In practice, you would extract features and compute proper alignment loss
        
        # For now, return a dummy loss
        batch_size = text_input_ids.size(0)
        dummy_loss = torch.tensor(0.1, device=self.device, requires_grad=True)
        
        return dummy_loss
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set.
        
        Returns:
            Dictionary containing evaluation metrics.
        """
        self.model.eval()
        
        # Collect predictions
        all_predictions = []
        all_references = []
        all_images = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                # Generate captions for images
                for i, image in enumerate(batch["images"]):
                    caption = self.model.image_to_text(image)
                    all_predictions.append(caption)
                    all_references.append(batch["captions"][i])
                    all_images.append(image)
        
        # Compute metrics
        metrics = self.evaluator.evaluate_image_to_text(
            images=all_images,
            reference_captions=all_references
        )
        
        return metrics
    
    def save_checkpoint(self) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_metric": self.best_metric,
        }
        
        checkpoint_path = Path(self.config["paths"]["checkpoints_dir"]) / f"checkpoint_step_{self.global_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Also save the best checkpoint
        if hasattr(self, 'current_metrics') and self.current_metrics.get("bleu", 0) > self.best_metric:
            self.best_metric = self.current_metrics["bleu"]
            best_path = Path(self.config["paths"]["checkpoints_dir"]) / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint.get("best_metric", 0.0)
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self) -> None:
        """Main training loop."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            if epoch % self.config["training"]["eval_every"] == 0:
                eval_metrics = self.evaluate()
                self.current_metrics = eval_metrics
                
                print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")
                print(f"Epoch {epoch} - Eval Metrics: {eval_metrics}")
                
                # Log to tensorboard
                if self.writer:
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f"eval/{key}", value, epoch)
            
            # Save checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint()
        
        print("Training completed!")
        
        # Final evaluation
        final_metrics = self.evaluate()
        print(f"Final metrics: {final_metrics}")
        
        # Save final model
        self.save_checkpoint()
        
        # Generate sample results
        self.generate_sample_results()
    
    def generate_sample_results(self) -> None:
        """Generate sample results for visualization."""
        print("Generating sample results...")
        
        # Sample prompts for demonstration
        sample_prompts = [
            "A beautiful sunset over the ocean with a sailboat on the horizon.",
            "A cute golden retriever puppy playing in a green meadow.",
            "A modern city skyline at night with tall buildings and lights.",
        ]
        
        # Generate images and captions
        generated_images = []
        generated_captions = []
        
        for prompt in sample_prompts:
            image, caption = self.model.cross_modal_translation(prompt)
            generated_images.append(image)
            generated_captions.append(caption)
        
        # Visualize results
        self.visualizer.visualize_cross_modal_pipeline(
            text_prompts=sample_prompts,
            generated_images=generated_images,
            generated_captions=generated_captions,
            save_name="sample_cross_modal_results.png"
        )
        
        print("Sample results saved to assets/")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train cross-modal translation model")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only run evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    if args.eval_only:
        # Run evaluation only
        metrics = trainer.evaluate()
        print(f"Evaluation metrics: {metrics}")
    else:
        # Run training
        trainer.train()


if __name__ == "__main__":
    main()
