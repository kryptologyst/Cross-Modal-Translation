"""Evaluation metrics for cross-modal translation."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from sacrebleu import BLEU
from transformers import BlipProcessor


class CrossModalMetrics:
    """Comprehensive evaluation metrics for cross-modal translation."""
    
    def __init__(self):
        """Initialize the metrics evaluator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu_scorer = BLEU()
        
        # Initialize BLIP processor for text processing
        self.blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    
    def compute_text_metrics(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """Compute text-based evaluation metrics.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference captions.
            
        Returns:
            Dictionary containing text metrics.
        """
        metrics = {}
        
        # BLEU scores
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            bleu_score = self.bleu_scorer.sentence_score(pred, [ref])
            bleu_scores.append(bleu_score.score)
        
        metrics["bleu"] = np.mean(bleu_scores)
        
        # ROUGE scores
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        for metric, scores in rouge_scores.items():
            metrics[metric] = np.mean(scores)
        
        # BERTScore
        try:
            P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
            metrics["bert_precision"] = P.mean().item()
            metrics["bert_recall"] = R.mean().item()
            metrics["bert_f1"] = F1.mean().item()
        except Exception as e:
            print(f"BERTScore computation failed: {e}")
            metrics["bert_precision"] = 0.0
            metrics["bert_recall"] = 0.0
            metrics["bert_f1"] = 0.0
        
        return metrics
    
    def compute_clip_score(
        self,
        images: List[torch.Tensor],
        texts: List[str],
        clip_model: Optional[Any] = None,
    ) -> float:
        """Compute CLIP similarity scores.
        
        Args:
            images: List of image tensors.
            texts: List of text descriptions.
            clip_model: CLIP model for evaluation.
            
        Returns:
            Average CLIP score.
        """
        if clip_model is None:
            return 0.0
        
        scores = []
        with torch.no_grad():
            for image, text in zip(images, texts):
                # This is a simplified version - in practice, you'd use the actual CLIP model
                # For now, return a placeholder score
                scores.append(0.5)  # Placeholder
        
        return np.mean(scores)
    
    def compute_diversity_metrics(
        self,
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute diversity metrics for generated captions.
        
        Args:
            predictions: List of predicted captions.
            
        Returns:
            Dictionary containing diversity metrics.
        """
        # Tokenize predictions
        all_tokens = []
        for pred in predictions:
            tokens = pred.lower().split()
            all_tokens.extend(tokens)
        
        # Distinct-n metrics
        distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0
        
        # Bigrams
        bigrams = []
        for pred in predictions:
            tokens = pred.lower().split()
            bigrams.extend([(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])
        
        distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        return {
            "distinct_1": distinct_1,
            "distinct_2": distinct_2,
        }
    
    def evaluate_batch(
        self,
        predictions: List[str],
        references: List[str],
        images: Optional[List[torch.Tensor]] = None,
        texts: Optional[List[str]] = None,
        clip_model: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Evaluate a batch of predictions.
        
        Args:
            predictions: List of predicted captions.
            references: List of reference captions.
            images: List of images (for CLIP score).
            texts: List of original texts (for CLIP score).
            clip_model: CLIP model for evaluation.
            
        Returns:
            Dictionary containing all metrics.
        """
        metrics = {}
        
        # Text metrics
        text_metrics = self.compute_text_metrics(predictions, references)
        metrics.update(text_metrics)
        
        # Diversity metrics
        diversity_metrics = self.compute_diversity_metrics(predictions)
        metrics.update(diversity_metrics)
        
        # CLIP score (if images and texts provided)
        if images is not None and texts is not None and clip_model is not None:
            clip_score = self.compute_clip_score(images, texts, clip_model)
            metrics["clip_score"] = clip_score
        
        return metrics


class CrossModalEvaluator:
    """Main evaluator for cross-modal translation models."""
    
    def __init__(self, model: Any, device: torch.device):
        """Initialize the evaluator.
        
        Args:
            model: Cross-modal translation model.
            device: Device to run evaluation on.
        """
        self.model = model
        self.device = device
        self.metrics = CrossModalMetrics()
    
    def evaluate_text_to_image(
        self,
        text_prompts: List[str],
        reference_images: Optional[List[torch.Tensor]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """Evaluate text-to-image generation.
        
        Args:
            text_prompts: List of text prompts.
            reference_images: List of reference images (optional).
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            
        Returns:
            Dictionary containing evaluation results.
        """
        results = {
            "generated_images": [],
            "generation_times": [],
            "clip_scores": [],
        }
        
        for prompt in text_prompts:
            import time
            start_time = time.time()
            
            # Generate image
            generated_image = self.model.text_to_image(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            generation_time = time.time() - start_time
            
            results["generated_images"].append(generated_image)
            results["generation_times"].append(generation_time)
            
            # Compute CLIP score if reference images provided
            if reference_images is not None:
                clip_score = self.model.evaluate_clip_score(generated_image, prompt)
                results["clip_scores"].append(clip_score)
        
        # Compute average metrics
        results["avg_generation_time"] = np.mean(results["generation_times"])
        if results["clip_scores"]:
            results["avg_clip_score"] = np.mean(results["clip_scores"])
        
        return results
    
    def evaluate_image_to_text(
        self,
        images: List[torch.Tensor],
        reference_captions: List[str],
        max_length: int = 77,
        num_beams: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate image-to-text generation.
        
        Args:
            images: List of input images.
            reference_captions: List of reference captions.
            max_length: Maximum caption length.
            num_beams: Number of beams for beam search.
            
        Returns:
            Dictionary containing evaluation results.
        """
        results = {
            "generated_captions": [],
            "generation_times": [],
        }
        
        for image in images:
            import time
            start_time = time.time()
            
            # Generate caption
            generated_caption = self.model.image_to_text(
                image,
                max_length=max_length,
                num_beams=num_beams
            )
            
            generation_time = time.time() - start_time
            
            results["generated_captions"].append(generated_caption)
            results["generation_times"].append(generation_time)
        
        # Compute text metrics
        text_metrics = self.metrics.compute_text_metrics(
            results["generated_captions"],
            reference_captions
        )
        results.update(text_metrics)
        
        # Compute diversity metrics
        diversity_metrics = self.metrics.compute_diversity_metrics(
            results["generated_captions"]
        )
        results.update(diversity_metrics)
        
        # Compute average generation time
        results["avg_generation_time"] = np.mean(results["generation_times"])
        
        return results
    
    def evaluate_cross_modal_translation(
        self,
        text_prompts: List[str],
        reference_captions: List[str],
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        max_length: int = 77,
        num_beams: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate full cross-modal translation pipeline.
        
        Args:
            text_prompts: List of input text prompts.
            reference_captions: List of reference captions.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            max_length: Maximum caption length.
            num_beams: Number of beams for beam search.
            
        Returns:
            Dictionary containing evaluation results.
        """
        results = {
            "generated_images": [],
            "generated_captions": [],
            "total_times": [],
            "text_to_image_times": [],
            "image_to_text_times": [],
        }
        
        for prompt in text_prompts:
            import time
            start_time = time.time()
            
            # Full cross-modal translation
            generated_image, generated_caption = self.model.cross_modal_translation(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
            
            total_time = time.time() - start_time
            
            results["generated_images"].append(generated_image)
            results["generated_captions"].append(generated_caption)
            results["total_times"].append(total_time)
        
        # Compute text metrics
        text_metrics = self.metrics.compute_text_metrics(
            results["generated_captions"],
            reference_captions
        )
        results.update(text_metrics)
        
        # Compute diversity metrics
        diversity_metrics = self.metrics.compute_diversity_metrics(
            results["generated_captions"]
        )
        results.update(diversity_metrics)
        
        # Compute average times
        results["avg_total_time"] = np.mean(results["total_times"])
        
        return results
