"""Tests for cross-modal translation project."""

import pytest
import torch
from unittest.mock import Mock, patch

from src.utils import get_device, get_dtype, set_seed, count_parameters
from src.data import CrossModalDataset, collate_fn
from src.models import CrossModalTranslationModel, ContrastiveLoss


class TestUtils:
    """Test utility functions."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]
    
    def test_get_dtype(self):
        """Test dtype selection."""
        assert get_dtype("fp16") == torch.float16
        assert get_dtype("fp32") == torch.float32
        assert get_dtype("bf16") == torch.bfloat16
        assert get_dtype("invalid") == torch.float32
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is a basic test - in practice you'd check random state
        assert True
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        count = count_parameters(model)
        assert count == 55  # 10*5 + 5 bias


class TestData:
    """Test data loading functionality."""
    
    def test_cross_modal_dataset(self):
        """Test dataset initialization."""
        with patch('src.data.Path.exists', return_value=False):
            dataset = CrossModalDataset("test_data", split="train")
            assert len(dataset) == 5  # Toy dataset has 5 samples
    
    def test_collate_fn(self):
        """Test collate function."""
        # Mock batch data
        batch = [
            {
                "text_input_ids": torch.tensor([1, 2, 3]),
                "text_attention_mask": torch.tensor([1, 1, 1]),
                "image": torch.randn(3, 224, 224),
                "id": "test_1",
                "text": "test text",
                "caption": "test caption",
                "image_path": "test.jpg"
            }
        ]
        
        result = collate_fn(batch)
        
        assert "text_input_ids" in result
        assert "text_attention_mask" in result
        assert "images" in result
        assert "ids" in result
        assert "texts" in result
        assert "captions" in result


class TestModels:
    """Test model functionality."""
    
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(temperature=0.1)
        
        text_features = torch.randn(4, 512)
        image_features = torch.randn(4, 512)
        
        loss = loss_fn(text_features, image_features)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    @patch('src.models.StableDiffusionPipeline.from_pretrained')
    @patch('src.models.BlipProcessor.from_pretrained')
    @patch('src.models.BlipForConditionalGeneration.from_pretrained')
    def test_cross_modal_model_init(self, mock_blip_model, mock_blip_processor, mock_sd):
        """Test model initialization."""
        # Mock the model loading
        mock_sd.return_value = Mock()
        mock_blip_processor.return_value = Mock()
        mock_blip_model.return_value = Mock()
        
        model = CrossModalTranslationModel(device=torch.device("cpu"))
        
        assert model.device == torch.device("cpu")
        assert model.text_to_image_pipe is not None
        assert model.image_to_text_model is not None


class TestIntegration:
    """Integration tests."""
    
    def test_basic_workflow(self):
        """Test basic workflow without actual model loading."""
        # Test that imports work
        from src.data import CrossModalDataset
        from src.models import CrossModalTranslationModel
        from src.eval import CrossModalMetrics
        from src.viz import CrossModalVisualizer
        
        assert True  # If imports work, test passes


if __name__ == "__main__":
    pytest.main([__file__])
