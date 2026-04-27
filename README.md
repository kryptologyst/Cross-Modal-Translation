# Cross-Modal Translation

A production-ready implementation of cross-modal translation between text and images using state-of-the-art AI models.

## Overview

This project demonstrates bidirectional translation between text and images using:
- **Text-to-Image**: Stable Diffusion for generating images from text descriptions
- **Image-to-Text**: BLIP (Bootstrapping Language-Image Pre-training) for generating captions from images
- **Cross-Modal Translation**: Complete pipeline from text → image → text

## Features

- **Modern Architecture**: Built with PyTorch 2.x and latest transformer models
- **Device Support**: Automatic device detection (CUDA → MPS → CPU)
- **Comprehensive Evaluation**: Multiple metrics including BLEU, ROUGE, BERTScore, CLIP score
- **Interactive Demo**: Streamlit-based web interface
- **Production Ready**: Proper configuration management, logging, and error handling
- **Safety Features**: Content filtering and safety disclaimers

## Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended) or Apple Silicon (MPS) or CPU
- 8GB+ RAM (16GB+ recommended for GPU)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Cross-Modal-Translation.git
cd Cross-Modal-Translation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev]"
```

3. Set up pre-commit hooks (optional):
```bash
pre-commit install
```

## Quick Start

### 1. Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

The demo provides four modes:
- **Text-to-Image**: Generate images from text prompts
- **Image-to-Text**: Generate captions from uploaded images
- **Cross-Modal Translation**: Complete text → image → text pipeline
- **Batch Generation**: Process multiple prompts at once

### 2. Command Line Usage

Generate an image from text:
```bash
python scripts/train.py --config configs/config.yaml --eval-only
```

### 3. Programmatic Usage

```python
from src.models import CrossModalTranslationModel
from src.utils import get_device, get_dtype

# Initialize model
device = get_device()
dtype = get_dtype("fp16")
model = CrossModalTranslationModel(device=device, dtype=dtype)

# Text-to-Image
text_prompt = "A beautiful sunset over the ocean with a sailboat on the horizon."
generated_image = model.text_to_image(text_prompt)

# Image-to-Text
caption = model.image_to_text(generated_image)

# Cross-Modal Translation
image, caption = model.cross_modal_translation(text_prompt)
```

## Dataset Schema

The project expects data in the following structure:

```
data/
├── train_annotations.json
├── val_annotations.json
├── test_annotations.json
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### Annotation Format

```json
[
  {
    "id": "sample_001",
    "text": "A beautiful sunset over the ocean with a sailboat on the horizon.",
    "image_path": "images/sunset_sailboat.jpg",
    "caption": "A sailboat sailing on calm ocean waters during sunset"
  }
]
```

If no dataset is provided, the system will automatically generate a toy dataset for testing.

## Training

### Configuration

Modify `configs/config.yaml` to adjust training parameters:

```yaml
training:
  batch_size: 8
  num_epochs: 10
  learning_rate: 1e-4
  weight_decay: 1e-5

model:
  text_to_image:
    num_inference_steps: 50
    guidance_scale: 7.5
  image_to_text:
    max_length: 77
    num_beams: 5
```

### Start Training

```bash
python scripts/train.py --config configs/config.yaml
```

### Resume Training

```bash
python scripts/train.py --config configs/config.yaml --resume checkpoints/best_model.pt
```

### Evaluation Only

```bash
python scripts/train.py --config configs/config.yaml --eval-only
```

## Evaluation Metrics

The system evaluates performance using multiple metrics:

### Text Generation Metrics
- **BLEU**: Bilingual Evaluation Understudy
- **ROUGE-1/2/L**: Recall-Oriented Understudy for Gisting Evaluation
- **BERTScore**: Contextual embedding-based similarity
- **Distinct-1/2**: Diversity metrics for generated text

### Cross-Modal Metrics
- **CLIP Score**: Semantic similarity between image and text
- **Generation Time**: Processing speed evaluation

### Example Results

```
Evaluation Metrics:
- BLEU: 0.234
- ROUGE-1: 0.456
- ROUGE-2: 0.234
- ROUGE-L: 0.345
- BERT F1: 0.567
- CLIP Score: 0.789
- Distinct-1: 0.123
- Distinct-2: 0.234
```

## Project Structure

```
cross-modal-translation/
├── src/                    # Source code
│   ├── data/               # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── losses/             # Loss functions
│   ├── eval/               # Evaluation metrics
│   ├── viz/                # Visualization utilities
│   └── utils/              # Utility functions
├── configs/                # Configuration files
├── scripts/                # Training and evaluation scripts
├── demo/                   # Interactive demo
├── tests/                  # Unit tests
├── assets/                 # Generated outputs and visualizations
├── data/                   # Dataset directory
├── checkpoints/            # Model checkpoints
├── logs/                   # Training logs
└── outputs/                # Output files
```

## Configuration

### Model Configuration

```yaml
model:
  text_to_image:
    name: "CompVis/stable-diffusion-v-1-4-original"
    num_inference_steps: 50
    guidance_scale: 7.5
  
  image_to_text:
    name: "Salesforce/blip-image-captioning-base"
    max_length: 77
    num_beams: 5
```

### Device Configuration

```yaml
device:
  type: "auto"  # auto, cuda, mps, cpu
  precision: "fp16"  # fp16, fp32, bf16
```

## Safety and Limitations

### Safety Features

- **Content Filtering**: Built-in safety checks for generated content
- **Prompt Filtering**: Automatic filtering of inappropriate prompts
- **Safety Disclaimers**: Clear warnings about AI-generated content

### Limitations

- **Computational Requirements**: GPU recommended for optimal performance
- **Model Biases**: Inherits biases from pre-trained models
- **Content Quality**: Generated content may not always be accurate or appropriate
- **Copyright**: Generated images may infringe on copyrighted material

### Disclaimer

**IMPORTANT**: This system is for research and educational purposes only. Generated content may not be suitable for all audiences. Users are responsible for ensuring appropriate use of generated content.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `pytest`
5. Format code: `black . && ruff check .`
6. Commit changes: `git commit -m "Add feature"`
7. Push to branch: `git push origin feature-name`
8. Submit a pull request

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
ruff check .
```

### Pre-commit Hooks

```bash
pre-commit run --all-files
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{cross_modal_translation,
  title={Cross-Modal Translation: Text-Image Bidirectional Translation},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Cross-Modal-Translation}
}
```

## Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) for text-to-image generation
- [BLIP](https://github.com/salesforce/BLIP) for image-to-text generation
- [Hugging Face Transformers](https://github.com/huggingface/transformers) for model implementations
- [Streamlit](https://streamlit.io/) for the interactive demo interface

## Support

For questions, issues, or contributions, please:
- Open an issue on GitHub
- Check the documentation
- Review the example notebooks

## Roadmap

- [ ] Support for video generation
- [ ] Multi-language support
- [ ] Advanced fine-tuning capabilities
- [ ] API server implementation
- [ ] Mobile app integration
- [ ] Advanced safety features
- [ ] Performance optimizations
# Cross-Modal-Translation
