"""Interactive Streamlit demo for cross-modal translation."""

import os
import time
from typing import List, Optional, Tuple

import streamlit as st
import torch
from PIL import Image

from src.models import CrossModalTranslationModel
from src.utils import get_device, get_dtype, load_config


class CrossModalDemo:
    """Streamlit demo for cross-modal translation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the demo.
        
        Args:
            config_path: Path to configuration file.
        """
        self.config = load_config(config_path)
        self.device = get_device()
        self.dtype = get_dtype(self.config["device"]["precision"])
        
        # Initialize model
        with st.spinner("Loading models..."):
            self.model = CrossModalTranslationModel(
                text_to_image_model=self.config["model"]["text_to_image"]["name"],
                image_to_text_model=self.config["model"]["image_to_text"]["name"],
                device=self.device,
                dtype=self.dtype
            )
        
        st.success("Models loaded successfully!")
    
    def text_to_image_generation(self) -> None:
        """Text-to-image generation interface."""
        st.subheader("Text-to-Image Generation")
        
        # Text input
        text_prompt = st.text_area(
            "Enter your text prompt:",
            value="A beautiful sunset over the ocean with a sailboat on the horizon.",
            height=100,
            max_chars=self.config["demo"]["max_prompt_length"]
        )
        
        # Generation parameters
        col1, col2 = st.columns(2)
        
        with col1:
            num_steps = st.slider(
                "Number of inference steps",
                min_value=10,
                max_value=100,
                value=self.config["demo"]["default_steps"],
                step=5
            )
        
        with col2:
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=1.0,
                max_value=20.0,
                value=self.config["demo"]["default_guidance"],
                step=0.5
            )
        
        # Generate button
        if st.button("Generate Image", type="primary"):
            if text_prompt.strip():
                with st.spinner("Generating image..."):
                    start_time = time.time()
                    
                    generated_image = self.model.text_to_image(
                        text_prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale
                    )
                    
                    generation_time = time.time() - start_time
                
                # Display results
                st.image(generated_image, caption="Generated Image", use_column_width=True)
                
                # Show generation info
                st.info(f"Generation time: {generation_time:.2f} seconds")
                
                # Download button
                st.download_button(
                    label="Download Image",
                    data=self._image_to_bytes(generated_image),
                    file_name=f"generated_{int(time.time())}.png",
                    mime="image/png"
                )
            else:
                st.error("Please enter a text prompt.")
    
    def image_to_text_generation(self) -> None:
        """Image-to-text generation interface."""
        st.subheader("Image-to-Text Generation")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to generate a caption"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Generation parameters
            col1, col2 = st.columns(2)
            
            with col1:
                max_length = st.slider(
                    "Maximum caption length",
                    min_value=20,
                    max_value=100,
                    value=self.config["model"]["image_to_text"]["max_length"],
                    step=5
                )
            
            with col2:
                num_beams = st.slider(
                    "Number of beams",
                    min_value=1,
                    max_value=10,
                    value=self.config["model"]["image_to_text"]["num_beams"],
                    step=1
                )
            
            # Generate button
            if st.button("Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    start_time = time.time()
                    
                    generated_caption = self.model.image_to_text(
                        image,
                        max_length=max_length,
                        num_beams=num_beams
                    )
                    
                    generation_time = time.time() - start_time
                
                # Display results
                st.success(f"Generated Caption: {generated_caption}")
                st.info(f"Generation time: {generation_time:.2f} seconds")
    
    def cross_modal_translation(self) -> None:
        """Cross-modal translation interface."""
        st.subheader("Cross-Modal Translation (Text → Image → Text)")
        
        # Text input
        text_prompt = st.text_area(
            "Enter your text prompt:",
            value="A cute golden retriever puppy playing in a green meadow.",
            height=100,
            max_chars=self.config["demo"]["max_prompt_length"]
        )
        
        # Generation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_steps = st.slider(
                "Inference steps",
                min_value=10,
                max_value=100,
                value=self.config["demo"]["default_steps"],
                step=5
            )
        
        with col2:
            guidance_scale = st.slider(
                "Guidance scale",
                min_value=1.0,
                max_value=20.0,
                value=self.config["demo"]["default_guidance"],
                step=0.5
            )
        
        with col3:
            max_length = st.slider(
                "Caption length",
                min_value=20,
                max_value=100,
                value=self.config["model"]["image_to_text"]["max_length"],
                step=5
            )
        
        # Generate button
        if st.button("Run Cross-Modal Translation", type="primary"):
            if text_prompt.strip():
                with st.spinner("Running cross-modal translation..."):
                    start_time = time.time()
                    
                    generated_image, generated_caption = self.model.cross_modal_translation(
                        text_prompt,
                        num_inference_steps=num_steps,
                        guidance_scale=guidance_scale
                    )
                    
                    total_time = time.time() - start_time
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(generated_image, caption="Generated Image", use_column_width=True)
                
                with col2:
                    st.write("**Original Text:**")
                    st.write(text_prompt)
                    st.write("**Generated Caption:**")
                    st.write(generated_caption)
                
                # Show timing info
                st.info(f"Total processing time: {total_time:.2f} seconds")
                
                # Download button
                st.download_button(
                    label="Download Generated Image",
                    data=self._image_to_bytes(generated_image),
                    file_name=f"cross_modal_{int(time.time())}.png",
                    mime="image/png"
                )
            else:
                st.error("Please enter a text prompt.")
    
    def batch_generation(self) -> None:
        """Batch generation interface."""
        st.subheader("Batch Generation")
        
        # Text prompts input
        prompts_text = st.text_area(
            "Enter multiple prompts (one per line):",
            value="A beautiful sunset over the ocean with a sailboat on the horizon.\nA cute golden retriever puppy playing in a green meadow.\nA modern city skyline at night with tall buildings and lights.",
            height=150
        )
        
        prompts = [p.strip() for p in prompts_text.split('\n') if p.strip()]
        
        if prompts:
            st.write(f"Number of prompts: {len(prompts)}")
            
            # Generation parameters
            col1, col2 = st.columns(2)
            
            with col1:
                num_steps = st.slider(
                    "Inference steps",
                    min_value=10,
                    max_value=100,
                    value=self.config["demo"]["default_steps"],
                    step=5,
                    key="batch_steps"
                )
            
            with col2:
                guidance_scale = st.slider(
                    "Guidance scale",
                    min_value=1.0,
                    max_value=20.0,
                    value=self.config["demo"]["default_guidance"],
                    step=0.5,
                    key="batch_guidance"
                )
            
            # Generate button
            if st.button("Generate All Images", type="primary"):
                if len(prompts) <= self.config["demo"]["max_images_per_request"]:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    generated_images = []
                    generated_captions = []
                    
                    for i, prompt in enumerate(prompts):
                        status_text.text(f"Generating image {i+1}/{len(prompts)}: {prompt[:50]}...")
                        
                        image, caption = self.model.cross_modal_translation(
                            prompt,
                            num_inference_steps=num_steps,
                            guidance_scale=guidance_scale
                        )
                        
                        generated_images.append(image)
                        generated_captions.append(caption)
                        
                        progress_bar.progress((i + 1) / len(prompts))
                    
                    status_text.text("Generation completed!")
                    
                    # Display results
                    for i, (prompt, image, caption) in enumerate(zip(prompts, generated_images, generated_captions)):
                        st.write(f"**Prompt {i+1}:** {prompt}")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
                        
                        with col2:
                            st.write(f"**Generated Caption:** {caption}")
                        
                        st.divider()
                else:
                    st.error(f"Too many prompts. Maximum allowed: {self.config['demo']['max_images_per_request']}")
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes for download.
        
        Args:
            image: PIL Image object.
            
        Returns:
            Image bytes.
        """
        import io
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def run(self) -> None:
        """Run the demo application."""
        st.set_page_config(
            page_title="Cross-Modal Translation Demo",
            page_icon="🔄",
            layout="wide"
        )
        
        st.title("🔄 Cross-Modal Translation Demo")
        st.markdown("Translate between text and images using state-of-the-art AI models")
        
        # Safety disclaimer
        st.warning("""
        **Safety Notice:** This demo uses AI models that may generate content. 
        Please use responsibly and be mindful of the content you generate.
        """)
        
        # Sidebar
        st.sidebar.title("Navigation")
        demo_mode = st.sidebar.selectbox(
            "Select Demo Mode:",
            ["Text-to-Image", "Image-to-Text", "Cross-Modal Translation", "Batch Generation"]
        )
        
        # Model info
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Model Information:**")
        st.sidebar.markdown(f"- Device: {self.device}")
        st.sidebar.markdown(f"- Precision: {self.config['device']['precision']}")
        st.sidebar.markdown(f"- Text-to-Image: Stable Diffusion")
        st.sidebar.markdown(f"- Image-to-Text: BLIP")
        
        # Main content
        if demo_mode == "Text-to-Image":
            self.text_to_image_generation()
        elif demo_mode == "Image-to-Text":
            self.image_to_text_generation()
        elif demo_mode == "Cross-Modal Translation":
            self.cross_modal_translation()
        elif demo_mode == "Batch Generation":
            self.batch_generation()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **About:** This demo showcases cross-modal translation between text and images.
        The system uses Stable Diffusion for text-to-image generation and BLIP for image-to-text generation.
        """)


def main():
    """Main function."""
    demo = CrossModalDemo()
    demo.run()


if __name__ == "__main__":
    main()
