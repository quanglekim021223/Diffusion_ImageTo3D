from download_weights import download_weights
from src.postprocess import ModelPostProcessor
from src.inference import load_model
from src.preprocess import ImagePreprocessor
import os
import sys
import torch
import argparse
from pathlib import Path
import time
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))


def setup_kaggle_environment():
    """
    Setup the Kaggle environment
    """
    # Check if running on Kaggle
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

    if is_kaggle:
        print("Running on Kaggle environment")

        # Create necessary directories
        os.makedirs('input', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Download pre-trained weights if not already present
        if not os.path.exists('models/model.pt'):
            download_weights()

        # Enable mixed precision for faster inference
        torch.backends.cudnn.benchmark = True
    else:
        print("Not running on Kaggle. This script is designed for Kaggle notebooks.")
        sys.exit(1)


def optimize_model(model):
    """
    Optimize the model for inference
    Args:
        model: The model to optimize
    Returns:
        Optimized model
    """
    # Enable gradient checkpointing
    if hasattr(model, 'unet'):
        model.unet.gradient_checkpointing_enable()

    # Convert to half precision
    model.half()

    # Move to GPU
    model.cuda()

    return model


def process_image(image_path, output_path, num_views=8, num_inference_steps=20):
    """
    Process a single image
    Args:
        image_path: Path to the input image
        output_path: Path to save the 3D model
        num_views: Number of novel views to generate
        num_inference_steps: Number of diffusion steps
    """
    # Initialize components
    preprocessor = ImagePreprocessor()
    model = load_model('models/model.pt')
    postprocessor = ModelPostProcessor()

    # Optimize model
    model = optimize_model(model)

    # Process image
    print(f"Processing image: {image_path}")
    start_time = time.time()

    image_tensor = preprocessor.preprocess(image_path)
    # Convert to half precision and move to GPU
    image_tensor = image_tensor.half().cuda()

    # Generate novel views
    print("Generating novel views...")
    novel_views = model.generate_novel_views(
        image_tensor, num_views, num_inference_steps)

    # Create 3D model
    print("Creating 3D model...")
    mesh = postprocessor.create_mesh_from_views(novel_views, output_path)

    end_time = time.time()
    print(f"3D model saved to: {output_path}")
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

    return mesh


def main():
    parser = argparse.ArgumentParser(
        description='Convert 2D image to 3D model on Kaggle')
    parser.add_argument('--input', type=str,
                        default='input/image.jpg', help='Path to input image')
    parser.add_argument(
        '--output', type=str, default='output/model.glb', help='Path to output 3D model')
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of novel views to generate')
    parser.add_argument('--num_inference_steps', type=int,
                        default=20, help='Number of diffusion steps')

    args = parser.parse_args()

    # Setup Kaggle environment
    setup_kaggle_environment()

    # Process image
    process_image(args.input, args.output, args.num_views,
                  args.num_inference_steps)


if __name__ == '__main__':
    main()
