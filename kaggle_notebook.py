from download_weights import download_weights
from src.postprocess import ModelPostProcessor
from src.inference import load_model, Zero123PlusPlus
from src.preprocess import ImagePreprocessor
import os
import sys
import torch
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import logging
import trimesh
import numpy as np
import open3d as o3d
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import gc

# Add the project root to the Python path
sys.path.append(os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('image_to_3d.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Configure mixed precision
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None


def setup_kaggle_environment():
    """
    Setup the Kaggle environment
    """
    # Check if running on Kaggle
    is_kaggle = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None

    if is_kaggle:
        logger.info("Running on Kaggle environment")

        # Create necessary directories
        os.makedirs('input', exist_ok=True)
        os.makedirs('output', exist_ok=True)
        os.makedirs('models', exist_ok=True)

        # Download pre-trained weights if not already present
        if not os.path.exists('models/model.pt'):
            try:
                logger.info("Downloading pre-trained weights...")
                download_weights()
                logger.info("Weights downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download weights: {e}")
                raise

        # Enable mixed precision for faster inference
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA benchmark enabled for faster inference")
    else:
        logger.warning(
            "Not running on Kaggle. This script is designed for Kaggle notebooks.")
        sys.exit(1)


def optimize_model(model):
    """
    Optimize the model for inference
    Args:
        model: The model to optimize
    Returns:
        Optimized model
    """
    try:
        # Enable gradient checkpointing
        if hasattr(model, 'unet'):
            model.unet.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")

        # Convert to half precision
        model.half()
        logger.info("Model converted to half precision")

        # Move to GPU
        model.cuda()
        logger.info("Model moved to GPU")

        return model
    except Exception as e:
        logger.error(f"Error optimizing model: {e}")
        raise


def process_image(input_path, output_path, num_views=8, elevation_range=(-20, 40), azimuth_range=(0, 360),
                  num_inference_steps=50, apply_texture=True, mesh_quality='high'):
    """
    Xử lý hình ảnh đầu vào để tạo mô hình 3D

    Args:
        input_path: Đường dẫn đến hình ảnh đầu vào
        output_path: Đường dẫn để lưu mô hình 3D
        num_views: Số lượng góc nhìn mới để tạo
        elevation_range: Phạm vi góc nâng (độ)
        azimuth_range: Phạm vi góc phương vị (độ)
        num_inference_steps: Số bước suy luận cho quá trình diffusion
        apply_texture: Có áp dụng kết cấu cho mô hình 3D không
        mesh_quality: Chất lượng mesh ('low', 'medium', 'high')

    Returns:
        Đường dẫn đến mô hình 3D đã tạo
    """
    try:
        # Kiểm tra đầu vào
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        # Kiểm tra tham số
        if num_views < 4:
            logger.warning(
                f"Number of views ({num_views}) is too low. Setting to 4.")
            num_views = 4

        if num_inference_steps < 20:
            logger.warning(
                f"Number of inference steps ({num_inference_steps}) is too low. Setting to 20.")
            num_inference_steps = 20

        if mesh_quality not in ['low', 'medium', 'high']:
            logger.warning(
                f"Invalid mesh quality: {mesh_quality}. Setting to 'high'.")
            mesh_quality = 'high'

        logger.info(f"Processing image: {input_path}")

        # Kiểm tra thiết bị
        if device.type == 'cuda':
            # Kiểm tra bộ nhớ GPU
            torch.cuda.empty_cache()
            logger.info(
                f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")

        # Khởi tạo mô hình
        logger.info("Initializing Zero123PlusPlus model with latent diffusion")
        model = Zero123PlusPlus(
            image_size=256,
            channels=3,
            dim=128,
            condition_dim=4,
            latent_dim=4  # Kích thước không gian latent
        ).to(device)

        # Tối ưu hóa mô hình cho suy luận
        model.optimize_for_inference()

        # Xử lý hình ảnh đầu vào
        logger.info(f"Loading input image: {input_path}")
        try:
            input_image = Image.open(input_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")

        # Tạo các góc nhìn mới
        logger.info(
            f"Generating {num_views} novel views with latent diffusion")
        try:
            with torch.cuda.amp.autocast() if device.type == 'cuda' else nullcontext():
                novel_views = model.generate_novel_views(
                    input_image=input_image,
                    num_views=num_views,
                    elevation_range=elevation_range,
                    azimuth_range=azimuth_range,
                    num_inference_steps=num_inference_steps
                )
        except Exception as e:
            raise RuntimeError(f"Failed to generate novel views: {e}")

        # Lưu các góc nhìn mới
        os.makedirs('novel_views', exist_ok=True)
        for i, view in enumerate(novel_views):
            view_path = f'novel_views/view_{i}.png'
            view.save(view_path)
            logger.info(f"Saved novel view {i+1}/{num_views} to {view_path}")

        # Tạo mô hình 3D từ các góc nhìn mới
        logger.info("Creating 3D model from novel views")
        post_processor = ModelPostProcessor()

        # Thiết lập chất lượng mesh dựa trên tham số
        if mesh_quality == 'low':
            use_poisson = False
            depth = 5
        elif mesh_quality == 'medium':
            use_poisson = True
            depth = 7
        else:  # high
            use_poisson = True
            depth = 9

        try:
            mesh = post_processor.create_mesh_from_views(
                novel_views,
                use_poisson=use_poisson,
                depth=depth,
                apply_texture=apply_texture
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create 3D mesh: {e}")

        # Lưu mô hình 3D
        logger.info(f"Saving 3D model to {output_path}")
        try:
            post_processor.save_mesh(mesh, output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to save 3D model: {e}")

        # Xóa bộ nhớ cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(
                f"GPU Memory after processing: {torch.cuda.memory_allocated() / 1024**2:.2f} MB allocated")

        return output_path

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


def main():
    """Hàm chính để chạy notebook"""
    parser = argparse.ArgumentParser(
        description='Convert 2D image to 3D model using Zero123++ with latent diffusion')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output 3D model')
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of novel views to generate')
    parser.add_argument('--elevation_min', type=float,
                        default=-20, help='Minimum elevation angle')
    parser.add_argument('--elevation_max', type=float,
                        default=40, help='Maximum elevation angle')
    parser.add_argument('--azimuth_min', type=float,
                        default=0, help='Minimum azimuth angle')
    parser.add_argument('--azimuth_max', type=float,
                        default=360, help='Maximum azimuth angle')
    parser.add_argument('--num_inference_steps', type=int,
                        default=50, help='Number of inference steps')
    parser.add_argument('--apply_texture', action='store_true',
                        help='Apply texture to 3D model')
    parser.add_argument('--mesh_quality', type=str, default='high', choices=['low', 'medium', 'high'],
                        help='Quality of the 3D mesh')

    args = parser.parse_args()

    # Xử lý hình ảnh
    output_path = process_image(
        input_path=args.input,
        output_path=args.output,
        num_views=args.num_views,
        elevation_range=(args.elevation_min, args.elevation_max),
        azimuth_range=(args.azimuth_min, args.azimuth_max),
        num_inference_steps=args.num_inference_steps,
        apply_texture=args.apply_texture,
        mesh_quality=args.mesh_quality
    )

    logger.info(f"3D model created successfully: {output_path}")

# Context manager cho việc sử dụng mixed precision


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


if __name__ == "__main__":
    main()
