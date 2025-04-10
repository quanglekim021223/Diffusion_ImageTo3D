import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


class ImagePreprocessor:
    def __init__(self, image_size=256):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess(self, image_path):
        """
        Preprocess the input image for the model
        Args:
            image_path: Path to the input image
        Returns:
            Preprocessed image tensor
        """
        # Read image
        image = Image.open(image_path).convert('RGB')

        # Apply transformations
        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    def postprocess(self, tensor):
        """
        Convert model output tensor back to image
        Args:
            tensor: Model output tensor
        Returns:
            PIL Image
        """
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).cpu()

        # Denormalize
        tensor = tensor * 0.5 + 0.5
        tensor = tensor.clamp(0, 1)

        # Convert to PIL Image
        image = transforms.ToPILImage()(tensor)

        return image
