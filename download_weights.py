import os
import requests
from tqdm import tqdm
import hashlib
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_weights():
    """
    Download pre-trained weights from Hugging Face
    """
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Define model repository and file
    repo_id = "sudo-ai/zero123plusplus"
    filename = "model.pt"

    # Download the model
    print(f"Downloading {filename} from {repo_id}...")
    try:
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded to {model_path}")
        return model_path
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download the model manually from: https://huggingface.co/sudo-ai/zero123plusplus")
        return None


if __name__ == "__main__":
    download_weights()
