# Image to 3D Model Converter

This project converts 2D images to 3D models using Zero123++ pre-trained model, optimized for game development.

## Overview

This project provides a complete pipeline for converting 2D images to 3D models using state-of-the-art AI techniques:

1. **Image Preprocessing**: Prepare the input image for the model
2. **Novel View Generation**: Generate multiple views of the object using Zero123++
3. **3D Reconstruction**: Create a 3D mesh from the generated views
4. **Mesh Optimization**: Optimize the mesh for better quality
5. **Export**: Export the 3D model in formats suitable for game engines

## Setup Instructions

### For Kaggle (Recommended):

1. Create a new notebook on Kaggle
2. Enable GPU accelerator (P100 or better recommended)
3. Clone this repository:
   ```bash
   !git clone https://github.com/yourusername/image-to-3d.git
   %cd image-to-3d
   ```
4. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
5. Download pre-trained weights:
   ```bash
   !python download_weights.py
   ```
6. Run the notebook:
   ```bash
   !python kaggle_notebook.py
   ```

### For Google Colab:

1. Create a new notebook
2. Enable GPU runtime
3. Clone this repository:
   ```bash
   !git clone https://github.com/yourusername/image-to-3d.git
   %cd image-to-3d
   ```
4. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
5. Download pre-trained weights:
   ```bash
   !python download_weights.py
   ```
6. Run the notebook:
   ```bash
   !python kaggle_notebook.py
   ```

## Usage

### Command Line Interface

```bash
python convert.py --input path/to/image.jpg --output output/model.glb --num_views 8
```

### Python API

```python
from src.preprocess import ImagePreprocessor
from src.inference import load_model
from src.postprocess import ModelPostProcessor

# Initialize components
preprocessor = ImagePreprocessor()
model = load_model('models/model.pt')
postprocessor = ModelPostProcessor()

# Process image
image_tensor = preprocessor.preprocess('input/image.jpg')
novel_views = model.generate_novel_views(image_tensor, num_views=8)
mesh = postprocessor.create_mesh_from_views(novel_views, 'output/model.glb')
```

## Project Structure

```
.
├── input/              # Input images
├── output/             # Generated 3D models
├── models/             # Pre-trained model weights
├── src/               # Source code
│   ├── preprocess.py  # Image preprocessing
│   ├── inference.py   # Model inference
│   └── postprocess.py # 3D model post-processing
├── convert.py         # Command-line interface
├── download_weights.py # Script to download pre-trained weights
├── kaggle_notebook.py # Script to run on Kaggle
├── kaggle_notebook.ipynb # Jupyter notebook for Kaggle
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## Technical Details

### Model Architecture

The project uses Zero123++, a state-of-the-art model for novel view synthesis. The model consists of:

1. **Encoder**: Converts the input image into a latent representation
2. **View Conditioner**: Conditions the latent representation on the desired view angle
3. **Decoder**: Generates a novel view from the conditioned latent representation

### 3D Reconstruction Pipeline

The 3D reconstruction pipeline consists of:

1. **Feature Detection**: Detect keypoints in the generated views
2. **Feature Matching**: Match keypoints between views
3. **Point Cloud Generation**: Triangulate 3D points from matched keypoints
4. **Mesh Reconstruction**: Create a mesh from the point cloud using Poisson surface reconstruction
5. **Mesh Optimization**: Optimize the mesh for better quality

## Requirements

- Python 3.8+
- CUDA-capable GPU (provided by Kaggle/Colab)
- 16GB+ RAM recommended
- PyTorch 2.0+
- Open3D 0.15+
- Trimesh 3.9+

## Performance Considerations

- **GPU Memory**: The model requires approximately 8GB of GPU memory
- **Processing Time**: Processing a single image takes approximately 1-2 minutes on a P100 GPU
- **Output Quality**: The quality of the 3D model depends on the input image quality and the number of novel views

## License

MIT License 