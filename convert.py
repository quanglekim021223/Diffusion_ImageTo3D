import argparse
from pathlib import Path
from src.preprocess import ImagePreprocessor
from src.inference import load_model
from src.postprocess import ModelPostProcessor


def main():
    parser = argparse.ArgumentParser(
        description='Convert 2D image to 3D model')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output 3D model')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model weights')
    parser.add_argument('--num_views', type=int, default=8,
                        help='Number of novel views to generate')

    args = parser.parse_args()

    # Initialize components
    preprocessor = ImagePreprocessor()
    model = load_model(args.model_path)
    postprocessor = ModelPostProcessor()

    # Process image
    print(f"Processing image: {args.input}")
    image_tensor = preprocessor.preprocess(args.input)

    # Generate novel views
    print("Generating novel views...")
    novel_views = model.generate_novel_views(image_tensor, args.num_views)

    # Create 3D model
    print("Creating 3D model...")
    postprocessor.create_mesh_from_views(novel_views, args.output)

    print(f"3D model saved to: {args.output}")


if __name__ == '__main__':
    main()
