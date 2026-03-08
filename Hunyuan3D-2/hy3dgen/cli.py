"""
Multiview to 3D Generation - Clean CLI
Command-line interface for generating 3D models from multiview images
"""
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from PIL import Image
import torch
import os
import argparse


def find_view_images(input_dir):
    """
    Find view images in the input directory.
    
    Args:
        input_dir: Directory containing view images
    
    Returns:
        List of paths to found images, list of view names
    """
    view_files = {
        'front': 'front.png',
        'back': 'back.png',
        'left': 'left.png',
        'right': 'right.png',
    }
    
    found_images = []
    found_views = []
    
    print(f"\n{'='*70}")
    print(f"Searching for view images in: {input_dir}")
    print(f"{'='*70}")
    
    for view_name, filename in view_files.items():
        filepath = os.path.join(input_dir, filename)
        if os.path.exists(filepath):
            found_images.append(filepath)
            found_views.append(view_name)
            print(f"✓ Found {view_name} view: {filename}")
        else:
            print(f"⊘ Missing {view_name} view: {filename}")
    
   
    
    if len(found_images) == 0:
        raise FileNotFoundError(
            f"No view images found in {input_dir}!\n"
            f"Expected: front.png, back.png, left.png, right.png"
        )
    
    if len(found_images) < 2:
        print(f"WARNING: Only {len(found_images)} view(s) found. "
              f"Recommend at least 2-3 views for best results.")
    
    print(f"\nTotal views found: {len(found_images)}")
    
    return found_images, found_views


def generate_3d(input_dir, output_path, with_texture=False, use_turbo=True):
    """
    Generate 3D model from multiview images.
    
    Args:
        input_dir: Directory containing view images
        output_path: Path to save output 3D model
        with_texture: Whether to generate texture
        use_turbo: Whether to use turbo models (faster)
    """
    # 1. Find view images
    image_paths, view_names = find_view_images(input_dir)
    
    # 2. Load images as dictionary with view tags
    print(f"\n{'='*70}")
    print("LOADING IMAGES")
    print(f"{'='*70}")
    images_dict = {}
    for path, view in zip(image_paths, view_names):
        img = Image.open(path).convert('RGBA')
        images_dict[view] = img
        print(f"Loaded {view} view: {img.size} pixels")
    

    print(f"SHAPE GENERATION ({'TURBO' if use_turbo else 'STANDARD'} MODE)")
  
    
    subfolder = 'hunyuan3d-dit-v2-mv-turbo' if use_turbo else 'hunyuan3d-dit-v2-mv'
    print(f"Loading model: tencent/Hunyuan3D-2mv/{subfolder}")
    
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'tencent/Hunyuan3D-2mv',
        subfolder=subfolder,
        torch_dtype=torch.float16,
    )
    
    print("Generating 3D geometry from multiview images...")
    print("This may take 1-2 minutes...")
    
    mesh = shape_pipeline(
        image=images_dict,  # Pass dictionary with view tags
        num_inference_steps=25 if use_turbo else 50,
        guidance_scale=3.0 if use_turbo else 7.5,
    )[0]
    
    print(f"Generated mesh: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    
    # Clean up to save memory
    del shape_pipeline
    torch.cuda.empty_cache()
    
    # 4. (Optional) Add texture
    if with_texture:
        print(f"TEXTURE GENERATION ({'TURBO' if use_turbo else 'STANDARD'} MODE)")
        print(" Warning: Texture generation requires compiled CUDA modules")
        print(" If not installed, use --no_texture flag")
        
        try:
            texture_subfolder = 'hunyuan3d-paint-v2-0-turbo' if use_turbo else 'hunyuan3d-paint-v2-0'
            print(f"Loading model: tencent/Hunyuan3D-2/{texture_subfolder}")
            
            texture_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                'tencent/Hunyuan3D-2',
                subfolder=texture_subfolder,
            )
            
            print("Generating texture...")
            print("Using first view as texture reference")
            print("This may take 2-3 minutes...")
            
            # Use first view as reference
            reference_image = list(images_dict.values())[0]
            mesh = texture_pipeline(
                mesh,
                image=reference_image,
                num_inference_steps=15 if use_turbo else 30
            )
            
            print("Texture generation complete")
            
            # Clean up
            del texture_pipeline
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Texture generation failed: {e}")
            print("  Continuing with geometry only...")
    
    # 5. Save the result
    print("SAVING OUTPUT")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    mesh.export(output_path)
    print(f"Saved 3D model to: {output_path}")
    
    # Print file info
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    print(f"  File size: {file_size:.2f} MB")

    
    return mesh


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='Generate 3D models from multiview images using Hunyuan3D-2mv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (geometry only, fastest)
  multiview-3d-gen --input_dir ./my_views --output model.glb --no_texture
  
  # With texture (requires compiled CUDA modules)
  multiview-3d-gen --input_dir ./my_views --output model.glb
  
  # High quality mode (slower)
  multiview-3d-gen --input_dir ./my_views --output model.glb --standard

Required files in input_dir:
  - front.png (required)
  - back.png (required)
  - left.png (optional)
  - right.png (optional)
  
Supported views: front, back, left, right (horizontal views only)
At least 2 views required, 3-4 recommended for best results.
        """
    )
    
    parser.add_argument(
        '--input_dir', '-i',
        type=str,
        required=True,
        help='Directory containing view images (front.png, back.png, left.png, right.png)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file path (e.g., output.glb, result.obj, result.stl)'
    )
    
    parser.add_argument(
        '--no_texture',
        action='store_true',
        help='Skip texture generation (geometry only, faster, no CUDA compilation needed)'
    )
    
    parser.add_argument(
        '--standard',
        action='store_true',
        help='Use standard models instead of turbo (higher quality but slower)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        print(f" Error: Input directory not found: {args.input_dir}")
        return 1
    
    # Print settings


    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output}")
    print(f"Texture generation: {'No' if args.no_texture else 'Yes'}")
    print(f"Model variant: {'Standard' if args.standard else 'Turbo'}")

    # Run generation
    try:
        generate_3d(
            input_dir=args.input_dir,
            output_path=args.output,
            with_texture=not args.no_texture,
            use_turbo=not args.standard
        )
        
     
        print(f"\nYour 3D model is ready: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())