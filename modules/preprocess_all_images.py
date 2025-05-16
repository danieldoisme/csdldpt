import os
import glob
import time
from src.preprocessing import preprocess_leaf_image

def preprocess_all_images():
    """
    Batch process all leaf images from raw folders to processed folders.
    
    The script looks for images in data/stored_images/raw/{tree_type}/ and 
    saves processed images to data/stored_images/processed/{tree_type}/
    """
    # Base directories
    raw_base_dir = "data/stored_images/raw"
    processed_base_dir = "data/stored_images/processed"
    
    # Create the processed base directory if it doesn't exist
    os.makedirs(processed_base_dir, exist_ok=True)
    
    # Get all tree type directories
    tree_types = [d for d in os.listdir(raw_base_dir) 
                 if os.path.isdir(os.path.join(raw_base_dir, d))]
    
    total_images = 0
    processed_images = 0
    errors = []
    
    start_time = time.time()
    
    print(f"Found {len(tree_types)} tree types: {', '.join(tree_types)}")
    
    # Process each tree type
    for tree_type in tree_types:
        print(f"\nProcessing {tree_type} leaves...")
        
        # Create the processed directory for this tree type if it doesn't exist
        processed_dir = os.path.join(processed_base_dir, tree_type)
        os.makedirs(processed_dir, exist_ok=True)
        
        # Get all images for this tree type
        raw_dir = os.path.join(raw_base_dir, tree_type)
        image_files = glob.glob(os.path.join(raw_dir, "*.jpg")) + \
                     glob.glob(os.path.join(raw_dir, "*.JPG")) + \
                     glob.glob(os.path.join(raw_dir, "*.jpeg")) + \
                     glob.glob(os.path.join(raw_dir, "*.png"))
        
        total_images += len(image_files)
        
        # Process each image
        for i, image_path in enumerate(image_files):
            try:
                # Get image filename
                filename = os.path.basename(image_path)
                output_path = os.path.join(processed_dir, filename)
                
                # Process the image
                preprocess_leaf_image(image_path, output_path)
                
                processed_images += 1
                print(f"  Processed [{i+1}/{len(image_files)}]: {filename}")
                
            except Exception as e:
                error_msg = f"Error processing {image_path}: {str(e)}"
                print(f"  {error_msg}")
                errors.append(error_msg)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("Processing Complete!")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Errors: {len(errors)}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Print errors if any
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")
