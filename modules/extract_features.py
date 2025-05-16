import os
import glob
import time
import pandas as pd
from src.feature_extraction import extract_all_features

def extract_features_for_all_images():
    """
    Extract features from all preprocessed leaf images and save them as CSV files.
    
    For each tree type, creates a CSV file containing features of all its images.
    Also creates a combined CSV with all images from all tree types.
    """
    # Base directory
    processed_base_dir = "data/stored_images/processed"
    features_dir = "data/features"
    
    # Create the features directory if it doesn't exist
    os.makedirs(features_dir, exist_ok=True)
    
    # Get all tree type directories
    tree_types = [d for d in os.listdir(processed_base_dir) 
                 if os.path.isdir(os.path.join(processed_base_dir, d))]
    
    total_images = 0
    processed_images = 0
    errors = []
    all_features = []
    
    start_time = time.time()
    
    print(f"Found {len(tree_types)} tree types: {', '.join(tree_types)}")
    
    # Process each tree type
    for tree_type in tree_types:
        print(f"\nProcessing {tree_type} leaves...")
        
        # Get all images for this tree type
        processed_dir = os.path.join(processed_base_dir, tree_type)
        image_files = glob.glob(os.path.join(processed_dir, "*.jpg")) + \
                      glob.glob(os.path.join(processed_dir, "*.JPG")) + \
                      glob.glob(os.path.join(processed_dir, "*.jpeg")) + \
                      glob.glob(os.path.join(processed_dir, "*.png"))
        
        tree_features = []
        total_images += len(image_files)
        
        # Process each image
        for i, image_path in enumerate(image_files):
            try:
                # Extract features
                features = extract_all_features(image_path)
                
                # Add tree type
                features['tree_type'] = tree_type
                
                # Add to lists
                tree_features.append(features)
                all_features.append(features)
                
                processed_images += 1
                print(f"  Processed [{i+1}/{len(image_files)}]: {features['image_name']}")
                
            except Exception as e:
                error_msg = f"Error processing {image_path}: {str(e)}"
                print(f"  {error_msg}")
                errors.append(error_msg)
        
        # Save features for this tree type
        if tree_features:
            df = pd.DataFrame(tree_features)
            csv_path = os.path.join(features_dir, f"{tree_type}_features.csv")
            df.to_csv(csv_path, index=False)
            print(f"Saved features for {tree_type} to {csv_path}")
    
    # Save combined features
    if all_features:
        df = pd.DataFrame(all_features)
        csv_path = os.path.join(features_dir, "all_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved combined features to {csv_path}")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*50)
    print("Feature Extraction Complete!")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Errors: {len(errors)}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    
    # Print errors if any
    if errors:
        print("\nErrors encountered:")
        for error in errors:
            print(f"  - {error}")
