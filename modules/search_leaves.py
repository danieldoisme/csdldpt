import os
import cv2
import matplotlib.pyplot as plt
from src.preprocessing import preprocess_leaf_image
import glob 
import numpy as np # Added for placeholder image in display_results

def display_results(query_image_path, similar_images, output_path=None):
    """
    Display the query image and its similar matches.
    
    Args:
        query_image_path: Path to the query image
        similar_images: DataFrame containing similar image information
        output_path: Path to save the result image (optional)
    """
    # Load the query image
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"Warning: Query image not found at {query_image_path} for display.")
        # Create a placeholder image or return if critical
        query_img = np.zeros((100,100,3), dtype=np.uint8) # Placeholder
        plt.text(0.5, 0.5, 'Query Image Not Found', ha='center', va='center')

    else:
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # Create figure
    # Adjust num_results to be at most the number of available similar_images + 1 for query
    1 + len(similar_images) if similar_images is not None and not similar_images.empty else 1
    # Ensure num_plots doesn't exceed a reasonable number for subplotting, e.g., 1 (query) + 3 (results)
    # This was previously hardcoded to 4 subplots (1 query + 3 results)
    # Let's make it dynamic up to a max, e.g., 5 (1 query + 4 results)
    max_display_results = 4 # Max similar results to display
    num_cols_display = 1 + min(len(similar_images) if similar_images is not None else 0, max_display_results)


    plt.figure(figsize=(3 * num_cols_display, 3)) # Adjusted figsize
    
    # Display query image
    plt.subplot(1, num_cols_display, 1)
    plt.imshow(query_img)
    plt.title('Query Image')
    plt.axis('off')
    
    # Display similar images
    if similar_images is not None and not similar_images.empty:
        for i, (_, row) in enumerate(similar_images.head(max_display_results).iterrows(), 1):
            # Get the image path
            img_path = os.path.join('data/stored_images/processed', row['tree_type'], row['image_name'])
            
            # Load and display image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Similar image not found at {img_path} for display.")
                img = np.zeros((100,100,3), dtype=np.uint8) # Placeholder
                title_text = f"Match {i}\nNot Found"
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                title_text = f"Match {i}\nType: {row['tree_type']}\nDistance: {row['distance']:.3f}"

            plt.subplot(1, num_cols_display, i + 1)
            plt.imshow(img)
            plt.title(title_text)
            plt.axis('off')
    
    plt.tight_layout()
    
    # Save the figure if output path is provided
    if output_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            plt.savefig(output_path)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving result image: {e}")

    
    # Show plot if in interactive mode (usually handled by matplotlib backend)
    # For CLI, if no output_path, plt.show() is desired.
    if not output_path:
        plt.show()

def process_search_directory():
    """
    Preprocess all images in the search directory.
    Reads from search_images/{tree_type}/ and saves to search_images/processed/{tree_type}/
    """
    # Base directories
    raw_search_base_dir = "search_images" # Updated: search_images is the base for tree_type folders
    processed_search_base_dir = os.path.join("search_images", "processed") # Processed images will go here
    
    # Create the processed base directory if it doesn't exist
    os.makedirs(processed_search_base_dir, exist_ok=True)
    
    # Get all tree type directories from the raw_search_base_dir
    if not os.path.exists(raw_search_base_dir):
        print(f"Base search directory not found: {raw_search_base_dir}")
        return
        
    tree_types = [d for d in os.listdir(raw_search_base_dir) 
                 if os.path.isdir(os.path.join(raw_search_base_dir, d) ) and d != "processed"] # Exclude 'processed' if it's a subdir
    
    if not tree_types:
        print(f"No tree type subdirectories found in {raw_search_base_dir}")
        return
    print(f"Found {len(tree_types)} tree types in search directory: {', '.join(tree_types)}")
    
    # Process each tree type
    for tree_type in tree_types:
        # Define raw directory for this tree type
        raw_tree_type_dir = os.path.join(raw_search_base_dir, tree_type)
        
        # Create the processed directory for this tree type
        processed_tree_type_dir = os.path.join(processed_search_base_dir, tree_type)
        os.makedirs(processed_tree_type_dir, exist_ok=True)
        
        # Get all images for this tree type
        image_files = []
        for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(raw_tree_type_dir, ext)))
        
        if not image_files:
            print(f"No images found for {tree_type} in {raw_tree_type_dir}")
            continue
            
        print(f"Processing {len(image_files)} images for {tree_type} from {raw_tree_type_dir}...")
        
        # Process each image
        for image_path in image_files:
            try:
                # Get image filename
                filename = os.path.basename(image_path)
                output_path = os.path.join(processed_tree_type_dir, filename)
                
                # Process the image
                preprocess_leaf_image(image_path, output_path)
                print(f"  Processed: {filename} -> {output_path}")
                
            except FileNotFoundError as fnf_error:
                print(f"  Error processing {image_path} (File Not Found): {str(fnf_error)}")
            except Exception as e:
                print(f"  Error processing {image_path}: {str(e)}")
