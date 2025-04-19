import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_result_visualization(query_image_path, results, output_path=None, show=True):
    """Create visualization of retrieval results"""
    # Read query image
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    
    # Check if query image was successfully loaded
    if query_image is None:
        raise ValueError(f"Could not read query image: {query_image_path}")
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Add query image
    plt.subplot(1, 4, 1)
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis('off')
    
    # Add result images
    for i, result in enumerate(results[:3]):  # Show top 3 results
        # Read result image
        result_image = cv2.imread(result['path'])
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        
        # Create subplot
        plt.subplot(1, 4, i+2)
        plt.imshow(result_image)
        
        # Create title with similarity score (percentage)
        similarity_percent = result['similarity'] * 100
        plt.title(f"Result {i+1}\n{result['label']}\nSimilarity: {similarity_percent:.1f}%")
        plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Result visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def create_preprocessing_visualization(original_image_path, processed_image_path, output_path=None, show=True):
    """Create visualization of preprocessing steps"""
    # Read images
    original = cv2.imread(original_image_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    processed = cv2.imread(processed_image_path)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    
    # Check if images were successfully loaded
    if original is None:
        raise ValueError(f"Could not read original image: {original_image_path}")
    if processed is None:
        raise ValueError(f"Could not read processed image: {processed_image_path}")
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Add original image
    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis('off')
    
    # Add processed image
    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title("Processed Image")
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Preprocessing visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def create_feature_visualization(image_path, output_path=None, show=True):
    """Visualize various feature extraction steps for a leaf image"""
    # Read image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Check if image was successfully loaded
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Create grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get binary mask
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours for shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create contour image
    contour_img = image_rgb.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Create edge image for veins
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(enhanced, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Create color histogram visualization
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Add original image
    plt.subplot(2, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Add binary mask
    plt.subplot(2, 3, 2)
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Mask")
    plt.axis('off')
    
    # Add contour visualization
    plt.subplot(2, 3, 3)
    plt.imshow(contour_img)
    plt.title("Shape Analysis")
    plt.axis('off')
    
    # Add edge/vein visualization
    plt.subplot(2, 3, 4)
    plt.imshow(edges, cmap='gray')
    plt.title("Vein Detection")
    plt.axis('off')
    
    # Add color histograms
    plt.subplot(2, 3, 5)
    plt.plot(h_hist, color='r', label='H')
    plt.plot(s_hist, color='g', label='S')
    plt.plot(v_hist, color='b', label='V')
    plt.title("Color Features")
    plt.legend()
    
    # Add enhanced contrast visualization
    plt.subplot(2, 3, 6)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced Contrast")
    plt.axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path provided
    if output_path:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature visualization saved to {output_path}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    # Example usage
    query_image = "test_images/test1.JPG"
    results = [
        {'path': 'data/processed/apples/sample1.jpg', 'label': 'apples', 'similarity': 0.85},
        {'path': 'data/processed/apples/sample2.jpg', 'label': 'apples', 'similarity': 0.75},
        {'path': 'data/processed/blueberry/sample1.jpg', 'label': 'blueberry', 'similarity': 0.65}
    ]
    
    create_result_visualization(query_image, results, "results/example_result.png")