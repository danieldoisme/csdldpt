import cv2
import numpy as np
import os

def preprocess_leaf_image(image_path, output_path):
    """
    Preprocess a leaf image by removing the background and isolating the leaf
    with well-preserved edges.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
    """
    # Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    # Step 2: Create a backup of the original image
    original = image.copy()

    # Step 3: Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Step 4: Define multiple HSV ranges to cover different green and brown shades
    # Standard green range
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Dark green range (for shadows and darker parts)
    lower_dark_green = np.array([25, 20, 20])
    upper_dark_green = np.array([90, 150, 150])
    
    # Create masks for each range
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    dark_green_mask = cv2.inRange(hsv, lower_dark_green, upper_dark_green)
    
    # Combine masks to capture both light and dark parts of the leaf
    combined_mask = cv2.bitwise_or(green_mask, dark_green_mask)

    # Step 5: Apply morphological operations to clean up the mask
    small_kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, small_kernel)
    
    # Use a larger kernel for dilation to ensure edges are included
    dilate_kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)
    
    # Step 6: Find the leaf contour
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Step 7: Create a refined mask with the precise contour
        filled_mask = np.zeros_like(combined_mask)
        cv2.drawContours(filled_mask, [largest_contour], 0, 255, -1)
        
        # Step 8: Apply flood fill from the center of the contour to ensure the whole leaf is captured
        # Calculate centroid of the contour
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Create a separate mask for flood filling
            flood_mask = filled_mask.copy()
            h, w = flood_mask.shape[:2]
            flood_fill_mask = np.zeros((h+2, w+2), np.uint8)
            cv2.floodFill(flood_mask, flood_fill_mask, (cX, cY), 255)
            
            # Combine the contour mask with the flood fill mask
            filled_mask = cv2.bitwise_or(filled_mask, flood_mask)
        
        # Step 9: Extract the edge mask for the leaf
        edge_mask = np.zeros_like(combined_mask)
        cv2.drawContours(edge_mask, [largest_contour], 0, 255, 1)  # Draw only the edge
        
        # Step 10: Apply the mask to the original image
        leaf_only = cv2.bitwise_and(original, original, mask=filled_mask)
    else:
        filled_mask = combined_mask
        edge_mask = np.zeros_like(combined_mask)
        leaf_only = cv2.bitwise_and(original, original, mask=filled_mask)
    
    # Step 11: Create white background instead of black
    bg_mask_inv = cv2.bitwise_not(filled_mask)
    white_bg = np.ones_like(original, np.uint8) * 255
    background = cv2.bitwise_and(white_bg, white_bg, mask=bg_mask_inv)
    final_image = cv2.add(leaf_only, background)
    
    # Step 12: Edge enhancement
    edge_enhanced = final_image.copy()
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(edge_enhanced, -1, kernel)
    
    # Only apply sharpening near the edges
    edge_dilated = cv2.dilate(edge_mask, np.ones((3,3), np.uint8), iterations=2)
    final_image = np.where(edge_dilated[:,:,np.newaxis] > 0, sharpened, final_image)

    # Step 13: Save the processed image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure output directory exists
    cv2.imwrite(output_path, final_image)
    
    return final_image