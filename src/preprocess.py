import cv2
import numpy as np
import os
from tqdm import tqdm

def remove_background(image):
    """Remove background from leaf image using GrabCut algorithm"""
    # Convert to RGB if in BGR format
    if len(image.shape) == 3 and image.shape[2] == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        img = image.copy()
    
    # Create initial mask
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # Set rectangular ROI based on finding leaf contours
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours of the leaf
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Use the largest contour for the rectangular ROI
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # Create a tighter rectangle for GrabCut
        rect = (x, y, w, h)
        
        # Initialize GrabCut's mask
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where definite and probable foreground are set to 1
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Create the resulting image
        result = img * mask2[:, :, np.newaxis]
        
        # Create white background
        white_background = np.ones_like(img) * 255
        background_area = np.where((mask2[:, :, np.newaxis] == 0), white_background, result)
        
        return background_area
    
    return img

def enhance_leaf_details(image):
    """Enhance leaf details like veins and edges"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE for enhanced contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Sharpen the image to enhance leaf veins and edges
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Return as RGB if input was RGB
    if len(image.shape) == 3:
        sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
        return sharpened_rgb
    
    return sharpened

def resize_image(image, target_size=(256, 256)):
    """Resize image to target size while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    # Calculate aspect ratio and resize
    aspect = w / h
    
    if aspect > 1:  # Width is larger
        new_w = target_size[0]
        new_h = int(new_w / aspect)
    else:  # Height is larger or equal
        new_h = target_size[1]
        new_w = int(new_h * aspect)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas of target size with white background
    canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
    
    # Center the image on the canvas
    y_offset = (target_size[1] - new_h) // 2
    x_offset = (target_size[0] - new_w) // 2
    
    # Place the resized image on the canvas
    if len(resized.shape) == 3:
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    else:
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    
    return canvas

def preprocess_image(image_path, output_path=None):
    """Full preprocessing pipeline for a single image"""
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize to 256x256
    image = resize_image(image)
    
    # Remove background
    image = remove_background(image)
    
    # Enhance leaf details
    enhanced_image = enhance_leaf_details(image)
    
    # Save if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, enhanced_image)
    
    return enhanced_image

def batch_process(input_dir, output_dir):
    """Process all leaf images in a directory"""
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} images to process.")
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing images"):
        # Construct output path
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        
        # Create subdirectories if needed
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        try:
            # Process image
            preprocess_image(img_path, out_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    print("Image preprocessing completed!")

if __name__ == "__main__":
    # Example usage
    input_dir = "data/raw"
    output_dir = "data/processed"
    batch_process(input_dir, output_dir)