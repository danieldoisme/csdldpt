import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
from tqdm import tqdm

def extract_color_features(image):
    """Extract color histograms as features"""
    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms for each channel
    h_hist = cv2.calcHist([hsv], [0], None, [30], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [32], [0, 256])
    
    # Normalize histograms
    h_hist = cv2.normalize(h_hist, h_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    
    # Concatenate histograms
    color_features = np.concatenate((h_hist, s_hist, v_hist))
    
    return color_features

def extract_shape_features(image):
    """Extract shape features using Hu moments and contour properties"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize features
    shape_features = np.zeros(15)
    
    if len(contours) > 0:
        # Find the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Calculate Hu moments
        moments = cv2.moments(max_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Apply log transform to Hu moments
        for i in range(7):
            if hu_moments[i] != 0:
                hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
        
        # Calculate contour properties
        area = cv2.contourArea(max_contour)
        perimeter = cv2.arcLength(max_contour, True)
        
        # Calculate shape properties
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate bounding rectangle
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        extent = float(area) / (w * h) if w > 0 and h > 0 else 0
        
        # Calculate minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        equi_diameter = np.sqrt(4 * area / np.pi)
        
        # Combine all shape features
        shape_features[:7] = hu_moments
        shape_features[7] = circularity
        shape_features[8] = aspect_ratio
        shape_features[9] = extent
        shape_features[10] = area / (binary.shape[0] * binary.shape[1])  # Relative area
        shape_features[11] = perimeter / (binary.shape[0] + binary.shape[1])  # Relative perimeter
        shape_features[12] = equi_diameter / max(binary.shape[0], binary.shape[1])  # Relative diameter
        
        # Calculate convex hull and convexity
        hull = cv2.convexHull(max_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        shape_features[13] = solidity
        
        # Calculate ellipse fit
        if len(max_contour) >= 5:  # Need at least 5 points to fit an ellipse
            ellipse = cv2.fitEllipse(max_contour)
            # Ratio of major to minor axis
            shape_features[14] = max(ellipse[1]) / min(ellipse[1]) if min(ellipse[1]) > 0 else 0
    
    return shape_features

def extract_texture_features(image):
    """Extract texture features using Local Binary Patterns (LBP)"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Parameters for LBP
    radius = 3
    n_points = 8 * radius
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    
    # Calculate LBP histogram
    n_bins = n_points + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_vein_features(image):
    """Extract features related to leaf veins"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Enhance veins
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply Canny edge detection to find veins
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Apply morphological operations to enhance veins
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Calculate HoG features on the vein image
    win_size = (64, 64)
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size = (8, 8)
    nbins = 9
    
    # Resize for HoG
    resized = cv2.resize(dilated, (128, 128))
    
    # Calculate HoG features
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    vein_features = hog.compute(resized)
    
    # Normalize and reduce dimensions for efficiency
    vein_features = normalize(vein_features, norm='l2')
    
    # Take a subset of HOG features to reduce dimensionality
    vein_features = vein_features[::4]  # Take every 4th element
    
    return vein_features.flatten()

def extract_edge_features(image):
    """Extract features related to leaf edges/margins"""
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Threshold to get a binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialize features
    edge_features = np.zeros(20)
    
    if len(contours) > 0:
        # Get the largest contour
        max_contour = max(contours, key=cv2.contourArea)
        
        # Simplify contour a bit
        epsilon = 0.005 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # Calculate curvature at each point
        if len(approx) >= 3:  # Need at least 3 points for curvature
            curvature = []
            contour_array = approx.reshape(-1, 2)
            n_points = len(contour_array)
            
            for i in range(n_points):
                # Get previous, current, and next point
                prev_idx = (i - 1) % n_points
                next_idx = (i + 1) % n_points
                
                prev = contour_array[prev_idx]
                curr = contour_array[i]
                next_pt = contour_array[next_idx]
                
                # Calculate vectors
                v1 = prev - curr
                v2 = next_pt - curr
                
                # Normalize vectors
                v1_norm = np.linalg.norm(v1)
                v2_norm = np.linalg.norm(v2)
                
                if v1_norm > 0 and v2_norm > 0:
                    v1 = v1 / v1_norm
                    v2 = v2 / v2_norm
                    
                    # Calculate the angle between vectors
                    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    
                    # Add to curvature list
                    curvature.append(angle)
            
            # Calculate curvature statistics
            if curvature:
                curvature = np.array(curvature)
                # Histogram of curvature values (10 bins)
                hist, _ = np.histogram(curvature, bins=10, range=(0, np.pi))
                edge_features[:10] = hist / sum(hist) if sum(hist) > 0 else hist
                
                # Statistical measures
                edge_features[10] = np.mean(curvature)
                edge_features[11] = np.std(curvature)
                edge_features[12] = np.max(curvature)
                edge_features[13] = np.min(curvature)
                edge_features[14] = np.median(curvature)
                
                # Measures of serration/smoothness
                edge_features[15] = len(approx) / cv2.arcLength(max_contour, True)
                
                # Calculate waviness
                perimeter = cv2.arcLength(max_contour, True)
                hull = cv2.convexHull(max_contour)
                hull_perimeter = cv2.arcLength(hull, True)
                
                edge_features[16] = perimeter / hull_perimeter if hull_perimeter > 0 else 0
                
                # Calculate fractal dimension approximation
                # (log(N) / log(d)) where N is number of points, d is step size
                edge_features[17] = np.log(len(max_contour)) / np.log(perimeter / len(max_contour)) if perimeter > len(max_contour) else 0
                
                # Entropy of angles
                hist_entropy = -np.sum((hist / sum(hist)) * np.log2(hist / sum(hist) + 1e-10)) if sum(hist) > 0 else 0
                edge_features[18] = hist_entropy
                
                # Standard deviation of consecutive angle differences
                angle_diffs = np.diff(curvature)
                edge_features[19] = np.std(angle_diffs) if len(angle_diffs) > 0 else 0
    
    return edge_features

def extract_features(image):
    """Extract all features from a leaf image"""
    # Extract individual feature types
    color_feats = extract_color_features(image)
    shape_feats = extract_shape_features(image)
    texture_feats = extract_texture_features(image)
    vein_feats = extract_vein_features(image)
    edge_feats = extract_edge_features(image)
    
    # Combine all features
    all_features = np.concatenate([
        color_feats,    # 94 features
        shape_feats,    # 15 features
        texture_feats,  # ~26 features
        vein_feats,     # ~81 features (reduced)
        edge_feats      # 20 features
    ])
    
    return all_features

def process_directory(input_dir):
    """Extract features from all images in a directory tree"""
    # Get all image files
    image_files = []
    labels = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                # Full path to the image
                img_path = os.path.join(root, file)
                
                # Extract label from directory structure
                label = os.path.basename(os.path.dirname(img_path))
                
                image_files.append(img_path)
                labels.append(label)
    
    print(f"Found {len(image_files)} images for feature extraction.")
    
    # Initialize feature matrix
    features = []
    valid_labels = []
    valid_paths = []
    
    # Extract features from each image
    for i, img_path in enumerate(tqdm(image_files, desc="Extracting features")):
        try:
            # Read and extract features
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
                
            feature_vector = extract_features(image)
            features.append(feature_vector)
            valid_labels.append(labels[i])
            valid_paths.append(img_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Convert lists to arrays
    features = np.array(features)
    
    return features, valid_labels, valid_paths

if __name__ == "__main__":
    # Example usage
    input_dir = "data/processed"
    features, labels, paths = process_directory(input_dir)
    print(f"Extracted features with shape: {features.shape}")
    print(f"Number of labels: {len(labels)}")
    print(f"First 5 labels: {labels[:5]}")