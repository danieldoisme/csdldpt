import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

def extract_color_features(image):
    """
    Extract simplified color features from a leaf image.
    
    Features:
    - 8-bin color histogram for hue channel
    - Mean and standard deviation for each HSV channel
    - 3 dominant colors (RGB values)
    
    Args:
        image: Preprocessed leaf image with white background
    
    Returns:
        dict: Dictionary containing color features
    """
    # Create mask to exclude white background
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Extract only leaf pixels
    leaf_pixels = image[mask > 0]
    
    if leaf_pixels.size == 0:
        # Handle empty leaf case
        return {
            'hue_hist': np.zeros(8).tolist(),
            'hsv_mean': [0, 0, 0],
            'hsv_std': [0, 0, 0],
            'dominant_colors': [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        }
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_leaf = hsv_image[mask > 0]
    
    # 1. Color histogram (8 bins for Hue channel)
    hist = cv2.calcHist([hsv_leaf], [0], None, [8], [0, 180])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX).flatten()
    
    # 2. Color statistics (mean and std for each channel)
    mean_h = np.mean(hsv_leaf[:, 0])
    mean_s = np.mean(hsv_leaf[:, 1])
    mean_v = np.mean(hsv_leaf[:, 2])
    
    std_h = np.std(hsv_leaf[:, 0])
    std_s = np.std(hsv_leaf[:, 1])
    std_v = np.std(hsv_leaf[:, 2])
    
    # 3. Dominant colors (simplified to 3 colors)
    pixels = leaf_pixels.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int).tolist()
    
    return {
        'hue_hist': hist.tolist(),
        'hsv_mean': [mean_h, mean_s, mean_v],
        'hsv_std': [std_h, std_s, std_v],
        'dominant_colors': dominant_colors
    }


def extract_shape_features(image):
    """
    Extract simplified shape features from a leaf image.
    
    Features:
    - Area and perimeter
    - Aspect ratio
    - Circularity
    - Solidity
    - First 4 Hu moments
    
    Args:
        image: Preprocessed leaf image with white background
    
    Returns:
        dict: Dictionary containing shape features
    """
    # Create binary mask of the leaf
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Handle case with no contours
        return {
            'area': 0,
            'perimeter': 0,
            'aspect_ratio': 0,
            'circularity': 0,
            'solidity': 0,
            'hu_moments': [0, 0, 0, 0]
        }
    
    contour = max(contours, key=cv2.contourArea)
    
    # Basic metrics
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Circularity
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    # Convexity measures
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Hu moments (shape descriptors)
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments).flatten()[:4]  # Use only first 4 moments
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))  # Log transform
    
    return {
        'area': area,
        'perimeter': perimeter,
        'aspect_ratio': aspect_ratio,
        'circularity': circularity,
        'solidity': solidity,
        'hu_moments': hu_moments.tolist()
    }


def extract_edge_features(image):
    """
    Extract simplified edge features from a leaf image.
    
    Features:
    - Edge roughness (convexity)
    - Number of significant concavities
    - Edge complexity (perimeter/sqrt(area))
    - Edge angle variance
    
    Args:
        image: Preprocessed leaf image with white background
    
    Returns:
        dict: Dictionary containing edge features
    """
    # Create binary mask of the leaf
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Handle case with no contours
        return {
            'convexity': 0,
            'significant_concavities': 0,
            'complexity': 0,
            'angle_variance': 0
        }
    
    contour = max(contours, key=cv2.contourArea)
    
    # Convexity = ratio of convex hull perimeter to contour perimeter
    hull = cv2.convexHull(contour)
    contour_perimeter = cv2.arcLength(contour, True)
    hull_perimeter = cv2.arcLength(hull, True)
    convexity = hull_perimeter / contour_perimeter if contour_perimeter > 0 else 0
    
    # Count significant concavities
    hull_indices = cv2.convexHull(contour, returnPoints=False)
    defects = None
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except:
        pass
        
    significant_concavities = 0
    if defects is not None:
        for i in range(defects.shape[0]):
            _, _, _, depth = defects[i, 0]
            if depth > 700:  # Threshold for significant concavity
                significant_concavities += 1
    
    # Edge complexity (perimeter/sqrt(area))
    area = cv2.contourArea(contour)
    complexity = contour_perimeter / np.sqrt(area) if area > 0 else 0
    
    # Edge angle variance
    angles = []
    if len(contour) >= 3:
        for i in range(len(contour)):
            p1 = contour[i][0]
            p2 = contour[(i+1) % len(contour)][0]
            p3 = contour[(i+2) % len(contour)][0]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle using dot product
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norm > 0:
                angle = np.arccos(np.clip(dot / norm, -1.0, 1.0))
                angles.append(angle)
        
    angle_variance = np.var(angles) if angles else 0
    
    return {
        'convexity': convexity,
        'significant_concavities': significant_concavities,
        'complexity': complexity,
        'angle_variance': angle_variance
    }


def extract_vein_features(image):
    """
    Extract simplified vein features from a leaf image.
    
    Features:
    - Vein density estimation
    - Directional consistency of veins
    - Mean and variance of vein width
    
    Args:
        image: Preprocessed leaf image with white background
    
    Returns:
        dict: Dictionary containing vein features
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask to exclude white background
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    
    # Apply CLAHE to enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    
    # Enhance veins using morphological operations and Laplacian
    kernel = np.ones((3,3), np.uint8)
    cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)
    
    # Apply Laplacian filter to detect edges (veins)
    laplacian = cv2.Laplacian(enhanced, cv2.CV_8U)
    laplacian = cv2.bitwise_and(laplacian, laplacian, mask=mask)
    
    # Threshold to get binary vein image
    _, veins = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
    
    # Calculate vein density
    leaf_area = np.sum(mask > 0)
    vein_area = np.sum(veins > 0)
    vein_density = vein_area / leaf_area if leaf_area > 0 else 0
    
    # Calculate directional consistency using Sobel filters
    sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    
    # Get gradient directions
    angles = np.arctan2(sobely, sobelx)
    angles = angles[mask > 0]
    
    # Measure directional consistency (lower variance = more consistent)
    direction_consistency = 1.0 - np.std(angles) / np.pi if angles.size > 0 else 0
    
    # Estimate vein widths
    # Simplified approach using distance transform
    dist = cv2.distanceTransform(cv2.bitwise_not(veins), cv2.DIST_L2, 3)
    dist = dist[mask > 0]
    
    mean_width = np.mean(dist) if dist.size > 0 else 0
    var_width = np.var(dist) if dist.size > 0 else 0
    
    return {
        'vein_density': vein_density,
        'direction_consistency': direction_consistency,
        'mean_vein_width': mean_width,
        'var_vein_width': var_width
    }


def extract_all_features(image_path):
    """
    Extract all features from a single leaf image and return them as a dictionary.
    
    Args:
        image_path: Path to the preprocessed leaf image
        
    Returns:
        dict: Dictionary containing all extracted features
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Extract all features
    color_features = extract_color_features(image)
    shape_features = extract_shape_features(image)
    edge_features = extract_edge_features(image)
    vein_features = extract_vein_features(image)
    
    # Combine all features
    all_features = {
        'image_name': os.path.basename(image_path),
        **flatten_dict(color_features, 'color_'),
        **flatten_dict(shape_features, 'shape_'),
        **flatten_dict(edge_features, 'edge_'),
        **flatten_dict(vein_features, 'vein_')
    }
    
    return all_features


def flatten_dict(d, prefix=''):
    """
    Flatten a nested dictionary by concatenating keys with underscores.
    
    Args:
        d: Dictionary to flatten
        prefix: Prefix to add to all keys
        
    Returns:
        dict: Flattened dictionary
    """
    result = {}
    for key, value in d.items():
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], list):
                # Handle nested lists (e.g., dominant_colors)
                for i, inner_list in enumerate(value):
                    for j, item in enumerate(inner_list):
                        result[f"{prefix}{key}_{i}_{j}"] = item
            else:
                # Handle simple lists
                for i, item in enumerate(value):
                    result[f"{prefix}{key}_{i}"] = item
        else:
            # Handle simple values
            result[f"{prefix}{key}"] = value
    return result