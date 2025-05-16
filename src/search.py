import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from src.feature_extraction import extract_all_features

def normalize_features(features_df):
    """
    Normalize numerical features to ensure fair comparison across different scales.
    
    Args:
        features_df: DataFrame containing feature data
        
    Returns:
        Tuple of (normalized DataFrame, scaler object)
    """
    # Separate non-numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = features_df.select_dtypes(exclude=[np.number]).columns
    
    # Create a scaler
    scaler = StandardScaler()
    
    # Normalize numeric columns
    normalized_data = features_df[numeric_cols].copy()
    
    # Fix: Apply transform and assign values properly
    scaled_values = scaler.fit_transform(normalized_data)
    normalized_data = pd.DataFrame(scaled_values, columns=numeric_cols, index=normalized_data.index)
    
    # Add back non-numeric columns
    for col in non_numeric_cols:
        normalized_data[col] = features_df[col]
    
    return normalized_data, scaler


def calculate_distance(query_features, db_features, feature_weights=None):
    """
    Calculate weighted Euclidean distance between query features and database features.
    
    Args:
        query_features: Series or DataFrame containing features of the query image
        db_features: Series or DataFrame containing features of a database image
        feature_weights: Dict of feature name to weight (optional)
        
    Returns:
        float: Distance score (lower is more similar)
    """
    # Handle Series vs DataFrame inputs
    if isinstance(db_features, pd.Series):
        # Extract only numeric values
        numeric_cols = [col for col in db_features.index 
                        if isinstance(db_features[col], (int, float)) 
                        and not pd.isna(db_features[col])]
    else:
        # Get only numeric columns
        numeric_cols = db_features.select_dtypes(include=[np.number]).columns
    
    # Default weights: equal weighting
    if feature_weights is None:
        feature_weights = {col: 1.0 for col in numeric_cols}
    
    # Calculate weighted Euclidean distance
    squared_diff = 0
    total_weight = 0
    
    for col in numeric_cols:
        # Skip columns that don't have a weight
        if col not in feature_weights:
            continue
            
        weight = feature_weights.get(col, 1.0)
        diff = query_features[col] - db_features[col]
        squared_diff += weight * (diff ** 2)
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        distance = np.sqrt(squared_diff / total_weight)
    else:
        distance = float('inf')
        
    return distance


def search_similar_leaves(query_image_path, features_df, n=3, feature_weights=None):
    """
    Find the n most similar leaves to the query image.
    
    Args:
        query_image_path: Path to the query leaf image
        features_df: DataFrame containing features of all database images
        n: Number of similar leaves to return (default: 3)
        feature_weights: Dict of feature weights (optional)
        
    Returns:
        DataFrame with the n most similar leaves and their similarity scores
    """
    # Extract features from the query image
    query_features = extract_all_features(query_image_path)
    
    # Convert to DataFrame for easier handling
    query_df = pd.DataFrame([query_features])
    
    # Get numeric columns
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    
    # Normalize the query features using the same scaler
    normalized_df, scaler = normalize_features(features_df)
    query_numeric = scaler.transform(query_df[numeric_cols])
    
    # Create normalized query DataFrame
    normalized_query = pd.DataFrame(query_numeric, columns=numeric_cols)
    for col in query_df.columns:
        if col not in numeric_cols:
            normalized_query[col] = query_df[col]
    
    # Calculate distance for each database image
    distances = []
    for idx, row in normalized_df.iterrows():
        distance = calculate_distance(
            normalized_query.iloc[0], 
            row, 
            feature_weights
        )
        
        distances.append({
            'index': idx,
            'image_name': row['image_name'],
            'tree_type': row['tree_type'],
            'distance': distance
        })
    
    # Sort by distance (lowest first)
    distances_df = pd.DataFrame(distances)
    distances_df = distances_df.sort_values('distance')
    
    # Return the top n results
    return distances_df.head(n)


def get_default_feature_weights():
    """
    Get the default weights for different feature categories.
    These can be adjusted based on what features are most important.
    
    Returns:
        Dict of feature weights
    """
    weights = {}
    
    # Color features (moderate importance)
    for i in range(8):
        weights[f'color_hue_hist_{i}'] = 0.6
    
    for i in range(3):
        weights[f'color_hsv_mean_{i}'] = 0.8
        weights[f'color_hsv_std_{i}'] = 0.7
    
    # Shape features (high importance)
    weights['shape_area'] = 0.4
    weights['shape_perimeter'] = 0.5
    weights['shape_aspect_ratio'] = 0.9
    weights['shape_circularity'] = 1.0
    weights['shape_solidity'] = 1.0
    
    for i in range(4):
        weights[f'shape_hu_moments_{i}'] = 1.0
    
    # Edge features (high importance)
    weights['edge_convexity'] = 0.9
    weights['edge_significant_concavities'] = 1.0
    weights['edge_complexity'] = 0.8
    weights['edge_angle_variance'] = 0.7
    
    # Vein features (moderate importance)
    weights['vein_density'] = 0.6
    weights['vein_direction_consistency'] = 0.5
    weights['vein_mean_vein_width'] = 0.4
    weights['vein_var_vein_width'] = 0.3
    
    return weights


def process_and_search(query_image_path, features_df, n=3, feature_weights=None, preprocess=True):
    """
    Process a new image (if needed) and search for similar leaves.
    
    Args:
        query_image_path: Path to the query leaf image
        features_df: DataFrame containing features of all database images
        n: Number of similar leaves to return (default: 3)
        feature_weights: Dict of feature weights (optional)
        preprocess: Whether to preprocess the image first (default: True)
        
    Returns:
        DataFrame with the n most similar leaves and their similarity scores
    """
    if preprocess:
        # Create a temporary file for the processed image
        temp_dir = "data/temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        filename = os.path.basename(query_image_path)
        processed_path = os.path.join(temp_dir, f"processed_{filename}")
        
        # Preprocess the image
        from src.preprocessing import preprocess_leaf_image
        preprocess_leaf_image(query_image_path, processed_path)
        
        # Use the processed image for searching
        return search_similar_leaves(processed_path, features_df, n, feature_weights)
    else:
        # Use the original image directly
        return search_similar_leaves(query_image_path, features_df, n, feature_weights)
