import numpy as np
import cv2
from src.preprocess import preprocess_image
from src.feature_extraction import extract_features

class LeafRetrieval:
    def __init__(self, database):
        """Initialize retrieval system with a database"""
        self.database = database
    
    def euclidean_distance(self, feature1, feature2):
        """Calculate Euclidean distance between feature vectors"""
        return np.sqrt(np.sum((feature1 - feature2) ** 2))
    
    def cosine_distance(self, feature1, feature2):
        """Calculate cosine distance between feature vectors"""
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance if either vector is zero
        
        return 1.0 - (dot_product / (norm1 * norm2))
    
    def chi_square_distance(self, feature1, feature2):
        """Calculate chi-square distance between feature vectors (good for histograms)"""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        
        # Compute chi-square distance
        return 0.5 * np.sum(((feature1 - feature2) ** 2) / (feature1 + feature2 + epsilon))
    
    def manhattan_distance(self, feature1, feature2):
        """Calculate Manhattan (L1) distance between feature vectors"""
        return np.sum(np.abs(feature1 - feature2))
    
    def weighted_distance(self, feature1, feature2, weights=None):
        """Calculate weighted distance with customizable weights for different feature types"""
        # Default weights if none provided (equal weighting)
        if weights is None:
            # Assuming the feature vector structure from feature_extraction.py:
            # [color_feats(94), shape_feats(15), texture_feats(~26), vein_feats(~81), edge_feats(20)]
            weights = np.ones(feature1.shape)
        
        # Calculate weighted Euclidean distance
        return np.sqrt(np.sum(weights * ((feature1 - feature2) ** 2)))
    
    def retrieve_similar_images(self, query_image_path, n=3, metric='euclidean', preprocess=True):
        """Retrieve n most similar images to the query image"""
        # Check if database is loaded
        if not self.database.is_built:
            raise ValueError("Database not built or loaded!")
        
        # Preprocess image if required
        if preprocess:
            # Preprocess the query image (without saving)
            processed_image = preprocess_image(query_image_path)
        else:
            # Just read the image (assuming it's already preprocessed)
            processed_image = cv2.imread(query_image_path)
        
        if processed_image is None:
            raise ValueError(f"Could not read or process query image: {query_image_path}")
        
        # Extract features from query image
        query_features = extract_features(processed_image)
        
        # Calculate distances to all images in the database
        distances = []
        
        # Choose distance metric
        if metric == 'euclidean':
            distance_func = self.euclidean_distance
        elif metric == 'cosine':
            distance_func = self.cosine_distance
        elif metric == 'chi_square':
            distance_func = self.chi_square_distance
        elif metric == 'manhattan':
            distance_func = self.manhattan_distance
        else:
            raise ValueError(f"Unsupported distance metric: {metric}")
        
        # Calculate distances
        for i in range(self.database.size()):
            db_features = self.database.get_feature_vector(i)
            distance = distance_func(query_features, db_features)
            distances.append((i, distance))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Get top n matches
        top_matches = []
        for i in range(min(n, len(distances))):
            idx, dist = distances[i]
            top_matches.append({
                'index': idx,
                'distance': dist,
                'path': self.database.get_image_path(idx),
                'label': self.database.get_label(idx),
                'similarity': 1.0 / (1.0 + dist)  # Convert distance to similarity score
            })
        
        return top_matches
    
    def retrieve_with_multiple_metrics(self, query_image_path, n=3, preprocess=True):
        """Retrieve similar images using an ensemble of distance metrics"""
        # Check if database is loaded
        if not self.database.is_built:
            raise ValueError("Database not built or loaded!")
        
        # Get results from different metrics
        euclidean_results = self.retrieve_similar_images(query_image_path, n=n, metric='euclidean', preprocess=preprocess)
        cosine_results = self.retrieve_similar_images(query_image_path, n=n, metric='cosine', preprocess=False)  # Skip preprocessing for subsequent calls
        chi_square_results = self.retrieve_similar_images(query_image_path, n=n, metric='chi_square', preprocess=False)
        
        # Combine results using rank aggregation (Borda count)
        # First, create a dictionary to store points for each image
        points = {}
        
        # Assign points based on rank
        for i, result in enumerate(euclidean_results):
            idx = result['index']
            points[idx] = points.get(idx, 0) + (n - i)
        
        for i, result in enumerate(cosine_results):
            idx = result['index']
            points[idx] = points.get(idx, 0) + (n - i)
        
        for i, result in enumerate(chi_square_results):
            idx = result['index']
            points[idx] = points.get(idx, 0) + (n - i)
        
        # Sort images by points
        sorted_indices = sorted(points.keys(), key=lambda idx: points[idx], reverse=True)
        
        # Get top n matches
        top_matches = []
        for i in range(min(n, len(sorted_indices))):
            idx = sorted_indices[i]
            top_matches.append({
                'index': idx,
                'points': points[idx],
                'path': self.database.get_image_path(idx),
                'label': self.database.get_label(idx),
                'similarity': points[idx] / (3 * n)  # Normalize to [0,1]
            })
        
        return top_matches
    
    def retrieve_by_label(self, query_label, n=3):
        """Retrieve n random images with the specified label"""
        # Check if database is loaded
        if not self.database.is_built:
            raise ValueError("Database not built or loaded!")
        
        # Find all indices with the matching label
        matching_indices = [i for i in range(self.database.size()) if self.database.get_label(i) == query_label]
        
        # Shuffle to randomize selection
        np.random.shuffle(matching_indices)
        
        # Get top n matches
        top_matches = []
        for i in range(min(n, len(matching_indices))):
            idx = matching_indices[i]
            top_matches.append({
                'index': idx,
                'path': self.database.get_image_path(idx),
                'label': self.database.get_label(idx),
                'similarity': 1.0  # Perfect match for label-based query
            })
        
        return top_matches

if __name__ == "__main__":
    # Example usage
    from src.database import LeafDatabase
    
    db = LeafDatabase()
    db.load("models/feature_database.pkl")
    
    retrieval = LeafRetrieval(db)
    results = retrieval.retrieve_similar_images("test_images/test1.JPG", n=3)
    
    for i, result in enumerate(results):
        print(f"Match {i+1}: {result['label']} (Similarity: {result['similarity']:.4f})")