import os
import pickle
import numpy as np
from tqdm import tqdm
import joblib
from src.feature_extraction import extract_features
from src.preprocess import preprocess_image
import cv2

class LeafDatabase:
    def __init__(self):
        self.features = []
        self.image_paths = []
        self.labels = []
        self.is_built = False
    
    def build_from_directory(self, input_dir):
        """Build database from a directory of processed images"""
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
        
        print(f"Found {len(image_files)} images for database.")
        
        # Extract features from each image
        features = []
        valid_labels = []
        valid_paths = []
        
        for i, img_path in enumerate(tqdm(image_files, desc="Building database")):
            try:
                # Read image
                image = cv2.imread(img_path)
                
                if image is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                
                # Extract features
                feature_vector = extract_features(image)
                
                # Store features and metadata
                features.append(feature_vector)
                valid_labels.append(labels[i])
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Store in object
        self.features = np.array(features)
        self.image_paths = valid_paths
        self.labels = valid_labels
        self.is_built = True
        
        print(f"Database built with {len(self.features)} feature vectors.")
        return self
    
    def add_image(self, image_path, label=None):
        """Add a single image to the database"""
        try:
            # Read and process image
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not read image {image_path}")
                return False
            
            # Extract features
            feature_vector = extract_features(image)
            
            # Determine label if not provided
            if label is None:
                label = os.path.basename(os.path.dirname(image_path))
            
            # Add to database
            self.features = np.vstack([self.features, feature_vector]) if len(self.features) > 0 else np.array([feature_vector])
            self.image_paths.append(image_path)
            self.labels.append(label)
            
            return True
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")
            return False
    
    def save(self, output_path):
        """Save database to disk"""
        if not self.is_built:
            print("Database not built yet!")
            return False
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create database object for saving
        database = {
            'features': self.features,
            'image_paths': self.image_paths,
            'labels': self.labels
        }
        
        # Save using joblib
        joblib.dump(database, output_path)
        print(f"Database saved to {output_path}")
        return True
    
    def load(self, input_path):
        """Load database from disk"""
        if not os.path.exists(input_path):
            print(f"Database file not found: {input_path}")
            return False
        
        try:
            # Load database object
            database = joblib.load(input_path)
            
            # Extract data
            self.features = database['features']
            self.image_paths = database['image_paths']
            self.labels = database['labels']
            self.is_built = True
            
            print(f"Database loaded with {len(self.features)} feature vectors.")
            return True
        except Exception as e:
            print(f"Error loading database: {e}")
            return False
    
    def get_feature_vector(self, index):
        """Get feature vector at specific index"""
        if not self.is_built or index >= len(self.features):
            return None
        return self.features[index]
    
    def get_image_path(self, index):
        """Get image path at specific index"""
        if not self.is_built or index >= len(self.image_paths):
            return None
        return self.image_paths[index]
    
    def get_label(self, index):
        """Get label at specific index"""
        if not self.is_built or index >= len(self.labels):
            return None
        return self.labels[index]
    
    def size(self):
        """Get number of images in database"""
        return len(self.features) if self.is_built else 0

if __name__ == "__main__":
    # Example usage
    db = LeafDatabase()
    db.build_from_directory("data/processed")
    db.save("models/feature_database.pkl")