import os
import argparse
import time
import cv2
import numpy as np
from tqdm import tqdm

from src.preprocess import preprocess_image, batch_process
from src.feature_extraction import extract_features, process_directory
from src.database import LeafDatabase
from src.retrieval import LeafRetrieval
from src.visualization import create_result_visualization, create_preprocessing_visualization, create_feature_visualization

def preprocess_data(input_dir="data/raw", output_dir="data/processed"):
    """Preprocess all leaf images"""
    print("Starting preprocessing...")
    
    # Start time
    start_time = time.time()
    
    # Process all images
    batch_process(input_dir, output_dir)
    
    # End time
    elapsed_time = time.time() - start_time
    print(f"Preprocessing completed in {elapsed_time:.2f} seconds.")

def build_database(input_dir="data/processed", output_path="models/feature_database.pkl"):
    """Build database from processed images"""
    print("Building database...")
    
    # Start time
    start_time = time.time()
    
    # Create and build database
    db = LeafDatabase()
    db.build_from_directory(input_dir)
    
    # Save database
    db.save(output_path)
    
    # End time
    elapsed_time = time.time() - start_time
    print(f"Database built in {elapsed_time:.2f} seconds with {db.size()} images.")
    
    return db

def retrieve_similar(query_image, database_path="models/feature_database.pkl", output_dir="results", n=3, metric="euclidean"):
    """Retrieve similar images to a query image"""
    # Load database
    db = LeafDatabase()
    db.load(database_path)
    
    # Create retrieval system
    retrieval = LeafRetrieval(db)
    
    # Determine output path
    query_filename = os.path.basename(query_image)
    output_path = os.path.join(output_dir, f"result_{os.path.splitext(query_filename)[0]}.png")
    
    # Retrieve similar images
    start_time = time.time()
    
    if metric == "ensemble":
        results = retrieval.retrieve_with_multiple_metrics(query_image, n=n)
    else:
        results = retrieval.retrieve_similar_images(query_image, n=n, metric=metric)
    
    elapsed_time = time.time() - start_time
    
    # Display results
    print(f"Retrieval completed in {elapsed_time:.2f} seconds.")
    print(f"Query image: {query_image}")
    
    for i, result in enumerate(results):
        similarity_percent = result['similarity'] * 100
        print(f"Result {i+1}: {result['path']} ({result['label']}) - Similarity: {similarity_percent:.1f}%")
    
    # Create visualization
    create_result_visualization(query_image, results, output_path, show=False)
    print(f"Result visualization saved to {output_path}")
    
    return results, output_path

def process_test_images(test_dir="test_images", database_path="models/feature_database.pkl", output_dir="results", n=3, metric="euclidean"):
    """Process all test images in a directory"""
    # Get all test images
    test_images = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            test_images.append(os.path.join(test_dir, file))
    
    # Process each test image
    for i, test_image in enumerate(test_images):
        print(f"\nProcessing test image {i+1}/{len(test_images)}: {test_image}")
        retrieve_similar(test_image, database_path, output_dir, n, metric)

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description="Leaf Image Retrieval System")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess raw images")
    parser.add_argument("--build", action="store_true", help="Build feature database")
    parser.add_argument("--retrieve", action="store_true", help="Retrieve similar images")
    parser.add_argument("--test_all", action="store_true", help="Test all images in test directory")
    parser.add_argument("--query", type=str, help="Query image path")
    parser.add_argument("--n", type=int, default=3, help="Number of similar images to retrieve")
    parser.add_argument("--metric", type=str, default="euclidean", choices=["euclidean", "cosine", "chi_square", "manhattan", "ensemble"], help="Distance metric to use")
    
    args = parser.parse_args()
    
    # Create required directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Execute requested operations
    if args.preprocess:
        preprocess_data()
    
    if args.build:
        build_database()
    
    if args.retrieve and args.query:
        retrieve_similar(args.query, n=args.n, metric=args.metric)
    
    if args.test_all:
        process_test_images(n=args.n, metric=args.metric)
    
    # If no arguments, show help
    if not (args.preprocess or args.build or args.retrieve or args.test_all):
        parser.print_help()

if __name__ == "__main__":
    main()