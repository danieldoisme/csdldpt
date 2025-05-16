#!/usr/bin/env python3
"""
Leaf Store-and-Search System

This program implements a system for preprocessing, feature extraction,
and similarity search on leaf images, accessible via a command-line interface.

Usage:
  python main.py [command] [options]

Commands:
  preprocess     - Preprocess all raw leaf images.
                 (Looks for images in data/stored_images/raw/{tree_type}/ and 
                  saves processed images to data/stored_images/processed/{tree_type}/)

  extract        - Extract features from preprocessed images.
                 (Reads from data/stored_images/processed/, saves features to data/features/)

  search         - Search for similar leaves to the given image.
    query_image_path : Path to the query leaf image. (e.g., search_images/apples/image.jpg or search_images/processed/apples/image.jpg)
    --output, -o     : Path to save the result image.
    --num_results, -n: Number of similar leaves to return (default: 3).
    --features_file, -f: Path to the features CSV file (default: data/features/all_features.csv).
    --process-search-dir: Process all images in search_images/{tree_type}/ directories, saving results to search_images/processed/{tree_type}/.
    --skip-preprocess-query: Skip preprocessing the query image (if it's already processed).

  app            - Start the web application interface.

Examples:
  python main.py preprocess
  python main.py extract
  python main.py search search_images/apples/image.jpg
  python main.py search search_images/cherry/image.jpg -n 5 -o results/search_output.png
  python main.py search --process-search-dir 
  python main.py search search_images/processed/apples/image.jpg --skip-preprocess-query
  python main.py app
"""

import os
import sys
import pandas as pd
import matplotlib
import argparse

# Import core functions from other modules
from modules.preprocess_all_images import preprocess_all_images
from modules.extract_features import extract_features_for_all_images
# Updated imports for search functionalities:
from src.search import process_and_search, get_default_feature_weights 
from modules.search_leaves import (
    display_results,
    process_search_directory as search_process_search_directory # Aliased for clarity
)
# Note: preprocess_leaf_image is used internally by search_leaves module functions
# and by process_and_search (if preprocess=True) via src.preprocessing

def preprocess_command_args(args):
    print("Starting preprocessing of stored images...")
    preprocess_all_images()
    print("Preprocessing of stored images complete.")

def extract_command_args(args):
    print("Starting feature extraction...")
    extract_features_for_all_images()
    print("Feature extraction complete.")

def search_command_args(args):
    if args.process_search_dir:
        print("Processing images in the 'search_images/{tree_type}/' directories...")
        search_process_search_directory()
        if not args.query_image:
            print("Search directory processing complete. No query image provided for search.")
            return

    if not args.query_image:
        # This should ideally be caught by argparse if query_image is made mandatory
        # when --process-search-dir is not the sole action.
        # The main_cli function has a check for this.
        print("Error: Query image path is required for search unless only --process-search-dir is used.")
        return

    print(f"Loading features from {args.features_file}...")
    try:
        features_df = pd.read_csv(args.features_file)
    except FileNotFoundError:
        print(f"Error: Features file not found at {args.features_file}")
        print("Please run 'python main.py extract' first, or check the file path.")
        return
    
    weights = get_default_feature_weights() # Using default weights
    
    preprocess_query_image = not args.skip_preprocess_query
    action_verb = "Preprocessing and searching" if preprocess_query_image else "Searching"
    print(f"{action_verb} for leaves similar to {args.query_image}...")
    
    try:
        similar_leaves = process_and_search(
            args.query_image,
            features_df,
            n=args.num_results,
            feature_weights=weights,
            preprocess=preprocess_query_image
        )
    except FileNotFoundError:
        print(f"Error: Query image not found at {args.query_image}")
        return
    except Exception as e:
        print(f"An error occurred during search: {e}")
        return
    
    if similar_leaves.empty:
        print("No similar leaves found.")
        return

    print("\nTop matches:")
    for i, (_, row) in enumerate(similar_leaves.iterrows(), 1):
        print(f"  {i}. {row['image_name']} (Type: {row['tree_type']}, Distance: {row['distance']:.3f})")
        
    output_file_path = args.output
    if output_file_path:
        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_file_path))
        if output_dir: # Check if output_dir is not empty (e.g. if path is just "file.png")
             os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving result image to {output_file_path}")
    else:
        print("\nDisplaying result image (close window to continue)...")

    # Determine the path of the query image to display (original or processed)
    query_image_to_display = args.query_image
    if preprocess_query_image:
        # process_and_search saves the processed image in "data/temp" if preprocess=True
        # It returns the path to the processed image if preprocessing occurred, otherwise original path.
        # For simplicity, let's assume process_and_search handles this detail or we reconstruct path.
        # The `process_and_search` function in `src/search.py` saves the processed query
        # image to `data/temp/processed_{filename}`.
        temp_processed_query_path = os.path.join("data/temp", f"processed_{os.path.basename(args.query_image)}")
        if os.path.exists(temp_processed_query_path):
            query_image_to_display = temp_processed_query_path
        else:
            # Fallback if processed image not found where expected, use original.
            # This might happen if process_and_search's internal saving changes.
            print(f"Warning: Expected processed query image at {temp_processed_query_path} not found. Displaying original.")


    display_results(
        query_image_to_display,
        similar_leaves,
        output_path=output_file_path # Pass None if no output path, display_results handles plt.show()
    )
    if not output_file_path:
        print("Result display finished.")


def app_command_args(args):
    print("Starting the Leaf Search web application...")
    print("Attempting to run: streamlit run leaf_search_app.py")
    print("If it fails, ensure Streamlit is installed ('pip install streamlit') and leaf_search_app.py is in the current directory.")
    
    try:
        import subprocess
        # Ensure leaf_search_app.py is in the same directory or provide a relative path
        # Assuming main.py and leaf_search_app.py are in the same root directory.
        app_file_path = os.path.join(os.path.dirname(__file__), "leaf_search_app.py")
        if not os.path.exists(app_file_path):
            app_file_path = "leaf_search_app.py" # Fallback to current dir if not found next to main.py

        subprocess.run(["streamlit", "run", app_file_path], check=True)
    except ImportError:
        print("\nError: Streamlit module is not installed.")
        print("Please install it with: pip install streamlit")
        print("Then run: python main.py app")
    except FileNotFoundError:
        print("\nError: 'streamlit' command not found. Is Streamlit installed and in your system's PATH?")
    except subprocess.CalledProcessError as e:
        print(f"\nError running Streamlit app: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while trying to start the app: {e}")

def main_cli():
    parser = argparse.ArgumentParser(
        description="Leaf Store-and-Search System CLI.",
        formatter_class=argparse.RawTextHelpFormatter # To preserve help text formatting
    )
    # Update parser description with more detailed usage from the module docstring
    parser.description = __doc__


    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess all raw leaf images in data/stored_images/raw/")
    preprocess_parser.set_defaults(func=preprocess_command_args)

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract features from preprocessed images in data/stored_images/processed/")
    extract_parser.set_defaults(func=extract_command_args)

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar leaves to a given image.")
    search_parser.add_argument(
        'query_image',
        nargs='?',
        help='Path to the query leaf image. Optional if only --process-search-dir is used.'
    )
    search_parser.add_argument(
        '--output', '-o',
        help='Path to save the result image (e.g., results/search.png).'
    )
    search_parser.add_argument(
        '--num_results', '-n',
        type=int,
        default=3,
        help='Number of similar leaves to return (default: 3).'
    )
    search_parser.add_argument(
        '--features_file', '-f',
        default='data/features/all_features.csv',
        help='Path to the features CSV file (default: data/features/all_features.csv).'
    )
    search_parser.add_argument(
        '--process-search-dir',
        action='store_true',
        help='Process all images in the search_images/{tree_type}/ directories, saving to search_images/processed/{tree_type}/. Can be used before searching (if a query_image is also provided) or as a standalone action.'
    )
    search_parser.add_argument(
        '--skip-preprocess-query',
        action='store_true',
        help='Skip preprocessing the query image (use if the query image is already preprocessed).'
    )
    search_parser.set_defaults(func=search_command_args)

    # App command
    app_parser = subparsers.add_parser("app", help="Start the web application interface (requires Streamlit).")
    app_parser.set_defaults(func=app_command_args)

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    
    # ADD CONDITIONAL BACKEND SETTING HERE:
    # If the command is 'search' and no output file is specified,
    # let Matplotlib use its default interactive backend.
    # Otherwise, use 'Agg'.
    if not (args.command == "search" and args.output is None):
        try:
            matplotlib.use('Agg')
        except Exception as e:
            # It's unlikely for 'Agg' to fail, but good to catch potential issues.
            print(f"Warning: Could not set Matplotlib backend to 'Agg': {e}", file=sys.stderr)
    # If it is the search command and no output is specified, we don't call matplotlib.use(),
    # allowing it to pick a default interactive backend suitable for your OS (e.g., 'MacOSX' or 'TkAgg' on macOS).

    # Handle cases for the search command specifically (argument validation)
    if args.command == "search":
        if not args.query_image and not args.process_search_dir:
            search_parser.error("A query_image path is required, or use --process-search-dir to process the search directory.")
            # error() typically exits, so sys.exit(1) might not be reached.

    args.func(args)

if __name__ == "__main__":
    main_cli()