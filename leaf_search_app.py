import streamlit as st
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.search import process_and_search, get_default_feature_weights

# Set page configuration
st.set_page_config(
    page_title="Leaf Search System",
    page_icon="🍃",
    layout="wide"
)

# Define helper functions
def load_image(image_file):
    """Load an uploaded image file"""
    img = Image.open(image_file)
    return np.array(img)

def convert_df_to_csv(df):
    """Convert DataFrame to CSV for download"""
    return df.to_csv().encode('utf-8')

# App title and description
st.title("🍃 Leaf Search System")
st.markdown("""
This application lets you upload a leaf image and find the most similar leaves in our database.
""")

# Sidebar controls
st.sidebar.header("Options")

# Number of results to display
num_results = st.sidebar.slider("Number of similar leaves to show", 1, 10, 3)

# Features file selection
features_file = st.sidebar.selectbox(
    "Select features dataset",
    options=[
        "data/features/all_features.csv",
        "data/features/apples_features.csv",
        "data/features/blueberry_features.csv",
        "data/features/cherry_features.csv",
        "data/features/grape_features.csv",
        "data/features/pepper-bell_features.csv",
        "data/features/potato_features.csv", 
        "data/features/raspberry_features.csv",
        "data/features/soybean_features.csv",
        "data/features/strawberry_features.csv",
        "data/features/tomato_features.csv"
    ],
    index=0
)

# Weight adjustment (advanced options)
show_weights = st.sidebar.checkbox("Adjust feature weights", False)

if show_weights:
    st.sidebar.subheader("Feature Weight Adjustment")
    color_weight = st.sidebar.slider("Color Features", 0.0, 2.0, 1.0, 0.1)
    shape_weight = st.sidebar.slider("Shape Features", 0.0, 2.0, 1.0, 0.1)
    edge_weight = st.sidebar.slider("Edge Features", 0.0, 2.0, 1.0, 0.1)
    vein_weight = st.sidebar.slider("Vein Features", 0.0, 2.0, 1.0, 0.1)
    
    # Get default weights and adjust them
    weights = get_default_feature_weights()
    
    # Modify weights based on sliders
    for key in weights:
        if key.startswith('color_'):
            weights[key] *= color_weight
        elif key.startswith('shape_'):
            weights[key] *= shape_weight
        elif key.startswith('edge_'):
            weights[key] *= edge_weight
        elif key.startswith('vein_'):
            weights[key] *= vein_weight
else:
    weights = get_default_feature_weights()

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        image = load_image(uploaded_file)
        st.image(image, use_container_width=True)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Save the uploaded image to a temp file
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, "uploaded_image.jpg")
    
    # Save the uploaded image
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    with col2:
        st.subheader("Preprocessing")
        status_text.text("Preprocessing the image...")
        progress_bar.progress(25)
        
        # Preprocess the image
        processed_path = os.path.join(temp_dir, "processed_uploaded_image.jpg")
        from src.preprocessing import preprocess_leaf_image
        preprocess_leaf_image(temp_path, processed_path)
        
        # Display the processed image
        processed_image = cv2.imread(processed_path)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(processed_image, use_container_width=True)
        status_text.text("Preprocessing complete!")
        progress_bar.progress(50)
    
    # Load feature data
    status_text.text("Loading feature database...")
    try:
        features_df = pd.read_csv(features_file)
        progress_bar.progress(75)
        
        # Search for similar leaves
        status_text.text("Searching for similar leaves...")
        similar_leaves = process_and_search(
            temp_path,
            features_df,
            n=num_results,
            feature_weights=weights,
            preprocess=True
        )
        
        progress_bar.progress(100)
        status_text.text("Search complete!")
        
        # Display results
        st.subheader("Similar Leaves")
        
        # Create a grid to display results
        cols = st.columns(min(num_results, 5))
        
        for i, (_, row) in enumerate(similar_leaves.iterrows()):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Get the image path
                img_path = os.path.join('data/stored_images/raw', row['tree_type'], row['image_name'])
                
                # Load and display image
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img)
                    st.caption(f"Match {i+1}: {row['tree_type']}\nDistance: {row['distance']:.3f}")
                except Exception as e:
                    st.write(f"Could not load image: {img_path}. Error: {e}")
        
        # Show the results in a table
        st.subheader("Detailed Results")
        st.dataframe(similar_leaves[['tree_type', 'image_name', 'distance']])
        
        # Provide a download button for the results
        csv = convert_df_to_csv(similar_leaves)
        st.download_button(
            "Download Results as CSV",
            csv,
            "leaf_search_results.csv",
            "text/csv",
            key='download-csv'
        )
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        progress_bar.empty()
else:
    # Show example
    st.info("👆 Upload a leaf image to find similar leaves in our database.")
    
    # Show some info about the dataset
    st.subheader("Dataset Information")
    try:
        features_df = pd.read_csv(features_file)
        tree_counts = features_df['tree_type'].value_counts()
        
        st.write(f"The database contains {len(features_df)} leaf images from {len(tree_counts)} different tree types.")
        
        # Plot the distribution of tree types
        fig, ax = plt.subplots(figsize=(10, 5))
        tree_counts.plot(kind='bar', ax=ax)
        plt.title('Number of Images per Tree Type')
        plt.xlabel('Tree Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    except Exception as e:
        st.write(f"Could not load dataset information. Please check if the features file exists. Error: {e}")

# Footer
st.markdown("---")
st.caption("Leaf Search System - Created for the leaf database project.")
