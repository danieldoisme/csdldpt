import pickle
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import preprocess_image
from .feature_extraction import extract_all_features, load_deep_model

def load_feature_database(database_file):
    """Tải cơ sở dữ liệu đặc trưng từ file"""
    try:
        with open(database_file, 'rb') as f:
            feature_database = pickle.load(f)
        return feature_database
    except Exception as e:
        print(f"Lỗi khi tải cơ sở dữ liệu: {e}")
        return None

def find_similar_images(query_image_path, feature_database, top_k=3, use_deep_features=True):
    """Tìm k ảnh tương tự nhất với ảnh truy vấn"""
    # Tải mô hình CNN nếu cần
    deep_model = None
    if use_deep_features:
        deep_model = load_deep_model()
    
    # Tiền xử lý ảnh truy vấn giống như các ảnh train
    processed_img = preprocess_image(query_image_path)
    
    if processed_img is None:
        print(f"Không thể xử lý ảnh truy vấn: {query_image_path}")
        return []
    
    # Trích xuất đặc trưng từ ảnh truy vấn đã được tiền xử lý
    query_features = extract_all_features(processed_img, deep_model)
    
    # Tính độ tương đồng với tất cả ảnh trong cơ sở dữ liệu
    database_features = feature_database['features']
    similarities = cosine_similarity([query_features], database_features)[0]
    
    # Sắp xếp theo độ tương đồng giảm dần
    indices = np.argsort(similarities)[::-1][:top_k]
    
    # Tạo danh sách kết quả
    results = []
    for idx in indices:
        results.append({
            'image_path': feature_database['file_paths'][idx],
            'similarity': similarities[idx],
            'label': feature_database['labels'][idx]
        })
    
    return results