import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from .preprocess import preprocess_image_improved as preprocess_image
from .feature_extraction import extract_all_features, load_deep_model

def compute_feature_lengths(image_rgb, deep_model=None, model_type='resnet'):
    """Tính toán độ dài của từng phần trong vector đặc trưng"""
    # Tạo ảnh xám và ảnh nhị phân
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_image)
    _, binary_otsu = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_adaptive = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    binary_image = cv2.bitwise_or(binary_otsu, binary_adaptive)
    
    # Áp dụng phép mở và đóng để loại bỏ nhiễu
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Tính độ dài từng phần
    from .feature_extraction import (extract_shape_features, 
                                            extract_texture_features,
                                            extract_color_features,
                                            extract_vein_features,
                                            extract_deep_features)
    
    shape_feats = extract_shape_features(binary_image)
    texture_feats = extract_texture_features(gray_image)
    color_feats = extract_color_features(image_rgb)
    vein_feats = extract_vein_features(gray_image)
    
    feature_sections = {
        'shape': len(shape_feats),
        'texture': len(texture_feats),
        'color': len(color_feats),
        'vein': len(vein_feats)
    }
    
    if deep_model is not None:
        deep_feats = extract_deep_features(image_rgb, deep_model)
        feature_sections['deep'] = len(deep_feats)
    
    return feature_sections

def build_feature_database_improved(image_folder, output_file, use_deep_features=True, model_type='resnet'):
    """Xây dựng cơ sở dữ liệu đặc trưng từ thư mục ảnh với các cải tiến"""
    # Tạo từ điển lưu trữ đặc trưng
    feature_database = {
        'file_paths': [],
        'features': [],
        'labels': [],
        'metadata': {}
    }
    
    # Tải mô hình CNN nếu cần
    deep_model = None
    if use_deep_features:
        print(f"Đang tải mô hình {model_type}...")
        deep_model = load_deep_model(model_type)
    
    # Duyệt qua các thư mục (mỗi thư mục là một loại lá)
    print(f"Đang xây dựng cơ sở dữ liệu từ {image_folder}...")
    
    feature_sections = None
    
    for label in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, label)
        
        if os.path.isdir(class_folder):
            print(f"Đang xử lý lá loại: {label}")
            
            # Duyệt qua từng ảnh trong thư mục
            image_files = [f for f in os.listdir(class_folder) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for filename in tqdm(image_files, desc=f"Xử lý {label}"):
                image_path = os.path.join(class_folder, filename)
                
                try:
                    # Tiền xử lý ảnh với phương pháp cải tiến
                    processed_img = preprocess_image(image_path)
                    
                    if processed_img is None:
                        print(f"Không thể xử lý ảnh: {image_path}")
                        continue
                    
                    # Trích xuất đặc trưng
                    features = extract_all_features(processed_img, deep_model, model_type)
                    
                    # Nếu chưa có thông tin về phần của vector đặc trưng, tính toán và lưu lại
                    if feature_sections is None:
                        feature_sections = compute_feature_lengths(processed_img, deep_model, model_type)
                        feature_database['feature_sections'] = feature_sections
                        print(f"Độ dài của các phần vector đặc trưng: {feature_sections}")
                    
                    # Thêm vào cơ sở dữ liệu
                    feature_database['file_paths'].append(image_path)
                    feature_database['features'].append(features)
                    feature_database['labels'].append(label)
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý {image_path}: {e}")
    
    # Thêm metadata
    feature_database['metadata']['use_deep_features'] = use_deep_features
    feature_database['metadata']['model_type'] = model_type if use_deep_features else None
    feature_database['metadata']['num_classes'] = len(set(feature_database['labels']))
    feature_database['metadata']['num_samples'] = len(feature_database['file_paths'])
    feature_database['metadata']['classes'] = list(set(feature_database['labels']))
    
    # Tính số lượng mẫu cho mỗi lớp
    class_counts = {}
    for label in feature_database['labels']:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    feature_database['metadata']['class_counts'] = class_counts
    
    # Chuyển danh sách đặc trưng thành mảng numpy
    feature_database['features'] = np.array(feature_database['features'])
    
    # Tạo thư mục cha nếu cần
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Lưu cơ sở dữ liệu
    with open(output_file, 'wb') as f:
        pickle.dump(feature_database, f)
    
    print(f"Đã xây dựng cơ sở dữ liệu với {len(feature_database['file_paths'])} ảnh")
    print(f"Đã lưu cơ sở dữ liệu vào: {output_file}")
    print(f"Thống kê lớp: {class_counts}")
    
    return feature_database