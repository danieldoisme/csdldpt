import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
from .preprocess import preprocess_image
from .feature_extraction import extract_all_features, load_deep_model

def build_feature_database(image_folder, output_file, use_deep_features=True):
    """Xây dựng cơ sở dữ liệu đặc trưng từ thư mục ảnh"""
    # Tạo từ điển lưu trữ đặc trưng
    feature_database = {
        'file_paths': [],
        'features': [],
        'labels': []
    }
    
    # Tải mô hình CNN nếu cần
    deep_model = None
    if use_deep_features:
        print("Đang tải mô hình CNN...")
        deep_model = load_deep_model()
    
    # Duyệt qua các thư mục (mỗi thư mục là một loại lá)
    print(f"Đang xây dựng cơ sở dữ liệu từ {image_folder}...")
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
                    # Đọc và tiền xử lý ảnh
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Không thể đọc ảnh: {image_path}")
                        continue
                    
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Trích xuất đặc trưng
                    features = extract_all_features(image_rgb, deep_model)
                    
                    # Thêm vào cơ sở dữ liệu
                    feature_database['file_paths'].append(image_path)
                    feature_database['features'].append(features)
                    feature_database['labels'].append(label)
                    
                except Exception as e:
                    print(f"Lỗi khi xử lý {image_path}: {e}")
    
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
    
    return feature_database