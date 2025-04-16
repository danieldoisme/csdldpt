import pickle
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from .preprocess import preprocess_image_simple as preprocess_image
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

def compute_weighted_similarity(query_features, database_features, weights=[0.7, 0.3]):
    """Tính độ tương đồng kết hợp giữa nhiều phương pháp đo độ tương đồng"""
    # Tính độ tương đồng cosine
    cosine_sims = cosine_similarity([query_features], database_features)[0]
    
    # Tính khoảng cách Euclidean và chuyển thành độ tương đồng
    euclidean_dists = euclidean_distances([query_features], database_features)[0]
    max_dist = np.max(euclidean_dists) if np.max(euclidean_dists) > 0 else 1
    euclidean_sims = 1 - (euclidean_dists / max_dist)
    
    # Kết hợp các độ tương đồng với trọng số
    similarities = weights[0] * cosine_sims + weights[1] * euclidean_sims
    
    return similarities

def compute_similarity_per_feature_type(query_features, database_features, feature_sections):
    """Tính độ tương đồng cho từng loại đặc trưng riêng biệt"""
    start_idx = 0
    section_similarities = {}
    
    for section_name, section_length in feature_sections.items():
        # Trích xuất phần đặc trưng tương ứng
        query_section = query_features[start_idx:start_idx + section_length]
        db_sections = database_features[:, start_idx:start_idx + section_length]
        
        # Tính độ tương đồng cho loại đặc trưng này
        section_sims = cosine_similarity([query_section], db_sections)[0]
        section_similarities[section_name] = section_sims
        
        # Di chuyển đến phần tiếp theo
        start_idx += section_length
    
    return section_similarities

def filter_by_shape(similarities_by_type, threshold=0.7):
    """Lọc kết quả dựa trên độ tương đồng hình dạng"""
    shape_sims = similarities_by_type.get('shape', np.array([]))
    if len(shape_sims) == 0:
        return np.ones(similarities_by_type.get(list(similarities_by_type.keys())[0], np.array([])).shape, dtype=bool)
    
    # Chỉ giữ lại các mẫu có độ tương đồng hình dạng vượt ngưỡng
    return shape_sims >= threshold

def find_similar_images_improved(query_image_path, feature_database, top_k=3, 
                                use_deep_features=True, 
                                model_type='resnet', 
                                similarity_weights=[0.7, 0.3],
                                feature_weights=None):
    """Tìm k ảnh tương tự nhất với ảnh truy vấn sử dụng phương pháp cải tiến"""
    # Tải mô hình CNN nếu cần
    deep_model = None
    if use_deep_features:
        deep_model = load_deep_model(model_type)
    
    # Tiền xử lý ảnh truy vấn với phương pháp cải tiến
    processed_img = preprocess_image(query_image_path)
    
    if processed_img is None:
        print(f"Không thể xử lý ảnh truy vấn: {query_image_path}")
        return []
    
    # Trích xuất đặc trưng từ ảnh truy vấn đã được tiền xử lý
    query_features = extract_all_features(processed_img, deep_model, model_type)
    
    # Tính độ tương đồng với tất cả ảnh trong cơ sở dữ liệu
    database_features = feature_database['features']
    
    # Nếu cơ sở dữ liệu có thông tin về phần của vector đặc trưng
    if 'feature_sections' in feature_database:
        feature_sections = feature_database['feature_sections']
        
        # Tính độ tương đồng theo từng loại đặc trưng
        similarities_by_type = compute_similarity_per_feature_type(
            query_features, database_features, feature_sections)
        
        # Lọc kết quả dựa trên độ tương đồng hình dạng
        shape_filter = filter_by_shape(similarities_by_type, threshold=0.65)
        
        # Tính toán độ tương đồng tổng hợp với trọng số động
        if feature_weights is None:
            # Trọng số mặc định
            feature_weights = {
                'shape': 0.35,      # Hình dạng
                'texture': 0.10,    # Kết cấu
                'color': 0.15,      # Màu sắc
                'vein': 0.25,       # Gân lá
                'deep': 0.15        # Deep features
            }
        
        # Kết hợp các đo lường tương đồng với trọng số
        weighted_similarities = np.zeros(database_features.shape[0])
        for feature_type, similarity in similarities_by_type.items():
            if feature_type in feature_weights:
                weighted_similarities += similarity * feature_weights[feature_type]
        
        # Áp dụng bộ lọc hình dạng
        filtered_similarities = np.copy(weighted_similarities)
        filtered_similarities[~shape_filter] *= 0.5  # Giảm 50% điểm cho những mẫu không đạt ngưỡng hình dạng
    else:
        # Phương pháp dự phòng: tính độ tương đồng tổng thể nếu không có thông tin phần
        weighted_similarities = compute_weighted_similarity(
            query_features, database_features, similarity_weights)
        filtered_similarities = weighted_similarities
    
    # Sắp xếp theo độ tương đồng giảm dần
    indices = np.argsort(filtered_similarities)[::-1][:top_k]
    
    # Tạo danh sách kết quả
    results = []
    for idx in indices:
        results.append({
            'image_path': feature_database['file_paths'][idx],
            'similarity': filtered_similarities[idx],
            'label': feature_database['labels'][idx]
        })
    
    return results

def analyze_similar_images(query_image_path, similar_images, output_dir=None):
    """Phân tích chi tiết sự giống nhau giữa ảnh truy vấn và các ảnh tương tự"""
    # Đọc ảnh truy vấn
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"Không thể đọc ảnh truy vấn: {query_image_path}")
        return
    
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # So sánh từng ảnh tương tự
    for i, result in enumerate(similar_images):
        # Đọc ảnh tương tự
        similar_img_path = result['image_path']
        similar_img = cv2.imread(similar_img_path)
        if similar_img is None:
            print(f"Không thể đọc ảnh tương tự: {similar_img_path}")
            continue
        
        similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
        
        # Resize ảnh nếu kích thước khác nhau
        if query_img.shape[:2] != similar_img.shape[:2]:
            similar_img = cv2.resize(similar_img, (query_img.shape[1], query_img.shape[0]))
        
        # Phân tích hình dạng
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_RGB2GRAY)
        similar_gray = cv2.cvtColor(similar_img, cv2.COLOR_RGB2GRAY)
        
        # Phân ngưỡng để lấy hình dạng
        _, query_binary = cv2.threshold(query_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        _, similar_binary = cv2.threshold(similar_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Tìm contour
        query_contours, _ = cv2.findContours(query_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        similar_contours, _ = cv2.findContours(similar_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Vẽ contour lên ảnh
        query_with_contour = query_img.copy()
        similar_with_contour = similar_img.copy()
        
        if query_contours:
            largest_query_contour = max(query_contours, key=cv2.contourArea)
            cv2.drawContours(query_with_contour, [largest_query_contour], 0, (255, 0, 0), 2)
        
        if similar_contours:
            largest_similar_contour = max(similar_contours, key=cv2.contourArea)
            cv2.drawContours(similar_with_contour, [largest_similar_contour], 0, (255, 0, 0), 2)
        
        # Tạo ảnh kết quả
        result_img = np.hstack((query_with_contour, similar_with_contour))
        
        # Hiển thị hoặc lưu kết quả
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"compare_{i+1}_{result['label']}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            print(f"Đã lưu ảnh so sánh: {output_path}")
        
        # Tính toán một số chỉ số tương đồng
        if query_contours and similar_contours:
            # So sánh hình dạng với cv2.matchShapes
            shape_distance = cv2.matchShapes(largest_query_contour, largest_similar_contour, 
                                            cv2.CONTOURS_MATCH_I3, 0)
            print(f"Khoảng cách hình dạng với {result['label']}: {shape_distance:.5f}")
            
            # So sánh histogram màu sắc
            query_hsv = cv2.cvtColor(query_img, cv2.COLOR_RGB2HSV)
            similar_hsv = cv2.cvtColor(similar_img, cv2.COLOR_RGB2HSV)
            
            h_bins = 50
            s_bins = 60
            histSize = [h_bins, s_bins]
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges
            channels = [0, 1]
            
            query_hist = cv2.calcHist([query_hsv], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(query_hist, query_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            similar_hist = cv2.calcHist([similar_hsv], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(similar_hist, similar_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            hist_comparison = cv2.compareHist(query_hist, similar_hist, cv2.HISTCMP_CORREL)
            print(f"Tương quan histogram với {result['label']}: {hist_comparison:.5f}")