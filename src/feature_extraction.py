import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def extract_shape_features(binary_image):
    """Trích xuất đặc trưng hình dạng từ ảnh nhị phân"""
    # Tìm contour
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu không tìm thấy contour, trả về vector 0
    if not contours:
        return np.zeros(13)  # 6 đặc trưng cơ bản + 7 moment Hu
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Diện tích và chu vi
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Độ tròn
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)
    
    # Tỷ lệ khung hình
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Độ chặt (solidity)
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Độ kéo dài (extent)
    extent = float(area) / (w * h) if w * h > 0 else 0
    
    # Moment Hu
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Logarit của moment Hu để giảm phạm vi giá trị
    for i in range(len(hu_moments)):
        if hu_moments[i] != 0:
            hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
    
    # Kết hợp các đặc trưng
    shape_features = np.concatenate([
        np.array([area, perimeter, circularity, aspect_ratio, solidity, extent]),
        hu_moments
    ])
    
    return shape_features

def extract_texture_features(gray_image):
    """Trích xuất đặc trưng kết cấu từ ảnh xám"""
    # Đảm bảo dữ liệu là uint8
    gray_uint8 = gray_image.astype(np.uint8)
    
    # Giảm cấp độ xám để giảm tính toán
    gray_reduced = (gray_uint8 // 16).astype(np.uint8)
    
    # Tính GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(gray_reduced, distances, angles, 16, symmetric=True, normed=True)
    
    # Tính các đặc trưng Haralick
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    
    # Tính LBP
    lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    
    # Kết hợp các đặc trưng
    texture_features = np.concatenate([
        contrast, dissimilarity, homogeneity, energy, correlation, lbp_hist
    ])
    
    return texture_features

def extract_color_features(rgb_image):
    """Trích xuất đặc trưng màu sắc từ ảnh RGB"""
    # Chuyển sang không gian màu HSV
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    # Tách kênh
    h, s, v = cv2.split(hsv_image)
    
    # Tính histogram cho từng kênh
    h_hist, _ = np.histogram(h, bins=16, range=(0, 180), density=True)
    s_hist, _ = np.histogram(s, bins=16, range=(0, 256), density=True)
    v_hist, _ = np.histogram(v, bins=16, range=(0, 256), density=True)
    
    # Tính giá trị thống kê cho từng kênh
    h_mean, h_std = np.mean(h), np.std(h)
    s_mean, s_std = np.mean(s), np.std(s)
    v_mean, v_std = np.mean(v), np.std(v)
    
    # Chuyển sang không gian màu LAB
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab_image)
    
    # Tính giá trị thống kê cho LAB
    l_mean, l_std = np.mean(l), np.std(l)
    a_mean, a_std = np.mean(a), np.std(a)
    b_mean, b_std = np.mean(b), np.std(b)
    
    # Kết hợp các đặc trưng
    color_features = np.concatenate([
        h_hist, s_hist, v_hist, 
        np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std,
                 l_mean, l_std, a_mean, a_std, b_mean, b_std])
    ])
    
    return color_features

def extract_vein_features(gray_image):
    """Trích xuất đặc trưng gân lá từ ảnh xám"""
    # Tăng cường tương phản
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image.astype(np.uint8))
    
    # Áp dụng bộ lọc Gabor
    gabor_features = []
    for theta in np.arange(0, np.pi, np.pi / 4):  # 8 hướng
        kernel = cv2.getGaborKernel((15, 15), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)
        
        # Tính năng lượng và entropy
        energy = np.sum(filtered**2) / (filtered.shape[0] * filtered.shape[1])
        histogram, _ = np.histogram(filtered, bins=10, range=(0, 256), density=True)
        entropy = -np.sum(histogram * np.log2(histogram + 1e-10))
        
        gabor_features.extend([energy, entropy])
    
    # Phát hiện cạnh
    edges = cv2.Canny(enhanced, 50, 150)
    
    # Biến đổi Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=20, maxLineGap=10)
    
    # Đếm số lượng đường thẳng
    num_lines = 0 if lines is None else len(lines)
    
    # Tính toán hướng của các đường thẳng
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Tránh chia cho 0
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                angles.append(angle)
    
    # Phân bố góc
    angle_hist = np.zeros(9)  # 9 khoảng góc
    for angle in angles:
        index = int((angle + 90) // 20)
        if 0 <= index < 9:
            angle_hist[index] += 1
    
    # Chuẩn hóa histogram góc
    if num_lines > 0:
        angle_hist = angle_hist / num_lines
    
    # Kết hợp các đặc trưng
    vein_features = np.concatenate([
        np.array(gabor_features),
        np.array([num_lines]),
        angle_hist
    ])
    
    return vein_features

def load_deep_model():
    """Tải mô hình VGG16 để trích xuất đặc trưng học sâu"""
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    return model

def extract_deep_features(rgb_image, model):
    """Trích xuất đặc trưng học sâu sử dụng mô hình CNN"""
    # Tiền xử lý ảnh
    img_array = np.copy(rgb_image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = img_array.astype(np.float32)
    
    # Chuẩn hóa dữ liệu
    img_array = preprocess_input(img_array)
    
    # Mở rộng chiều batch
    img_array = np.expand_dims(img_array, axis=0)
    
    # Trích xuất đặc trưng
    features = model.predict(img_array, verbose=0)
    
    return features.flatten()

def extract_all_features(rgb_image, deep_model=None):
    """Trích xuất tất cả đặc trưng từ một ảnh"""
    # Tạo ảnh xám và ảnh nhị phân
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Trích xuất các loại đặc trưng
    shape_feats = extract_shape_features(binary_image)
    texture_feats = extract_texture_features(gray_image)
    color_feats = extract_color_features(rgb_image)
    vein_feats = extract_vein_features(gray_image)
    
    # Trích xuất đặc trưng học sâu nếu có mô hình
    if deep_model is not None:
        deep_feats = extract_deep_features(rgb_image, deep_model)
        
        # Kết hợp tất cả đặc trưng
        all_features = np.concatenate([
            shape_feats * 0.2,        # Trọng số 20%
            texture_feats * 0.2,      # Trọng số 20%
            color_feats * 0.2,        # Trọng số 20%
            vein_feats * 0.1,         # Trọng số 10%
            deep_feats * 0.3          # Trọng số 30%
        ])
    else:
        # Nếu không có mô hình học sâu
        all_features = np.concatenate([
            shape_feats * 0.3,        # Trọng số 30%
            texture_feats * 0.3,      # Trọng số 30%
            color_feats * 0.3,        # Trọng số 30%
            vein_feats * 0.1          # Trọng số 10%
        ])
    
    return all_features