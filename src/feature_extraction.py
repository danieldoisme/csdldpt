import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from tensorflow.keras.applications.efficientnet import EfficientNetB0 # type: ignore
from scipy.stats import entropy
import math

def extract_shape_features(binary_image):
    """Trích xuất đặc trưng hình dạng từ ảnh nhị phân với các chỉ số cải tiến"""
    # Tìm contour
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Nếu không tìm thấy contour, trả về vector 0
    if not contours:
        return np.zeros(17)  # 10 đặc trưng cơ bản + 7 moment Hu
    
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
    
    # Độ lồi (convexity)
    convexity = float(hull_area) / (perimeter * perimeter + 1e-6)
    
    # Trục chính và trục phụ
    if len(largest_contour) >= 5:  # Cần ít nhất 5 điểm để tính ellipse
        ellipse = cv2.fitEllipse(largest_contour)
        major_axis = max(ellipse[1])
        minor_axis = min(ellipse[1])
        eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
    else:
        major_axis = max(w, h)
        minor_axis = min(w, h)
        eccentricity = np.sqrt(1 - (minor_axis/major_axis)**2) if major_axis > 0 else 0
    
    # Số lượng góc (dựa trên phép xấp xỉ đa giác)
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    num_corners = len(approx)
    
    # Moment Hu
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # Logarit của moment Hu để giảm phạm vi giá trị
    for i in range(len(hu_moments)):
        if hu_moments[i] != 0:
            hu_moments[i] = -np.sign(hu_moments[i]) * np.log10(abs(hu_moments[i]))
    
    # Kết hợp các đặc trưng
    shape_features = np.concatenate([
        np.array([area, perimeter, circularity, aspect_ratio, solidity, extent, 
                 convexity, eccentricity, major_axis/minor_axis, num_corners]),
        hu_moments
    ])
    
    return shape_features

def extract_texture_features(gray_image):
    """Trích xuất đặc trưng kết cấu từ ảnh xám với các tính năng bổ sung"""
    # Đảm bảo dữ liệu là uint8
    gray_uint8 = gray_image.astype(np.uint8)
    
    # Giảm cấp độ xám để giảm tính toán
    gray_reduced = (gray_uint8 // 16).astype(np.uint8)
    
    # Tính GLCM với nhiều khoảng cách và góc hơn
    distances = [1, 2, 3, 4]  # Thêm khoảng cách 4
    angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]  # 8 góc thay vì 4
    glcm = graycomatrix(gray_reduced, distances, angles, 16, symmetric=True, normed=True)
    
    # Tính các đặc trưng Haralick
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    ASM = graycoprops(glcm, 'ASM').flatten()  # Thêm Angular Second Moment
    
    # Tính LBP với nhiều tham số hơn
    lbp_patterns = []
    # Thử với các bán kính khác nhau
    for r in [1, 2]:
        # Thử với số điểm khác nhau
        for p in [8, 16]:
            lbp = local_binary_pattern(gray_image, P=p, R=r, method='uniform')
            # Thêm các phân vị thay vì histogram
            lbp_hist, _ = np.histogram(lbp, bins=p+2, density=True)
            lbp_patterns.append(lbp_hist)
    
    # Tính Sobel gradient để nắm bắt thông tin về biên
    sobelx = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
    
    # Histogram của gradient magnitude và direction
    mag_hist, _ = np.histogram(gradient_magnitude, bins=10, density=True)
    dir_hist, _ = np.histogram(gradient_direction, bins=18, range=(-180, 180), density=True)
    
    # Kết hợp các đặc trưng
    texture_features = np.concatenate([
        contrast, dissimilarity, homogeneity, energy, correlation, ASM, 
        np.concatenate(lbp_patterns),
        mag_hist, dir_hist
    ])
    
    return texture_features

def extract_color_features(rgb_image):
    """Trích xuất đặc trưng màu sắc từ ảnh RGB với nhiều không gian màu hơn"""
    # Chuyển sang các không gian màu khác nhau
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
    ycrcb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)  # Thêm không gian màu YCrCb
    
    # Tách kênh
    r, g, b = cv2.split(rgb_image)
    h, s, v = cv2.split(hsv_image)
    l, a, b_lab = cv2.split(lab_image)
    y, cr, cb = cv2.split(ycrcb_image)
    
    # Tính histogram cho từng kênh với số bins khác nhau
    r_hist, _ = np.histogram(r, bins=16, range=(0, 256), density=True)
    g_hist, _ = np.histogram(g, bins=16, range=(0, 256), density=True)
    b_hist, _ = np.histogram(b, bins=16, range=(0, 256), density=True)
    
    h_hist, _ = np.histogram(h, bins=18, range=(0, 180), density=True)  # Tăng số bins
    s_hist, _ = np.histogram(s, bins=16, range=(0, 256), density=True)
    v_hist, _ = np.histogram(v, bins=16, range=(0, 256), density=True)
    
    l_hist, _ = np.histogram(l, bins=16, range=(0, 256), density=True)
    a_hist, _ = np.histogram(a, bins=16, range=(0, 256), density=True)
    b_hist, _ = np.histogram(b_lab, bins=16, range=(0, 256), density=True)
    
    # Thêm histogram cho Y, Cr, Cb
    y_hist, _ = np.histogram(y, bins=16, range=(0, 256), density=True)
    cr_hist, _ = np.histogram(cr, bins=16, range=(0, 256), density=True)
    cb_hist, _ = np.histogram(cb, bins=16, range=(0, 256), density=True)
    
    # Tính giá trị thống kê cho từng kênh
    # RGB
    r_mean, r_std = np.mean(r), np.std(r)
    g_mean, g_std = np.mean(g), np.std(g)
    b_mean, b_std = np.mean(b), np.std(b)
    
    # Tỷ lệ giữa các kênh RGB (đặc biệt hữu ích cho lá cây)
    r_g_ratio = r_mean / (g_mean + 1e-10)
    r_b_ratio = r_mean / (b_mean + 1e-10)
    g_b_ratio = g_mean / (b_mean + 1e-10)
    
    # HSV
    h_mean, h_std = np.mean(h), np.std(h)
    s_mean, s_std = np.mean(s), np.std(s)
    v_mean, v_std = np.mean(v), np.std(v)
    
    # LAB
    l_mean, l_std = np.mean(l), np.std(l)
    a_mean, a_std = np.mean(a), np.std(a)
    b_lab_mean, b_lab_std = np.mean(b_lab), np.std(b_lab)
    
    # YCrCb
    y_mean, y_std = np.mean(y), np.std(y)
    cr_mean, cr_std = np.mean(cr), np.std(cr)
    cb_mean, cb_std = np.mean(cb), np.std(cb)
    
    # Thêm chỉ số khác
    # Chỉ số xanh lá - Đây là đặc trưng đặc biệt quan trọng cho việc phân loại lá cây
    exg = 2 * g_mean - r_mean - b_mean  # Excess Green Index
    exr = 1.4 * r_mean - g_mean  # Excess Red Index
    ndvi = (r_mean - g_mean) / (r_mean + g_mean + 1e-10)  # Normalized Difference Vegetation Index mô phỏng
    
    # Kết hợp các đặc trưng
    color_features = np.concatenate([
        # Histograms
        r_hist, g_hist, b_hist, h_hist, s_hist, v_hist, l_hist, a_hist, b_hist,
        y_hist, cr_hist, cb_hist,
        
        # Statistical features
        np.array([
            r_mean, r_std, g_mean, g_std, b_mean, b_std,
            r_g_ratio, r_b_ratio, g_b_ratio,
            h_mean, h_std, s_mean, s_std, v_mean, v_std,
            l_mean, l_std, a_mean, a_std, b_lab_mean, b_lab_std,
            y_mean, y_std, cr_mean, cr_std, cb_mean, cb_std,
            exg, exr, ndvi
        ])
    ])
    
    return color_features

def extract_vein_features(gray_image):
    """Trích xuất đặc trưng gân lá từ ảnh xám với các cải tiến"""
    # Tăng cường tương phản
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image.astype(np.uint8))
    
    # Áp dụng bộ lọc Gabor với nhiều tham số hơn
    gabor_features = []
    for theta in np.arange(0, np.pi, np.pi / 8):  # 8 hướng
        for sigma in [3.0, 5.0]:  # Thêm nhiều độ lệch chuẩn
            for lambd in [9.0, 13.0]:  # Thêm nhiều bước sóng
                kernel = cv2.getGaborKernel((21, 21), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(enhanced, cv2.CV_8UC3, kernel)
                
                # Tính năng lượng và entropy
                energy = np.sum(filtered**2) / (filtered.shape[0] * filtered.shape[1])
                histogram, _ = np.histogram(filtered, bins=10, range=(0, 256), density=True)
                ent = entropy(histogram + 1e-10)
                
                # Tính phương sai của phản hồi bộ lọc
                variance = np.var(filtered)
                
                gabor_features.extend([energy, ent, variance])
    
    # Phát hiện cạnh với nhiều thuật toán
    edges_canny = cv2.Canny(enhanced, 30, 120)
    sobel_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
    edges_sobel = np.sqrt(sobel_x**2 + sobel_y**2).astype(np.uint8)
    
    # Tạo ảnh nhị phân từ cạnh Sobel
    _, edges_sobel_binary = cv2.threshold(edges_sobel, 40, 255, cv2.THRESH_BINARY)
    
    # Biến đổi Hough trên cả hai ảnh cạnh
    lines_canny = cv2.HoughLinesP(edges_canny, 1, np.pi/180, threshold=40, minLineLength=15, maxLineGap=10)
    lines_sobel = cv2.HoughLinesP(edges_sobel_binary, 1, np.pi/180, threshold=40, minLineLength=15, maxLineGap=10)
    
    # Đếm số lượng đường thẳng
    num_lines_canny = 0 if lines_canny is None else len(lines_canny)
    num_lines_sobel = 0 if lines_sobel is None else len(lines_sobel)
    total_lines = num_lines_canny + num_lines_sobel
    
    # Tính toán hướng của các đường thẳng
    angles = []
    
    # Xử lý đường thẳng từ Canny
    if lines_canny is not None:
        for line in lines_canny:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Tránh chia cho 0
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                angles.append(angle)
    
    # Xử lý đường thẳng từ Sobel
    if lines_sobel is not None:
        for line in lines_sobel:
            x1, y1, x2, y2 = line[0]
            if x2 != x1:  # Tránh chia cho 0
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                angles.append(angle)
    
    # Phân bố góc với số lượng bin nhiều hơn
    angle_hist = np.zeros(12)  # 12 khoảng góc thay vì 9
    for angle in angles:
        index = int((angle + 90) // 15)  # Chia thành các khoảng 15 độ
        if 0 <= index < 12:
            angle_hist[index] += 1
    
    # Chuẩn hóa histogram góc
    if total_lines > 0:
        angle_hist = angle_hist / total_lines
    
    # Tính mật độ gân lá (số đường/diện tích)
    vein_density = total_lines / (gray_image.shape[0] * gray_image.shape[1])
    
    # Tính độ cong trung bình của các đường gân
    curvature = 0
    if lines_canny is not None:
        for line in lines_canny:
            x1, y1, x2, y2 = line[0]
            # Tính độ cong gần đúng bằng tỷ lệ giữa chiều dài thẳng và khoảng cách Euclidean
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            curvature += length
    
    avg_curvature = curvature / (total_lines + 1e-10)
    
    # Kết hợp các đặc trưng
    vein_features = np.concatenate([
        np.array(gabor_features),
        np.array([num_lines_canny, num_lines_sobel, vein_density, avg_curvature]),
        angle_hist
    ])
    
    return vein_features

def load_deep_model(model_type='resnet'):
    """Tải mô hình học sâu để trích xuất đặc trưng"""
    if model_type == 'resnet':
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    elif model_type == 'efficientnet':
        model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
    else:
        # Mặc định sử dụng ResNet50
        model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
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

def extract_all_features(rgb_image, deep_model=None, model_type='resnet'):
    """Trích xuất tất cả đặc trưng từ một ảnh với trọng số cải tiến"""
    # Tạo ảnh xám và ảnh nhị phân
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    
    # Áp dụng CLAHE để cải thiện tương phản trước khi phân ngưỡng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray_image)
    
    # Thử nghiệm với nhiều phương pháp phân ngưỡng khác nhau
    # 1. Phương pháp Otsu
    _, binary_otsu = cv2.threshold(enhanced_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. Thử phân ngưỡng thích ứng
    binary_adaptive = cv2.adaptiveThreshold(enhanced_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
    
    # Kết hợp hai phương pháp
    binary_image = cv2.bitwise_or(binary_otsu, binary_adaptive)
    
    # Áp dụng phép mở và đóng để loại bỏ nhiễu
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Trích xuất các loại đặc trưng
    shape_feats = extract_shape_features(binary_image)
    texture_feats = extract_texture_features(gray_image)
    color_feats = extract_color_features(rgb_image)
    vein_feats = extract_vein_features(gray_image)
    
    # Trích xuất đặc trưng học sâu nếu có mô hình
    if deep_model is not None:
        deep_feats = extract_deep_features(rgb_image, deep_model)
        
        # Kết hợp tất cả đặc trưng với trọng số mới
        # Ưu tiên đặc trưng hình dạng và gân lá cho việc phân loại lá cây
        all_features = np.concatenate([
            shape_feats * 0.35,
            texture_feats * 0.10,
            color_feats * 0.15,
            vein_feats * 0.25,
            deep_feats * 0.15
        ])
    else:
        # Nếu không có mô hình học sâu
        all_features = np.concatenate([
            shape_feats * 0.40,
            texture_feats * 0.15,
            color_feats * 0.15,
            vein_feats * 0.30
        ])
    
    return all_features