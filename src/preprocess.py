import os
import cv2
import numpy as np
from skimage import exposure

def create_directories(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def normalize_image(image):
    """Chuẩn hóa ảnh để giảm ảnh hưởng của điều kiện ánh sáng"""
    # Chuẩn hóa từng kênh màu riêng biệt
    normalized = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        channel = image[:,:,i].astype(np.float32)
        if np.max(channel) > np.min(channel):
            normalized[:,:,i] = (channel - np.min(channel)) / (np.max(channel) - np.min(channel))
    
    # Chuyển về dạng uint8
    normalized = (normalized * 255).astype(np.uint8)
    return normalized

def remove_shadow_improved(image, mask):
    """Cải tiến thuật toán loại bỏ bóng"""
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Điều chỉnh độ sáng để giảm bóng với CLAHE cải tiến
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Giảm bão hòa của vùng tối
    s_adjusted = s.copy()
    dark_regions = (v < 128)
    s_adjusted[dark_regions] = s_adjusted[dark_regions] * 0.7  # Giảm độ bão hòa ở vùng tối
    
    # Tái tạo ảnh sau khi đã điều chỉnh
    hsv_enhanced = cv2.merge([h, s_adjusted, v])
    enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    # Áp dụng mặt nạ để loại bỏ hoàn toàn nền
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    
    # Tạo nền trắng thay vì nền đen
    background = np.ones_like(image, np.uint8) * 255
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background, background, mask=mask_inv)
    
    # Kết hợp ảnh lá với nền trắng
    final = cv2.add(result, background)
    
    return final

def enhance_leaf_details(image, mask):
    """Tăng cường chi tiết của lá"""
    # Áp dụng bộ lọc làm sắc nét
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpen)
    
    # Tăng cường độ tương phản cho vùng lá
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Áp dụng CLAHE chỉ trên kênh độ sáng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Tái tạo ảnh
    enhanced_lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Áp dụng chỉ trong vùng lá
    result = cv2.bitwise_and(enhanced, enhanced, mask=mask)
    
    # Kết hợp với phần nền
    background = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    final = cv2.add(result, background)
    
    return final

def extract_leaf_vein_mask(gray_image, mask):
    """Trích xuất một mask riêng cho gân lá"""
    # Cải thiện độ tương phản cho ảnh xám
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_image)
    
    # Áp dụng bộ lọc làm nổi bật gân lá
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Áp dụng bộ lọc Gabor để phát hiện gân lá theo các hướng khác nhau
    gabor_responses = []
    for theta in np.arange(0, np.pi, np.pi/4):
        kern = cv2.getGaborKernel((15, 15), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(sharpened, cv2.CV_8UC3, kern)
        gabor_responses.append(filtered)
    
    # Kết hợp các phản hồi của bộ lọc Gabor
    vein_response = np.zeros_like(gray_image)
    for response in gabor_responses:
        vein_response = cv2.max(vein_response, response)
    
    # Áp dụng phân ngưỡng thích ứng
    vein_binary = cv2.adaptiveThreshold(vein_response, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY_INV, 11, 2)
    
    # Chỉ giữ lại vein trong vùng lá
    vein_mask = cv2.bitwise_and(vein_binary, vein_binary, mask=mask)
    
    # Áp dụng phép mở để loại bỏ nhiễu
    kernel = np.ones((2, 2), np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return vein_mask

def preprocess_image_improved(image_path, output_path=None, target_size=(224, 224)):
    """Tiền xử lý ảnh với các cải tiến"""
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Chuẩn hóa ảnh để giảm ảnh hưởng của điều kiện ánh sáng
    image = normalize_image(image)
    
    # Chuyển sang RGB và lưu lại ảnh gốc cho các bước xử lý sau
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chuyển sang không gian màu LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Tạo mask lá cây sử dụng kết hợp kênh a và b
    # Kênh a phân biệt màu xanh-đỏ, kênh b phân biệt màu xanh-vàng
    green_mask = cv2.subtract(b, a)  # Tạo mask nhấn mạnh màu xanh lá
    
    # Làm mịn mask với bộ lọc Gaussian
    blurred_green = cv2.GaussianBlur(green_mask, (5, 5), 0)
    
    # Áp dụng phân ngưỡng Otsu
    _, binary_otsu = cv2.threshold(blurred_green, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Thêm phân ngưỡng thích ứng để nắm bắt chi tiết
    binary_adaptive = cv2.adaptiveThreshold(
        blurred_green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Kết hợp hai phương pháp
    binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
    
    # Áp dụng các phép toán hình thái học để cải thiện mask
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Tìm contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Không tìm thấy lá trong ảnh: {image_path}")
        resized = cv2.resize(image_rgb, target_size)
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        return resized
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tạo mask từ contour
    mask = np.zeros_like(l)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Chuyển sang ảnh xám cho việc xử lý chi tiết
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Bảo toàn chi tiết gân lá và biên với phương pháp cải tiến
    mask = preserve_details_improved(mask, gray)
    
    # Trích xuất mask gân lá riêng để sử dụng sau này
    vein_mask = extract_leaf_vein_mask(gray, mask)
    
    # Loại bỏ bóng và nâng cao chất lượng ảnh
    result = remove_shadow_improved(image, mask)
    
    # Tăng cường chi tiết gân lá
    result = enhance_leaf_details(result, mask)
    
    # Cắt ra vùng chứa lá
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Mở rộng vùng cắt để đảm bảo không mất chi tiết
    padding = 10  # Tăng padding từ 5 lên 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(result.shape[1] - x, w + 2*padding)
    h = min(result.shape[0] - y, h + 2*padding)
    roi = result[y:y+h, x:x+w]
    
    # Thay đổi kích thước
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)  # INTER_AREA tốt hơn cho giảm kích thước
    
    # Lưu ảnh nếu cần
    if output_path:
        cv2.imwrite(output_path, resized)
    
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def preserve_details_improved(mask, original_gray):
    """Cải tiến thuật toán bảo toàn chi tiết"""
    # Phát hiện cạnh từ ảnh gốc với nhiều phương pháp
    edges_canny = cv2.Canny(original_gray, 40, 140)
    
    # Phát hiện cạnh bằng Sobel
    sobelx = cv2.Sobel(original_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(original_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobelx, sobely)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, sobel_binary = cv2.threshold(sobel_edges, 40, 255, cv2.THRESH_BINARY)
    
    # Kết hợp các phương pháp phát hiện cạnh
    combined_edges = cv2.bitwise_or(edges_canny, sobel_binary)
    
    # Mở rộng mask một chút để bao gồm các chi tiết biên
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Kết hợp các cạnh phát hiện được với mask
    edge_mask = cv2.bitwise_and(combined_edges, combined_edges, mask=dilated_mask)
    
    # Thêm các chi tiết cạnh vào mask
    final_mask = cv2.bitwise_or(mask, edge_mask)
    
    # Làm mịn mask cuối cùng để loại bỏ nhiễu
    final_mask = cv2.medianBlur(final_mask, 3)
    
    return final_mask