import os
import cv2
from tqdm import tqdm
import numpy as np

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

def preprocess_image_simple(image_path, output_path=None, target_size=(224, 224)):
    """Tiền xử lý ảnh với phương pháp đơn giản nhưng ổn định"""
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Chuyển sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chuyển sang HSV để phân đoạn lá dễ dàng hơn
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    
    # Tạo mask cho lá cây (màu xanh lá trong không gian màu HSV)
    # Dải màu xanh lá trong HSV
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    
    # Tạo mask
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Cải thiện mask với xử lý hình thái học
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Tìm contour lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Không phát hiện lá trong ảnh: {image_path}")
        resized = cv2.resize(image_rgb, target_size)
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        return resized
    
    # Lấy contour lớn nhất
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tạo mask từ contour
    refined_mask = np.zeros_like(mask)
    cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
    
    # Áp dụng mask để lấy chỉ lá
    result = cv2.bitwise_and(image_rgb, image_rgb, mask=refined_mask)
    
    # Tạo background trắng
    white_bg = np.ones_like(image_rgb) * 255
    inv_mask = cv2.bitwise_not(refined_mask)
    background = cv2.bitwise_and(white_bg, white_bg, mask=inv_mask)
    
    # Kết hợp lá với background trắng
    result = cv2.add(result, background)
    
    # Cắt ra vùng chứa lá
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Mở rộng vùng cắt để đảm bảo không mất chi tiết
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(result.shape[1] - x, w + 2*padding)
    h = min(result.shape[0] - y, h + 2*padding)
    roi = result[y:y+h, x:x+w]
    
    # Thay đổi kích thước
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
    
    # Lưu ảnh nếu cần
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    
    return resized

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

def preprocess_all_images_simple(raw_data_dir, processed_data_dir):
    """Tiền xử lý tất cả ảnh trong thư mục với phương pháp đơn giản"""
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    # Duyệt qua tất cả các thư mục con
    for class_name in os.listdir(raw_data_dir):
        class_dir = os.path.join(raw_data_dir, class_name)
        
        if os.path.isdir(class_dir):
            # Tạo thư mục đầu ra cho loại lá này
            output_class_dir = os.path.join(processed_data_dir, class_name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
            
            print(f"Đang tiền xử lý ảnh lá loại {class_name}...")
            
            # Duyệt qua tất cả ảnh trong thư mục
            image_files = [f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for img_file in tqdm.tqdm(image_files, desc=f"Xử lý {class_name}"):
                # Đường dẫn đầy đủ đến ảnh
                img_path = os.path.join(class_dir, img_file)
                
                # Đường dẫn đầu ra
                output_path = os.path.join(output_class_dir, img_file)
                
                # Tiền xử lý ảnh
                try:
                    preprocess_image_simple(img_path, output_path)
                except Exception as e:
                    print(f"Lỗi khi xử lý {img_path}: {e}")
    
    print(f"Đã hoàn thành tiền xử lý cho {len(os.listdir(raw_data_dir))} loại lá.")