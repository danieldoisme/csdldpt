import os
import cv2
import numpy as np
from skimage import exposure

def create_directories(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def remove_shadow(image, mask):
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Điều chỉnh độ sáng để giảm bóng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    
    # Tái tạo ảnh sau khi đã điều chỉnh độ sáng
    hsv_enhanced = cv2.merge([h, s, v])
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

def preserve_details(mask, original_gray):
    # Phát hiện cạnh từ ảnh gốc
    edges = cv2.Canny(original_gray, 50, 150)
    
    # Mở rộng mask một chút để bao gồm các chi tiết biên
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Kết hợp các cạnh phát hiện được với mask
    edge_mask = cv2.bitwise_and(edges, edges, mask=dilated_mask)
    
    # Thêm các chi tiết cạnh vào mask
    final_mask = cv2.bitwise_or(mask, edge_mask)
    
    return final_mask

def preprocess_image(image_path, output_path=None, target_size=(224, 224)):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Chuyển sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chuyển sang không gian màu LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Sử dụng kênh a để phân đoạn lá (kênh a phân biệt màu xanh lá tốt hơn)
    blurred_a = cv2.GaussianBlur(a, (5, 5), 0)
    _, binary = cv2.threshold(blurred_a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Áp dụng các phép toán hình thái học để cải thiện mask
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
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
    
    # Bảo toàn chi tiết gân lá và biên
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = preserve_details(mask, gray)
    
    # Loại bỏ bóng và thay đổi nền
    result = remove_shadow(image, mask)
    
    # Cắt ra vùng chứa lá
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Mở rộng vùng cắt một chút để đảm bảo không mất chi tiết
    padding = 5
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(result.shape[1] - x, w + 2*padding)
    h = min(result.shape[0] - y, h + 2*padding)
    roi = result[y:y+h, x:x+w]
    
    # Thay đổi kích thước
    resized = cv2.resize(roi, target_size)
    
    # Lưu ảnh nếu cần
    if output_path:
        cv2.imwrite(output_path, resized)
    
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

def preprocess_all_images(input_folder, output_folder, target_size=(224, 224)):
    """Tiền xử lý tất cả ảnh trong thư mục và lưu vào thư mục đầu ra"""
    create_directories(output_folder)
    
    # Duyệt qua các thư mục con (mỗi thư mục là một loại lá)
    for label in os.listdir(input_folder):
        input_label_path = os.path.join(input_folder, label)
        output_label_path = os.path.join(output_folder, label)
        
        if os.path.isdir(input_label_path):
            create_directories(output_label_path)
            
            # Xử lý từng ảnh trong thư mục
            for filename in os.listdir(input_label_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_image_path = os.path.join(input_label_path, filename)
                    output_image_path = os.path.join(output_label_path, filename)
                    
                    try:
                        preprocess_image(input_image_path, output_image_path, target_size)
                        print(f"Đã xử lý: {input_image_path}")
                    except Exception as e:
                        print(f"Lỗi khi xử lý {input_image_path}: {e}")