import os
import cv2
import numpy as np
from skimage import exposure

def create_directories(directory):
    """Tạo thư mục nếu chưa tồn tại"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_image(image_path, output_path=None, target_size=(224, 224)):
    """Tiền xử lý một ảnh lá cây"""
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print(f"Không thể đọc ảnh: {image_path}")
        return None
    
    # Chuyển sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chuyển sang ảnh xám để phân đoạn
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Phân ngưỡng để tách lá khỏi nền
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Tìm contour
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print(f"Không tìm thấy lá trong ảnh: {image_path}")
        # Nếu không tìm thấy contour, chỉ thay đổi kích thước ảnh gốc
        resized = cv2.resize(image_rgb, target_size)
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        return resized
    
    # Lấy contour lớn nhất (giả sử là lá cây)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Tạo mask từ contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    
    # Áp dụng mask
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    # Cắt ra vùng chứa lá
    x, y, w, h = cv2.boundingRect(largest_contour)
    roi = masked_image[y:y+h, x:x+w]
    
    # Nếu ROI rỗng, sử dụng ảnh gốc
    if roi.size == 0:
        roi = image_rgb
    
    # Cân bằng histogram để cải thiện độ tương phản
    roi_enhanced = exposure.equalize_hist(roi)
    roi_enhanced = (roi_enhanced * 255).astype(np.uint8)
    
    # Thay đổi kích thước
    resized = cv2.resize(roi_enhanced, target_size)
    
    # Lưu ảnh nếu cần
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    
    return resized

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