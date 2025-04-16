import matplotlib.pyplot as plt
import cv2
import os

def display_results(query_image_path, similar_images, save_path=None):
    """Hiển thị ảnh truy vấn và các ảnh tương tự"""
    # Đọc ảnh truy vấn
    query_img = cv2.imread(query_image_path)
    if query_img is None:
        print(f"Không thể đọc ảnh truy vấn: {query_image_path}")
        return
    
    query_img = cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB)
    
    # Tạo figure với kích thước phù hợp
    plt.figure(figsize=(16, 5))
    
    # Hiển thị ảnh truy vấn
    plt.subplot(1, 4, 1)
    plt.imshow(query_img)
    plt.title("Ảnh truy vấn")
    plt.axis('off')
    
    # Hiển thị các ảnh tương tự
    for i, result in enumerate(similar_images):
        # Đọc ảnh tương tự
        similar_img = cv2.imread(result['image_path'])
        if similar_img is None:
            print(f"Không thể đọc ảnh tương tự: {result['image_path']}")
            continue
        
        similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
        
        # Hiển thị ảnh
        plt.subplot(1, 4, i + 2)
        plt.imshow(similar_img)
        plt.title(f"#{i+1}: {result['label']}\nĐộ tương đồng: {result['similarity']:.3f}")
        plt.axis('off')
    
    # Thêm padding
    plt.tight_layout(pad=2.0)
    
    # Lưu kết quả nếu cần
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        print(f"Đã lưu kết quả vào: {save_path}")
    
    # Hiển thị
    plt.show()

def create_result_folder(result_dir):
    """Tạo thư mục lưu kết quả nếu chưa tồn tại"""
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir