import os
import argparse
from src.preprocess import preprocess_all_images_improved
from src.database import build_feature_database_improved
from src.retrieval import load_feature_database, find_similar_images_improved
from src.visualization import display_results, create_result_folder

def main():
    # Phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description='Hệ thống tìm kiếm ảnh lá cây')
    parser.add_argument('--mode', type=str, default='search', 
                       choices=['preprocess', 'train', 'search', 'all'],
                       help='Chế độ chạy: preprocess (tiền xử lý), train (xây dựng CSDL), search (tìm kiếm), all (tất cả các bước)')
    parser.add_argument('--raw_data', type=str, default='data/raw',
                       help='Thư mục chứa dữ liệu gốc')
    parser.add_argument('--processed_data', type=str, default='data/processed',
                       help='Thư mục lưu dữ liệu đã xử lý')
    parser.add_argument('--database', type=str, default='models/feature_database.pkl',
                       help='Đường dẫn tới file cơ sở dữ liệu')
    parser.add_argument('--query', type=str, default='test_images/test1.jpg',
                       help='Đường dẫn tới ảnh truy vấn')
    parser.add_argument('--top_k', type=int, default=3,
                       help='Số lượng ảnh tương tự muốn tìm')
    parser.add_argument('--use_deep', action='store_true',
                       help='Sử dụng đặc trưng học sâu (CNN)')
    parser.add_argument('--model_type', type=str, default='resnet', choices=['resnet', 'efficientnet'],
                       help='Loại mô hình CNN sử dụng cho trích xuất đặc trưng học sâu')
    parser.add_argument('--result_dir', type=str, default='results',
                       help='Thư mục lưu kết quả')
    
    args = parser.parse_args()
    
    # Tạo các thư mục cần thiết
    os.makedirs('models', exist_ok=True)
    os.makedirs('test_images', exist_ok=True)
    create_result_folder(args.result_dir)
    
    # Chạy theo chế độ được chọn
    if args.mode in ['preprocess', 'all']:
        print("=== Bắt đầu tiền xử lý dữ liệu ===")
        preprocess_all_images_improved(args.raw_data, args.processed_data)
    
    if args.mode in ['train', 'all']:
        print("=== Bắt đầu xây dựng cơ sở dữ liệu đặc trưng ===")
        build_feature_database_improved(args.processed_data, args.database, 
                                       use_deep_features=args.use_deep, 
                                       model_type=args.model_type)
    
    if args.mode in ['search', 'all']:
        print("=== Bắt đầu tìm kiếm ảnh tương tự ===")
        
        # Tải cơ sở dữ liệu
        feature_database = load_feature_database(args.database)
        if feature_database is None:
            print("Không tìm thấy cơ sở dữ liệu. Hãy chạy với --mode train để xây dựng CSDL trước.")
            return
        
        # Định nghĩa trọng số cho từng loại đặc trưng
        feature_weights = {
            'shape': 0.40,     # Hình dạng
            'texture': 0.10,   # Kết cấu
            'color': 0.15,     # Màu sắc
            'vein': 0.30,      # Gân lá
            'deep': 0.05       # Đặc trưng học sâu
        }
        
        # Tìm kiếm ảnh tương tự với phương pháp cải tiến
        similar_images = find_similar_images_improved(
            args.query, 
            feature_database, 
            top_k=args.top_k,
            use_deep_features=args.use_deep,
            model_type=args.model_type,
            feature_weights=feature_weights
        )
        
        # Hiển thị kết quả
        if similar_images:
            # Tạo tên file kết quả
            query_name = os.path.splitext(os.path.basename(args.query))[0]
            result_path = os.path.join(args.result_dir, f"result_{query_name}.png")
            
            # Hiển thị và lưu kết quả
            display_results(args.query, similar_images, save_path=result_path)
        else:
            print("Không tìm thấy ảnh tương tự.")

if __name__ == "__main__":
    main()