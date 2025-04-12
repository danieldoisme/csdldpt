# Hướng Dẫn Thiết Lập Dự Án

## Yêu Cầu Hệ Thống

- Python 3.12 hoặc cao hơn
- Git

## Thiết Lập Môi Trường Phát Triển

### Cài Đặt uv

`uv` là công cụ cài đặt và quản lý gói Python nhanh mà chúng ta sử dụng để quản lý các phụ thuộc.

#### Trên macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Trên Windows:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Kiểm tra cài đặt:

```bash
uv --version
```

### Cài Đặt Các Gói Phụ Thuộc

1. Sao chép kho lưu trữ (nếu bạn chưa làm):

   ```bash
   git clone <repository-url>
   cd csdldpt
   ```

2. Tạo và kích hoạt môi trường ảo bằng uv:

   ```bash
   uv venv
   source .venv/bin/activate  # Trên Linux/macOS
   # Hoặc trên Windows:
   # .venv\Scripts\activate
   ```

3. Cài đặt các phụ thuộc từ requirements.txt:
   ```bash
   uv pip install -r requirements.txt
   ```

## Chạy Dự Án

Dự án là một hệ thống tìm kiếm ảnh lá cây tương tự với ba chế độ hoạt động chính:

### 1. Tiền Xử Lý Ảnh

Chạy lệnh sau để tiền xử lý tất cả ảnh:

```bash
uv run python main.py --mode preprocess
```

Lệnh này sẽ xử lý tất cả ảnh trong data/raw và lưu kết quả vào data/processed, giữ nguyên cấu trúc thư mục.

### 2. Xây Dựng Cơ Sở Dữ Liệu Đặc Trưng

Để xây dựng cơ sở dữ liệu mà không sử dụng đặc trưng học sâu (nhanh hơn):

```bash
uv run python main.py --mode train
```

Hoặc để xây dựng cơ sở dữ liệu có sử dụng đặc trưng học sâu (chính xác hơn):

```bash
uv run python main.py --mode train --use_deep
```

Quá trình này sẽ trích xuất đặc trưng từ tất cả ảnh đã xử lý và lưu vào models/feature_database.pkl.

### 3. Tìm Kiếm Ảnh Tương Tự

Để tìm kiếm ảnh tương tự với một ảnh truy vấn:

```bash
uv run python main.py --mode search --query test_images/test1.jpg
```

Để sử dụng đặc trưng học sâu khi tìm kiếm (đảm bảo CSDL cũng được xây dựng với đặc trưng học sâu):

```bash
uv run python main.py --mode search --query test_images/test1.jpg --use_deep
```

Để tìm nhiều hơn 3 ảnh tương tự:

```bash
uv run python main.py --mode search --query test_images/test1.jpg --top_k 5
```

Kết quả tìm kiếm sẽ được hiển thị trên màn hình và lưu vào thư mục results.

### 4. Chạy Tất Cả Các Bước Cùng Lúc

Để chạy toàn bộ quy trình từ tiền xử lý đến tìm kiếm:

```bash
uv run python main.py --mode all --query test_images/test1.jpg --use_deep
```

### Các Tham Số Bổ Sung

- `--top_k`: Số lượng ảnh tương tự muốn trả về (mặc định: 3)
- `--result_dir`: Thư mục lưu kết quả (mặc định: results)
