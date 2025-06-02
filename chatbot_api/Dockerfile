# Dockerfile

# Sử dụng một base image có sẵn Python và các công cụ build cần thiết
FROM python:3.10-slim

# Thiết lập môi trường
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Thiết lập thư mục làm việc bên trong container
WORKDIR /code

# Cài đặt các thư viện cần thiết cho PyTorch và Transformers
# Tách riêng việc cài đặt torch để tận dụng cache của Docker
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Sao chép tệp requirements và cài đặt các thư viện còn lại
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn của ứng dụng vào container
COPY ./app /code/app

# Mở cổng 8000 để API có thể được truy cập từ bên ngoài
EXPOSE 8000

# Lệnh để chạy ứng dụng khi container khởi động
# Chạy uvicorn, trỏ đến đối tượng 'app' trong file 'app.main'
# --host 0.0.0.0 để API có thể được truy cập từ bên ngoài container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]