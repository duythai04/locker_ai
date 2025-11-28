# Dockerfile
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Cài lib hệ điều hành cho OpenCV, TensorFlow Lite...
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng cache
COPY requirements.txt .

# Cài các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code cần thiết vào container
COPY backend ./backend
COPY frontend ./frontend
COPY models ./models

# Nếu code load model từ root, copy thêm:
COPY best.pt ./

# Nếu cần SSL cert trong container (tùy):
# COPY ssl ./ssl

# Biến môi trường cơ bản
ENV PYTHONUNBUFFERED=1

# Expose port FastAPI
EXPOSE 8000

# Chạy thẳng uvicorn thay vì run_server.py (đỡ rắc rối SSL/host)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
