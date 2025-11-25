🚀 Smart Locker System using Facial Recognition
Mở tủ thông minh sử dụng nhận diện khuôn mặt (FastAPI + YOLO + TFLite)

Hệ thống Smart Locker cho phép người dùng đăng ký khuôn mặt và mở tủ chỉ bằng việc đứng trước camera. Backend sử dụng FastAPI + mô hình AI để phát hiện khuôn mặt, tính toán embedding và so khớp với dữ liệu trong MongoDB. Frontend là web đơn giản hỗ trợ camera trực tiếp.

1. 🎯 Mục tiêu hệ thống

Đăng ký khuôn mặt của người dùng (Enroll Face)

Kiểm tra trùng khuôn mặt khi đăng ký (nếu similarity > 95% → báo trùng)

Nhận diện khuôn mặt để mở tủ (Unlock Locker)

Hiển thị trạng thái hệ thống theo thời gian thực

Xử lý AI trên backend (YOLO + Face Recognition)

Lưu embedding khuôn mặt vào MongoDB

2. 🧠 Công nghệ sử dụng
Backend

FastAPI

Uvicorn

OpenCV

TensorFlow Lite (Face Embedding Model)

YOLO (face detection)

MongoDB Atlas (lưu embedding)

python-dotenv

Frontend

HTML / CSS / JavaScript

WebRTC Camera API

Fetch API (gửi frame → backend)

3. 🏗 Kiến trúc hệ thống
┌────────────────┐      ┌──────────────────┐      ┌────────────────┐
│ Frontend (Web) │◄────►│   FastAPI API    │◄────►│  AI Models     │
│  Camera/WebRTC │      │  Face Processing │      │ YOLO + TFLite  │
└────────────────┘      └──────────────────┘      └────────────────┘
                             │
                             ▼
                      MongoDB Atlas
               (Lưu embedding khuôn mặt)

4. 🔄 Luồng hoạt động
4.1. Đăng ký khuôn mặt

Người dùng đứng trước camera → nhấn Đăng ký

Frontend gửi ảnh qua API /enroll_face

Backend:

Phát hiện khuôn mặt bằng YOLO

Tạo vector embedding

So sánh với database (nếu similarity > 95% → báo trùng)

Nếu không trùng → lưu embedding + user_id vào MongoDB

4.2. Mở tủ bằng khuôn mặt

Người dùng đứng trước camera → nhấn Mở tủ

Backend:

Phát hiện khuôn mặt

So khớp với embeddings trong DB

Nếu similarity >= 95% → mở tủ

Nếu không → báo lỗi

5. 📂 Cấu trúc thư mục
project/
│
├── backend/
│   ├── main.py
│   ├── db_utils.py
│
├── frontend
├   index.html         # Trang chính của ứng dụng
├   css/               # Các file CSS
│   ├── style.css      # CSS chính (nhập khẩu các file CSS khác)
│   ├── base.css       # Biến và kiểu cơ bản
│   ├── layout.css     # Layout chính và responsive
│   └── components/    # CSS cho từng thành phần
│       ├── header-footer.css
│       ├── video.css
│       ├── stats.css
│       ├── controls.css
│       ├── toggles.css
│       ├── buttons.css
│       └── loading.css
└── js/                # Các file JavaScript
    ├── main.js        # Điểm khởi đầu ứng dụng
    ├── camera.js      # Xử lý camera
    ├── detection.js   # Xử lý kết quả nhận diện
    ├── stats.js       # Cập nhật thống kê
    ├── ui.js          # Xử lý giao diện người dùng
    ├── state.js       # Quản lý trạng thái ứng dụng
    └── config.js      # Cấu hình ứng dụng
├── .env.example
└── README.md

6. ⚙️ Cài đặt & chạy hệ thống
6.1. Clone project
git clone <your-repo>
cd lock-detect-ai

6.2. Tạo môi trường
python -m venv venv311
source venv311/Scripts/activate

6.3. Cài dependency
pip install -r requirements.txt

6.4. Tạo file .env

Tạo file .env:

MONGODB_URI=your_mongodb_uri
MONGODB_DB_NAME=face_recognition_db
MONGODB_FACE_COLLECTION=faces

6.5. Chạy server
python run_server.py

7. 🧬 API Backend
7.1. Đăng ký khuôn mặt

POST /enroll_face
Gửi: image/jpeg hoặc image/png

Response:
{
  "success": true,
  "message": "Face enrolled successfully"
}


Hoặc nếu trùng:

{
  "success": false,
  "message": "Face already exists (similarity > 95%)"
}

7.2. Mở tủ bằng khuôn mặt

POST /unlock

Response:
{
  "success": true,
  "user_id": "user123",
  "message": "Locker unlocked"
}


Nếu không nhận diện được:

{
  "success": false,
  "message": "Face not recognized"
}

7.3. Kiểm tra sức khỏe

GET /health

{ "status": "ok" }

8. 🧠 Mô hình AI
Face Detection

YOLOv8n (rút gọn, chỉ lấy layer face)

Face Embedding

TensorFlow Lite 256-dim embedding vector
→ dùng Dot Product + Cosine Similarity so khớp

Ngưỡng nhận diện

Đăng ký trùng mặt: similarity ≥ 0.95

Mở tủ: similarity ≥ 0.95

9. 🛠 Giao diện Web

Có hỗ trợ camera trực tiếp

Nút Start Camera

Nút Enroll Face

Nút Unlock Locker

Khung hiển thị khuôn mặt đã detect

10. 🛡 Bảo mật

Backend chạy HTTPS

Không lưu ảnh (chỉ lưu embedding)

Lưu vector đã chuẩn hóa (không thể khôi phục ảnh gốc)

MongoDB Atlas + mật khẩu được ẩn qua .env

11. 🐞 Debug
Frontend

F12 → Console

Backend

Terminal chạy FastAPI