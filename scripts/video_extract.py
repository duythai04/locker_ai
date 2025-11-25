from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0/")

import cv2
import os
from ultralytics import YOLO

model = YOLO("yolo8p-face/train4/weights/best.pt")
video_path = "test_video/Khai.MOV"

# Tạo đường dẫn output dựa trên tên video
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_dir = os.path.join("faces", video_name)
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
# Tính toán bước nhảy dựa trên FPS gốc
original_fps = cap.get(cv2.CAP_PROP_FPS)
desired_fps = 3
step = max(1, int(original_fps // desired_fps))  # Ví dụ: 30 FPS -> step = 3

# Xử lý xoay video
rotation_code = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))

def rotate_frame(frame, code):
    if code == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif code == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif code == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame

frame_count = 0
processed_count = 0  # Đếm số frame đã xử lý

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Chỉ xử lý frame khi thỏa điều kiện step
    if frame_count % step == 0:
        corrected_frame = rotate_frame(frame, rotation_code)
        results = model.predict(corrected_frame, conf=0.5)
        
        max_conf = 0
        best_box = None
        
        for box in results[0].boxes:
            current_conf = box.conf[0].item()
            if current_conf > max_conf:
                max_conf = current_conf
                best_box = box

        if best_box:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
            face_roi = corrected_frame[y1:y2, x1:x2]
            save_path = os.path.join(output_dir, f"frame_{processed_count}_conf_{max_conf:.2f}.jpg")
            cv2.imwrite(save_path, face_roi)
            processed_count += 1  # Tăng counter cho output

    frame_count += 1

cap.release()
print(f"Đã xử lý xong ở tốc độ ~{desired_fps}FPS!")