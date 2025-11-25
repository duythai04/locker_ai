from os import putenv, listdir
import os
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0")

import cv2
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras import layers, Model

# def build_embedding_model(input_shape=(160,160,3), embedding_dim=128):
#     inputs = layers.Input(shape=input_shape)

#     x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D()(x)

#     x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D()(x)
    
#     x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPooling2D()(x)
    
#     x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
#     x = layers.GlobalAveragePooling2D()(x)
#     x = layers.Dropout(0.2)(x)
    
#     embeddings = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)    
#     embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embeddings)

#     return Model(inputs, embeddings, name='embedding_model')

def load_embedding_model(input_shape=(160, 160, 3), embedding_dim=256):
    """Load mô hình embedding khuôn mặt."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    embeddings = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)    
    embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embeddings)
    model = Model(inputs, embeddings, name='embedding_model')
    model.load_weights("models/face_embedding_model_256.h5")
    return model

def get_face_embedding(face_img, model):
    """Trích xuất embedding từ ảnh khuôn mặt."""
    try:
        resized_face = cv2.resize(face_img, (160, 160))
        normalized_face = resized_face.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(normalized_face, axis=0)
        embedding = model.predict(input_tensor, verbose=0)[0].tolist()
        return embedding
    except Exception as e:
        print(f"Error getting face embedding: {e}")
        return None

def store_face_data(user_id, name, face_embedding, collection):
    """Lưu trữ dữ liệu khuôn mặt vào MongoDB."""
    try:
        face_data = {
            "user_id": user_id,
            "name": name,
            "face_embedding": face_embedding,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        result = collection.insert_one(face_data)
        print(f"Stored face data for user_id: {user_id}, inserted_id: {result.inserted_id}")
        return True
    except Exception as e:
        print(f"Error storing face data: {e}")
        return False

def process_images_from_directory(image_dir, user_id, name, model, collection):
    """Xử lý tất cả ảnh trong thư mục"""
    processed = 0
    skipped = 0
    
    for filename in listdir(image_dir):
        # Kiểm tra định dạng ảnh
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(image_dir, filename)
        
        # Đọc ảnh
        face_img = cv2.imread(image_path)
        if face_img is None:
            print(f"⚠️ Could not read image: {filename}")
            skipped += 1
            continue
        
        # Trích xuất embedding
        embedding = get_face_embedding(face_img, model)
        if not embedding:
            print(f"⚠️ Failed to extract embedding: {filename}")
            skipped += 1
            continue
        
        # Lưu vào database
        if store_face_data(user_id, name, embedding, collection):
            processed += 1
        else:
            skipped += 1
            
    return processed, skipped

def main():
    # Kết nối MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["face_recognition_db"]
    face_collection = db["faces"]

    # Load mô hình
    embedding_model = load_embedding_model()

    # Thông tin cố định
    USER_ID = "2"          # ID người dùng cố định
    USER_NAME = "Nguyen Thanh Tung" # Tên người dùng cố định
    IMAGE_DIR = "test_imgs/Tung"    # Thư mục chứa ảnh

    # Xử lý hàng loạt
    print(f"🔄 Starting processing images from {IMAGE_DIR}")
    processed, skipped = process_images_from_directory(
        IMAGE_DIR, 
        USER_ID, 
        USER_NAME, 
        embedding_model, 
        face_collection
    )
    
    print(f"\n✅ Processing completed!")
    print(f"Total images processed: {processed}")
    print(f"Skipped images: {skipped}")

if __name__ == "__main__":
    main()