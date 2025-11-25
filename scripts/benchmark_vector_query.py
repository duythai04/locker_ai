from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("ROCM_PATH", "/opt/rocm-6.3.0")

import time
import numpy as np
import cv2
from pymongo import MongoClient
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras import layers, Model
from fastapi import HTTPException

# Kết nối MongoDB
global_client = None
def get_db_client(uri="mongodb://localhost:27017/"):
    global global_client
    if global_client is None:
        global_client = MongoClient(uri)
    return global_client

def load_embedding_model(input_shape=(160, 160, 3), embedding_dim=64):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    embeddings = layers.Dense(embedding_dim, activation=None, name='embeddings')(x)
    embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(embeddings)
    model = Model(inputs, embeddings, name='embedding_model')
    model.load_weights("models/face_embedding_model_64.h5")
    return model

def get_face_embedding(face_img, model):
    try:
        resized = cv2.resize(face_img, (160, 160))
        arr = resized.astype(np.float32) / 255.0
        inp = np.expand_dims(arr, 0)
        return model.predict(inp)[0].tolist()
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def measure_latency_and_query(image_path, top_k=5, num_runs=100, mongo_uri="mongodb://localhost:27017/"):
    try:
        client = get_db_client(mongo_uri)
        db = client["face_recognition_db"]
        face_collection = db["faces"]

        model = load_embedding_model()
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        q_emb = get_face_embedding(img, model)
        if q_emb is None:
            raise ValueError("Failed to extract embedding.")
        query_vec = np.array(q_emb, dtype=np.float32).tolist()

        total = 0.0
        for _ in range(num_runs):
            start = time.time()
            pipeline = [
                # dot product
                {"$addFields": {
                    "dot": {"$sum": {"$map": {
                        "input": {"$range": [0, {"$size": "$face_embedding"}]},
                        "as": "idx",
                        "in": {"$multiply": [
                            {"$arrayElemAt": ["$face_embedding", "$$idx"]},
                            {"$arrayElemAt": [query_vec, "$$idx"]}
                        ]}
                    }}}
                }},
                # magnitudes
                {"$addFields": {
                    "magDoc": {"$sqrt": {"$sum": {"$map": {
                        "input": "$face_embedding",
                        "as": "val",
                        "in": {"$multiply": ["$$val", "$$val"]}
                    }}}},
                    "magQuery": {"$sqrt": {"$sum": {"$map": {
                        "input": query_vec,
                        "as": "qv",
                        "in": {"$multiply": ["$$qv", "$$qv"]}
                    }}}}
                }},
                # cosineSim
                {"$addFields": {
                    "cosineSim": {
                        "$cond": [
                            {"$eq": ["$magDoc", 0]}, 0,
                            {"$divide": ["$dot", {"$multiply": ["$magDoc", "$magQuery"]}]}
                        ]
                    }
                }},
                {"$sort": {"cosineSim": -1}},
                {"$limit": top_k},
                {"$project": {"_id": 0, "name": 1, "cosineSim": 1}}
            ]
            list(face_collection.aggregate(pipeline))
            total += (time.time() - start) * 1000
        avg = total / num_runs
        print(f"Avg latency: {avg:.2f} ms over {num_runs} runs")
        return avg
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    measure_latency_and_query("test_imgs/HoangDinhHaiAnh_2.jpg")
