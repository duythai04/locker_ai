from pymongo import MongoClient
from datetime import datetime, timezone
from fastapi import HTTPException
import numpy as np

# Kết nốis MongoDB
client = MongoClient("mongodb://localhost:27017/")  # Thay đổi nếu cần
db = client["face_recognition_db"]
face_collection = db["faces"]

def store_face_data(user_id, name, face_embedding):
    """Lưu trữ dữ liệu khuôn mặt vào MongoDB."""
    try:
        if not isinstance(user_id, str):
            raise ValueError(f"user_id must be a string, got {type(user_id)}")
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")
        if not isinstance(face_embedding, list) or not all(isinstance(x, (int, float)) for x in face_embedding):
            raise ValueError(f"face_embedding must be a list of numbers, got {type(face_embedding)}")
        face_data = {
            "user_id": user_id,
            "name": name,
            "face_embedding": face_embedding,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        result = face_collection.insert_one(face_data)
        print(f"Stored face data for user_id: {user_id}, inserted_id: {result.inserted_id}")
        return True
    except Exception as e:
        print(f"Error storing face data: {e}")
        return False


def find_similar_faces(query_embedding, top_k=1):
    """Tìm kiếm các khuôn mặt tương đồng bằng cosine similarity trong MongoDB."""
    try:
        # Chuyển vector truy vấn về list thuần để đưa vào pipeline
        query_vec = np.array(query_embedding, dtype=np.float32).tolist()

        # Xây dựng aggregation pipeline để tính cosine similarity
        pipeline = [
        {
            "$addFields": {
                "cosineSim": {
                    "$reduce": {
                        "input": {
                            "$map": {
                                "input": {"$range": [0, 256]},
                                "as": "i",
                                "in": {
                                    "$multiply": [
                                        {"$arrayElemAt": ["$face_embedding", "$$i"]},
                                        {"$arrayElemAt": [query_vec, "$$i"]}
                                    ]
                                }
                            }
                        },
                        "initialValue": 0,
                        "in": {"$add": ["$$value", "$$this"]}
                    }
                }
            }
        },
        {"$match": {"cosineSim": {"$gt": 0.96}}},  # Điều chỉnh ngưỡng
        {"$sort": {"cosineSim": -1}},
        {"$limit": top_k},
        {"$project": {"name": 1, "cosineSim": 1, "_id": 1}}
    ]


        results = list(face_collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Error finding similar faces: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar faces")
