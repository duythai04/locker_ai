# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import cv2
from typing import List

from backend.db_utils import lockers_collection
from app.box_detector import Detector
from backend import db_utils

app = FastAPI(
    title="Smart Locker System using Facial Recognition",
    description="FastAPI backend for real-time face recognition and smart locker control.",
    version="1.0.0",
)

detector = Detector()

# Khởi tạo danh sách tủ nếu cần
db_utils.init_lockers_if_empty(num_lockers=12)

# ----------------- CORS -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- Pydantic Models -----------------
class StoreResponse(BaseModel):
    status: str              # "granted" | "denied"
    locker_id: str | None
    confidence: float | None
    message: str


class RetrieveResponse(BaseModel):
    status: str              # "granted" | "denied"
    locker_id: str | None
    confidence: float | None
    message: str


# Ngưỡng để quyết định cho mở tủ khi LẤY ĐỒ
UNLOCK_THRESHOLD = 0.93
# Ngưỡng để coi là "mặt đã có tủ đang gửi đồ"
EXISTING_FACE_THRESHOLD = 0.95


# ----------------- API: process_frame (debug) -----------------
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh từ file upload")

    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

    face_boxes_for_response = []
    for (coords, conf, emotion, embedding) in face_boxes:
        if embedding is not None:
            similar_faces = db_utils.find_similar_faces(embedding, top_k=3)
            face_names = [face.get("name") for face in similar_faces]
            face_boxes_for_response.append(
                {
                    "coords": coords,
                    "confidence": conf,
                    "emotion": emotion,
                    "similar_faces": face_names,
                }
            )
        else:
            face_boxes_for_response.append(
                {
                    "coords": coords,
                    "confidence": conf,
                    "emotion": emotion,
                    "similar_faces": [],
                }
            )

    return {
        "persons": person_count,
        "faces": face_count,
        "person_boxes": [
            {"coords": coords, "confidence": conf}
            for (coords, conf, action) in person_boxes
        ],
        "face_boxes": face_boxes_for_response,
    }


# ----------------- API: LƯU ĐỒ (STORE) -----------------
@app.post("/store", response_model=StoreResponse)
async def store_item(
    files: List[UploadFile] = File(
        None, description="Danh sách frame chụp khuôn mặt (tùy chọn nhiều frame)"
    ),
    file: UploadFile = File(
        None, description="1 frame chụp khuôn mặt (fallback, tương thích đơn giản)"
    ),
):
    """
    Flow LƯU ĐỒ:
    - FE bấm 'Lưu đồ' -> bật camera -> gửi 1 hoặc nhiều frame lên endpoint này.
    - BE:
      + Trích embedding khuôn mặt (lấy trung bình nhiều frame).
      + CHECK: nếu mặt này đã có session active (đang gửi đồ) -> từ chối.
      + Nếu OK -> tìm tủ free, tạo session, đánh dấu occupied.
    """

    uploads: List[UploadFile] = []
    if files:
        uploads.extend(files)
    if file is not None:
        uploads.append(file)

    if not uploads:
        raise HTTPException(status_code=400, detail="Không nhận được file ảnh nào để lưu đồ")

    embeddings: list[np.ndarray] = []

    for upload in uploads:
        contents = await upload.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("[STORE] Bỏ qua frame: không đọc được ảnh")
            continue

        person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

        if face_count == 0:
            print("[STORE] Frame không có khuôn mặt, bỏ qua")
            continue

        coords, conf, emotion, embedding = face_boxes[0]

        if embedding is None:
            print("[STORE] Không trích xuất được embedding, bỏ qua frame này")
            continue

        embeddings.append(np.array(embedding, dtype=np.float32))

    if len(embeddings) == 0:
        raise HTTPException(
            status_code=400,
            detail="Không thu được khuôn mặt hợp lệ nào. Hãy thử lại và đảm bảo mặt rõ, đủ sáng.",
        )

    embed_stack = np.stack(embeddings, axis=0)
    avg_embedding = np.mean(embed_stack, axis=0)
    print(f"[STORE] Collected {len(embeddings)} embeddings, using averaged template.")

    # 🔴 CHECK: mặt này đã có session active chưa?
    existing_session = db_utils.find_active_session_by_face(avg_embedding)
    if existing_session and float(existing_session["cosineSim"]) >= EXISTING_FACE_THRESHOLD:
        locker_id = existing_session["locker_id"]
        # Trả về 400 để FE show lỗi
        raise HTTPException(
            status_code=400,
            detail=f"Khuôn mặt này đang có đồ tại tủ {locker_id}. "
                   f"Vui lòng lấy đồ hoặc đóng phiên hiện tại trước khi gửi thêm.",
        )

    # Tìm 1 tủ đang free
    locker = db_utils.find_free_locker()
    if not locker:
        return StoreResponse(
            status="denied",
            locker_id=None,
            confidence=None,
            message="Hiện không còn tủ trống, vui lòng thử lại sau.",
        )

    locker_id = locker["locker_id"]

    # Tạo session mới
    session_id = db_utils.create_locker_session(locker_id=locker_id, face_embedding=avg_embedding)

    # Đánh dấu tủ đang bị chiếm
    db_utils.mark_locker_occupied(locker_id=locker_id, session_id=session_id)

    return StoreResponse(
        status="granted",
        locker_id=locker_id,
        confidence=None,
        message=f"Tủ {locker_id} đã được cấp. Vui lòng gửi đồ vào tủ.",
    )


# ----------------- API: LẤY ĐỒ (RETRIEVE) -----------------
@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_item(file: UploadFile = File(...)):
    """
    Flow LẤY ĐỒ:
    - FE bấm 'Lấy đồ' -> bật camera -> gửi 1 frame khuôn mặt hiện tại.
    - BE:
      + Trích embedding khuôn mặt.
      + Tìm session active có cosineSim cao nhất.
      + Nếu cosineSim >= UNLOCK_THRESHOLD -> mở tủ, đóng session, free locker.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh từ file upload")

    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

    if face_count == 0:
        raise HTTPException(status_code=400, detail="Không phát hiện khuôn mặt nào trong ảnh")

    coords, conf, emotion, embedding = face_boxes[0]

    if embedding is None:
        raise HTTPException(
            status_code=400, detail="Không trích xuất được embedding khuôn mặt"
        )

    best_session = db_utils.find_active_session_by_face(embedding)

    if not best_session:
        return RetrieveResponse(
            status="denied",
            locker_id=None,
            confidence=None,
            message="Không tìm thấy tủ tương ứng với khuôn mặt này.",
        )

    locker_id = best_session["locker_id"]
    cosineSim = float(best_session["cosineSim"])

    if cosineSim < UNLOCK_THRESHOLD:
        return RetrieveResponse(
            status="denied",
            locker_id=locker_id,
            confidence=cosineSim,
            message="Độ tương đồng khuôn mặt chưa đủ để mở tủ.",
        )

    # Đủ ngưỡng -> cho mở tủ, đóng session & free locker
    session_id = best_session["session_id"]
    db_utils.close_locker_session(session_id=session_id)
    db_utils.mark_locker_free(locker_id=locker_id)

    return RetrieveResponse(
        status="granted",
        locker_id=locker_id,
        confidence=cosineSim,
        message=f"Đã mở tủ {locker_id}. Vui lòng lấy đồ.",
    )


@app.get("/lockers/summary")
async def lockers_summary():
    total = lockers_collection.count_documents({})
    free = lockers_collection.count_documents({"status": "free"})
    occupied = lockers_collection.count_documents({"status": "occupied"})
    return {
        "total_lockers": total,
        "free_lockers": free,
        "occupied_lockers": occupied,
    }


@app.post("/init_lockers")
async def init_lockers(count: int = 12):
    created = db_utils.create_lockers(count)
    return {
        "requested": count,
        "created": created,
        "message": f"Đã tạo {created} tủ mới",
    }


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ----------------- STATIC FRONTEND -----------------
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
