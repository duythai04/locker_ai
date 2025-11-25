# backend/main.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import numpy as np
import cv2

from app.box_detector import Detector
from backend import db_utils  # Import các hàm từ db_utils.py

app = FastAPI(
    title="Smart Locker System using Facial Recognition",
    description="FastAPI backend for real-time face recognition and smart locker control.",
    version="1.0.0",
)

detector = Detector()

# ----------------- CORS -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể siết lại origin khi deploy
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Pydantic Models -----------------
class EnrollResponse(BaseModel):
    success: bool
    user_id: str
    name: str
    message: str


class UnlockResponse(BaseModel):
    status: str              # "granted" | "denied"
    user_id: str | None
    name: str | None
    locker_id: str | None
    confidence: float | None
    message: str


# Ngưỡng để quyết định cho mở tủ
UNLOCK_THRESHOLD = 0.95
EXISTING_FACE_THRESHOLD = 0.95

# ----------------- API CŨ: process_frame (giữ để debug/thống kê) -----------------
@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...)):
    """
    Nhận 1 frame từ camera, detect person & face,
    trả về bounding boxes + tên khuôn mặt (nếu match DB).
    """
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
            face_names = [face["name"] for face in similar_faces]
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


# ----------------- API MỚI: ĐĂNG KÝ KHUÔN MẶT -----------------
@app.post("/enroll_face", response_model=EnrollResponse)
async def enroll_face(
    file: UploadFile = File(...),
    user_id: str = Query(..., description="Mã người dùng / mã tủ (vd: locker_01)"),
    name: str = Query(..., description="Tên người dùng"),
):
    """
    Đăng ký khuôn mặt cho 1 user (tương ứng 1 locker).

    Bổ sung:
    - Check xem khuôn mặt này đã tồn tại trong DB chưa (theo embedding).
    - Nếu có mặt nào trong DB có cosineSim >= EXISTING_FACE_THRESHOLD
      thì báo lỗi "Khuôn mặt đã tồn tại" và không lưu thêm.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh từ file upload")

    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

    if face_count == 0:
        raise HTTPException(status_code=400, detail="Không phát hiện khuôn mặt nào trong ảnh")

    # Lấy khuôn mặt đầu tiên
    coords, conf, emotion, embedding = face_boxes[0]

    if embedding is None:
        raise HTTPException(
            status_code=400, detail="Không trích xuất được embedding khuôn mặt"
        )

    # ✅ BƯỚC MỚI: kiểm tra xem khuôn mặt này đã tồn tại trong DB chưa
    similar_faces = db_utils.find_similar_faces(embedding, top_k=1)

    if similar_faces:
        best = similar_faces[0]
        existing_sim = float(best["cosineSim"])

        if existing_sim >= EXISTING_FACE_THRESHOLD:
            # Đã có người khác có khuôn mặt rất giống (trùng)
            existing_user_id = best.get("user_id")
            existing_name = best.get("name")

            msg = (
                f"Khuôn mặt này đã được đăng ký trong hệ thống "
                f"cho người dùng '{existing_name}' (mã: {existing_user_id}), "
                # f"độ tương đồng {existing_sim:.3f} ≥ {EXISTING_FACE_THRESHOLD}."
            )
            raise HTTPException(status_code=400, detail=msg)

    # ✅ Nếu không trùng mặt, tiến hành lưu như cũ
    success = db_utils.store_face_data(
        user_id=user_id, name=name, face_embedding=embedding
    )

    if not success:
        raise HTTPException(
            status_code=500, detail="Lưu khuôn mặt vào cơ sở dữ liệu thất bại"
        )

    return EnrollResponse(
        success=True,
        user_id=user_id,
        name=name,
        message="Đăng ký khuôn mặt thành công",
    )


# ----------------- API MỚI: MỞ TỦ BẰNG KHUÔN MẶT -----------------
@app.post("/unlock", response_model=UnlockResponse)
async def unlock(file: UploadFile = File(...)):
    """
    Nhận 1 frame từ camera, nhận diện xem là user nào.
    Nếu cosineSim >= UNLOCK_THRESHOLD thì cho 'mở tủ'.

    Tạm thời:
    - locker_id = user_id (1 user gắn với 1 tủ)
    - Chưa điều khiển phần cứng, chỉ trả JSON.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Không đọc được ảnh từ file upload")

    person_count, face_count, person_boxes, face_boxes = detector.process_frame(frame)

    if face_count == 0:
        raise HTTPException(status_code=400, detail="Không phát hiện khuôn mặt nào trong ảnh")

    # Chọn khuôn mặt đầu tiên
    coords, conf, emotion, embedding = face_boxes[0]

    if embedding is None:
        raise HTTPException(
            status_code=400, detail="Không trích xuất được embedding khuôn mặt"
        )

    similar_faces = db_utils.find_similar_faces(embedding, top_k=1)

    if not similar_faces:
        return UnlockResponse(
            status="denied",
            user_id=None,
            name=None,
            locker_id=None,
            confidence=None,
            message="Không tìm thấy khuôn mặt tương ứng trong hệ thống",
        )

    best = similar_faces[0]
    user_id = best["user_id"]
    name = best["name"]
    cosineSim = float(best["cosineSim"])

    if cosineSim < UNLOCK_THRESHOLD:
        return UnlockResponse(
            status="denied",
            user_id=user_id,
            name=name,
            locker_id=None,
            confidence=cosineSim,
            message="Độ tương đồng khuôn mặt chưa đủ để mở tủ",
        )

    # Ở bản đơn giản này: locker_id = user_id
    locker_id = user_id

    # TODO: Sau này có thể gửi lệnh ra phần cứng ở đây
    # ví dụ: gpio_unlock(locker_id)

    return UnlockResponse(
        status="granted",
        user_id=user_id,
        name=name,
        locker_id=locker_id,
        confidence=cosineSim,
        message=f"Đã mở tủ cho {name}",
    )


# ----------------- HEALTH CHECK -----------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}


# ----------------- STATIC FRONTEND -----------------
# Giả định cấu trúc: project_root/frontend/...
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
