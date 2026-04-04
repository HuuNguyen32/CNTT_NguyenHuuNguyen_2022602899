import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- BẢN VÁ LỖI CẬP NHẬT CỦA PYTORCH 2.6+ ---
# PyTorch 2.6 chặn load các file model chứa definitions bên trong nếu không có weights_only=False
# Đây là cách chèn đè để ép thư viện Ultralytics (YOLO) vượt qua lỗi bảo mật UnpicklingError
safe_load = torch.load
torch.load = lambda *a, **k: safe_load(*a, **{**k, 'weights_only': False})
# ---------------------------------------------

from ultralytics import YOLO

from Config.config import YOLO_MODEL

yolo = YOLO(YOLO_MODEL)

tracker = DeepSort(max_age=30)


def track(frame):
    # Rút gọn độ phân giải YOLO xuống 320x320 để tăng Tốc độ tính toán (FPS), bỏ log Terminal rác
    results = yolo(frame, imgsz=320, verbose=False)[0]

    detections = []

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = box.conf[0].cpu().numpy()

        # Lọc nhiễu: Chỉ bắt Nhãn Người (0) và Độ Tự Tin phải lớn hơn 35%
        if cls != 0 or conf < 0.35:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    tracks = tracker.update_tracks(detections, frame=frame)

    return tracks
