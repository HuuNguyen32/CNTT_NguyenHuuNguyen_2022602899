import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

# Giảm max_age từ 30 xuống 10 để giết chết các Khung hình bóng ma (Ghost Track) ngay lập tức nếu YOLO bị mất dấu
# Tăng max_iou_distance lên 0.8 để thuật toán bao dung hơn khi ghép đôi khung hình cũ và mới
tracker = DeepSort(max_age=10, max_iou_distance=0.8)

def reset_tracker():
    global tracker
    tracker = DeepSort(max_age=10, max_iou_distance=0.8)


def track(frame):
    # Trả lại IOU mặc định của YOLO để tránh nó chém nhầm Box chính xác
    results = yolo(frame, imgsz=640, verbose=False)[0]

    detections = []

    for box in results.boxes:

        cls = int(box.cls[0])
        conf = box.conf[0].cpu().numpy()

        # Bắt dính cả những vật chỉ có 25% giống người (Xóa nạn rớt Tracking khi gục đầu ngủ)
        if cls != 0 or conf < 0.25:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

    # LƯỚI LỌC BOX LỒNG NHAU THÔNG MINH (CHỐNG NHẦM BẠN ĐẰNG SAU)
    filtered_detections = []
    for i, det_A in enumerate(detections):
        box_A = det_A[0]
        x_A, y_A, w_A, h_A = box_A
        area_A = w_A * h_A
        center_x_A = x_A + w_A / 2
        is_ghost = False

        for j, det_B in enumerate(detections):
            if i == j: continue
            box_B = det_B[0]
            x_B, y_B, w_B, h_B = box_B
            area_B = w_B * h_B
            center_x_B = x_B + w_B / 2

            # Thuật toán bắt Ghost Box (Nửa người):
            # 1. Box A nhỏ hơn Box B
            # 2. Hai Box có cột sống thẳng hàng (Sai số tâm X < 30%)
            # 3. MẤU CHỐT: Hai Box phải TRÙNG KHÍT ĐỈNH ĐẦU NHAU (Thu hẹp sai số lại còn < 4%)
            # Sai số 15% h_B đôi khi bằng cả cái đầu người, phải siết xuống 4% để tránh giết đằng sau.
            if area_A < area_B:
                if abs(center_x_A - center_x_B) < (w_B * 0.3) and abs(y_A - y_B) < (h_B * 0.04):
                    is_ghost = True
                    break

        if not is_ghost:
            filtered_detections.append(det_A)

    tracks = tracker.update_tracks(filtered_detections, frame=frame)

    return tracks
