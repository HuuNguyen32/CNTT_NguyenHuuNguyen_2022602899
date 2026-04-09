import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import collections

from tensorflow.keras.models import load_model

from src.feature_extractor import PoseFeatureExtractor
from src.tracking import track
from src.utils import draw_box
from Config.config import *

# 1. TẢI MÔ HÌNH LSTM
model = load_model(LSTM_MODEL)

# 2. KHỞI TẠO BỘ TRÍCH XUẤT ĐẶC TRƯNG MEDIAPIPE (66 Features)
extractor = PoseFeatureExtractor()

buffers = {}  # Lưu trữ chuỗi frames cho từng ID sinh viên

# --- CƠ CHẾ CHUYỂN TRẠNG THÁI REAL-TIME (DEBOUNCER) ---
# Tự động thay đổi nhãn ngay lập tức nếu phát hiện hành động mới 2 lần liên tiếp (Không phân biệt nhãn nào)
stable_labels = {}
streak_labels = {}
streak_counts = {}

cap = cv2.VideoCapture(VIDEO_INPUT)

# --- CƠ CHẾ SKIP FRAME CHỐNG LAG CPU ---
frame_count = 0
FRAME_SKIP = 2 # Giống hệt thiết lập lúc lấy Data báo cáo

while True:

    # --- PIPELINE: 1. ĐỌC VIDEO ---
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue # Bỏ qua frame không xử lý AI để Laptop lẹ hơn

    # --- PIPELINE: 2. YOLOv11 DETECT PERSON + 3. DEEPSORT TRACKING ---
    # Đầu ra là danh sách các sinh viên đang được theo dõi
    tracks = track(frame)

    for track_obj in tracks:
        if not track_obj.is_confirmed():
            continue

        track_id = track_obj.track_id
        l, t, w, h = track_obj.to_ltrb()
        l, t, w, h = max(0, int(l)), max(0, int(t)), int(w), int(h)

        # --- PIPELINE: 4. CROP TỪNG SINH VIÊN ---
        roi = frame[t:h, l:w]
        if roi.size == 0:
            continue

        if track_id not in stable_labels:
            stable_labels[track_id] = "Vui long doi..."
            streak_labels[track_id] = ""
            streak_counts[track_id] = 0

        # --- PIPELINE: 5. MEDIAPIPE POSE -> 66 FEATURES ---
        features = extractor.extract_pose(roi)

        # Xử lý trường hợp MediaPipe không nhận diện được khung xương (bị che khuất)
        if features is None:
            # Nếu đã có frame trước đó, nhân bản frame cuối lên để đệm (Padding/Zero-order hold)
            if track_id in buffers and len(buffers[track_id]) > 0:
                features = buffers[track_id][-1]
            else:
                continue

        if track_id not in buffers:
            buffers[track_id] = []

        # Thêm 66 điểm đặc trưng vào dãy sequence
        buffers[track_id].append(features)

        label = stable_labels.get(track_id, "Vui long doi...")

        # --- PIPELINE: 6. SEQUENCE 30 FRAMES ---
        if len(buffers[track_id]) == SEQUENCE_LENGTH:
            # Shape chuẩn bị cho LSTM: (1, 30, 66)
            input_data = np.expand_dims(buffers[track_id], axis=0)

            # --- PIPELINE: 7. LSTM PREDICT BEHAVIOR ---
            prediction = model.predict(input_data, verbose=0)[0]
            class_id = np.argmax(prediction)
            current_label = LABEL_MAP[class_id]

            # --- LƯỚI KHỬ NHIỄU GIƠ TAY (Heuristic Filter) ---
            if current_label == "HAND RAISING" or current_label == "HAND_RAISING":
                current_features = buffers[track_id][-1] # Tọa độ chuẩn hóa khung hình hiện tại
                # Theo Code 46 điểm: 15 là cổ tay trái (Tọa độ y: index 31), 16: cổ tay phải (Tọa độ y: index 33)
                left_wrist_y = current_features[31]
                right_wrist_y = current_features[33]
                
                # Trục Y của MediaPipe: Số ÂM càng lớn là càng vươn lên cao.
                # Ngưỡng -0.4 là mức nới lỏng: Tay nhấc qua cằm (Không cần phải thẳng đuột qua đầu).
                if left_wrist_y > -0.4 and right_wrist_y > -0.4:
                    # Phủ quyết hành vi Giơ tay (Đưa xác suất Giơ tay về 0)
                    prediction[0] = 0.0  
                    # Bắt AI lấy hành vi đứng hạng 2 (Reading / Writing / Sleeping) để làm kết quả
                    class_id = np.argmax(prediction)
                    current_label = LABEL_MAP[class_id]

            # --- CƠ CHẾ CẬP NHẬT REALTIME (STATE MACHINE) ---
            # Nếu nhãn mới giống với streak hiện tại, tăng điểm xác nhận
            if current_label == streak_labels.get(track_id):
                streak_counts[track_id] += 1
            else:
                streak_labels[track_id] = current_label
                streak_counts[track_id] = 1

            # CHUYỂN TRẠNG THÁI NGAY TỨC KHẮC nếu hành động được xác nhận 2 lần liên tiếp
            # Vừa đáp ứng nhanh (Real-time), vừa chống chớp giật nhãn (Flicker)
            if streak_counts[track_id] >= 2:
                stable_labels[track_id] = current_label

            label = stable_labels[track_id]

            # Trượt (Sliding Window): Bỏ 5 frames cũ nhất để dự đoán mượt hơn
            buffers[track_id] = buffers[track_id][5:]

        # Vẽ Bounding Box và Nhãn bằng hàm trong utils.py
        draw_box(frame, (l, t, w, h), track_id=track_id, label=label)

    # Scale khung hình nhỏ lại cho vừa màn hình Laptop (Ví dụ: HD 1280x720)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Student Behavior Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
