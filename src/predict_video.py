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

# --- CƠ CHẾ LÀM MƯỢT NHÃN (SMOOTHING / MAJORITY VOTE) ---
# Dùng bộ nhớ đệm 10 dự đoán gần nhất để chọn ra nhãn xuất hiện nhiều nhất
last_predictions = {}
prediction_history = {} 

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

        if track_id not in prediction_history:
            prediction_history[track_id] = collections.deque(maxlen=15) # Buffer 15 dự đoán cũ

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

        label = last_predictions.get(track_id, "Vui long doi...")

        # --- PIPELINE: 6. SEQUENCE 30 FRAMES ---
        if len(buffers[track_id]) == SEQUENCE_LENGTH:
            # Shape chuẩn bị cho LSTM: (1, 30, 66)
            input_data = np.expand_dims(buffers[track_id], axis=0)

            # --- PIPELINE: 7. LSTM PREDICT BEHAVIOR ---
            prediction = model.predict(input_data, verbose=0)[0]
            class_id = np.argmax(prediction)

            # --- CƠ CHẾ LÀM MƯỢT ---
            current_label = LABEL_MAP[class_id]
            prediction_history[track_id].append(current_label)
            
            # Lấy nhãn xuất hiện nhiều nhất trong 15 frames gần nhất (Majority Voting)
            most_common_label = collections.Counter(prediction_history[track_id]).most_common(1)[0][0]

            # --- PIPELINE: 8. DISPLAY CẬP NHẬT NHÃN ---
            label = most_common_label
            last_predictions[track_id] = label  # Lưu lại để hiển thị mượt

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
