import os
import cv2
import numpy as np
import collections
import streamlit as st
from tensorflow.keras.models import load_model

from src.feature_extractor import PoseFeatureExtractor
from src.tracking import track
from src.utils import draw_box
from Config.config import LSTM_MODEL, SEQUENCE_LENGTH, LABEL_MAP

# Load model tĩnh 1 lần khi import để tránh load lại nhồi CPU
try:
    model = load_model(LSTM_MODEL)
except:
    model = None


def process_video_for_streamlit(video_path, output_path, st_placeholder):
    from src.tracking import reset_tracker
    
    if model is None:
        raise Exception(f"Không tìm thấy não bộ AI tại {LSTM_MODEL}. Xin vui lòng Train Model trước!")

    # Dọn dẹp hoàn toàn rác Bộ Nhớ AI (DeepSORT Tracker) từ Video trước
    reset_tracker()

    # Xóa sạch các Biến Trạng Thái Màn Hình (Session State) cặn bã từ Video Cũ
    st.session_state.stable_labels = {}
    st.session_state.streak_labels = {}
    st.session_state.streak_counts = {}

    extractor = PoseFeatureExtractor()
    buffers = {}
    
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Ghi đè kích thước để khớp với kích thước Frame thu nhỏ (800x600) hỗ trợ chống giật
    width = 800
    height = 600
    
    # Khởi tạo bộ lưu Video chuẩn mp4v cho OpenCV
    # Codec 'avc1' (h264) tốt hơn cho web browser, hoặc giữ nguyên 'mp4v'. Có thể dùng mp4v cho Windows Media Player
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    FRAME_SKIP = 2
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800, 600))
        frame_count += 1

        # Nếu muốn Streamlit mượt hơn, có thể thu nhỏ frame chiếu lên web
        # frame_display = cv2.resize(frame, (800, 600))

        # AI Processing
        # Khắc phục lỗi chớp nháy (Flickering Boxes): Phải dùng "continue" để bỏ qua hoàn toàn việc Render lên màn hình
        if frame_count % FRAME_SKIP != 0:
            continue

        tracks = track(frame)

        for track_obj in tracks:
            if not track_obj.is_confirmed():
                continue

            track_id = track_obj.track_id
            l, t, w, h = track_obj.to_ltrb()
            l, t, w, h = max(0, int(l)), max(0, int(t)), int(w), int(h)

            roi = frame[t:h, l:w]
            if roi.size == 0:
                continue

            if track_id not in st.session_state.stable_labels:
                st.session_state.stable_labels[track_id] = "Vui long doi..."
                st.session_state.streak_labels[track_id] = ""
                st.session_state.streak_counts[track_id] = 0

            features = extractor.extract_pose(roi)

            if features is None:
                if track_id in buffers and len(buffers[track_id]) > 0:
                    features = buffers[track_id][-1]
                else:
                    continue

            if track_id not in buffers:
                buffers[track_id] = []

            buffers[track_id].append(features)
            label = st.session_state.stable_labels.get(track_id, "Vui long doi...")

            if len(buffers[track_id]) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(buffers[track_id], axis=0)

                prediction = model.predict(input_data, verbose=0)[0]
                class_id = np.argmax(prediction)
                current_label = LABEL_MAP[class_id]

                # --- LƯỚI KHỬ NHIỄU GIƠ TAY (Heuristic Filter) ---
                if current_label == "HAND RAISING" or current_label == "HAND_RAISING":
                    current_features = buffers[track_id][-1]
                    left_wrist_y = current_features[31]
                    right_wrist_y = current_features[33]
                    
                    if left_wrist_y > -0.8 and right_wrist_y > -0.8:
                        prediction[0] = 0.0  
                        class_id = np.argmax(prediction)
                        current_label = LABEL_MAP[class_id]
                
                # --- CƠ CHẾ CẬP NHẬT REALTIME (STATE MACHINE) ---
                if current_label == st.session_state.streak_labels.get(track_id):
                    st.session_state.streak_counts[track_id] += 1
                else:
                    st.session_state.streak_labels[track_id] = current_label
                    st.session_state.streak_counts[track_id] = 1

                if st.session_state.streak_counts[track_id] >= 2:
                    st.session_state.stable_labels[track_id] = current_label

                label = st.session_state.stable_labels[track_id]
                buffers[track_id] = buffers[track_id][5:]

            draw_box(frame, (l, t, w, h), track_id=track_id, label=label)

        # Trực tiếp render Frame lên Web (Chuyển BGR -> RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Ghi frame vào video tĩnh (.mp4)
        writer.write(frame)

    cap.release()
    writer.release()
    try:
        extractor.close()
    except:
        pass
