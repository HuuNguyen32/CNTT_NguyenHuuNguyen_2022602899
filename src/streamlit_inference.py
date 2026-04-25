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
    import traceback


def process_video_for_streamlit(video_path, output_path, st_placeholder, progress_bar=None, status_text=None, frame_skip: int = 1):
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
    
    # NEW: Bảng đếm thống kê hành vi cho file Export
    behavior_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Xóa bỏ hoàn toàn việc ép cứng width = 800 và height = 600
    # Để thuật toán Tự động tính tỷ lệ 16:9 ở bên dưới làm việc.
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Máy cấu hình Gaming (RTX 3050) -> Bật chế độ Native Resolution (Tỷ lệ gốc)
    # Không cần Resize, giữ nguyên 100% độ phân giải gốc để AI đạt độ chính xác cao nhất
    
    # Khởi tạo bộ lưu Video chuẩn H264 (avc1) hỗ trợ HTML5 Web Player.
    # Vì vứt bỏ bớt Frame để giảm giật (frame_skip), nên tốc độ khung hình đầu ra phải chia đôi theo tỷ lệ tương ứng.
    # Nhờ vậy Video xuất ra sẽ giữ nguyên vẹn thời lượng gốc (6s -> 6s) thay vì bị nén lại như Tua Nhanh (Fast Forward).
    output_fps = fps / frame_skip 
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Cập nhật thanh Loading Mượt mà
        if progress_bar and status_text:
            percent = min(frame_count / total_frames, 1.0)
            progress_bar.progress(percent)
            status_text.write(f"🔄 **Đang phân tích:** {frame_count}/{total_frames} khung hình ({(percent*100):.1f}%)")

        # Nếu muốn Streamlit mượt hơn, có thể thu nhỏ frame chiếu lên web
        # frame_display = cv2.resize(frame, (800, 600))

        # AI Processing Khắc phục lỗi chớp nháy (Flickering Boxes): Phải dùng "continue" để bỏ qua hoàn toàn việc
        # Render lên màn hình theo tùy chọn Web
        if frame_count % frame_skip != 0:
            continue
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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
                st.session_state.stable_labels[track_id] = "Analyzing..."
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
            label = st.session_state.stable_labels.get(track_id, "Analyzing...")

            if len(buffers[track_id]) == SEQUENCE_LENGTH:
                input_data = np.expand_dims(buffers[track_id], axis=0)

                prediction = model.predict(input_data, verbose=0)[0]
                class_id = np.argmax(prediction)
                confidence = prediction[class_id]
                
                # --- BỘ LỌC NGƯỠNG TỰ TIN (CONFIDENCE THRESHOLD) ---
                if confidence < 0.70:
                    current_label = "UNKNOWN"
                else:
                    current_label = LABEL_MAP[class_id]

                # --- LƯỚI KHỬ NHIỄU GIƠ TAY (Heuristic Filter) ---
                if current_label == "HAND RAISING" or current_label == "HAND_RAISING":
                    current_features = buffers[track_id][-1]
                    left_wrist_y = current_features[31]
                    right_wrist_y = current_features[33]
                    
                    if left_wrist_y > -0.8 and right_wrist_y > -0.8:
                        prediction[class_id] = 0.0  
                        class_id = np.argmax(prediction)
                        confidence = prediction[class_id]
                        
                        if confidence < 0.70:
                            current_label = "UNKNOWN"
                        else:
                            current_label = LABEL_MAP[class_id]
                
                # --- CƠ CHẾ CẬP NHẬT REALTIME (STATE MACHINE) ---
                if current_label == st.session_state.streak_labels.get(track_id):
                    st.session_state.streak_counts[track_id] += 1
                else:
                    st.session_state.streak_labels[track_id] = current_label
                    st.session_state.streak_counts[track_id] = 1

                if st.session_state.streak_counts[track_id] >= 2:
                    old_label = st.session_state.stable_labels.get(track_id, "Analyzing...")
                    st.session_state.stable_labels[track_id] = current_label
                    
                    # LOGIC KIỂM ĐẾM (STATE TRANSITION TICKER)
                    if current_label != old_label and current_label != "Analyzing...":
                        behavior_counts[track_id][current_label] += 1

                label = st.session_state.stable_labels.get(track_id, "Analyzing...")
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
        
    return behavior_counts
