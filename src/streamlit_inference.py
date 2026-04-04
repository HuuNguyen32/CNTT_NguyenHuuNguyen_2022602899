import os
import cv2
import numpy as np
import collections
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
    if model is None:
        raise Exception(f"Không tìm thấy não bộ AI tại {LSTM_MODEL}. Xin vui lòng Train Model trước!")

    extractor = PoseFeatureExtractor()
    buffers = {}
    last_predictions = {}
    prediction_history = {}

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Khởi tạo bộ lưu Video chuẩn mp4v cho OpenCV
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
        if frame_count % FRAME_SKIP == 0:
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

                if track_id not in prediction_history:
                    prediction_history[track_id] = collections.deque(maxlen=15)

                features = extractor.extract_pose(roi)

                if features is None:
                    if track_id in buffers and len(buffers[track_id]) > 0:
                        features = buffers[track_id][-1]
                    else:
                        continue

                if track_id not in buffers:
                    buffers[track_id] = []

                buffers[track_id].append(features)
                label = last_predictions.get(track_id, "Vui long doi...")

                if len(buffers[track_id]) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(buffers[track_id], axis=0)

                    prediction = model.predict(input_data, verbose=0)[0]
                    class_id = np.argmax(prediction)

                    current_label = LABEL_MAP[class_id]
                    prediction_history[track_id].append(current_label)

                    most_common_label = collections.Counter(prediction_history[track_id]).most_common(1)[0][0]
                    label = most_common_label
                    last_predictions[track_id] = label
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
