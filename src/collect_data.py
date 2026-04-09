import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import cv2
import numpy as np
import time

from src.feature_extractor import PoseFeatureExtractor
from Config.config import SEQUENCE_LENGTH


def collect(video_path, label, output_dir):
    extractor = PoseFeatureExtractor()

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    buffer = []
    count = 0
    frame_count = 0
    FRAME_SKIP = 2
    
    # KÍCH THƯỚC BƯỚC TRƯỢT (STEP_SIZE): Số Frame xử lý sẽ bị trừ đi sau mỗi lần cắt Data.
    # 8 Frame xử lý x 2 (Skip) = 16 Frame Gốc -> Tương đương Máy sẽ cắt Data định kỳ mỗi 0.54 giây! (Chuẩn tỷ lệ rọc thịt)
    STEP_SIZE = 8

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        features = extractor.extract_pose(frame)

        if features is None:
            buffer = []
            continue

        buffer.append(features)

        if len(buffer) == SEQUENCE_LENGTH:
            data = np.array(buffer)

            filename = f"{label}_{int(time.time())}_{count}.npy"

            np.save(os.path.join(output_dir, filename), data)

            # Trượt cửa sổ: Xóa bỏ STEP_SIZE khung hình cũ nhất để giãn cách Thời gian (Tránh sinh rác Data bị đè lên nhau quá nhiều)
            buffer = buffer[STEP_SIZE:]

            count += 1

    cap.release()

    extractor.close()

    print("Saved sequences:", count)


if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # SỬ DỤNG AUTO-GOM DATA HÀNG LOẠT CHO TẤT CẢ NHÃN:

    labels_to_extract = ["hand_raising", "reading", "writing", "sleeping"]

    for label in labels_to_extract:
        input_folder = f"data/videos/{label}"
        output_dir = f"data/processed_npy/{label}"

        print(f"\n=======================================================")
        print(f"BẮT ĐẦU QUÉT THƯ MỤC: {input_folder}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Tự động dọn sạch rác Data cũ trước khi tạo Data mới để chống lệch pha điểm xương
            import shutil

            for f in os.listdir(output_dir):
                file_path = os.path.join(output_dir, f)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    pass

        if not os.path.exists(input_folder):
            print(f"[-] BỎ QUA: Không tìm thấy thư mục {input_folder}!")
            continue

        video_files = [f for f in os.listdir(input_folder) if
                       f.endswith('.mp4') or f.endswith('.avi') or f.endswith('.mov')]
        if len(video_files) == 0:
            print(f"[-] BỎ QUA: Thư mục {input_folder} đang trống (Không có video).")
            continue

        print(f"[+] TÌM THẤY {len(video_files)} video cho hành vi '{label}'...")
        for video_file in video_files:
            video_path = os.path.join(input_folder, video_file)
            print(f" -> Đang tiến hành trích xuất Khung Xương: {video_file}...")
            collect(video_path, label, output_dir)

        print(f"*** HOÀN TẤT THU THẬP DỮ LIỆU CHO NHÃN: {label.upper()} ***")

    print("\n================ TỐNG KẾT ================")
    print("ĐÃ HOÀN THÀNH NHIỆM VỤ THU THẬP DATA CHO TẤT CẢ CÁC NHÃN!")
