import cv2
import numpy as np
import mediapipe as mp


class PoseFeatureExtractor:

    def __init__(self):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_pose(self, roi):

        if roi is None or roi.size == 0:
            return None

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        results = self.pose.process(roi_rgb)

        pose_landmarks = getattr(results, 'pose_landmarks', None)
        if not pose_landmarks:
            return None

        # CẮT ĐỨT NHIỄU RÁC NGAY DƯỚI RỐN:
        # Chỉ lấy 23 điểm xương Nửa Thân Trên (Từ đầu đến ngón tay). Bỏ toàn bộ đùi và gót chân bị lấp dưới bàn.
        landmarks = pose_landmarks.landmark[:23]

        features = []

        for lm in landmarks:
            # MediaPipe trả về tọa độ chuẩn hóa [0.0, 1.0] theo kích thước của roi_rgb
            features.append(lm.x)
            features.append(lm.y)

        features = np.array(features)

        if len(features) != 46:
            return None

        # --- CHUẨN HÓA (NORMALIZATION) ---
        # Tính toán Scale của người dựa trên thân mình (Khoảng cách hai vai)
        # Các điểm vai trong MediaPipe: Vai phải (12), Vai trái (11)
        # Vì Mảng features lưu dạng: [x0, y0, x1, y1, ...]
        # x_right_shoulder = features[11 * 2], y_right_shoulder = features[11 * 2 + 1]
        
        right_shoulder_x, right_shoulder_y = features[24], features[25]
        left_shoulder_x, left_shoulder_y = features[22], features[23]
        
        # Khoảng cách giữa 2 vai (Dùng làm thước đo)
        torso_width = np.sqrt((left_shoulder_x - right_shoulder_x)**2 + (left_shoulder_y - right_shoulder_y)**2)
        
        if torso_width < 0.01: # Tránh lỗi chia cho số 0 nếu do MediaPipe lỗi
            torso_width = 0.01
            
        # Chọn gốc tọa độ mới (Origin) nằm ở giữa 2 vai
        origin_x = (right_shoulder_x + left_shoulder_x) / 2
        origin_y = (right_shoulder_y + left_shoulder_y) / 2

        # Dời gốc và Chuẩn hóa (Scale)
        normalized_features = []
        for i in range(0, 46, 2):
            norm_x = (features[i] - origin_x) / torso_width
            norm_y = (features[i+1] - origin_y) / torso_width
            normalized_features.extend([norm_x, norm_y])

        return np.array(normalized_features)

    def close(self):
        self.pose.close()
