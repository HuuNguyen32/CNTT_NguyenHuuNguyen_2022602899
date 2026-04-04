import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.feature_extractor import PoseFeatureExtractor
from Config.config import LSTM_MODEL

model = load_model(LSTM_MODEL)

extractor = PoseFeatureExtractor()

LABEL_MAP = {
    0: "HAND RAISING",
    1: "READING",
    2: "WRITING",
    3: "SLEEPING"
}

buffer = []

cap = cv2.VideoCapture("data/videos/test.mp4")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    features = extractor.extract_pose(frame)

    if features is not None:

        buffer.append(features)

        if len(buffer) == 30:
            input_data = np.expand_dims(buffer, axis=0)

            prediction = model.predict(input_data)[0]

            class_id = np.argmax(prediction)

            label = LABEL_MAP[class_id]

            confidence = prediction[class_id]

            cv2.putText(frame,
                        f"{label} {confidence:.2f}",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            buffer = buffer[5:]

    cv2.imshow("Student Behavior Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
