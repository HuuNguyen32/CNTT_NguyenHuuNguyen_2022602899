import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

DATA_DIR = "data/processed_npy"
LSTM_MODEL_PATH = "models/lstm/student_behavior_lstm.h5"

LABELS = {
    "hand_raising": 0,
    "reading": 1,
    "writing": 2,
    "sleeping": 3
}

# Đảo ngược Dict để in tên nhãn cho đẹp
REVERSE_LABELS = {v: k for k, v in LABELS.items()}

def load_data():
    X = []
    y = []
    for label_name, label_id in LABELS.items():
        folder = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                path = os.path.join(folder, file)
                data = np.load(path)
                X.append(data)
                y.append(label_id)
    return np.array(X), np.array(y)

def evaluate_model():
    print("[*] Đang tải mô hình đã huấn luyện...")
    if not os.path.exists(LSTM_MODEL_PATH):
        print(f"[-] Lỗi: Không tìm thấy mô hình tại {LSTM_MODEL_PATH}")
        return

    model = tf.keras.models.load_model(LSTM_MODEL_PATH)

    print("[*] Đang tải dữ liệu kiểm thử (Evaluation Data)...")
    X, y_true = load_data()
    
    if len(X) == 0:
        print("[-] Không có dữ liệu để đánh giá.")
        return

    print(f"[+] Loaded {len(X)} mẫu dữ liệu. Bắt đầu Predict...")
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n================ TỔNG KẾT ĐÁNH GIÁ (CLASSIFICATION REPORT) ================\n")
    target_names = [REVERSE_LABELS[i] for i in range(len(LABELS))]
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    print("\n================ MA TRẬN NHẦM LẪN (CONFUSION MATRIX) ================\n")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Đoạn này sẽ tự động lưu ảnh Confusion Matrix ra giúp bạn chèn lên báo cáo
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Thực tế (True Label)')
    plt.xlabel('Dự đoán của AI (Predicted Label)')
    plt.title('Ma trận nhầm lẫn (Confusion Matrix) của mô hình LSTM')
    plt.savefig('confusion_matrix_results.png')
    print("\n[+] Đã lưu hình ảnh biểu đồ vào file 'confusion_matrix_results.png' (Bạn hãy chèn ảnh này vào Word).")

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    evaluate_model()
