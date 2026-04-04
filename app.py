import streamlit as st
import os
import tempfile

# Cấu hình giao diện trang Web luôn full viền
st.set_page_config(page_title="AI Nhận Diện Hành Vi Sinh Viên", layout="wide")

# Súp port đường dẫn động
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from src.streamlit_inference import process_video_for_streamlit

# --- GIAO DIỆN WEB ---
st.title("🎓 Hệ Thống Trí Tuệ Nhân Tạo (Đồ Án Tốt Nghiệp)")
st.markdown("### Ứng dụng AI phân tích hành vi lớp học tự động")
st.markdown("**Công nghệ lõi:** `YOLOv8` (Nhận diện dáng người) $\\to$ `MediaPipe` (Bắt khung xương) $\\to$ `LSTM` (Phân định hành vi).")

# Khung Upload File
st.info("Hãy kéo thả Video lớp học vào ô bên dưới để bắt đầu:")
uploaded_file = st.file_uploader("📂 Tải lên Video Lớp Học (.mp4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 1. Lưu file Video người dùng vừa Upload xuống ổ cứng làm nháp
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, "input_video.mp4")
    output_path = os.path.join(temp_dir, "output_video.mp4")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.success(f"✅ Đã tải file `{uploaded_file.name}` lên máy chủ thành công. Sẵn sàng phân tích!")

    # 2. Nút Bấm Xử lý
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH HÀNH VI (AI INFERENCE)", use_container_width=True):
        st.warning("⏳ Hệ thống đang quét Xương khớp và Tính toán Mô Hình. Bạn có thể xem trực tiếp quá trình phân tích ngay bên dưới màng hình Tivi. Vui lòng không đóng trang web!")
        
        # Màn hình Tivi để làm hiệu ứng Live-Streaming
        st_placeholder = st.empty()
        
        # Chạy Trùm Cuối!
        try:
            process_video_for_streamlit(input_path, output_path, st_placeholder)
            st.success("🎉 Phân tích hoàn tất! Dưới đây là kết quả của bạn.")
            
            # Nút Download File Video mp4
            with open(output_path, "rb") as f:
                video_bytes = f.read()
                st.download_button(
                    label="⬇️ TẢI VIDEO PHÂN TÍCH VỀ MÁY",
                    data=video_bytes,
                    file_name="AI_Result_Video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"❌ Xảy ra lổi trong quá trình phân tích: {e}")
