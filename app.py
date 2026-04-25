import streamlit as st
import os
import collections

st.set_page_config(page_title="AI Nhận Diện Hành Vi Sinh Viên", layout="wide")

# Hỗ trợ đường dẫn động
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Sửa lỗi DLL do đụng độ Matplotlib/PyTorch/Pandas
from src.streamlit_inference import process_video_for_streamlit

# --- GIAO DIỆN WEB ---
st.title("🎓 EduVision AI - Hệ Thống Phân Tích Hành Vi Sinh Viên")
st.markdown("### Ứng dụng AI nhận diện và phân tích hành vi sinh viên trong lớp học")
st.markdown("**Công nghệ lõi:** `YOLOv8` (Nhận diện dáng người) $\\to$ `DeepSort` (Theo dõi đối tượng) $\\to$ `MediaPipe` (Bắt khung xương) $\\to$ `LSTM` (Phân định hành vi).")

# Tránh tạo Temp liên tục gây tràn RAM
os.makedirs("temp", exist_ok=True)
input_path = os.path.join("temp", "input_video.mp4")
output_path = os.path.join("temp", "output_video.mp4")

# Khung Upload File
st.info("Hãy kéo thả Video lớp học vào ô bên dưới để bắt đầu:")
uploaded_file = st.file_uploader("📂 Tải lên Video Lớp Học (.mp4)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # --- RESET TRẠNG THÁI KHI CÓ VIDEO MỚI ---
    if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.analysis_complete = False
        st.session_state.behavior_data = None
        
    # Lưu file Video đè lên thư mục Temp tĩnh
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
        
    st.success(f"✅ Đã tải file `{uploaded_file.name}` lên máy chủ thành công!")

    # --- KHUNG HIỂN THỊ CHUỖI DỌC TRÊN DƯỚI ---
    st.markdown("---")
    st.markdown("#### 📥 Video Gốc Đầu Vào")
    st.video(input_path) 
    
    st.markdown("---")
    
    # Gộp Tiêu đề và Nút Cài Đặt (Popover) lên cùng 1 Dòng
    head_col1, head_col2 = st.columns([0.85, 0.15])
    with head_col1:
        st.markdown("#### 🧠 Trình xử lý AI (AI Processing Engine)")
        
    with head_col2:
        # Hộp thoại Cài Đặt (Popover) sẽ nổi lên khi bấm vào thay vì tràn Lan ra màn hình
        with st.popover("⚙️ Cài đặt"):
            frame_skip_val = st.slider(
                label="Điều chỉnh (Frame Skip)", 
                min_value=1, max_value=5, value=1, 
                help=(
                    "Điều chỉnh số frame bị bỏ qua khi xử lý video.\n\n"
                    "• 1: Phân tích toàn bộ frame → độ chính xác cao nhất (chậm hơn)\n\n"
                    "• 2-3: Cân bằng giữa tốc độ và độ chính xác\n\n"
                    "• 4-5: Xử lý nhanh hơn → có thể bỏ sót hành vi ngắn"
                )
            )
            
    # CHIA CỘT DASHBOARD: VIDEO (TRÁI) - LEGEND (PHẢI)
    left_col, right_col = st.columns([0.85, 0.15])
    
    with left_col:
        st_placeholder = st.empty()
        
    with right_col:
        legend_placeholder = st.empty()
    progress_bar = st.empty()
    status_text = st.empty()

    # Nút Bấm Xử lý
    if st.button("🚀 BẮT ĐẦU PHÂN TÍCH (AI INFERENCE)", use_container_width=True):
        status_text.warning("⏳ Đang phân tích video...")
        
        # Chỉ hiện Legend khi Video bắt đầu chạy
        legend_placeholder.markdown(
            "### 🎨 Legend\n\n"
            "🟢 Writing\n\n"
            "🟦 Reading\n\n"
            "🔴 Sleeping\n\n"
            "🟣 Raise Hand\n\n"
            "🟨 Unknown\n\n"
            "⬜ Analyzing"
        )
        
        try:
            # Trỏ về Cấu trúc thanh loading
            pg_bar = progress_bar.progress(0.0)
            
            # Biến behavior_counts chứa dữ liệu từ File Inference Test
            # Truyền tham số frame_skip_val vào tầng thuật toán Lõi
            behavior_counts = process_video_for_streamlit(input_path, output_path, st_placeholder, pg_bar, status_text, frame_skip_val)
            
            # Lưu Dữ liệu vào Mạng Lưới Nhớ để chịu tải Rerun
            st.session_state.behavior_data = behavior_counts
            st.session_state.analysis_complete = True
            
            # Dọn dẹp thanh UI Live process
            st_placeholder.empty()
            progress_bar.empty()
            status_text.empty()
            legend_placeholder.empty()
                
        except Exception as e:
            import traceback
            st.error(f"❌ Xảy ra lỗi trong quá trình phân tích: {e}")
            st.error(traceback.format_exc())

    # --- KHU VỰC HIỂN THỊ KẾT QUẢ ĐỘC LẬP ---
    if st.session_state.get('analysis_complete', False):

        st.success("🎉 Phân tích hoàn tất! Dưới đây là kết quả của bạn.")
        
        st.markdown("---")
        # MÀN HÌNH 1: Phát Video Output
        st.markdown("#### 🎬 Kết Quả Phân Tích (H264 Playback)")
        st.video(output_path)
        
        st.markdown("---")
        # MÀN HÌNH 2: Bảng Thống Kê Hành Vi
        st.markdown("#### 📊 Bảng thống kê hành vi (State Transition Table)")
        
        behavior_counts = st.session_state.behavior_data
        if behavior_counts:
            # Rút Data Dict sang Pandas DataFrame
            import pandas as pd
            df_data = []
            for track_id, counts in behavior_counts.items():
                row = {"ID Máy Thu": f"Student_{track_id}"}
                row.update(counts)
                df_data.append(row)
                
            # fillna(0) để lấp đầy bảng excel chuẩn mẫu
            df = pd.DataFrame(df_data).fillna(0).astype(int, errors='ignore') 
            st.dataframe(df, use_container_width=True)
            
            # --- START: RENDER BẢNG PANDAS THÀNH ẢNH PNG ---
            import matplotlib.pyplot as plt
            import io
            
            # Tạo Figure vừa khít với kích thước dòng của Bảng
            fig, ax = plt.subplots(figsize=(8, 1 + len(df) * 0.4))
            ax.axis('off')
            ax.axis('tight')
            
            # Vẽ bảng
            table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.8)
            
            # Sơn màu cho Tiêu đề (Header)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#2980b9')
                else:
                    cell.set_facecolor('#ecf0f1' if row % 2 == 0 else 'white')
                    
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format='png', bbox_inches='tight', dpi=300)
            plt.close(fig)
            img_buf.seek(0)
            # --- END: RENDER BẢNG THÀNH PNG ---
            
            # --- TẠO THANH CÔNG CỤ (BAR) ĐỂ ÉP 2 NÚT NẰM CÙNG 1 ROW (TRIỆT KHE HỞ) ---
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                st.download_button(
                    label="🖼️ LƯU ẢNH BẢNG (.PNG)",
                    data=img_buf.read(),
                    file_name="Behavior_Table.png",
                    mime="image/png",
                    use_container_width=True
                )
                
            with btn_col2:
                try:
                    with open(output_path, "rb") as f:
                        video_bytes = f.read()
                        st.download_button(
                            label="⬇️ TẢI XUỐNG VIDEO FULL",
                            data=video_bytes,
                            file_name="AI_Result_Video.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                except FileNotFoundError:
                    st.error("Không tìm thấy Video đầu ra.")
        else:
            st.warning("⚠️ Không ghi nhận được bất kỳ thao tác nào đủ 2 giây liên tiếp.")
