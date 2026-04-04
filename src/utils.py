import cv2


def draw_label(frame, label, confidence, x, y):
    text = f"{label} {confidence:.2f}"

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )


def draw_box(frame, box, track_id=None, label=None):
    x1, y1, x2, y2 = box

    # CHỌN MÀU THEO NHÃN HÀNH VI (OpenCV dùng chuẩn BGR: Blue - Green - Red)
    color = (0, 255, 0) # Màu Xanh Lá Cây (Mặc định cho Reading hoặc chữ Vui Lòng Đợi...)
    
    if label is not None:
        label_lower = label.lower()
        if "writing" in label_lower:
            color = (255, 0, 0) # Xanh Dương (Blue)
        elif "sleeping" in label_lower:
            color = (0, 0, 255) # Đỏ (Red)
        elif "raise" in label_lower or "raising" in label_lower:
            color = (0, 0, 0) # Đen (Black)
        elif "reading" in label_lower:
            color = (0, 255, 0) # Xanh Lá (Green)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = ""

    if track_id is not None:
        text += f"ID {track_id}"

    if label is not None:
        text += f" {label}"

    if text != "":
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )


def create_output_video(cap, output_path):
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter.fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    return writer


def resize_frame(frame, width=640):
    h, w = frame.shape[:2]

    scale = width / w

    new_height = int(h * scale)

    return cv2.resize(frame, (width, new_height))
