import cv2
import os 

def extract_frames(video_path, output_path, frame_interval=30):
    """
    Lấy hình ảnh từ video và lưu thành ảnh tại các frame cụ thể.

    Parameters:
        - video_path: Đường dẫn đến file video.
        - output_path: Thư mục đầu ra để lưu các hình ảnh.
        - frame_interval: Khoảng cách giữa các frame mà bạn muốn lấy (mặc định là mỗi 30 frame).

    Returns:
        None
    """
    # Mở video
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra xem video có mở thành công không
    if not cap.isOpened():
        print("Không thể mở video.")
        return

    # Đảm bảo thư mục đầu ra tồn tại
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Đọc và lấy frame từ video
    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Lấy mỗi frame_interval frame
        if frame_count % frame_interval == 0:
            # Lưu frame thành file ảnh
            img_name = f"{output_path}/frame_{frame_count}.png"
            cv2.imwrite(img_name, frame)

        frame_count += 1

    # Đóng video và giải phóng tài nguyên
    cap.release()

if __name__ == "__main__":
    video_path = "home/toe/Documents/video.mp4"
    output_path = "hoe/toe/Documents/folder"
    extract_frames(video_path, output_path)
