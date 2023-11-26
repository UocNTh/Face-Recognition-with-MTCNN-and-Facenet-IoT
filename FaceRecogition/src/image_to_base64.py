import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # Đọc dữ liệu từ file ảnh
        image_data = image_file.read()
        # Mã hóa dữ liệu thành chuỗi Base64
        encoded_image = base64.b64encode(image_data)
        # Chuyển đổi từ bytes sang chuỗi
        encoded_image_str = encoded_image.decode("utf-8")
        return encoded_image_str
