import streamlit as st
import torch
from PIL import Image
import numpy as np

# Thiết lập đường dẫn tới model đã train
model_path = 'runs/train/exp6/weights/best.pt'  # Thay đổi theo đường dẫn của bạn
# Tải model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Hàm để đếm số heo trong hình ảnh
def count_pigs(image):
    results = model(image)
    detections = results.xyxy[0]  # Kết quả dự đoán cho hình ảnh đầu tiên
    pig_count = 0
    
    for *box, conf, cls in detections:
        if int(cls) == 0:  # Giả sử 'heo' có class ID là 0
            pig_count += 1

    return pig_count, results.render()[0]  # Trả về số lượng heo và hình ảnh đã đánh dấu

# Tiêu đề ứng dụng
st.title("ĐẾM HEO")

# Upload ảnh
uploaded_file = st.file_uploader("Select Picture", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Chuyển đổi file thành ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Đếm số heo
    pig_count, result_image = count_pigs(np.array(image))

    # Hiển thị kết quả
    st.image(result_image, caption=f"Result Detect", use_column_width=True)
    st.write(f"Số heo nhận diện được: {pig_count}")