import os
import torch
from PIL import Image
from torchvision.transforms import functional as F
import cv2

# Bước 1: Detect và cắt phần vùng chứa dãy số
def detect_water_meter(image_path):
    model1 = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/20222/project_AI2022/yolov5/runs/train/exp2/weights/best.torchscript' , force_reload=True)
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image)
    image_tensor = F.pad(image_tensor, (0, 0, 0, 0), fill=0)
    results = model1(image_tensor)
    output = results.pandas().xyxy[0]
    bbox = output[['xmin', 'ymin', 'xmax', 'ymax']].values[0]
    cropped_image = image.crop(bbox)
    cropped_image.save("cropped_image.jpg")

# Bước 2: Resize và detect chữ số trong ảnh đã cắt
def detect_digits(cropped_image_path):
    model2 = torch.hub.load('ultralytics/yolov5', 'custom', path='E:/20222/project_AI2022/yolov5/runs/train/exp/weights/best.torchscript', force_reload=True)
    cropped_image = Image.open(cropped_image_path)
    resized_image = cropped_image.resize((640, 640))
    image_tensor = F.to_tensor(resized_image)
    image_tensor = F.pad(image_tensor, (0, 0, 0, 0), fill=0)
    results = model2(image_tensor)
    output = results.pandas().xyxy[0]
    with open("label.txt", "w") as label_file:
        for _, row in output.iterrows():
            label = row['class']
            label_file.write(f"{label}\n")

# Bước 3: Sắp xếp và in kết quả lên bounding box
def display_result(image_path, label_path):
    image = cv2.imread(image_path)
    with open(label_path, "r") as label_file:
        labels = label_file.read().splitlines()
    sorted_labels = sorted(labels)
    x_min, y_min, x_max, y_max = 10, 10, 100, 100  # Tùy chỉnh tọa độ và kích thước bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    text_offset_x = 10
    text_offset_y = 40
    for i, label in enumerate(sorted_labels):
        text = label
        text_position = (x_min + text_offset_x, y_min + text_offset_y + i * 30)
        cv2.putText(image, text, text_position, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Xác định đường dẫn tới thư mục chứa ảnh hoặc đường dẫn tuyệt đối của ảnh
def process_water_meter(input_path):
    if os.path.isdir(input_path):
        image_files = [os.path.join(input_path, file) for file in os.listdir(input_path)]
    elif os.path.isfile(input_path):
        image_files = [input_path]
    else:
        print("Đường dẫn không hợp lệ!")
        return
    
    for image_file in image_files:
        detect_water_meter(image_file)
        detect_digits("cropped_image.jpg")
        display_result("cropped_image.jpg", "label.txt")

# Nhập đường dẫn tuyệt đối hoặc tương đối đến ảnh
image_path = input("Nhập đường dẫn đến ảnh: ")
process_water_meter(image_path)
