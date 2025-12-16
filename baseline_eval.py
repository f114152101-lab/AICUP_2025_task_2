import os
import shutil
import torch
from ultralytics import YOLO

print("CUDA 是否可用 (torch.cuda.is_available):", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = YOLO('./runs/detect/train15/weights/best.pt')
model = YOLO(r'D:\Course\AI Friday\task2\runs\detect\train55\weights\last.pt')
model.to(device)
results = model.predict(source=r"D:\Course\AI Friday\task2\datasets\test\images_2.5d_stacked",
              save=True,
              imgsz=640,
              device=device,
              max_det=1
              )

#預測數量
print(len(results))

#將偵測框數值寫進.txt檔
output_file = open('./predict_txt/merged.txt', 'w')
for i in range(len(results)):
    # 取得圖片檔名（不含副檔名）
    filename = results[i].path.split('/')[-1].split('.png')[0]

    # 取得預測框數量
    boxes = results[i].boxes
    box_num = len(boxes.cls.tolist())

    # 如果有預測框
    if box_num > 0:
        for j in range(box_num):
            # 提取資訊
            label = int(boxes.cls[j].item())  # 類別
            conf = boxes.conf[j].item()       # 信心度
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()  # 邊界框座標

            # 建立一行資料
            line = f"{filename} {label} {conf:.4f} {int(x1)} {int(y1)} {int(x2)} {int(y2)}\n"
            output_file.write(line)

# 關閉輸出檔案
output_file.close()