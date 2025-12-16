import os
import shutil
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    print('訓練集圖片數量 : ',len(os.listdir("./datasets/train/images")))
    print('訓練集標記數量 : ',len(os.listdir("./datasets/train/labels")))
    print('驗證集圖片數量 : ',len(os.listdir("./datasets/val/images")))
    print('驗證集標記數量 : ',len(os.listdir("./datasets/val/labels")))

    print("CUDA 是否可用 (torch.cuda.is_available):", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO('yolov8x.pt')
    model.to(device)

    results = model.train(data="./aortic_valve_colab.yaml",
                epochs=200, #跑幾個epoch
                batch=8, #batch_size
                box=10.0,
                # (A) 關閉無用的色彩增強 (因為是灰階影像)
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.0,
                nbs=64,

                imgsz=640, #圖片大小640*640
                device=device, #使用GPU進行訓練
                workers=8,
                # --- 建議加入的資料增強參數 ---
                degrees=15,       # 隨機旋轉 +/- 15 度
                translate=0.1,    # 隨機平移 +/- 10%
                scale=0.3,        # 隨機縮放 +/- 10%
                shear=5,          # 隨機錯切 +/- 5 度
                perspective=0.001,# 隨機透視變換
                
                flipud=0.5,       # 50% 的機率進行垂直翻轉
                fliplr=0.5,       # 50% 的機率進行水平翻轉
                
                mosaic=1.0,       # (強烈推薦) 啟用馬賽克增強，將四張圖拼接成一張進行訓練
                mixup=0.3,        # (可選) 啟用 MixUp 增強，將兩張圖混合
                copy_paste=0.3,    # (可選) 啟用 Copy-Paste 增強，將圖像中的物件複製貼上

                patience=50 # 提前停止的耐心值
                )