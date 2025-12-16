'''
training yolov8x model with 1:2 ratio of positive and negative samples, then all teh samples
'''
import os
import shutil
import torch
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    print("CUDA 是否可用 (torch.cuda.is_available):", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = YOLO('yolov8x.pt')
    model = YOLO(r'D:\Course\AI Friday\task2\runs\detect\train33\weights\last.pt')
    model.to(device)

    results = model.train(data="./aortic_valve_colab_neg_samples.yaml",
                epochs=100, #跑幾個epoch
                batch=8, #batch_size
                # (A) 關閉無用的色彩增強 (因為是灰階影像)
                hsv_h=0.0,
                hsv_s=0.0,
                hsv_v=0.4,
                nbs=64,

                lr0=0.002, 
                lrf=0.01,

                imgsz=640, #圖片大小640*640
                device=device, #使用GPU進行訓練
                workers=8,
                # --- 建議加入的資料增強參數 ---
                degrees=5.0,       # 隨機旋轉 +/- 15 度
                translate=0.2,    # 隨機平移 +/- 10%
                scale=0.6,        # 隨機縮放 +/- 10%   
                
                flipud=0.5,       # 50% 的機率進行垂直翻轉
                fliplr=0.5,       # 50% 的機率進行水平翻轉
                
                mosaic=0.5,       # (強烈推薦) 啟用馬賽克增強，將四張圖拼接成一張進行訓練
                mixup=0.1,        # (可選) 啟用 MixUp 增強，將兩張圖混合

                box=15.0, 
                cls=0.3,
                dfl=3.5,

                #patience=20 # 提前停止的耐心值
                )