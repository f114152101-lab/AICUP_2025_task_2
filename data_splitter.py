import os
import random
import shutil
from pathlib import Path
import math

# --- 1. 設定路徑 ---
base_dir = Path.cwd()
raw_dir = base_dir / 'raw'
image_source_dir = raw_dir / 'training_image'
label_source_dir = raw_dir / 'training_label'

# 檢查來源資料夾是否存在
if not raw_dir.exists() or not image_source_dir.exists() or not label_source_dir.exists():
    print(f"錯誤：找不到 'raw' 資料夾或其子資料夾 'training_image', 'training_label'")
    print(f"請確保您的資料夾結構如下：")
    print(f"your_project_folder/")
    print(f"├─ raw/")
    print(f"│  ├─ training_image/")
    print(f"│  └─ training_label/")
    print(f"└─ split_dataset_ratio.py (此腳本)")
    exit()

# 新的資料集輸出資料夾
output_dir = base_dir / 'datasets_1_2_ratio' # 換個新名稱以免覆蓋

# 定義訓練集和驗證集的目標資料夾
train_dir = output_dir / 'train'
val_dir = output_dir / 'val'

train_img_dest = train_dir / 'images'
train_lbl_dest = train_dir / 'labels'
val_img_dest = val_dir / 'images'
val_lbl_dest = val_dir / 'labels'

# --- 2. 設定分割與比例 ---
val_split_ratio = 0.3  # 30% 驗證集
target_neg_ratio = 2   # 負樣本是正樣本的 2 倍 (1:2)
random.seed(42)      # 為了可重複性，設定隨機種子

# --- 3. 搜尋、分類和過濾 ---
print("步驟 1: 正在搜尋並分類所有圖片...")

all_image_paths = list(image_source_dir.rglob('*.png'))

if not all_image_paths:
    print(f"錯誤：在 {image_source_dir} 中找不到任何 .png 圖片。")
    exit()

print(f"總共找到 {len(all_image_paths)} 張圖片。")

positive_samples = []  # 存圖片路徑 (Path)
negative_samples = []  # 存圖片路徑 (Path)

for img_path in all_image_paths:
    relative_path = img_path.relative_to(image_source_dir)
    relative_label_path = relative_path.with_suffix('.txt')
    label_path = label_source_dir / relative_label_path
    
    if label_path.exists() and label_path.stat().st_size > 0:
        positive_samples.append(img_path)
    else:
        negative_samples.append(img_path)

print("分類完成：")
print(f" - 原始正樣本: {len(positive_samples)} 張")
print(f" - 原始負樣本: {len(negative_samples)} 張")

# --- 4. 【新】根據 1:2 比例 (正:負) 調整樣本數 ---
print(f"\n步驟 2: 正在根據 1:{target_neg_ratio} (正:負) 比例調整樣本...")

num_positive = len(positive_samples)
num_negative_available = len(negative_samples)

num_negative_needed = num_positive * target_neg_ratio

if num_negative_available < num_negative_needed:
    # 負樣本不足，以負樣本為基準，反過來減少正樣本
    print(f"警告：負樣本不足 (僅 {num_negative_available} 張)，無法達成 1:{target_neg_ratio}。")
    print(f"將以所有 {num_negative_available} 張負樣本為基準。")
    num_negative_to_use = num_negative_available
    num_positive_to_use = int(num_negative_available / target_neg_ratio)
    
    random.shuffle(positive_samples) # 隨機抽取正樣本
    final_positive_list = positive_samples[:num_positive_to_use]
    final_negative_list = negative_samples # 使用所有可用負樣本
    print(f"將使用 {num_positive_to_use} 張正樣本和 {num_negative_to_use} 張負樣本。")

else:
    # 負樣本充足 (您的情況)，隨機抽取所需數量的負樣本
    print(f"負樣本充足 (有 {num_negative_available} 張)。")
    num_negative_to_use = num_negative_needed
    num_positive_to_use = num_positive
    
    random.shuffle(negative_samples) # 隨機抽取負樣本
    final_positive_list = positive_samples # 使用所有正樣本
    final_negative_list = negative_samples[:num_negative_to_use]
    print(f"將使用 {num_positive_to_use} 張正樣本和 {num_negative_to_use} 張負樣本 (從 {num_negative_available} 張中抽取)。")

print(f" -> 最終使用樣本池： {len(final_positive_list)} 正 / {len(final_negative_list)} 負 (共 {len(final_positive_list) + len(final_negative_list)} 張)")


# --- 5. 隨機打亂與分割 ---
print(f"\n步驟 3: 正在隨機打亂並按 {1-val_split_ratio:.0%}/{val_split_ratio:.0%} 比例分割...")

# 我們分別分割正負樣本，以確保兩個集合中的比例 *完全* 一致
random.shuffle(final_positive_list)
random.shuffle(final_negative_list)

# --- 分割正樣本 ---
pos_split_idx = int(len(final_positive_list) * (1 - val_split_ratio))
train_pos = final_positive_list[:pos_split_idx]
val_pos = final_positive_list[pos_split_idx:]

# --- 分割負樣本 ---
neg_split_idx = int(len(final_negative_list) * (1 - val_split_ratio))
train_neg = final_negative_list[:neg_split_idx]
val_neg = final_negative_list[neg_split_idx:]

# --- 組合 ---
train_image_list = train_pos + train_neg
val_image_list = val_pos + val_neg

# 再次打亂，確保訓練時正負樣本混合
random.shuffle(train_image_list)
random.shuffle(val_image_list)

# --- 6. 建立輸出資料夾 ---
print("\n步驟 4: 正在建立輸出資料夾...")
os.makedirs(train_img_dest, exist_ok=True)
os.makedirs(train_lbl_dest, exist_ok=True)
os.makedirs(val_img_dest, exist_ok=True)
os.makedirs(val_lbl_dest, exist_ok=True)
print(f" - 已建立/確認 {output_dir}")


# --- 7. 複製檔案的輔助函式 ---
# (此函式無需修改)
def process_and_copy_files(image_list, img_dest, lbl_dest, set_name):
    count_pos = 0
    count_neg = 0
    
    print(f"\n正在處理 {set_name} 集...")
    
    for img_src_path in image_list:
        img_filename = img_src_path.name
        lbl_filename = img_src_path.with_suffix('.txt').name
        
        img_dest_path = img_dest / img_filename
        lbl_dest_path = lbl_dest / lbl_filename
        
        # 1. 複製圖片
        shutil.copy(img_src_path, img_dest_path)
        
        # 2. 處理標籤
        relative_path = img_src_path.relative_to(image_source_dir)
        relative_label_path = relative_path.with_suffix('.txt')
        lbl_src_path = label_source_dir / relative_label_path
        
        if lbl_src_path.exists() and lbl_src_path.stat().st_size > 0:
            shutil.copy(lbl_src_path, lbl_dest_path)
            count_pos += 1
        else:
            with open(lbl_dest_path, 'w') as f:
                pass
            count_neg += 1
            
    print(f" - {set_name} 集：複製了 {len(image_list)} 張圖片")
    print(f"   - {count_pos} 個正樣本 (複製標註)")
    print(f"   - {count_neg} 個負樣本 (建立空標註)")
    return (count_pos, count_neg)

# --- 8. 執行複製 ---
print("\n步驟 5: 開始複製檔案...")

# --- 處理訓練集 ---
(train_pos_count, train_neg_count) = process_and_copy_files(train_image_list, train_img_dest, train_lbl_dest, 'train')

# --- 處理驗證集 ---
(val_pos_count, val_neg_count) = process_and_copy_files(val_image_list, val_img_dest, val_lbl_dest, 'val')

# --- 9. 最終總結 ---
print("\n--- 分割完成 ---")
print(f"原始樣本: {len(positive_samples)} 正 / {len(negative_samples)} 負")
print(f"使用樣本池: {num_positive_to_use} 正 / {num_negative_to_use} 負 (比例 1:{num_negative_to_use/num_positive_to_use:.1f})")
print(f"\n訓練集 (train): {len(train_image_list)} 張 ({train_pos_count} 正 / {train_neg_count} 負)")
print(f"驗證集 (val):   {len(val_image_list)} 張 ({val_pos_count} 正 / {val_neg_count} 負)")
print(f"\n新的 1:2 比例資料集已建立在 '{output_dir.absolute()}' 資料夾中。")