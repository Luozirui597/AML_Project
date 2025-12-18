import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import re

LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-07_19-13-12"
Z_DATA_PATH = os.path.join(LOG_DIR, "z_data.torch")
INLIERS_DIR = os.path.join(LOG_DIR, "preds_superpoint-lg")

def numerical_sort_key(filename):
    """
    从文件名中提取数字用于排序，避免 '10' 排在 '2' 前面的问题。
    假设文件名类似 'query_123.torch' 或只是数字。
    """
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else filename

def plot_final_histogram():
    print("🚀 Parsing Ground Truth Data...")
    
    if not os.path.exists(Z_DATA_PATH):
        print("❌ Error: z_data.torch not found.")
        return
        
    try:
        z_data = torch.load(Z_DATA_PATH, weights_only=False)
        predictions = z_data['predictions']          
        positives = z_data['positives_per_query']    
        
        is_correct_mask = []
        for i in range(len(predictions)):
            top_pred = predictions[i][0]
            true_matches = positives[i]
            
            if isinstance(top_pred, torch.Tensor):
                top_pred = top_pred.item()
            
            # Check if the Top-1 prediction is correct
            if len(true_matches) > 0:
                if isinstance(true_matches, torch.Tensor):
                    hit = (top_pred == true_matches).any().item()
                else:
                    hit = top_pred in true_matches
            else:
                hit = False 
            
            is_correct_mask.append(hit)
            
        print(f"📊 Parsing Complete: Top-1 Correct Queries {sum(is_correct_mask)}, Wrong Queries {len(is_correct_mask) - sum(is_correct_mask)}.")
        
    except Exception as e:
        print(f"❌ Failed to parse z_data: {e}")
        return

    # 2. Read inliers data
    print("Reading Inliers Data...")
    
    # === 🔥 FIX: 使用数字排序 ===
    files = sorted([f for f in os.listdir(INLIERS_DIR) if f.endswith(".torch")], key=numerical_sort_key)
    
    # 检查长度是否一致，如果不一致打印警告
    if len(files) != len(is_correct_mask):
        print(f"⚠️ Warning: File count ({len(files)}) does not match query count ({len(is_correct_mask)}). Truncating to minimum.")

    min_len = min(len(files), len(is_correct_mask))
    files = files[:min_len]
    is_correct_mask = is_correct_mask[:min_len]

    correct_inliers = []
    wrong_inliers = []

    print(f"Processing {min_len} files...")

    for idx, filename in enumerate(files):
        try:
            filepath = os.path.join(INLIERS_DIR, filename)
            data = torch.load(filepath, weights_only=False)
            
            max_val = 0
            if isinstance(data, list):
                # 过滤掉 None 或异常值
                counts = [x['num_inliers'] for x in data if isinstance(x, dict) and 'num_inliers' in x]
                max_val = max(counts) if counts else 0
            
            if is_correct_mask[idx]:
                correct_inliers.append(max_val)
            else:
                wrong_inliers.append(max_val)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 3. Plotting
    print("Generating Plot...")
    plt.figure(figsize=(10, 6))
    
    # 设置 bins
    max_inlier_found = max(max(correct_inliers, default=0), max(wrong_inliers, default=0))
    bins = np.linspace(0, min(200, max_inlier_found + 10), 50) # 动态设置范围，上限 200 或最大值

    # 绘制直方图
    # alpha 设置透明度，方便看到重叠部分
    plt.hist(correct_inliers, bins=bins, color='#4CAF50', alpha=0.6, label='Correct Queries', density=False)
    plt.hist(wrong_inliers, bins=bins, color='#F44336', alpha=0.6, label='Wrong Queries', density=False)
    
    # 如果想看“归一化”后的概率密度（因为 Wrong 往往很多，会把 Correct 压得看不见），可以将上面的 density 改为 True
    
    plt.title('Inliers Distribution: Correct vs Wrong (SuperPoint+LightGlue)')
    plt.xlabel('Max Number of Inliers')
    plt.ylabel('Frequency (Count)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(LOG_DIR, "inliers_histogram_fixed.png")
    plt.savefig(save_path, dpi=300) # 提高清晰度
    print(f"✅ Success! Image saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_final_histogram()