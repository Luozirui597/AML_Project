import os
import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score
import re

# === 配置路径 ===
LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-07_19-13-12"
Z_DATA_PATH = os.path.join(LOG_DIR, "z_data.torch")
INLIERS_DIR = os.path.join(LOG_DIR, "preds_superpoint-lg")

def numerical_sort_key(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[0]) if numbers else filename

def calculate_metrics():
    print("🚀 Loading Data...")
    
    # 1. 加载 Ground Truth 和 检索预测
    if not os.path.exists(Z_DATA_PATH):
        print("❌ Error: z_data.torch not found.")
        return

    z_data = torch.load(Z_DATA_PATH, weights_only=False)
    predictions = z_data['predictions']          # Shape: [Num_Queries, Top_K]
    positives = z_data['positives_per_query']    # Ground Truth indices
    
    # === PART 1: 计算标准 VPR Recall@N ===
    print("\n📊 Calculating Standard VPR Recall@N...")
    
    recalls = {1: [], 5: [], 10: [], 20: []}
    
    # 遍历每个查询
    for i in range(len(predictions)):
        # 获取该查询的 Top-K 预测 (通常 K=20)
        preds = predictions[i]
        if isinstance(preds, torch.Tensor):
            preds = preds.tolist()
            
        true_matches = positives[i] # 该查询对应的真实正确索引列表
        
        # 检查 Top-N 是否命中
        for n in recalls.keys():
            # 取前 N 个预测
            top_n = preds[:n]
            # 判断是否有任意一个在 true_matches 里
            hit = any(p in true_matches for p in top_n)
            recalls[n].append(1 if hit else 0)

    # 输出 Recall 结果
    print("-" * 30)
    for n in sorted(recalls.keys()):
        avg_recall = np.mean(recalls[n]) * 100
        print(f"Recall@{n}: {avg_recall:.2f}%")
    print("-" * 30)

    # === PART 2: 计算不确定性指标 (AUPRC/AUROC) ===
    # 这一步评估“内点数”是否是一个好的置信度指标
    print("\n📉 Calculating Uncertainty Metrics (based on Inliers)...")
    
    # 1. 读取内点数 (作为 Score)
    files = sorted([f for f in os.listdir(INLIERS_DIR) if f.endswith(".torch")], key=numerical_sort_key)
    
    # 确保数据对齐
    min_len = min(len(files), len(predictions))
    files = files[:min_len]
    binary_labels = recalls[1][:min_len] # 使用 R@1 的结果作为标签 (1=Correct, 0=Wrong)
    
    inlier_scores = []
    
    for filename in files:
        try:
            data = torch.load(os.path.join(INLIERS_DIR, filename), weights_only=False)
            # 获取最大内点数作为该查询的置信度
            if isinstance(data, list):
                counts = [x['num_inliers'] for x in data if isinstance(x, dict) and 'num_inliers' in x]
                score = max(counts) if counts else 0
            else:
                score = 0
            inlier_scores.append(score)
        except:
            inlier_scores.append(0)

    # 2. 计算指标
    # 注意：AUPRC 需要 inputs 是 numpy array
    y_true = np.array(binary_labels)
    y_scores = np.array(inlier_scores)
    
    if len(y_true) > 0:
        # AUPRC (Average Precision)
        auprc = average_precision_score(y_true, y_scores)
        
        # AUROC
        try:
            auroc = roc_auc_score(y_true, y_scores)
        except:
            auroc = 0.5 # 如果只有一个类别，AUROC 无法计算

        print(f"AUPRC (Average Precision): {auprc:.4f} (Higher is better)")
        print(f"AUROC: {auroc:.4f}")
        print("Interpretation: High AUPRC means Inliers are a good predictor of correctness.")
    else:
        print("⚠️ Not enough data to calculate AUPRC.")

if __name__ == "__main__":
    calculate_metrics()