import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# ================= 🔧 配置区域 (修改这里即可) =================

# 1. 训练集 (Teacher) -> SVOX (21:01:31)
TRAIN_LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\2025-12-23_21-01-31" 
TRAIN_FOLDER = "preds_superpoint-lg"

# 2. 测试集 (Student) -> SF-XS (21:04:37)
TEST_LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\2025-12-23_21-04-37" 
TEST_FOLDER = "preds_superpoint-lg"

# ==========================================================

def get_data(log_dir, folder_name):
    """
    通用数据提取函数：
    从 Log 文件夹中提取 (连线数量, 是否正确)
    """
    print(f"📂 正在读取: {log_dir} ...")
    
    # 1. 检查文件是否存在
    z_path = os.path.join(log_dir, "z_data.torch")
    preds_path = os.path.join(log_dir, folder_name)
    
    if not os.path.exists(z_path):
        print(f"❌ 错误: 找不到 z_data.torch！请检查路径。")
        return None, None
    if not os.path.exists(preds_path):
        print(f"❌ 错误: 找不到匹配文件夹 {folder_name}！请先运行 match_queries_preds.py。")
        return None, None

    # 2. 加载数据
    z_data = torch.load(z_path, weights_only=False)
    files = sorted([f for f in os.listdir(preds_path) if f.endswith(".torch")], 
                   key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    X_inliers = []
    y_labels = []
    
    limit = min(len(z_data['predictions']), len(files))
    
    for i in range(limit):
        # --- 获取 Label (0/1) ---
        top_pred = z_data['predictions'][i][0]
        if isinstance(top_pred, torch.Tensor): top_pred = top_pred.item()
        
        true_matches = z_data['positives_per_query'][i]
        if isinstance(true_matches, torch.Tensor): true_matches = true_matches.tolist()
        
        is_correct = 1 if top_pred in true_matches else 0
        y_labels.append(is_correct)
        
        # --- 获取 Feature (Inliers) ---
        data = torch.load(os.path.join(preds_path, files[i]), weights_only=False)
        max_inliers = 0
        # 兼容不同的存储格式
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                counts = [x['num_inliers'] for x in data]
            else:
                counts = [x.item() for x in data] # 如果直接是 tensor
            max_inliers = max(counts)
            
        X_inliers.append(max_inliers)
        
    return np.array(X_inliers).reshape(-1, 1), np.array(y_labels)

def main():
    # --- 1. 准备数据 ---
    print("--- 正在准备数据 ---")
    X_train, y_train = get_data(TRAIN_LOG_DIR, TRAIN_FOLDER)
    X_test, y_test = get_data(TEST_LOG_DIR, TEST_FOLDER)
    
    if X_train is None or X_test is None:
        print("程序终止：数据加载失败。")
        return

    print(f"✅ 训练集 (SVOX Sun-Night): {len(y_train)} 个样本 (正样本率: {y_train.mean():.1%})")
    print(f"✅ 测试集 (SF-XS): {len(y_test)} 个样本")

    # --- 2. 训练逻辑回归 ---
    print("\n🧠 正在训练逻辑回归 (Logistic Regression)...")
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # 获取学到的参数
    coef = clf.coef_[0][0]
    intercept = clf.intercept_[0]
    print(f"💡 模型学到的公式: Probability = Sigmoid({coef:.3f} * Inliers + {intercept:.3f})")

    # --- 3. 预测与评估 ---
    probs_test = clf.predict_proba(X_test)[:, 1]
    
    # 计算分数
    score_raw = average_precision_score(y_test, X_test)       # 原始连线数
    score_learned = average_precision_score(y_test, probs_test) # 预测概率
    
    print("\n" + "="*40)
    print(f"📊 最终结果 (AUPRC)")
    print(f"1. Baseline (仅数连线): {score_raw:.4f}")
    print(f"2. Proposed (逻辑回归): {score_learned:.4f}")
    print("="*40)

    # --- 4. 可视化 (PDF要求的曲线图) ---
    plt.figure(figsize=(10, 6))
    
    # 画出 SVOX 的 S 形曲线
    x_range = np.linspace(0, 150, 300).reshape(-1, 1)
    y_prob = clf.predict_proba(x_range)[:, 1]
    plt.plot(x_range, y_prob, color='red', linewidth=3, label='Learned Uncertainty Model (on SVOX)')
    
    # 画出 SF-XS 的数据分布
    plt.scatter(X_test, y_test, color='gray', alpha=0.1, label='SF-XS Test Data')
    
    plt.title("Uncertainty Estimation: Trained on SVOX(Sun/Night) -> Tested on SF-XS")
    plt.xlabel("Number of Inliers (LightGlue)")
    plt.ylabel("Probability of Correctness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 150)
    
    save_path = "final_uncertainty_plot.png"
    plt.savefig(save_path)
    print(f"\n🖼️ 图表已保存为: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()