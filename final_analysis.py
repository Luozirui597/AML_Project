import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, auc

# ================= 🔧 修改这里：加入了第三组 SuperGlue =================
EXPERIMENTS = [
    {
        "name": "CosPlace + SP-LightGlue",
        "log_dir": r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-18_12-51-25",
        "inliers_folder": "preds_superpoint-lg",
        "color": "blue"
    },
    {
        "name": "NetVLAD + SP-LightGlue",
        "log_dir": r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-19_12-27-49",
        "inliers_folder": "preds_superpoint-lg",
        "color": "orange"
    },
    {
        # 👇 新增的 SuperGlue 实验
        "name": "CosPlace + SuperGlue",
        "log_dir": r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-18_12-51-25", # 依然用 CosPlace 的日志目录
        "inliers_folder": "preds_superglue",  # 指向刚才新生成的文件夹
        "color": "green"
    }
]
# ======================================================================

def load_data(log_dir, inliers_folder_name):
    z_path = os.path.join(log_dir, "z_data.torch")
    inliers_dir = os.path.join(log_dir, inliers_folder_name)
    
    if not os.path.exists(inliers_dir):
        print(f"⚠️ Warning: Folder {inliers_dir} not found. Skipping.")
        return None, None

    z_data = torch.load(z_path, weights_only=False)
    predictions = z_data['predictions']
    positives = z_data['positives_per_query']
    
    files = sorted([f for f in os.listdir(inliers_dir) if f.endswith(".torch")], 
                   key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    X_inliers = []
    y_labels = [] # 1=Correct, 0=Wrong
    
    limit = min(len(predictions), len(files))
    
    for i in range(limit):
        # 1. Ground Truth
        top_pred = predictions[i][0]
        if isinstance(top_pred, torch.Tensor): top_pred = top_pred.item()
        true_matches = positives[i]
        if isinstance(true_matches, torch.Tensor): true_matches = true_matches.tolist()
        
        is_correct = 1 if top_pred in true_matches else 0
        y_labels.append(is_correct)
        
        # 2. Max Inliers
        data = torch.load(os.path.join(inliers_dir, files[i]), weights_only=False)
        max_inliers = 0
        if isinstance(data, list) and len(data) > 0:
            counts = [x['num_inliers'] for x in data if isinstance(x, dict)]
            if counts: max_inliers = max(counts)
        X_inliers.append(max_inliers)
        
    return np.array(X_inliers).reshape(-1, 1), np.array(y_labels)

def compute_ausc(confidence, errors):
    # Confidence should be higher for more certain predictions
    indices = np.argsort(-confidence) 
    sorted_errors = errors[indices]
    
    n = len(sorted_errors)
    error_curve = []
    retention_rates = np.arange(1, n + 1) / n
    
    current_sum = 0
    for i in range(n):
        current_sum += sorted_errors[i]
        error_rate = current_sum / (i + 1)
        error_curve.append(error_rate)
        
    ausc = auc(retention_rates, error_curve)
    return retention_rates, error_curve, ausc

def main():
    plt.figure(figsize=(10, 7))
    
    print(f"{'Method':<30} | {'AUPRC':<8} | {'Spearman':<8} | {'AUSC':<8}")
    print("-" * 65)

    for exp in EXPERIMENTS:
        X, y = load_data(exp['log_dir'], exp['inliers_folder'])
        if X is None: continue
        
        # 1. Logistic Regression
        clf = LogisticRegression()
        clf.fit(X, y)
        probs = clf.predict_proba(X)[:, 1] # P(Correct)
        
        # 2. Metrics
        auprc = average_precision_score(y, probs)
        sp_corr, _ = spearmanr(probs, y)
        
        # 3. AUSC
        errors = 1 - y
        retention, error_curve, ausc_val = compute_ausc(probs, errors)
        
        print(f"{exp['name']:<30} | {auprc:.4f}   | {sp_corr:.4f}   | {ausc_val:.4f}")
        
        # Plot
        plt.plot(retention, error_curve, label=f"{exp['name']} (AUSC={ausc_val:.3f})", 
                 color=exp['color'], linewidth=2)

    # Plot Baseline (Random)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Guessing')
    
    # Update Title for 3 lines
    plt.title("Uncertainty Estimation: Sparsification Curves\n(Compare Method & Matcher)")
    plt.xlabel("Retention Rate (Keep Top X% Confident Queries)")
    plt.ylabel("Error Rate in Retained Queries")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "final_uncertainty_comparison_3lines.png"
    plt.savefig(save_path)
    print(f"\n✅ Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()