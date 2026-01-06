import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, r2_score
from scipy.stats import spearmanr

# Ignore warnings
warnings.filterwarnings('ignore')

# ================= 🔧 CONFIGURATION AREA =================

# 1. Training Set (Teacher) -> SVOX (Sun vs Night)
TRAIN_LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\2025-12-23_21-01-31"

# 2. Test Sets (Student) -> SF-XS
# The keys will be used as labels in the plots
TEST_LOG_DIRS = {
    "CosPlace": r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_18-45-46",
    "NetVLAD":  r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_23-01-01",
    "MixVPR":   r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_08-24-08",
    "MegaLoc":  r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_10-47-59"
}

# 3. Matcher Folder Name
MATCHER_FOLDER = "preds_superpoint-lg"

# =======================================================

def get_data(log_dir, folder_name):
    """
    Reads log files to extract inlier counts (X) and correctness labels (y).
    """
    z_path = os.path.join(log_dir, "z_data.torch")
    preds_path = os.path.join(log_dir, folder_name)
    
    if not os.path.exists(z_path) or not os.path.exists(preds_path):
        print(f"   ❌ Skipping: Incomplete files -> {log_dir}")
        return None, None

    z_data = torch.load(z_path, weights_only=False)
    # Sort files numerically to ensure alignment
    files = sorted([f for f in os.listdir(preds_path) if f.endswith(".torch")], 
                   key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    X_inliers = []
    y_labels = []
    limit = min(len(z_data['predictions']), len(files))
    
    for i in range(limit):
        # 1. Get Label (Correctness)
        top_pred = z_data['predictions'][i][0]
        if isinstance(top_pred, torch.Tensor): top_pred = top_pred.item()
        true_matches = z_data['positives_per_query'][i]
        if isinstance(true_matches, torch.Tensor): true_matches = true_matches.tolist()
        
        is_correct = 1 if top_pred in true_matches else 0
        y_labels.append(is_correct)
        
        # 2. Get Feature (Inliers)
        data = torch.load(os.path.join(preds_path, files[i]), weights_only=False)
        max_inliers = 0
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                counts = [x['num_inliers'] for x in data]
            else:
                counts = [x.item() for x in data]
            max_inliers = max(counts)
        X_inliers.append(max_inliers)
        
    return np.array(X_inliers).reshape(-1, 1), np.array(y_labels)

def compute_ause(y_true, y_pred_prob):
    """
    Computes the Area Under the Sparsification Error (AUSE) curve.
    Returns the score and the curve data for plotting.
    """
    errors = (y_true == 0).astype(int)
    uncertainty = 1 - y_pred_prob
    # Sort by uncertainty (descending)
    sort_idx = np.argsort(uncertainty)[::-1]
    errors_sorted = errors[sort_idx]
    
    n = len(y_true)
    model_curve = []
    oracle_curve = []
    num_errors = np.sum(errors)
    
    for i in range(n):
        remaining_errors = errors_sorted[i:]
        model_curve.append(np.mean(remaining_errors) if len(remaining_errors)>0 else 0)
        
        remaining_count = n - i
        if remaining_count > 0:
            current_errors = max(0, num_errors - i)
            oracle_curve.append(current_errors / remaining_count)
        else:
            oracle_curve.append(0)
            
    x = np.arange(n) / n
    # Calculate area using trapezoidal rule
    ause = np.trapz(model_curve, x) - np.trapz(oracle_curve, x)
    return ause, model_curve, oracle_curve

def main():
    print("🚀 === Training SVOX (Teacher) Model ===")
    X_train, y_train = get_data(TRAIN_LOG_DIR, MATCHER_FOLDER)
    
    if X_train is None:
        print("❌ Failed to load training set! Please check paths.")
        return

    # 1. Train Logistic Regression
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print(f"✅ Model trained successfully! (Samples: {len(y_train)})")
    
    print("\n⚔️ === Starting Batch Testing (including Spearman & R2) ===")
    
    # Print Table Header
    print(f"{'Method':<10} | {'AUPRC (↑)':<10} | {'AUSE (↓)':<10} | {'Spearman (↑)':<12} | {'R2 Score (↑)':<10}")
    print("-" * 75)
    
    # Store statistics for plotting
    stats = {
        'names': [], 'auprc': [], 'ause': [], 'spearman': []
    }
    
    # Variables to track the "best performer" for detailed curve plotting
    best_ause = 1.0
    best_method_data = None 

    for name, path in TEST_LOG_DIRS.items():
        X_test, y_test = get_data(path, MATCHER_FOLDER)
        
        if X_test is None:
            continue
            
        probs_test = clf.predict_proba(X_test)[:, 1]
        
        # === Core Metric Calculation ===
        # 1. AUPRC (Robustness)
        auprc = average_precision_score(y_test, probs_test)
        
        # 2. AUSE (Reliability/Sparsification)
        ause_val, model_curve, oracle_curve = compute_ause(y_test, probs_test)
        
        # 3. Spearman (Rank Correlation)
        spearman_val, _ = spearmanr(probs_test, y_test)
        
        # 4. R2 Score (Goodness of Fit)
        r2_val = r2_score(y_test, probs_test)
        
        # Print result row
        print(f"{name:<10} | {auprc:.4f}     | {ause_val:.4f}     | {spearman_val:.4f}       | {r2_val:.4f}")
        
        # Append to lists
        stats['names'].append(name)
        stats['auprc'].append(auprc)
        stats['ause'].append(ause_val)
        stats['spearman'].append(spearman_val)
        
        # Update best method data (lowest AUSE) for Fig 1 and Fig 2
        if ause_val < best_ause:
            best_ause = ause_val
            best_method_data = {
                'name': name, 'X': X_test, 'y': y_test, 'probs': probs_test,
                'model_curve': model_curve, 'oracle_curve': oracle_curve
            }

    # ================= 🎨 Plotting Section =================
    print("\n📊 Generating three core figures...")
    plt.style.use('ggplot')

    # --- Fig 1: Logistic Regression Curve (S-Curve) ---
    plt.figure(figsize=(8, 6))
    x_range = np.linspace(0, 150, 300).reshape(-1, 1)
    y_prob_learned = clf.predict_proba(x_range)[:, 1]
    
    plt.plot(x_range, y_prob_learned, color='#e74c3c', linewidth=3, label='Learned Probability')
    if best_method_data:
        plt.scatter(best_method_data['X'], best_method_data['y'], 
                   color='#3498db', alpha=0.3, s=20, label=f"Data ({best_method_data['name']})")
    
    plt.title('Fig 1. Logistic Regression Uncertainty Model', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Inliers', fontsize=12)
    plt.ylabel('Probability of Correctness', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 150)
    plt.tight_layout()
    plt.savefig('Fig1_Logistic_Curve.png', dpi=300)
    print("✅ Fig 1 Saved")

    # --- Fig 2: Sparsification Curve (Showing AUSE) ---
    plt.figure(figsize=(8, 6))
    if best_method_data:
        mc = best_method_data['model_curve']
        oc = best_method_data['oracle_curve']
        x_axis = np.linspace(0, 1, len(mc))
        
        plt.plot(x_axis, mc, label='Model Error Curve', color='#3498db', linewidth=2)
        plt.plot(x_axis, oc, label='Oracle (Ideal) Curve', color='#2ecc71', linestyle='--', linewidth=2)
        plt.fill_between(x_axis, mc, oc, color='gray', alpha=0.2, label=f'AUSE = {best_ause:.4f}')
        
        plt.title(f"Fig 2. Sparsification Curve ({best_method_data['name']})", fontsize=14, fontweight='bold')
        plt.xlabel('Fraction of Removed Samples', fontsize=12)
        plt.ylabel('Error Rate', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Fig2_Sparsification_Curve.png', dpi=300)
    print("✅ Fig 2 Saved")

    # --- Fig 3: Bar Charts (Comparison across methods) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar 1: AUPRC
    bars1 = ax1.bar(stats['names'], stats['auprc'], color=['#95a5a6', '#3498db', '#9b59b6', '#2ecc71'], alpha=0.9)
    ax1.set_title('Robustness (AUPRC ↑)', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.1)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{bar.get_height():.1%}', ha='center', fontweight='bold')

    # Bar 2: AUSE
    bars2 = ax2.bar(stats['names'], stats['ause'], color=['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71'], alpha=0.9)
    ax2.set_title('Reliability Error (AUSE ↓)', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(stats['ause']) * 1.3)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                 f'{bar.get_height():.4f}', ha='center', fontweight='bold')

    plt.suptitle('Fig 3. Performance Comparison across Methods', fontsize=16)
    plt.tight_layout()
    plt.savefig('Fig3_Bar_Chart.png', dpi=300)
    print("✅ Fig 3 Saved")
    
    plt.show()

if __name__ == "__main__":
    main()
