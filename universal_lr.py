import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

# ================= 🔧 Configuration Area =================

# 1. Training Set (Teacher) -> SVOX (Sun vs Night)
# Path from your run at 21:01:31
TRAIN_LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\2025-12-23_21-01-31" 
TRAIN_FOLDER = "preds_superpoint-lg"

# 2. Test Set (Student) -> SF-XS (Day)
# Path from your run at 21:04:37
TEST_LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\2025-12-23_21-04-37" 
TEST_FOLDER = "preds_superpoint-lg"

# ==========================================================

def get_data(log_dir, folder_name):
    """
    Generic Data Extraction Function:
    Extracts (Number of Inliers, Correctness Label) from the Log folder.
    """
    print(f"📂 Loading: {log_dir} ...")
    
    # 1. Check if files exist
    z_path = os.path.join(log_dir, "z_data.torch")
    preds_path = os.path.join(log_dir, folder_name)
    
    if not os.path.exists(z_path):
        print(f"❌ Error: z_data.torch not found! Please check the path.")
        return None, None
    if not os.path.exists(preds_path):
        print(f"❌ Error: Matcher folder {folder_name} not found! Please run match_queries_preds.py first.")
        return None, None

    # 2. Load data
    z_data = torch.load(z_path, weights_only=False)
    # Sort files numerically to ensure alignment
    files = sorted([f for f in os.listdir(preds_path) if f.endswith(".torch")], 
                   key=lambda x: int(''.join(filter(str.isdigit, x))))
    
    X_inliers = []
    y_labels = []
    
    # Limit loop to the smaller count (in case processing was interrupted)
    limit = min(len(z_data['predictions']), len(files))
    
    for i in range(limit):
        # --- Get Label (0/1) ---
        top_pred = z_data['predictions'][i][0]
        if isinstance(top_pred, torch.Tensor): top_pred = top_pred.item()
        
        true_matches = z_data['positives_per_query'][i]
        if isinstance(true_matches, torch.Tensor): true_matches = true_matches.tolist()
        
        # Label is 1 if the prediction is in the ground truth set, else 0
        is_correct = 1 if top_pred in true_matches else 0
        y_labels.append(is_correct)
        
        # --- Get Feature (Inliers) ---
        data = torch.load(os.path.join(preds_path, files[i]), weights_only=False)
        max_inliers = 0
        
        # Handle different storage formats (list of dicts vs list of tensors)
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], dict):
                counts = [x['num_inliers'] for x in data]
            else:
                counts = [x.item() for x in data] # If it is a tensor directly
            max_inliers = max(counts)
            
        X_inliers.append(max_inliers)
        
    return np.array(X_inliers).reshape(-1, 1), np.array(y_labels)

def main():
    # --- 1. Prepare Data ---
    print("--- Preparing Data ---")
    X_train, y_train = get_data(TRAIN_LOG_DIR, TRAIN_FOLDER)
    X_test, y_test = get_data(TEST_LOG_DIR, TEST_FOLDER)
    
    if X_train is None or X_test is None:
        print("Terminating: Data loading failed.")
        return

    print(f"✅ Training Set (SVOX Sun-Night): {len(y_train)} samples (Positive Rate: {y_train.mean():.1%})")
    print(f"✅ Test Set (SF-XS): {len(y_test)} samples")

    # --- 2. Train Logistic Regression ---
    print("\n🧠 Training Logistic Regression...")
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    
    # Get learned parameters
    coef = clf.coef_[0][0]
    intercept = clf.intercept_[0]
    print(f"💡 Learned Formula: Probability = Sigmoid({coef:.3f} * Inliers + {intercept:.3f})")

    # --- 3. Prediction & Evaluation ---
    probs_test = clf.predict_proba(X_test)[:, 1]
    
    # Calculate scores
    score_raw = average_precision_score(y_test, X_test)       # Baseline: Raw inlier count
    score_learned = average_precision_score(y_test, probs_test) # Proposed: Probabilistic score
    
    print("\n" + "="*40)
    print(f"📊 Final Results (AUPRC)")
    print(f"1. Baseline (Raw Inliers): {score_raw:.4f}")
    print(f"2. Proposed (Logistic Reg): {score_learned:.4f}")
    print("="*40)

    # --- 4. Visualization (S-Curve required for the report) ---
    plt.figure(figsize=(10, 6))
    
    # Plot the learned S-Curve (from SVOX)
    x_range = np.linspace(0, 150, 300).reshape(-1, 1)
    y_prob = clf.predict_proba(x_range)[:, 1]
    plt.plot(x_range, y_prob, color='red', linewidth=3, label='Learned Uncertainty Model (on SVOX)')
    
    # Plot SF-XS data distribution
    plt.scatter(X_test, y_test, color='gray', alpha=0.1, label='SF-XS Test Data')
    
    plt.title("Uncertainty Estimation: Trained on SVOX(Sun/Night) -> Tested on SF-XS")
    plt.xlabel("Number of Inliers (LightGlue)")
    plt.ylabel("Probability of Correctness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 150)
    
    save_path = "final_uncertainty_plot.png"
    plt.savefig(save_path)
    print(f"\n🖼️ Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()