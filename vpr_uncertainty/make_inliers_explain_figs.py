import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr

# ================== CONFIG (use your existing paths) ==================
TRAIN_LOG_DIR = r"D:\AML\Visual-Place-Recognition-Project\logs\2025-12-23_21-01-31"
TEST_LOG_DIRS = {
    "CosPlace": r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_18-45-46",
    "NetVLAD":  r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_23-01-01",
    "MixVPR":   r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_08-24-08",
    "MegaLoc":  r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_10-47-59"
}
MATCHER_FOLDER = "preds_superpoint-lg"
RESULTS_DIR = r"D:\AML\Visual-Place-Recognition-Project\results"
OUT_DIR = os.path.join(RESULTS_DIR, "explain_inliers")
# =====================================================================

def safe_int_sort_key(name: str) -> int:
    digits = ''.join([c for c in name if c.isdigit()])
    return int(digits) if digits else 10**18

def extract_inlier_list(one_file_obj):
    """
    Returns a list[int] of inlier counts from a loaded .torch object.
    Compatible with your existing format:
      - list of dict with 'num_inliers'
      - list of tensors/ints
    """
    if not isinstance(one_file_obj, list) or len(one_file_obj) == 0:
        return []
    if isinstance(one_file_obj[0], dict) and "num_inliers" in one_file_obj[0]:
        return [int(x.get("num_inliers", 0)) for x in one_file_obj]
    # fallback: list of tensors/ints
    out = []
    for x in one_file_obj:
        if hasattr(x, "item"):
            out.append(int(x.item()))
        else:
            out.append(int(x))
    return out

def load_xy_with_multiple_summaries(log_dir: str, matcher_folder: str):
    z_path = os.path.join(log_dir, "z_data.torch")
    preds_path = os.path.join(log_dir, matcher_folder)

    if not os.path.exists(z_path) or not os.path.isdir(preds_path):
        return None

    z_data = torch.load(z_path, weights_only=False)
    files = sorted([f for f in os.listdir(preds_path) if f.endswith(".torch")],
                   key=safe_int_sort_key)

    n = min(len(z_data["predictions"]), len(files))
    if n == 0:
        return None

    y = []
    max_x, mean_x, med_x, p90_x = [], [], [], []

    for i in range(n):
        # label y: top-1 correct?
        top_pred = z_data["predictions"][i][0]
        if isinstance(top_pred, torch.Tensor):
            top_pred = top_pred.item()

        positives = z_data["positives_per_query"][i]
        if isinstance(positives, torch.Tensor):
            positives = positives.tolist()

        yi = 1 if top_pred in positives else 0
        y.append(yi)

        # inliers list for this query
        obj = torch.load(os.path.join(preds_path, files[i]), weights_only=False)
        inliers = extract_inlier_list(obj)

        if len(inliers) == 0:
            # keep zeros to avoid dropping alignment
            max_x.append(0); mean_x.append(0); med_x.append(0); p90_x.append(0)
        else:
            in_arr = np.array(inliers, dtype=np.float32)
            max_x.append(float(np.max(in_arr)))
            mean_x.append(float(np.mean(in_arr)))
            med_x.append(float(np.median(in_arr)))
            p90_x.append(float(np.percentile(in_arr, 90)))

    y = np.array(y, dtype=np.int32)
    feats = {
        "max": np.array(max_x, dtype=np.float32),
        "mean": np.array(mean_x, dtype=np.float32),
        "median": np.array(med_x, dtype=np.float32),
        "p90": np.array(p90_x, dtype=np.float32),
    }
    return y, feats

def save_hist_correct_wrong(x, y, title, out_path):
    x1 = x[y == 1]
    x0 = x[y == 0]
    plt.figure(figsize=(10, 6))
    xmax = max(10, int(np.max(x)))
    bins = np.linspace(0, xmax, 40)

    plt.hist(x1, bins=bins, alpha=0.6, label=f"Correct (y=1), n={len(x1)}")
    plt.hist(x0, bins=bins, alpha=0.6, label=f"Wrong (y=0), n={len(x0)}")

    plt.title(title)
    plt.xlabel("inlier summary value")
    plt.ylabel("count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_box_violin(x, y, title, out_path):
    x1 = x[y == 1]
    x0 = x[y == 0]
    plt.figure(figsize=(10, 6))

    parts = plt.violinplot([x1, x0], showmeans=True, showmedians=True, showextrema=False)
    plt.xticks([1, 2], ["Correct (y=1)", "Wrong (y=0)"])
    plt.title(title)
    plt.ylabel("inlier summary value")
    plt.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_binned_accuracy_curve(x, y, title, out_path, nbins=12):
    # bin by x quantiles to ensure each bin has samples
    q = np.linspace(0, 1, nbins + 1)
    edges = np.quantile(x, q)
    # make edges strictly increasing
    edges = np.unique(edges)
    if len(edges) < 3:
        return

    centers, accs, counts = [], [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        mask = (x >= lo) & (x <= hi) if i == len(edges) - 2 else (x >= lo) & (x < hi)
        if np.sum(mask) < 10:
            continue
        centers.append((lo + hi) / 2.0)
        accs.append(float(np.mean(y[mask])))
        counts.append(int(np.sum(mask)))

    plt.figure(figsize=(10, 6))
    plt.plot(centers, accs, marker="o")
    for cx, ay, c in zip(centers, accs, counts):
        plt.text(cx, ay + 0.02, f"n={c}", ha="center", fontsize=9)

    plt.ylim(-0.05, 1.05)
    plt.title(title)
    plt.xlabel("inlier summary (binned)")
    plt.ylabel("empirical P(correct)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def save_summary_comparison(y, feats, title, out_path):
    # Use x as a scoring function directly (higher x => more likely correct)
    names = ["max", "p90", "mean", "median"]
    auprc_vals = []
    spr_vals = []
    for k in names:
        x = feats[k]
        auprc_vals.append(average_precision_score(y, x))
        spr_vals.append(spearmanr(x, y).correlation)

    fig = plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(names, auprc_vals)
    ax1.set_title("AUPRC using x as score (↑)")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    ax2.bar(names, spr_vals)
    ax2.set_title("Spearman(x, y) (↑)")
    ax2.set_ylim(-1.05, 1.05)
    ax2.axhline(0, linewidth=1)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

def run_for_one_dir(tag, log_dir):
    out_prefix = os.path.join(OUT_DIR, tag)
    os.makedirs(OUT_DIR, exist_ok=True)

    loaded = load_xy_with_multiple_summaries(log_dir, MATCHER_FOLDER)
    if loaded is None:
        print(f"[SKIP] {tag}: missing files")
        return

    y, feats = loaded

    # 1) Show what inliers "look like" via distribution of max-inliers
    save_hist_correct_wrong(
        feats["max"], y,
        title=f"{tag}: Correct vs Wrong distribution (x = max #inliers)",
        out_path=f"{out_prefix}_FigA_hist_max.png"
    )

    # 2) violin plot (cleaner than histogram on slides)
    save_box_violin(
        feats["max"], y,
        title=f"{tag}: Violin (x = max #inliers)",
        out_path=f"{out_prefix}_FigB_violin_max.png"
    )

    # 3) binned empirical P(correct) vs inliers
    save_binned_accuracy_curve(
        feats["max"], y,
        title=f"{tag}: Empirical P(correct) increases with inliers (binned)",
        out_path=f"{out_prefix}_FigC_binned_acc.png"
    )

    # 4) Why summary? compare max/mean/median/p90
    save_summary_comparison(
        y, feats,
        title=f"{tag}: Why summarization? Compare different inlier summaries",
        out_path=f"{out_prefix}_FigD_summary_compare.png"
    )

    print(f"[OK] Saved figures for {tag} -> {OUT_DIR}")

def main():
    # Teacher (training) directory
    run_for_one_dir("TRAIN_SVOX", TRAIN_LOG_DIR)

    # Test directories (optional; can pick one to avoid too many figs)
    for name, d in TEST_LOG_DIRS.items():
        run_for_one_dir(f"TEST_{name}", d)

if __name__ == "__main__":
    main()
