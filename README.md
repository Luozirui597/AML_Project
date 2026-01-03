


# 🚀 Update: Section 6.2 Uncertainty Estimation (Started)

**Date:** 2025-12-19
**Status:** ✅ start

## 1. Summary of Work
I have successfully completed the tasks for **Section 6.2 (Uncertainty Estimation)**. To satisfy the project requirement of "picking two VPR methods," I performed a comparative analysis between **CosPlace** and **NetVLAD**.

**Key Achievements:**
1.  **NetVLAD Implementation:** Successfully ran NetVLAD (VGG16) on the Tokyo-XS test set and fixed a Windows compatibility bug regarding weight downloads.
2.  **Uncertainty Module:** Implemented a **Logistic Regression** model that uses the number of geometric inliers (from SuperPoint+LightGlue) to predict the probability of a correct match.
3.  **New Metrics:** Implemented the required advanced metrics: **AUSC** (Area Under Sparsification Curve) and **Spearman’s Rank Correlation**.
4.  **Comparison:** Generated comparative plots showing the robustness of our uncertainty estimation across different backbones.

## 2. Key Results for the Report
**💡 Note:** The data and plot below are ready to be inserted directly into **Section 6.2** of the final report.

### 2.1 Quantitative Comparison
We compared the uncertainty estimation performance on two different VPR methods using the same geometric verifier (SuperPoint + LightGlue):

| VPR Method | Matcher | AUPRC (Higher is better) | Spearman (Higher is better) | AUSC (Lower is better) |
| :--- | :--- | :--- | :--- | :--- |
| **CosPlace** (ResNet18) | SP + LightGlue | **0.8659** | 0.5231 | **0.1643** |
| **NetVLAD** (VGG16) | SP + LightGlue | 0.8216 | **0.6123** | 0.2535 |

**Conclusions:**
* **High Reliability:** Both methods achieve high AUPRC scores (>0.82), confirming that the "Number of Inliers" is a robust proxy for prediction correctness.
* **Generalizability:** Even with NetVLAD (which has a lower base retrieval recall of ~50%), the uncertainty module effectively identifies potential errors.

### 2.2 Sparsification Curve
The generated plot shows how the error rate decreases as we reject uncertain queries.

![Uncertainty Comparison](final_uncertainty_comparison.png)
*(Image file: `final_uncertainty_comparison.png` is located in the root directory)*

## 3. Code Changes & Usage

### 📂 Modified/New Files
* `final_analysis.py`: **[New Script]** Loads the experiment logs, calculates the Logistic Regression/AUSC metrics, and generates the comparison plot.
* `vpr_models/netvlad.py`: **[Bug Fix]** Replaced the Linux-specific `wget` command with Python's native `urllib` to fix weight download errors on Windows.

### 🏃‍♂️ How to Reproduce
To regenerate the statistics and the plot, simply run the following command in the root directory:

```bash
python final_analysis.py
```

### 📊 Data Logs
If you need to inspect the raw prediction files, they are located here:

CosPlace Data: logs/log_dir/2025-12-18_12-51-25

NetVLAD Data: logs/log_dir/2025-12-19_12-27-49

Next Step: Please integrate the table and the plot into the final report document.



---

## 🔍 Uncertainty Estimation (Section 6.2)

This module implements a probabilistic uncertainty estimation mechanism for Visual Place Recognition. Instead of relying solely on the similarity score from the VPR model (e.g., CosPlace), we use **Geometric Verification** as a proxy for confidence.

### 🛠 Methodology

We treat the uncertainty estimation as a binary classification problem solvable via **Logistic Regression**:
1.  **Retrieval:** The VPR model retrieves the top candidate.
2.  **Matching:** We apply **SuperPoint + LightGlue** to match the query with the retrieved image.
3.  **Feature:** The number of geometrically verified inliers is used as the input feature ($x$).
4.  **Prediction:** A Logistic Regression model predicts the probability of correctness ($P(y=1|x)$).

### 🧪 Experiment Design: Cross-Domain Robustness

To ensure the robustness of our uncertainty estimator, we designed a challenging **Cross-Domain** experiment:

* **Training Set (The "Teacher"):** **SVOX (Sun vs. Night)**
    * We trained the model on a difficult split where queries are taken during the day and matched against a database at night.
    * *Reasoning:* This forces the model to learn a conservative decision boundary, as even correct matches in extreme lighting changes yield fewer inliers.
* **Testing Set (The "Student"):** **SF-XS (San Francisco)**
    * The trained model was directly evaluated on the SF-XS dataset without fine-tuning.
    * *Goal:* To validate the generalization capability of the estimator across different cities and environments.

---

## 🚀 Execution Flow: Uncertainty Estimation (Section 6.2)

This section details the step-by-step pipeline to reproduce the Uncertainty Estimation results. The process involves generating raw VPR predictions, extracting geometric inliers via LightGlue, and training a Logistic Regression model for cross-domain evaluation.

### 1. Generate Raw Predictions & Uncertainty Data
First, run the VPR model (e.g., CosPlace) on both the **Training Set (SVOX Sun/Night)** and the **Test Set (SF-XS)**. The flag `--save_for_uncertainty` is crucial as it saves the ground truth labels and prediction indices.

```bash
# A. Generate Training Data (SVOX: Sun query vs Night database)
python ./VPR-methods-evaluation/main.py \
    --method cosplace \
    --database_folder <path_to_svox_night> \
    --queries_folder <path_to_svox_sun> \
    --save_for_uncertainty --num_preds_to_save 2000 \
    --log_dir ./logs/svox_train

# B. Generate Test Data (SF-XS)
python ./VPR-methods-evaluation/main.py \
    --method cosplace \
    --database_folder <path_to_sf_xs_db> \
    --queries_folder <path_to_sf_xs_query> \
    --save_for_uncertainty --num_preds_to_save 2000 \
    --log_dir ./logs/sfxs_test
```

### 2. Extract Geometric Features (Inliers)
Run the matching script to calculate the number of inliers for the retrieved candidates using SuperPoint + LightGlue. This converts raw images into quantitative "confidence scores" (inlier counts).

```bash
# Process SVOX Logs
python match_queries_preds.py \
    --preds-dir ./logs/svox_train/<timestamp>/preds \
    --matcher superpoint-lg --num-preds 10

# Process SF-XS Logs
python match_queries_preds.py \
    --preds-dir ./logs/sfxs_test/<timestamp>/preds \
    --matcher superpoint-lg --num-preds 10
```

### 3.Logistic Regression Model

This module implements the **Post-hoc Uncertainty Estimation** for Visual Place Recognition (VPR). It addresses the deterministic nature of VPR methods by learning a probabilistic mapping from **Geometric Inliers** to the **Probability of Correct Localization**.

To ensure robustness, the model is trained in a **Cross-Domain** setting:
* **Training (Teacher):** SVOX Night (Challenging illumination).
* **Testing (Student):** SF-XS (Urban scenes) across multiple VPR backbones.

### 📋 Features

* **Logistic Regression Model:** Maps geometric consistency (inlier counts) to confidence scores.
* **Cross-Domain Evaluation:** Validates generalization from difficult to easy domains.
* **Multi-Method Support:** Batch testing for **CosPlace, NetVLAD, MixVPR, and MegaLoc**.
* **Comprehensive Metrics:**
    * **AUPRC:** Area Under Precision-Recall Curve (Robustness).
    * **AUSE:** Area Under Sparsification Error (Calibration).
    * **Spearman's $\rho$:** Rank Correlation.
    * **$R^2$ Score:** Goodness of Fit.
* **Automated Visualization:** Generates S-Curves, Sparsification Curves, and Comparison Bar Charts.

### 🛠️ Prerequisites

The script requires the following Python libraries:

```bash
pip install torch numpy matplotlib scikit-learn scipy
```


