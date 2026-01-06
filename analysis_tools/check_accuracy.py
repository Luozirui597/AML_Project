import torch
import os

LOG_DIRS = [
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_18-45-46",
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_23-01-01",
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_08-24-08",
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_10-47-59"
]

def calculate_accuracy(path):
    z_path = os.path.join(path, "z_data.torch")
    folder_name = os.path.basename(path)
    
    if not os.path.exists(z_path):
        print(f"❌ {folder_name}: 缺少 z_data")
        return

    try:
        z_data = torch.load(z_path, weights_only=False)
        correct_count = 0
        total = len(z_data['predictions'])
        
        for i in range(total):
            top_pred = z_data['predictions'][i][0]
            if isinstance(top_pred, torch.Tensor): top_pred = top_pred.item()
            
            true_matches = z_data['positives_per_query'][i]
            if isinstance(true_matches, torch.Tensor): true_matches = true_matches.tolist()
            
            if top_pred in true_matches:
                correct_count += 1
                
        acc = (correct_count / total) * 100
        print(f"📂 {folder_name} -> 正确率: {acc:.2f}%")
        
        # 智能推测
        if acc < 70:
            print(f"   👉 可能是 SVOX (难度高/训练集)")
        else:
            print(f"   👉 可能是 SF-XS (难度低/测试集)")
            
    except Exception as e:
        print(f"❌ {folder_name}: 读取错误 {e}")

if __name__ == "__main__":
    print("=== 正在计算 R@1 正确率以区分数据集 ===")
    for p in LOG_DIRS:
        calculate_accuracy(p)
        