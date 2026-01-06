import torch
import os
import json

# === 你的四个文件夹路径 ===
LOG_DIRS = [
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_18-45-46",
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-30_23-01-01",
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_08-24-08",
    r"D:\AML\Visual-Place-Recognition-Project\logs\log_dir\2025-12-31_10-47-59"
]

def inspect_folder(path):
    print(f"\n📂 正在检查: {os.path.basename(path)}")
    
    # 1. 尝试读取配置 (flags.json 或 args.json) 以确定数据集名称
    config_file = os.path.join(path, "flags.json")
    dataset_name = "未知"
    method_name = "未知"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                args = json.load(f)
                # 尝试找数据集名字
                dataset_name = args.get('dataset_name') or args.get('queries_folder') or "未知"
                method_name = args.get('method') or "未知"
        except:
            pass
    
    print(f"   ℹ️  推测数据集: {dataset_name}")
    print(f"   ℹ️  使用方法: {method_name}")

    # 2. 检查 z_data.torch (VPR 结果)
    z_path = os.path.join(path, "z_data.torch")
    if os.path.exists(z_path):
        try:
            z_data = torch.load(z_path, weights_only=False)
            count = len(z_data['predictions'])
            print(f"   ✅ z_data.torch: 包含 {count} 个查询结果")
        except:
            print(f"   ⚠️ z_data.torch 损坏或无法读取")
    else:
        print(f"   ❌ 缺少 z_data.torch (无法用于训练/测试)")

    # 3. 检查匹配结果 (preds_...)
    match_folders = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith("preds_")]
    
    if match_folders:
        for mf in match_folders:
            num_files = len(os.listdir(os.path.join(path, mf)))
            print(f"   ✅ 发现匹配文件夹: {mf} ({num_files} 个文件)")
    else:
        print(f"   ❌ 未发现匹配文件夹 (preds_...) -> 无法用于 LR 模型")

if __name__ == "__main__":
    print("=== 开始扫描日志文件夹 ===")
    for p in LOG_DIRS:
        if os.path.exists(p):
            inspect_folder(p)
        else:
            print(f"\n❌ 路径不存在: {p}")