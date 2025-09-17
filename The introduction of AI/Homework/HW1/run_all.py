import subprocess

# 設定要跑的次數
num_runs = 3  # 你可以改成任何數字

# 輸出檔名（eval.py 需要這個檔案）
output_file = "predicted_label.npy"

for i in range(1, num_runs + 1):
    print(f"\n=== 第 {i} 次執行 ===")

    # 1. 執行 sample.py 產生分群結果
    subprocess.run(["python", "sample.py"], check=True)

    # 2. 執行 eval.py 評分
    subprocess.run(["python", "eval.py", output_file], check=True)

print("\n✅ 全部流程完成！")