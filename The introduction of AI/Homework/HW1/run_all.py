import subprocess

output_file = "predicted_label.npy"
seeds = list(range(0, 100))

for i, seed in enumerate(seeds, start=1):
    print(f"\n=== 第 {i} 次執行，seed = {seed} ===")

    # 執行 sample.py 並傳入 seed
    subprocess.run(["python", "sample.py", "--seed", str(seed)], check=True)

    # 執行 eval.py 評分
    subprocess.run(["python", "eval_tech.py", output_file], check=True)

print("\n✅ 所有 seed 測試完成！")