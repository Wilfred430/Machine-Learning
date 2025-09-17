import cupy as cp
print(cp.__version__)       # 應該會顯示 13.6.0 或更新版本
print(cp.cuda.runtime.runtimeGetVersion())  # 顯示 CUDA runtime 版本