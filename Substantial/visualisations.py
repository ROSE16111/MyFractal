# -*- coding: utf-8 -*-
"""
Newton Fractal (GPU vectorised) — 带三种可视化（basins / steps / residual）
主要看点：
- vectorised on GPU：整张网格张量一次性在 CUDA 上迭代；
- mask：只对“未收敛像素”继续计算，并支持提前结束；
- no_grad：推理模式，关闭 autograd，省内存/开销；
- root basins：按收敛到的根着色 + 用步数控制亮度。
"""

import numpy as np                     # 数值工具（构网格、后处理）/ numerical utils
import torch                           # 张量与 GPU / tensors & CUDA
import matplotlib.pyplot as plt        # 可视化 / plotting
from pathlib import Path               # 路径管理 / paths

# ---------------------------
# 0) Device & dtype / 设备与类型
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选 GPU/CPU
torch.set_default_dtype(torch.float32)                                  # 默认实数精度 float32
complex_dtype = torch.complex64                                         # 复数精度 complex64（省显存）
print("Device:", device)                                                # 打印设备用于 demo

# ---------------------------
# 1) Build complex grid / 生成复平面网格（向量化）
# ---------------------------
def build_grid(x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), step=0.0025):
    """
    返回复数网格张量（H×W）放在 device 上。
    Return complex grid tensor (H×W) on the chosen device.
    """
    # np.mgrid 先 Y 后 X；step 越小分辨率越高、越慢
    y_np, x_np = np.mgrid[y_range[0]:y_range[1]:step, x_range[0]:x_range[1]:step]
    # from_numpy 生成 CPU 张量（零拷贝共享内存），随后 .to(device) 拷到 GPU
    x = torch.from_numpy(x_np).to(device=device)
    y = torch.from_numpy(y_np).to(device=device)
    # 组合为复数网格（x + i y），并设为 complex_dtype
    grid = torch.complex(x, y).to(dtype=complex_dtype)
    return grid

# ---------------------------
# 2) Newton iteration setup / 牛顿迭代的函数与三根
# f(z) = z^3 - 1, f'(z) = 3 z^2
# 三个根：1，(-1±i√3)/2

# Make the three roots into complex tensors on the GPU so that they do not need 
# to be repeatedly transmitted when broadcasting and comparing distances.
# ---------------------------
ROOTS = torch.tensor(
    [1.0 + 0.0j,
     -0.5 + (np.sqrt(3)/2)*1j,
     -0.5 - (np.sqrt(3)/2)*1j],
    dtype=complex_dtype, device=device
)

# ---------------------------
# 3) Newton fractal core / 核心计算（GPU + 向量化 + mask + no_grad）
# ---------------------------
def newton_fractal(x_range=(-2,2), y_range=(-2,2), step=0.0025,
                   max_iter=50, tol=1e-6):
    """
    计算 Newton Fractal：
    - 向量化在 GPU 上进行（vectorised on GPU）
    - 仅更新未收敛像素（mask）
    - 关闭 autograd（no_grad）
    返回：
      root_idx (H,W,int8): 收敛到哪个根（0/1/2；-1 表示未收敛）
      iters    (H,W,int16): 收敛所需步数（或到达上限）
      z_final  (H,W)complex: 最终 z（用于残差/相位可视化）
      f_final  (H,W)complex: 最终 f(z)
    """
    # 初始状态：z0 为复平面网格
    z = build_grid(x_range, y_range, step)           # (H,W) complex64 on device
    H, W = z.shape

    # 结果张量：根编号、步数；起始为 -1 / 0
    root_idx = torch.full((H, W), -1, dtype=torch.int8, device=device)   # basin labels
    iters    = torch.zeros((H, W), dtype=torch.int16, device=device)     # steps to converge

    # 活跃遮罩：True 表示“尚未收敛、需要继续计算”
    mask     = torch.ones((H, W), dtype=torch.bool, device=device)

    eps = 1e-12  # 避免 f'(z)=0 时除零 / small epsilon for numerical safety

    # 推理模式：不需要梯度 / inference mode: no autograd graph
    with torch.no_grad():
        for k in range(max_iter):
            if not mask.any():               # 全部收敛可提前停止 / early exit
                break

            # 仅取出活跃像素的 z 与 c（此处 c ≡ 0，因为牛顿法只是更新 z）
            z_alive = z[mask]               # shape: (N_alive,)
            # 计算 f(z) 与 f'(z)（向量化逐元素）/ vectorised elementwise ops
            f  = z_alive*z_alive*z_alive - 1.0
            fp = 3.0 * z_alive*z_alive
            # 牛顿步：z ← z - f/f'
            z_alive = z_alive - f / (fp + eps)
            # 写回到原网格位置 / scatter back
            z[mask] = z_alive

            # 与三个根的距离（广播计算）/ distance to 3 roots (broadcasting)
            diffs = z_alive.unsqueeze(-1) - ROOTS            # (N_alive, 3)
            d2 = (diffs.real**2 + diffs.imag**2)             # squared distance
            min_d2, min_idx = torch.min(d2, dim=1)           # 最近的根及其距离^2

            # 收敛判据：距离最近根 < tol
            conv = (min_d2 < (tol*tol))                      # shape: (N_alive,)

            # 将刚收敛像素的根编号与步数写回 / write labels & steps for newly converged
            alive_y, alive_x = torch.where(mask)             # 活跃像素的全局坐标
            root_idx[alive_y[conv], alive_x[conv]] = min_idx[conv].to(torch.int8)
            iters[alive_y[conv],    alive_x[conv]] = (k+1)

            # 更新遮罩：收敛则剔除，不再参与下一轮 / shrink active set
            mask[alive_y[conv], alive_x[conv]] = False

    # 循环结束：保存最终 z 与 f(z) 以便额外可视化（残差/相位）
    f_final = (z*z*z - (1.0 + 0.0j))
    # 转 CPU/NumPy 供绘图或分析 / move to CPU for plotting/analysis
    return (root_idx.detach().cpu().numpy(),
            iters.detach().cpu().numpy(),
            z.detach().cpu().numpy(),
            f_final.detach().cpu().numpy())

# ---------------------------
# 4) Visualisations / 三种可视化
# ---------------------------
def viz_root_basins(root_idx, iters, gamma=0.75):
    """
    根域图（basins）+ 速度明暗：
    - 不同根：不同基色（RGB）
    - 收敛步数 iters：步数越少越亮（shade = steps^{-gamma}）

    gamma was large
    """
    H, W = root_idx.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    colors = np.array([[0.95, 0.25, 0.25],   # root 0 → red-ish
                       [0.25, 0.95, 0.35],   # root 1 → green-ish
                       [0.30, 0.35, 0.95]],  # root 2 → blue-ish
                      dtype=np.float32)
    for i in range(3):
        m = (root_idx == i)
        shade = (iters[m].astype(np.float32) + 1e-3)**(-gamma) # 亮度=步数^{-gamma}
        if shade.size > 0:
            img[m] = colors[i] * shade[:, None]
    img[root_idx < 0] = 0.0  # 未收敛像素黑色
    return np.uint8(np.clip(img*255, 0, 255))

def viz_speed_heatmap(iters):
    """
    收敛步数热图（Convergence steps）：
    - 仅以步数反映速度：越快越亮
    - 做归一化和伽马压缩以增强对比
    """
    arr = iters.astype(np.float32)
    # 归一化到 [0,1]
    m, M = arr.min(), arr.max() if arr.max() > arr.min() else (arr.min()+1)
    norm = (arr - m) / (M - m + 1e-9)
    # 快=亮（1 - norm）并做 gamma = 0.6
    # # 亮度=步数^{-gamma}
    heat = 1.0 - norm**0.6
    return np.uint8(np.clip(255*heat, 0, 255))

def viz_residual_log(f_final):
    """
    残差图（Residual log10|f(z)|）：
    - 显示最终残差的数量级；边界/难收敛区域通常更明亮或变化剧烈

    映射方向：现在是“残差小=亮”。若想反过来，改成 255*norm 即“残差大=亮”。
    cmap：换色图能突出不同的环纹/等值线风格。
    """
    mag = np.abs(f_final).astype(np.float32)
    logm = np.log10(mag + 1e-12)
    # 线性拉伸到 [0,255]
    m, M = logm.min(), logm.max() if logm.max() > logm.min() else (logm.min()+1)
    norm = (logm - m) / (M - m + 1e-9)
    img = np.uint8(np.clip(255*(1.0 - norm), 0, 255))  # 残差小=亮
    return img

# ---------------------------
# 5) Main demo / 主程序演示
# ---------------------------
if __name__ == "__main__":
    # 参数可按机器性能调整 / tune for your GPU
    x_range = (-2.0, 2.0)
    y_range = (-2.0, 2.0)
    step    = 0.0025
    max_iter= 50
    tol     = 1e-6 # 到根距离

    # 核心计算（GPU 向量化 + mask + no_grad）
    root_idx, iters, z_final, f_final = newton_fractal(
        x_range=x_range, y_range=y_range, step=step, max_iter=max_iter, tol=tol
    )

    # 三种可视化
    img_basins = viz_root_basins(root_idx, iters, gamma=0.75)
    img_speed  = viz_speed_heatmap(iters)
    img_resid  = viz_residual_log(f_final)

    # 并排展示
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1); plt.title("Basins + speed"); plt.imshow(img_basins); plt.axis("off")
    plt.subplot(1,3,2); plt.title("Convergence steps"); plt.imshow(img_speed, cmap="magma"); plt.axis("off")
    plt.subplot(1,3,3); plt.title("log10|f(z)|"); plt.imshow(img_resid, cmap="viridis"); plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    # 保存到 outputs 目录，便于 README 展示 / save outputs for your README
    out = Path("outputs"); out.mkdir(exist_ok=True, parents=True)
    plt.imsave(out/"newton_basins.png", img_basins)
    plt.imsave(out/"newton_steps.png",  img_speed, cmap="magma")
    plt.imsave(out/"newton_residual.png", img_resid, cmap="viridis")

    # 可选：保存中间数组（供 box-counting 或其他分析）
    np.save(out/"root_idx.npy", root_idx)
    np.save(out/"iters.npy",    iters)
    np.save(out/"z_final.npy",  z_final)
    np.save(out/"f_final.npy",  f_final)
