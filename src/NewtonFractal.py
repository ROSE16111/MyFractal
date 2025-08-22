# src/NewtonFractal.py
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------------------------
# 0) Device & dtype
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float32)
complex_dtype = torch.complex64
print("Device:", device)

# ---------------------------
# 1) Build complex grid (vectorised on GPU)
# ---------------------------
def build_grid(x_range=(-2.0, 2.0), y_range=(-2.0, 2.0), step=0.0025):
    y_np, x_np = np.mgrid[y_range[0]:y_range[1]:step, x_range[0]:x_range[1]:step]
    x = torch.from_numpy(x_np).to(device=device)
    y = torch.from_numpy(y_np).to(device=device)
    grid = torch.complex(x, y).to(dtype=complex_dtype)
    return grid

# ---------------------------
# 2) Newton iteration for f(z) = z^3 - 1
# roots: 1, -1/2 ± i*sqrt(3)/2
# ---------------------------
ROOTS = torch.tensor(
    [1.0 + 0.0j,
     -0.5 + np.sqrt(3)/2*1j,
     -0.5 - np.sqrt(3)/2*1j],
    dtype=complex_dtype, device=device
)

def newton_fractal(x_range=(-2,2), y_range=(-2,2), step=0.0025,
                   max_iter=50, tol=1e-6):
    """
    Vectorised Newton fractal on GPU.
    Returns:
      root_idx (H,W,int8): 收敛到哪个根（-1 表示未收敛）
      iters    (H,W,int16): 收敛所需步数（或到达上限）
    """
    z = build_grid(x_range, y_range, step)   # z0
    H, W = z.shape

    root_idx = torch.full((H, W), -1, dtype=torch.int8, device=device)   # basin label
    iters    = torch.zeros((H, W), dtype=torch.int16, device=device)     # steps to converge
    mask     = torch.ones((H, W), dtype=torch.bool, device=device)       # still not converged

    with torch.no_grad():
        for k in range(max_iter):
            if not mask.any(): break

            # Iterative core
            #(the entire graph is updated in parallel in each round)
            z_alive = z[mask]

            # f(z) = z^3 - 1, f'(z) = 3 z^2
            f  = z_alive*z_alive*z_alive - 1.0
            fp = 3.0 * z_alive*z_alive

            # 避免除零：加一个小 epsilon
            z_alive = z_alive - f / (fp + 1e-12)

            # 写回
            z[mask] = z_alive

            # 收敛判据：距离任一根 < tol  或 |f(z)| < tol
            # 先对每个像素分别计算到三个根的距离，选最近的
            diffs = z_alive.unsqueeze(-1) - ROOTS  # shape: (N_alive, 3)
            d2 = (diffs.real**2 + diffs.imag**2)
            min_d2, min_idx = torch.min(d2, dim=1)

            # 仍然在活跃集合里的像素的全局位置
            # 根据 min_d2 < tol^2 标记收敛
            conv = (min_d2 < (tol*tol))
            # 给刚收敛的像素写入根编号与迭代步数
            root_idx[mask] = torch.where(conv, min_idx.to(torch.int8), root_idx[mask])
            iters[mask]    = torch.where(conv, torch.tensor(k+1, dtype=torch.int16, device=device), iters[mask])

            # 更新活跃遮罩：未收敛的继续迭代
            alive_indices = torch.where(mask)
            new_mask = mask.clone()
            new_mask[alive_indices[0][conv], alive_indices[1][conv]] = False
            mask = new_mask
            # print(z.device, f.device, root_idx.device)
    return root_idx.detach().cpu().numpy(), iters.detach().cpu().numpy()

# ---------------------------
# 3) Colouring / Visualisation
# ---------------------------
def color_root_basins(root_idx, iters, gamma=0.75):
    """
    根域着色（basins of attraction）+ 迭代步数做亮度：steps^{-gamma}
    """
    idx = root_idx.copy()
    H, W = idx.shape
    img = np.zeros((H, W, 3), dtype=np.float32)

    # 三个根的基色
    colors = np.array([[0.9, 0.2, 0.2],
                       [0.2, 0.9, 0.2],
                       [0.2, 0.2, 0.9]], dtype=np.float32)

    for i in range(3):
        mask = (idx == i)
        shade = (iters[mask] + 1e-3)**(-gamma)   #The fewer steps, the brighter
        img[mask] = colors[i] * shade[:, None]   

    # Unconverged black
    img[idx < 0] = 0.0
    return np.uint8(np.clip(img*255, 0, 255))

# ---------------------------
# 4) Demo run
# ---------------------------
if __name__ == "__main__":
    root_idx, iters = newton_fractal(
        x_range=(-2.0, 2.0), y_range=(-2.0, 2.0),
        step=0.0025, max_iter=50, tol=1e-6
    )

    img = color_root_basins(root_idx, iters, gamma=0.75)
    # root_idx (H,W,int8)  Which root does each pixel converge to（0/1/2，或 -1 = 未收敛）
    # iters     # (H,W,int16) Number of steps to convergence收敛用的步数（速度）
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout(pad=0)
    # plt.savefig("outputs/newton.png", dpi=200)  # 保存
    plt.show()
