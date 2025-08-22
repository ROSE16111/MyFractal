# MyFractal
COMP3710_DEMO1
## Project introduction
### Newton Fractal（PyTorch/GPU，vectorised + mask）
Newton iterative update + convergence determination

![alt text](pic/newton1.png)
## operation method
* /src/NewtonFractal.py
1. vectorized on GPU (the entire grid tensor is updated in parallel)
   * I don’t have a pixel-by-pixel for loop; instead, I represent the entire image as a tensor, and all updates are element-wise + broadcasting + reduction, which triggers GPU kernels to execute in parallel.
2. mask (iterate only non-converged pixels, can stop early)
3. no grad (inference mode)
    * Fractal rendering is not training and does not require backpropagation of the gradient graph (computation graph) ; 分形渲染不是训练，不需要反向传播的 梯度图（computation graph）。
    * `torch.no_grad()` close autograd：
    * reduce storage 减少内存（不保存中间变量）
    * Reduce scheduling/meta-operation overhead 减少调度/元操作开销
    * Slightly improve throughput (especially noticeable at high iterations);小幅提升吞吐（尤其在高迭代时更明显）
4. root basins (coloring by root + velocity shading).
    * Different initial value points will eventually converge to different roots (attraction domains)
    * **root_idx** determines the base color (red/green/blue), and **iters** determines the brightness (faster convergence → brighter).root_idx 决定基色（红/绿/蓝），用 iters 决定亮度（收敛更快 → 更亮）。
    * Unconverged (-1) pixels are set to black for clear distinction未收敛（-1）像素设为黑色，清晰区分
## dependencies
* torch
* numpy 
* matplotlib
## what analysis/visualization have done
## location of AI usage records
/AI_usage_record.md

