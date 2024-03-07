import numpy as np
from typing import Optional
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg
from time import time
from utils import adjacency_4, timer



def gauss_seidel_sparse(
    A_data: np.ndarray,
    A_rows: np.ndarray,
    A_cols: np.ndarray,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    max_iter: int = 100
) -> tuple[np.ndarray, str]:
    """
    用 Gauss-Seidel 方法解稀疏矩阵方程组 Ax = b
    """
    A = csr_matrix((A_data, (A_rows, A_cols)))
    n = len(b)
    x = x0 if x0 is not None else np.zeros(n)

    for k in tqdm(range(max_iter)):
        x_old = np.copy(x)

        for i in range(n):
            row_start = A.indptr[i]
            row_end = A.indptr[i + 1]
            # sigma = A_data[row_start:row_end] @ x[A_cols[row_start:row_end]] - A[i, i] * x[i]
            sigma = np.dot(A_data[row_start:row_end], x[A_cols[row_start:row_end]]) - A[i, i] * x[i]
            x[i] = (b[i] - sigma) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, f"Gauss-Seidel converged within {k+1} iterations."

    return x, f"Gauss-Seidel did not converge within {max_iter} iterations."


def conjugate_gradient_sparse(
    A: csr_matrix,
    b: np.ndarray,
    x0: Optional[np.ndarray] = None,
    tol: float = 1e-5,
    max_iter: Optional[int] = None
) -> tuple[np.ndarray, str]:
    """
    用共轭梯度法解稀疏矩阵方程组 Ax = b
    """

    max_iter = max_iter if max_iter is not None else 10*b.shape[0]
    x = x0 if x0 is not None else np.zeros_like(b)
    r = b - A.dot(x)
    p = r
    rsold = np.dot(r, r)

    for i in range(max_iter):
        Ap = A.dot(p)
        alpha = rsold / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        if np.sqrt(rsnew) < tol:
            return x, f"Conjugate gradient converged within {i+1} iterations (tol = {tol})."
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew

    return x, f"Conjugate gradient did not converge within {max_iter} iterations (tol = {tol})."



def poisson_blending(
    patch_area: np.ndarray,  # 指示每个像素是否使用 patch，是(1)，否(0)
    scene: np.ndarray,
    patch: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 1000,
    verbose: bool = False
) -> np.ndarray:
    """
    使用 Poisson Blending （泊松融合）来把 patch 融合到 scene 的某个区域内（该区域由 patch_area 指示）。
    
    patch_area: shape = (h, w)
    scene: shape = (h, w, 3)
    patch: shape = (h, w, 3)，其中 h, w 为裁剪后的大小
    """

    start_time = time()

    result: np.ndarray = np.zeros_like(scene, dtype=np.int32)
    patch = patch.astype(np.int32)

    # 要解的方程组是 Ax = b，方程个数、未知数个数为 n
    n = np.sum(patch_area)
    n_channel = scene.shape[2]  # assert scene.shape[2] == patch.shape[2]
    b = np.zeros((n, n_channel), dtype=np.int32)
    # 迭代法解方程组时的初始解，取为 patch
    x0 = np.zeros((n, n_channel), dtype=np.float64)

    # patch_area 的每个像素对应 1 个未知数（先将整个像素值视为 1 个未知数；实际上 1 个像素对应 3 个未知数，这会在后面处理）
    xs, ys = np.where(patch_area == 1)
    # 给每个未知数一个编号
    coor_to_id = np.zeros_like(patch_area)
    for i, (x, y) in enumerate(zip(xs, ys)):
        coor_to_id[x, y] = i

    # 方程 Ax = b 中的 A 是稀疏矩阵，故只需记录非零元的 row, col, data
    rows = []
    cols = []
    data = []
    # A 的所有对角元素都有数值（在对角线上不“稀疏”），故专门用 diag_data 记录对角线的元素值
    diag_data = [0 for _ in range(n)]

    # 逐个列出所有方程，本次循环列出第 i 个方程；方程个数与未知数个数相同
    for i, (x, y) in enumerate(zip(xs, ys)):
        # 方程参照论文 https://www.cs.jhu.edu/~misha/Fall07/Papers/Perez03.pdf 的公式 (7) 构建
        for qx, qy in adjacency_4(x, y):  # 邻居节点 q
            if not (0 <= qx < patch_area.shape[0] and 0 <= qy < patch_area.shape[1]):
                continue
            diag_data[i] += 1
            b[i] += patch[x, y] - patch[qx, qy]
            # 如果这个邻居节点也属于 patch_area（即论文中的 Ω）
            if patch_area[qx, qy] == 1:
                rows.append(i)
                cols.append(coor_to_id[qx, qy])
                data.append(-1)
            # 如果这个邻居节点不属于 patch_area，那么这个节点在边界上（即论文中的 ∂Ω）
            else:
                b[i] += scene[qx, qy]
        # 迭代法解方程组时的初始解
        x0[i] = patch[x, y]
                
    # 把所有（对角元和非对角元的）row, col, data 整理到一起 
    data: np.ndarray = np.concatenate((data, diag_data))
    rows: np.ndarray = np.concatenate((np.array(rows), np.arange(n)))
    cols: np.ndarray = np.concatenate((np.array(cols), np.arange(n)))

    # 思路A：对每个颜色通道，解一次方程组
    eqa_start_time = time()
    A = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    for ci in range(n_channel):
        # # 方法1：使用开源库 scipy.sparse.linalg.cg 函数求解稀疏矩阵方程组
        # X, _ = cg(A, b[:, ci], x0=x0[:, ci])  # time: 5.82s (5.88, 5.83, 5.66, 6.23, 5.50); without x0: 8.01s (8.05, 7.28, 7.93, 9.35, 7.43)
        
        # # 方法2：自主实现 Gauss-Seidel 方法解稀疏矩阵方程组 Ax = b
        # X, info = gauss_seidel_sparse(data, rows, cols, b[:, ci], x0=x0[:, ci])
        
        # 方法3：自主实现共轭梯度法解稀疏矩阵方程组 Ax = b
        X, info = conjugate_gradient_sparse(A, b[:, ci], x0=x0[:, ci], tol=tol, max_iter=max_iter)
        """
        在第 1 个测例下的时间（其它数据同）：
        time:       6.23s (6.10, 6.78, 6.19, 5.91, 6.19)
        without x0: 6.63s (6.44, 6.99, 6.28, 7.31, 6.11)
        由于在 tol=1e-5, mas_iter=1000 条件下，是否有 x0 都未达到收敛，所以是否有 x0 的耗时应该相近
        """

        for i, (x, y) in enumerate(zip(xs, ys)):
            result[x, y, ci] = X[i]

    if verbose:
        end_time = time()
        print(f"Time taken to solve sparse system of equations: {end_time-eqa_start_time:.4f} s, entire Poisson blending: {end_time-start_time:.4f} s.")
        print(info)

    result = np.clip(result, 0, 255).astype(np.uint8)  # 这里必须先 clip 到 [0, 255] ，再转为 uint8
    return result

    # # 思路B：把所有颜色通道拼成一个 3n x 3n 的大方程组
    
    # eqa_start_time = time()
    # data = np.concatenate(tuple(data for _ in range(n_channel)))
    # rows = np.concatenate(tuple(rows + i*n for i in range(n_channel)))
    # cols = np.concatenate(tuple(cols + i*n for i in range(n_channel)))
    # b = np.concatenate(tuple(b[:, i] for i in range(n_channel)))
    # x0 = np.concatenate(tuple(x0[:, i] for i in range(n_channel)))
    # A = csr_matrix((data, (rows, cols)), shape=(n_channel*n, n_channel*n), dtype=np.float64)
    
    # # # 方法1：使用开源库 scipy.sparse.linalg.cg 函数求解稀疏矩阵方程组
    # # X, _ = cg(A, b, x0=x0)  # time: 9.34s (8.76, 10.57, 8.85, 9.15, 9.37); without x0: 12.20s (11.9, 10.95, 10.92, 14.24, 13.0)
    
    # # # 方法2：自主实现稀疏矩阵方程组求解
    # # X, info = gauss_seidel_sparse(data, rows, cols, b, x0=x0)
    
    # # 方法3：自主实现共轭梯度法解稀疏矩阵方程组 Ax = b
    # X, info = conjugate_gradient_sparse(A, b, x0=x0, tol=tol, max_iter=max_iter)
    # """
    # time:       6.95s (7.12, 6.72, 6.91, 6.81, 7.18)
    # without x0: 7.15s (7.13, 7.26, 7.83, 6.78, 6.74)
    # """
    # for i, (x, y) in enumerate(zip(xs, ys)):
    #     for ci in range(n_channel):
    #         result[x, y, ci] = X[i + ci*n]

    # if verbose:
    #     end_time = time()
    #     print(f"Time taken to solve sparse system of equations: {end_time-eqa_start_time:.4f} s, entire Poisson blending: {end_time-start_time:.4f} s.")
    #     print(info)

    # result = np.clip(result, 0, 255).astype(np.uint8)
    # return result
