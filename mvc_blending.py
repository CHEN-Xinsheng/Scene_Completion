import cv2
import math
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from time import time


def clip(a, min, max):
    if a < min:
        return min
    elif a > max:
        return max
    else:
        return a


def angle(a, p, b):
    """
    对平面上三点 apb ，计算 ∠apb ，结果以弧度给出
    """
    # 计算向量AP和向量BP的坐标
    ap = (p[0] - a[0], p[1] - a[1])
    bp = (p[0] - b[0], p[1] - b[1])

    # 计算向量AP和向量BP的长度
    length_ap = math.sqrt(ap[0] ** 2 + ap[1] ** 2)
    length_bp = math.sqrt(bp[0] ** 2 + bp[1] ** 2)

    # 计算向量AP和向量BP的点积
    dot_product = ap[0] * bp[0] + ap[1] * bp[1]

    # 计算夹角的弧度
    angle_rad = math.acos(clip(dot_product / (length_ap * length_bp), 0, 1))

    # 有向角
    if ap[0] * bp[1] - ap[1] * bp[0] < 0:
        angle_rad = 2*math.pi -angle_rad

    return angle_rad


def distance(a, b):
    """
    计算平面上两点的距离
    """
    return math.sqrt(
        (a[0] - b[0])**2 + (a[1] - b[1])**2
    )


def mvc_blending(
    patch_area: np.ndarray,  # 指示每个像素是否使用 patch，是(1)，否(0)
    scene: np.ndarray,
    patch: np.ndarray,
    mask: np.ndarray,        # 是 scene 中缺失的区域(False)，是 scene 中有内容的区域(True)
    verbose: bool = False,
    inter_result: str = ''
) -> np.ndarray:
    """
    使用 Mean-Value Coordinates 来把 patch 融合到 scene 的某个区域内（该区域由 patch_area 指示）。
    
    patch_area: shape = (h, w)
    scene: shape = (h, w, 3)
    patch: shape = (h, w, 3)，其中 h, w 为裁剪后的大小
    """

    start_time = time()

    scene = scene.astype(np.int32)
    patch = patch.astype(np.int32)

    # 先计算出 patch 的边界
    contours, _ = cv2.findContours(patch_area.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 将轮廓的坐标提取出来
    boundary_coors = contours[0][:, 0, :].tolist()
    # # 删除其中位于缺失区域的点
    # id_to_delete = []
    # # for col, row in boundary_coors:
    # for i in range(len(boundary_coors)):
    #     col, row = boundary_coors[i]
    #     if mask[row, col] == False:  # 是 scene 中缺失的区域(False)
    #         id_to_delete.append(i)
    # for i in id_to_delete:
    #     # np.delete(boundary_coors, i)
    #     del boundary_coors[i]

    boundary_corrs_set = {(col, row) for col, row in boundary_coors}
    
    if inter_result:
        dir = Path(inter_result) / 'step3'
        os.makedirs(dir, exist_ok=True)

        boundary_np = np.zeros(shape=scene.shape, dtype=np.uint8)
        for col, row in boundary_coors:
            boundary_np[row, col] = 255
        plt.imsave(dir / 'boundary.png', boundary_np)

    residual = np.zeros(shape=patch.shape, dtype=np.int32)

    # 计算所有边界点的 diff
    diff = np.array([
        scene[row, col] - patch[row, col] for col, row in boundary_coors
    ]).astype(np.int32)

    # 计算每个边界像素 p 融合后的值
    # i_debug = 0   # [debug]
    # cnt_debug = 0   # [debug]
    for p in np.argwhere(patch_area == 1):
        # i_debug += 1   # [debug]
        # if i_debug % 5000 == 0:  # [debug]
        #     print(i_debug, f'doing MVC..., total = {len(np.argwhere(patch_area == 1))}')

        # if [p[1], p[0]] in boundary_coors:
        if (p[1], p[0]) in boundary_corrs_set:
            # residual[p[0], p[1]] = scene[p[0], p[1]] - patch[p[0], p[1]]
            residual[p[0], p[1]] = np.array([0, 0, 0])
            # cnt_debug += 1   # [debug]
            continue

        w = np.zeros(shape=len(boundary_coors))
        for i in range(len(boundary_coors)):
            a = boundary_coors[i - 1]
            b = boundary_coors[i]
            c = boundary_coors[i + 1] if i + 1 < len(boundary_coors) else boundary_coors[0]

            # w_i = ( tan(∠apb /2) + tan(∠bpc /2) ) / ||b-p||
            pp = [p[1], p[0]]
            w[i] = (math.tan(angle(a, pp, b) / 2) + math.tan(angle(b, pp, c) / 2)) / distance(b, pp)

        lambd = w / w.sum()

        residual[p[0], p[1]] = np.sum(
            lambd[:, np.newaxis] * diff,
            axis=0
        )

    # print(f"cnt_debug = {cnt_debug}, len(boundary_coors) = {len(boundary_coors)}")    # [debug]
    if verbose:
        print(f"Time taken in MVC blending: {time()-start_time:.4f} s.")

    return np.clip(patch + residual, 0, 255).astype(np.uint8)
