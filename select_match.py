import numpy as np
import jittor as jt
from typing import Tuple, List, Deque
from PIL import Image
from collections import deque
from utils import adjacency_4


def calc_local_context(mask: np.ndarray, ctx_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算 mask 的 'local context'，用 BFS 
    """
    # Get the coordinates (x, y) where mask[x, y] == False, i.e. the pixel (x, y) is masked.
    mask_pos: List[List[int, int]] = np.argwhere(mask == False).tolist()
    queue: Deque[List[int, int]] = deque(mask_pos)
    dis: np.ndarray = np.zeros_like(mask, dtype=np.int32)  # dis[x, y] is the L1 distance between (x, y) and the masked region.

    while queue:
        x, y = queue.popleft()
        # 这个点已经到达 local context 的边界了，不能再扩展
        if dis[x, y] >= ctx_size:
            continue
        for newx, newy in adjacency_4(x, y):
            if not (0 <= newx < mask.shape[0] and 0 <= newy < mask.shape[1]):
                continue
            # 只能向非 mask 的区域去扩展
            if mask[newx, newy] == False:
                continue
            # 这个邻居节点已经访问过了
            if dis[newx, newy] > 0:
                continue
            # 把邻居节点加入队列
            dis[newx, newy] = dis[x, y] + 1
            queue.append([newx, newy])

    # 所有 (x, y) s.t. dis[x, y] > 0 构成的区域即为 local context
    is_local_ctx: np.ndarray = np.where(dis > 0, 1, 0).astype(np.uint8)
    ctx_ibound: np.ndarray = np.where(dis == 1, 1, 0).astype(np.uint8)  # local context 的内部边界
    ctx_obound: np.ndarray = np.where(dis == dis.max(), 1, 0).astype(np.uint8)  # local context 的外部边界

    return is_local_ctx, ctx_ibound, ctx_obound


def crop_scene(
    scene: np.ndarray,
    mask: np.ndarray,
    is_local_ctx: np.ndarray,
    ctx_ibound: np.ndarray,
    ctx_obound: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    """
    裁剪 scene 图像，只保留 local context 部分；对 scene 图像做 mask ，只保留 local context 部分。
    会修改传入的 scene, mask, is_local_ctx ，返回修改后的值
    """
    # 所有 (x, y) s.t. mask[x, y] == False or dis[x, y] > 0, 分别对应 mask 区域和 local context 区域  # TODO? 这里只要保留 local context 区域就可以了？
    coordinates = np.argwhere((mask == False) | (is_local_ctx > 0))
    # 用一个最小矩形框包围住这个区域
    min_x, max_x = np.min(coordinates[:, 0]), np.max(coordinates[:, 0])
    min_y, max_y = np.min(coordinates[:, 1]), np.max(coordinates[:, 1])
    # scene 只需要保留住这个区域
    scene: np.ndarray = scene[min_x: max_x+1, min_y: max_y+1]
    mask: np.ndarray = mask[min_x: max_x+1, min_y: max_y+1]
    is_local_ctx: np.ndarray = is_local_ctx[min_x: max_x+1, min_y: max_y+1]
    ctx_ibound: np.ndarray = ctx_ibound[min_x: max_x+1, min_y: max_y+1]
    ctx_obound: np.ndarray = ctx_obound[min_x: max_x+1, min_y: max_y+1]
    # # 将非 local context 的区域置 0
    # is_local_ctx: np.ndarray = is_local_ctx[:, :, np.newaxis]
    # scene: np.ndarray = scene * is_local_ctx

    """
    scene.shape = (h, w, 3)
    mask.shape = (h, w, 3)
    is_local_ctx.shape = (h, w)
    ctx_ibound.shape = (h, w)
    ctx_obound.shape = (h, w)
    其中 h, w 为裁剪后的值
    """
    return scene, mask, is_local_ctx, ctx_ibound, ctx_obound, \
            min_x, max_x, min_y, max_y


def calc_err(
    scene: np.ndarray,
    is_local_ctx: np.ndarray,
    patch: np.ndarray
):
    """
    计算 scene 和 patch 在 local context 区域的误差，使用计组实现卷积
    """
    h, w, _ = patch.shape
    kh, kw, _ = scene.shape  # assert kh, kw == new_h, new_w
    if kh > h or kw > w:
        return float('inf'), -1, -1

    # 使用 jittor 实现卷积
    patch_jt = jt.array(patch, dtype=jt.float32).reindex(
        [h - kh + 1, w - kw + 1, kh, kw, 3],
        ['i0 + i2', 'i1 + i3', 'i4']
    )
    is_local_ctx_jt = jt.array(is_local_ctx[:, :, np.newaxis].repeat(3, axis=2), dtype=jt.float32).broadcast_var(patch_jt)
    scene_jt = jt.array(scene, dtype=jt.float32).broadcast_var(patch_jt)
    error_jt = ((scene_jt - patch_jt) * is_local_ctx_jt) ** 2
    error_jt = error_jt.sum([2, 3, 4])
    error = error_jt.fetch_sync()
    # # equivalent naive implementation (without jittor convolution)
    # shape = (h - kh + 1, w - kw + 1, kh, kw, 3)
    # error = np.zeros((shape[0], shape[1]), dtype=np.float32)
    # for i0 in range(shape[0]):
    #     for i1 in range(shape[1]):
    #         for i2 in range(shape[2]):
    #             for i3 in range(shape[3]):
    #                 for i4 in range(shape[4]):
    #                     error[i0, i1] += ((patch[i0 + i2, i1 + i3, i4] - scene[i2, i3, i4]) * is_local_ctx[i2, i3, i4])**2
    
    x, y = np.unravel_index(np.argmin(error, axis=None), error.shape)

    return error[x, y], x, y


def select_pos(
    scene: np.ndarray,
    patch: np.ndarray,
    is_local_ctx: np.ndarray,
    scale_min: float,
    scale_max: float,
    verbose: bool = False
) -> np.ndarray:
    """
    将待补全区域在 patch 图上移动，选择最佳匹配位置。
    会修改传入的 `patch`，返回修改后的值。
    """
    
    result = np.zeros_like(scene)

    # 选择最佳匹配位置
    best_mean_error = None
    best_x, best_y, best_scale = None, None, None
    # for scale in [1.0]:
    for scale in np.arange(scale_min, scale_max+0.09, 0.2):
        new_h = int(patch.shape[0] / scale)
        new_w = int(patch.shape[1] / scale)
        patch_resized = np.asarray(
            Image.fromarray(patch)
            .resize((new_w, new_h))
        )

        err, x, y = calc_err(scene, is_local_ctx, patch_resized)
        if best_mean_error is None or err < best_mean_error:
            best_mean_error = err
            best_x, best_y = x, y
            best_scale = scale

            result = patch_resized[
                best_x: best_x + scene.shape[0],
                best_y: best_y + scene.shape[1]
            ]

    if verbose:
        print(f'The scaling ratio for scene images: {best_scale}.')

    return result
