import argparse
import os
import numpy as np
import jittor as jt
import matplotlib.pyplot as plt
from typing import Tuple
from PIL import Image
from pathlib import Path
from select_match import calc_local_context, crop_scene, select_pos
from graph_cut import graph_cut, graph_cut_my
from poisson_blending import poisson_blending
from mvc_blending import mvc_blending

jt.flags.use_cuda = jt.has_cuda
print("jt.flags.use_cuda =", jt.flags.use_cuda)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True, help='The scene image to be complemented.')
    parser.add_argument("--mask", type=str, required=True, help='The mask image, indicating the area to be complemented.')
    parser.add_argument("--patch", type=str, required=True, help='The patch image serving as a source in the completion.')
    parser.add_argument("--result", type=str, required=True, help='The path to save the result file.')
    parser.add_argument("--ctx_size", type=int, default=80, help='The size of local context, default=80.')
    parser.add_argument("--scale_min", type=float, default=1.0, help='The minimum of the scaling ratio for scene images.')
    parser.add_argument("--scale_max", type=float, default=1.1, help='The maximum of the scaling ratio for scene images.')
    parser.add_argument("--inter_result", type=str, default='', help='The path to save the intermediate results. If this option is not provided, the intermediate results will not be saved.')
    parser.add_argument("--maxflow", action='store_true', help='Use "maxflow" package to calculate graph-cut.')
    parser.add_argument("--my_graphcut_max_bfs", type=int, default=2000, help='In my implementation of graph-cut, the maximum number of BFS.')
    parser.add_argument("--mvc", action='store_true', help='Use MVC.')
    parser.add_argument("--tol", type=float, default=1e-5, help='The maximum number of iterations when solving a system of linear equations')
    parser.add_argument("--max_iter", type=int, default=1000, help='The maximum number of iterations when solving a system of linear equations')
    parser.add_argument("--verbose", action='store_true', help='Print intermediate information during program execution.')
    args = parser.parse_args()
    return args


def load_images(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert os.path.exists(args.scene)
    assert os.path.exists(args.mask)
    assert os.path.exists(args.patch)
    scene: np.ndarray = np.asarray(Image.open(args.scene), dtype=np.uint8)
    mask: np.ndarray = np.asarray(Image.open(args.mask).convert('1'))  # mask(False), non-mask(True)
    patch: np.ndarray = np.asarray(Image.open(args.patch), dtype=np.uint8)

    return scene, mask, patch


def main():
    # step 0: 导入图片
    args = parse_args()
    scene, mask, patch = load_images(args)
    result = scene.copy()
    if args.inter_result:
        print(f'The intermediate results will be saved to "{args.inter_result}".')
        dir = Path(args.inter_result) / 'step0'
        os.makedirs(dir, exist_ok=True)
        plt.imsave(dir / 'mask.png', mask, cmap='gray')


    # step 1: 精细匹配
    is_local_ctx, ctx_ibound, ctx_obound = calc_local_context(mask, args.ctx_size)
    scene, mask, is_local_ctx, ctx_ibound, ctx_obound, min_x, max_x, min_y, max_y = \
        crop_scene(scene, mask, is_local_ctx, ctx_ibound, ctx_obound)
    patch = select_pos(scene, patch, is_local_ctx, args.scale_min, args.scale_max, verbose=args.verbose)

    if args.inter_result:
        dir = Path(args.inter_result) / 'step1'
        os.makedirs(dir, exist_ok=True)
        plt.imsave(dir / 'patch.png', patch)
        plt.imsave(dir / 'scene.png', scene)
        plt.imsave(dir / 'scene_local_ctx.png', scene * is_local_ctx[:, :, np.newaxis])
        plt.imsave(dir / 'mask.png', mask, cmap='gray')
        plt.imsave(dir / 'is_local_ctx.png', np.where(is_local_ctx > 0, 255, 0).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2), cmap='gray')  # 0 - 黑，255 - 白


    # step 2: 计算融合边界
    if args.verbose:
        print('Step 2 graph-cut: using ' +
            ('"maxflow" package.' if args.maxflow else f'my implementation (max_bfs={args.my_graphcut_max_bfs}).'))
    if args.maxflow:
        # 方法 1：使用开源库计算 Graph-cut
        cut_result: np.ndarray = graph_cut(
            scene, patch, ctx_ibound, ctx_obound
        )  # 指示每个像素是否将要使用 patch，是(0)，否(1)
    else:
        # 方法 2：自主实现计算 Graph-cut
        cut_result: np.ndarray = graph_cut_my(
            scene, patch, is_local_ctx, mask, ctx_ibound, ctx_obound, max_bfs=args.my_graphcut_max_bfs
        )  # 指示每个像素是否将要使用 patch，是(0)，否(1)

    if args.inter_result:
        dir = Path(args.inter_result) / 'step2'
        os.makedirs(dir, exist_ok=True)
        plt.imsave(dir / 'cut_result.png', np.where(cut_result == True, 255, 0).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2), cmap='gray')
        # plt.imsave(dir / 'ctx_ibound.png', np.where(ctx_ibound, 255, 0).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2), cmap='gray')
        # plt.imsave(dir / 'ctx_obound.png', np.where(ctx_obound, 255, 0).astype(np.uint8)[:, :, np.newaxis].repeat(3, axis=2), cmap='gray')
        plt.imsave(dir / 'patch.png', np.where(cut_result[:, :, np.newaxis] == True, 0, patch))


    # step 3: 自然融合两张图像
    if args.verbose:
        print('Step 3 blending: using ' + ('MVC.' if args.mvc else 'Poisson blending.'))
    if not args.mvc:
        # 方法 1：泊松融合，然后转化为解稀疏矩阵方程组
        patch_result = poisson_blending(
            1-cut_result, scene, patch, \
            tol=args.tol, max_iter=args.max_iter, verbose=args.verbose
        )
    else:
        # 方法 2：使用 MVC 
        patch_result = mvc_blending(
            1-cut_result, scene, patch, mask, \
            verbose=args.verbose, inter_result=args.inter_result
        )

    result[
        min_x: max_x + 1,
        min_y: max_y + 1
    ] = np.where(cut_result[:, :, np.newaxis] == True, scene, patch_result)
    plt.imsave(args.result, result)

    if args.inter_result:
        dir = Path(args.inter_result) / 'step3'
        os.makedirs(dir, exist_ok=True)
        plt.imsave(dir / 'patch_result.png', patch_result.astype(np.uint8))


if __name__ == '__main__':
    main()
