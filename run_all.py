import os
import subprocess
import argparse
from pathlib import Path
from time import time
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help='Print all information during subprocess execution and save all intermediate results.')
    parser.add_argument('-m', "--maxflow", action='store_true', help='Use "maxflow" package to calculate graph-cut.')
    args = parser.parse_args()
    return args


def run(scene, mask, patch, result, inter_result='', verbose=False, maxflow=False):
    # 检查要读取的文件是否存在
    for dir in [scene, mask, patch]:
        if not os.path.exists(dir):
            print(f'The input path {dir} is invalid, there is no such file.')
            exit()
    # 创建将要写的文件
    for dir in [result, inter_result]:
        os.mkdir(dir, exist_ok=True)

    cmd = ['python', 'main.py', '--scene', str(scene), '--mask', str(mask), '--patch', str(patch), '--result', str(result)]
    if verbose:
        cmd += ['--inter_result', str(inter_result), '--verbose']
    if maxflow:
        cmd += ['--maxflow']
    print('running ', cmd)

    start_time = time()
    if verbose:
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(f'OK, execution time = {time()-start_time:.2f} s.')
    print('===========================================================')


if __name__ == '__main__':
    args = parse_args()

    input = Path('data') / 'completion'
    result = Path('result')

    if not os.path.exists(input):
        print(f'The input path {input} is invalid, there is no such path.')
        exit()

    for i in range(1, 6):
        run(
            scene=input / f'input{i}.jpg',
            mask=input / f'input{i}_mask.jpg',
            patch=input / f'input{i}_patch.jpg',
            result=result / f'{i}.jpg',
            inter_result=result / f'inter_{i}',
            verbose=args.verbose,
            maxflow=args.maxflow
        )
