import maxflow
import numpy as np
from collections import deque
from utils import adjacency_4


def graph_cut(
    scene: np.ndarray,
    patch: np.ndarray,
    ctx_ibound: np.ndarray,
    ctx_obound: np.ndarray
) -> np.ndarray:
    """
    用 Graph-cut 计算融合边界，调用 maxflow
    """

    # 建图
    g = maxflow.GraphFloat()
    nodeids = g.add_grid_nodes((scene.shape[0], scene.shape[1]))  # assert scene.shape == patch.shape
    weights = np.sqrt(np.sum((scene - patch)**2, axis=-1))
    # 两个像素 s, t 之间的边权 = ||A(s)-B(s)|| + ||A(t)-B(t)||，其中 A, B 为两幅图像
    # 先计算所有 ||A(s)-B(s)||
    g.add_grid_edges(nodeids, weights=weights, structure=np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]), symmetric=True)
    # 再加上所有 ||A(t)-B(t)||
    g.add_grid_edges(nodeids, weights=weights, structure=np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]), symmetric=True)
    FLOAT_INF = 1e20
    g.add_grid_tedges(nodeids, sourcecaps=ctx_ibound*FLOAT_INF, sinkcaps=ctx_obound*FLOAT_INF)

    # 计算最大流，即为最小割
    g.maxflow()
    cut_result: np.ndarray = g.get_grid_segments(nodeids)

    return cut_result  # shape = (h, w), dtype = dtype('bool')


class Graph:
    def __init__(self, n_vertices, source, sink, max_neighbors=5):
        self.V = n_vertices
        self.source = source
        self.sink = sink
        self.max_neighbors = max_neighbors
        
        self.edge = [{} for _ in range(self.V)]  # edge[u][v] = [capacity, flow]
        self.a = [0.0 for _ in range(n_vertices)]

    def add_edge(self, u, v, capacity):
        if v not in self.edge[u]:
            self.edge[u][v] = [capacity, 0.0]
        else:
            self.edge[u][v][0] += capacity
        
        if u not in [self.source, self.sink]:  # 除了 source 和 sink 节点之外，每个节点的邻居数有上限
            assert len(self.edge[u]) <= self.max_neighbors 

    def add_edge_bidirect(self, u, v, capacity):
        self.add_edge(u, v, capacity)
        self.add_edge(v, u, capacity)

    def bfs(self, thereshold):
        parent = [-1 for _ in range(self.V)]  # -1 表示该节点还没有被 BFS 到
        visited = [False for _ in range(self.V)]
        positive_node = [False for _ in range(self.V)]

        q = deque()
        q.append(self.source)
        parent[self.source] = -9  #  != -1
        self.a[self.source] = float('inf')

        while q and parent[self.sink] == -1:
            b = q.popleft()
            visited[b] = True

            for neighbor, (c, f) in self.edge[b].items():
                if visited[neighbor]:
                    continue
                # with open('a.txt', 'a') as file:  # [debug]
                #     file.write(f'f-c = {f-c}\n')
                if c-f > thereshold and parent[neighbor] == -1:
                    q.append(neighbor)
                    parent[neighbor] = b
                    positive_node[neighbor] = True
                    self.a[neighbor] = min(self.a[b], c - f)
                
                if b in self.edge[neighbor]:
                    _, f2 = self.edge[neighbor][b]
                    # with open('a.txt', 'a') as file:  # [debug]
                    #     file.write(f'f2 = {f2}\n')
                    if f2 > thereshold and parent[neighbor] == -1:
                        q.append(neighbor)
                        parent[neighbor] = b
                        positive_node[neighbor] = False
                        self.a[neighbor] = min(self.a[b], f2)

        return parent, positive_node

    def min_cut(self, max_bfs):
        # i_debug = 0  # [debug]
        cnt = 0  # 增流次数
        thereshold = 35

        while True and cnt < max_bfs:
            cnt += 1

            parent, positive_node = self.bfs(thereshold)

            # i_debug += 1  # [debug]
            # if i_debug % 20 == 0:
            #     with open('a.txt', 'a') as f:  # [debug]
            #         f.write(f'{i_debug} bfs finished\n')
            # print(i_debug, 'bfs finished') # [debug]

            if parent[self.sink] != -1:
                i = self.sink
                while i != self.source:
                    if positive_node[i]:
                        self.edge[parent[i]][i][1] += self.a[self.sink]
                        # with open('a.txt', 'a') as f:  # [debug]
                        #     f.write(f"self.edge[{parent[i]}][sink={i}][1] += {self.a[self.sink]}\n")
                    else:
                        self.edge[i][parent[i]][1] -= self.a[self.sink]
                        # with open('a.txt', 'a') as f:  # [debug]
                        #     f.write(f"self.edge[sink={i}][{parent[i]}][1] -= {self.a[self.sink]}\n")
                    i = parent[i]
            elif cnt < max_bfs:
                thereshold *= 0.75
                continue
            else:
                # with open('a.txt', 'a') as f:  # [debug]
                #     f.write(f"parent[self.sink] == -1: break\n")
                break

        # 标记每个点属于哪一边
        print('begin dfs_mark of graph_cut')
        cut = [False] * self.V

        self.dfs_mark(self.source, cut, 6)

        return np.array(cut)

    def dfs_mark(self, v, cut, thereshold):
        """
        找到所有和 v（初始为 source）同一边的点。
        """
        cut[v] = True

        for neighbor, (c, f) in self.edge[v].items():
            if c-f > thereshold and not cut[neighbor]:
                self.dfs_mark(neighbor, cut, thereshold)


def graph_cut_my(
    scene: np.ndarray,         # dtype: uint8
    patch: np.ndarray,         # dtype: uint8
    is_local_ctx: np.ndarray,  # dtype: uint8
    mask: np.ndarray,          # dtype: bool
    ctx_ibound: np.ndarray,    # dtype: uint8
    ctx_obound: np.ndarray,    # dtype: uint8
    max_bfs: int
) -> np.ndarray:
    """
    用 Graph-cut 计算融合边界，自主实现
    """

    # 建图
    # 统计所有 is_local_ctx == 1 的像素个数，即为 local context 的大小
    n = int(is_local_ctx.sum())
    # n 个像素，加上 2 个虚拟节点（编号从 1 开始使用）
    node_ids = np.zeros(shape=is_local_ctx.shape, dtype=np.int32)
    for i, (x, y) in enumerate(np.argwhere(is_local_ctx == 1)):
        node_ids[x, y] = i+1
    source_id, sink_id = n+1, n+2

    g = Graph(n+3, source=source_id, sink=sink_id, max_neighbors=5)

    weights = np.sqrt(np.sum((scene - patch)**2, axis=-1))
    # 两个像素 s, t 之间的边权 = ||A(s)-B(s)|| + ||A(t)-B(t)||，其中 A, B 为两幅图像
    for x in range(scene.shape[0]):
        for y in range(scene.shape[1]):
            if is_local_ctx[x, y] == 1:
                for qx, qy in adjacency_4(x, y):
                    if not (0 <= qx < scene.shape[0] and 0 <= qy < scene.shape[1]):
                        continue
                    if not (is_local_ctx[qx, qy] == 1):
                        continue
                    g.add_edge_bidirect(node_ids[x, y], node_ids[qx, qy], weights[x, y])

    # 再加入和 source, sink 之间的边
    for x, y in np.argwhere(ctx_obound == 1):
        # assert is_local_ctx[x, y] == 1
        g.add_edge(source_id, node_ids[x, y], 1e18)
    for x, y in np.argwhere(ctx_ibound == 1):
        # assert is_local_ctx[x, y] == 1
        g.add_edge(node_ids[x, y], sink_id, 1e18)

    # 计算最小割
    cut = g.min_cut(max_bfs)

    # cut[idx] 记录的是 local context 中第 idx 个像素属于哪一边，需要再转换成 (h, w)
    # patch 区域包括两部分：一是 mask 区域，所以先给 mask 区域标记上 False
    result = mask.copy()  # 0 - mask 区域, 1 - 非 mask 区域
    # 二是 local context 区域中被划入 patch 的部分，已经记录在 cut 中
    for i, (x, y) in enumerate(np.argwhere(is_local_ctx == 1)):
        result[x, y] = cut[i]

    return result
