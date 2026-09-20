# ===========================================================
# utils.py — Hàm tiện ích: logging, build coords, shared pop
# ===========================================================

import json
import numpy as np


def jsonl_append(path, obj):
    """Ghi thêm 1 dòng JSON vào file JSONL."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def compute_total_energy(all_nodes):
    """Tính tổng năng lượng còn lại của tất cả node."""
    return float(sum(all_nodes[n]["residual_energy"] for n in all_nodes))


def build_center_coords(clusters, node_positions, bs=(200, 200, 400)):
    """
    Xây dựng mảng tọa độ [BS, CH1, CH2, ...] cho bài toán TSP.

    Args:
        clusters: dict - thông tin các cluster
        node_positions: dict - vị trí các node
        bs: tuple - vị trí base station

    Returns:
        np.ndarray (n x 3) - tọa độ BS + các cluster head
    """
    sorted_keys = sorted(clusters.keys())
    centers = [tuple(bs)]
    for k in sorted_keys:
        ch = clusters[k]["cluster_head"]
        centers.append(tuple(node_positions[ch]))
    return np.array(centers, dtype=np.float64)


def make_shared_population(n_cities, pop_size, seed=0):
    """
    Tạo shared population/swarm cho GA & PSO:
    - list[list[int]]
    - Mỗi route chứa 0..n_cities-1 đúng 1 lần
    - Depot 0 ở vị trí đầu

    Args:
        n_cities: int - số thành phố (bao gồm depot)
        pop_size: int - kích thước population
        seed: int - random seed

    Returns:
        list[list[int]] - population
    """
    rng = np.random.default_rng(int(seed))
    base = np.arange(1, n_cities, dtype=np.int64)

    pop = []
    for _ in range(int(pop_size)):
        perm = base.copy()
        rng.shuffle(perm)
        pop.append([0] + perm.tolist())
    return pop
