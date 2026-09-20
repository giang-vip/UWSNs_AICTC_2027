# ===========================================================
# physics.py — Hàm @njit: vận tốc, thời gian di chuyển AUV
# ===========================================================

import math
import numpy as np
from numba import njit


@njit
def compute_vs(p1, p2, v_f, v_AUV):
    """
    Tính vận tốc thực tế (resultant speed) của AUV di chuyển từ p1 đến p2,
    có tính đến ảnh hưởng dòng chảy (heave velocity v_f dọc trục +z).
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    if L == 0.0:
        return v_AUV

    # beta = angle(L, +z) => cos(beta) = dz / L
    cosb = dz / L
    if cosb > 1.0:
        cosb = 1.0
    elif cosb < -1.0:
        cosb = -1.0

    # v_s = v_f*cos(beta) + sqrt(v_AUV^2 - v_f^2 * sin^2(beta))
    sin2 = 1.0 - cosb * cosb
    sqrt_term = v_AUV * v_AUV - v_f * v_f * sin2

    if sqrt_term < 0.0:
        sqrt_term = 0.0

    v_s = v_f * cosb + math.sqrt(sqrt_term)

    if v_s < 1e-9:
        v_s = 1e-9

    return v_s


@njit
def travel_time(path, coords, v_f, v_AUV):
    """
    Tính tổng thời gian di chuyển theo đường đi cho trước (bao gồm quay về).

    Args:
        path: array - danh sách các index của các điểm theo thứ tự
        coords: array (n x 3) - mảng tọa độ các điểm
        v_f: float - vận tốc dòng chảy
        v_AUV: float - vận tốc của AUV

    Returns:
        float - tổng thời gian di chuyển (bao gồm quay về điểm xuất phát)
    """
    total_time = 0.0
    n = len(path)
    if n <= 1:
        return 0.0

    # Các cạnh trong chu trình
    for i in range(n - 1):
        i1 = path[i]
        i2 = path[i + 1]

        p1 = coords[i1]
        p2 = coords[i2]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)

        v_s = compute_vs(p1, p2, v_f, v_AUV)
        if v_s < 1e-9:
            v_s = 1e-9

        total_time += d / v_s

    # Quay về điểm bắt đầu
    p1 = coords[path[-1]]
    p2 = coords[path[0]]

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    d = math.sqrt(dx * dx + dy * dy + dz * dz)

    v_s = compute_vs(p1, p2, v_f, v_AUV)
    if v_s < 1e-9:
        v_s = 1e-9

    total_time += d / v_s

    return total_time


@njit
def build_time_matrix(coords, v_f, v_AUV):
    """
    Xây dựng ma trận thời gian di chuyển giữa tất cả các cặp điểm.
    T[i, j] = thời gian di chuyển từ điểm i đến điểm j.
    """
    n = coords.shape[0]
    T = np.empty((n, n), dtype=np.float64)

    for i in range(n):
        T[i, i] = 0.0
        x1, y1, z1 = coords[i, 0], coords[i, 1], coords[i, 2]

        for j in range(n):
            if i == j:
                continue
            dx = coords[j, 0] - x1
            dy = coords[j, 1] - y1
            dz = coords[j, 2] - z1

            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            if d == 0.0:
                T[i, j] = 0.0
                continue

            cosb = dz / d
            if cosb > 1.0:
                cosb = 1.0
            elif cosb < -1.0:
                cosb = -1.0

            sin2 = 1.0 - cosb * cosb
            sqrt_term = v_AUV * v_AUV - v_f * v_f * sin2
            if sqrt_term < 0.0:
                sqrt_term = 0.0

            v_s = v_f * cosb + math.sqrt(sqrt_term)
            if v_s < 1e-9:
                v_s = 1e-9

            T[i, j] = d / v_s

    return T


@njit
def tour_time_from_T(path, T):
    """
    Tính thời gian tour từ ma trận thời gian đã tính trước.
    """
    n = path.shape[0]
    if n <= 1:
        return 0.0

    s = 0.0
    for k in range(n - 1):
        s += T[path[k], path[k + 1]]
    s += T[path[n - 1], path[0]]
    return s
