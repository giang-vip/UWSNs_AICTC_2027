# ===========================================================
# routing/pso_2opt.py — Memetic PSO (PSO + Adaptive Noise + 2-Opt ATSP)
# Thuật toán lai ghép đề xuất cho bài báo/đồ án
# ===========================================================

import random
import numpy as np
from numba import njit

from ..physics import build_time_matrix, tour_time_from_T
from ._base import get_swap_sequence, apply_velocity, validate_and_clone_swarm

# ── THUẬT TOÁN 2-OPT CHO ATSP (Được biên dịch C bằng Numba) ──
@njit(cache=True)
def two_opt_atsp_numba(path, T):
    """
    Thuật toán 2-Opt chuyên dụng cho bài toán Asymmetric TSP (Có dòng chảy).
    Đảm bảo giữ nguyên điểm bắt đầu (Depot = 0).
    """
    n = len(path)
    improved = True
    best_path = path.copy()
    
    # Tính chi phí ban đầu
    best_cost = 0.0
    for i in range(n - 1):
        best_cost += T[best_path[i], best_path[i+1]]
    best_cost += T[best_path[n-1], best_path[0]]
    
    while improved:
        improved = False
        # Bỏ qua index 0 vì Depot luôn phải nằm ở đầu
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                nxt_j = (j + 1) % n
                
                delta = 0.0
                
                # 1. Trừ đi chi phí của các cạnh cũ
                delta -= T[best_path[i-1], best_path[i]]
                delta -= T[best_path[j], best_path[nxt_j]]
                for k in range(i, j):
                    delta -= T[best_path[k], best_path[k+1]]
                    
                # 2. Cộng thêm chi phí của các cạnh mới (khi đoạn [i:j] bị đảo ngược)
                delta += T[best_path[i-1], best_path[j]]
                delta += T[best_path[i], best_path[nxt_j]]
                for k in range(j, i, -1):
                    delta += T[best_path[k], best_path[k-1]]
                    
                # Nếu tiết kiệm được thời gian đáng kể
                if delta < -1e-6: 
                    # Đảo ngược đoạn [i:j]
                    left = i
                    right = j
                    while left < right:
                        temp = best_path[left]
                        best_path[left] = best_path[right]
                        best_path[right] = temp
                        left += 1
                        right -= 1
                        
                    best_cost += delta
                    improved = True
                    
    return best_path, best_cost


class Pso_2opt:
    """
    Memetic PSO: Kết hợp Adaptive Noise và Local Search (2-Opt) cho ATSP.
    """

    @staticmethod
    def pso_tsp_3d_time_2opt(
        coords, v_f=1.0, v_AUV=3.0,
        n_particles=50, max_iter=200,
        w=0.5, c1=0.25, c2=0.25,
        init_gbest=None,
        init_swarm=None,
        verbose=True
    ):
        n_cities = len(coords)
        if n_cities < 2:
            stats = [{
                "step": 0, "best_time": 0.0, "avg_time": 0.0,
                "std_time": 0.0, "worst_time": 0.0, "noise": 0.0
            }]
            return [0], 0.0, stats

        # Tính toán Time Matrix 1 lần duy nhất để tối ưu tốc độ cực đại
        T_mat = build_time_matrix(coords, v_f, v_AUV)

        if n_cities == 2:
            path_arr = np.array([0, 1], dtype=np.int64)
            tcost = float(tour_time_from_T(path_arr, T_mat))
            stats = [{
                "step": 0, "best_time": tcost, "avg_time": tcost,
                "std_time": 0.0, "worst_time": tcost, "noise": 0.0
            }]
            return [0, 1], tcost, stats

        # INIT SWARM
        cities = list(range(1, n_cities))
        if init_swarm is not None:
            swarm = validate_and_clone_swarm(init_swarm, n_cities)
            n_particles = len(swarm)
        else:
            swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]

        velocities = [[] for _ in range(n_particles)]

        # Đánh giá quần thể ban đầu
        costs = []
        for p in swarm:
            p_arr = np.array(p, dtype=np.int64)
            costs.append(tour_time_from_T(p_arr, T_mat))
            
        pbest = [p.copy() for p in swarm]
        pbest_cost = costs.copy()

        # Xác định gbest
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = init_gbest.copy()
            gbest_cost = float(tour_time_from_T(np.array(gbest, dtype=np.int64), T_mat))
        else:
            best_idx = int(np.argmin(pbest_cost))
            gbest = pbest[best_idx].copy()
            gbest_cost = float(pbest_cost[best_idx])

        # ── ADAPTIVE NOISE SETUP ──
        no_improve = 0
        last_best = gbest_cost
        p_noise = 0.1
        max_noise_swaps = 2
        
        # Xác suất áp dụng 2-Opt để cân bằng thời gian chạy (Memetic Rate)
        p_local_search = 0.3 

        iter_stats = []
        iter_stats.append({
            "step": -1,
            "best_time": float(gbest_cost),
            "avg_time": float(np.mean(costs)),
            "std_time": float(np.std(costs)),
            "worst_time": float(np.max(costs)),
            "noise": float(p_noise),
        })

        for t in range(max_iter):
            inertia = 0.7 - 0.5 * (t / max_iter)

            for i in range(n_particles):
                xi, vi = swarm[i], velocities[i]
                v_new = vi[:int(inertia * len(vi))]

                # Hướng về pbest
                if random.random() < c1:
                    seq_pb = get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))

                # Hướng về gbest
                if random.random() < c2:
                    seq_gb = get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))

                # ── BƠM NHIỄU (ADAPTIVE NOISE) ──
                if random.random() < p_noise:
                    k = random.randint(1, max_noise_swaps)
                    for _ in range(k):
                        i1, i2 = random.sample(range(1, n_cities), 2)
                        v_new.append((i1, i2))

                # Áp dụng di chuyển
                new_x = apply_velocity(xi, v_new)
                new_x_arr = np.array(new_x, dtype=np.int64)
                
                # ── VŨ KHÍ BÍ MẬT: LOCAL SEARCH (2-OPT) ──
                if random.random() < p_local_search:
                    new_x_arr, new_cost = two_opt_atsp_numba(new_x_arr, T_mat)
                    new_x = new_x_arr.tolist()
                else:
                    new_cost = tour_time_from_T(new_x_arr, T_mat)

                swarm[i], velocities[i] = new_x, v_new
                costs[i] = new_cost

                # Cập nhật pbest và gbest
                if new_cost < pbest_cost[i]:
                    pbest[i], pbest_cost[i] = new_x, new_cost
                    if new_cost < gbest_cost:
                        gbest, gbest_cost = new_x, float(new_cost)

            # ── CẬP NHẬT TỶ LỆ NHIỄU ──
            if gbest_cost >= last_best - 1e-6:
                no_improve += 1
            else:
                no_improve = 0
                last_best = gbest_cost

            if no_improve > 15:
                p_noise = min(0.4, p_noise * 1.3)
            else:
                p_noise = max(0.05, p_noise * 0.95)

            iter_stats.append({
                "step": int(t),
                "best_time": float(gbest_cost),
                "avg_time": float(np.mean(costs)),
                "std_time": float(np.std(costs)),
                "worst_time": float(np.max(costs)),
                "noise": float(p_noise),
            })

            if verbose and t % 50 == 0:
                print(f"    [PSO3-Memetic Iter {t:3d}]: Best time = {gbest_cost:.4f}, Noise = {p_noise:.3f}")

        if verbose:
            print(f"    [PSO3-Memetic Iter {max_iter:3d}]: Final Best time = {gbest_cost:.4f}")

        return gbest, float(gbest_cost), iter_stats

    @staticmethod
    def multi_pso_tsp(
        coords, v_f=1.2, v_AUV=3.0,
        n_outer=5, verbose=True,
        init_swarm=None,
        **kwargs
    ):
        prev_gbest, prev_cost = None, float("inf")
        best_stats = None

        for outer in range(1, n_outer + 1):
            if verbose:
                print(f"   [PSO3-Memetic Outer loop {outer}/{n_outer}]")

            gbest, cost, stats = Pso_2opt.pso_tsp_3d_time_2opt(
                coords,
                v_f=v_f,
                v_AUV=v_AUV,
                init_gbest=prev_gbest,
                init_swarm=init_swarm,
                verbose=verbose,
                **kwargs,
            )

            if cost < prev_cost:
                prev_gbest, prev_cost = gbest, float(cost)
                best_stats = stats

        return prev_gbest, float(prev_cost), best_stats