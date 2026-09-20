# ===========================================================
# routing/aco.py — Thuật toán Ant Colony Optimization (ACO) cho TSP
#
# Kiến trúc bài toán giống hệt GA/PSO:
#   - coords = center_coords = [BS, CH1, CH2, ...] (index 0 = depot)
#   - Trả về (path, cost, stats) để runner.py ghi log như các thuật toán khác
#   - Dùng build_time_matrix / tour_time_from_T (numba) từ physics.py để
#     tính "khoảng cách" là THỜI GIAN di chuyển của AUV (đồng nhất với GA)
#
# TỐI ƯU TỐC ĐỘ:
#   - Phần "xây tour" (roulette wheel) và "bồi đắp pheromone" là phần tốn
#     thời gian nhất (chạy n_ants * max_iter * n_cities lần), nên được
#     compile bằng numba @njit — giống cách physics.py đang làm cho
#     travel_time / tour_time_from_T. Nhờ vậy ACO nhanh ngang GA/PSO.
# ===========================================================

import numpy as np
from numba import njit

from ..physics import build_time_matrix, tour_time_from_T


# ── Xây tour cho TOÀN BỘ đàn kiến trong 1 vòng lặp (JIT) ────
@njit(cache=True)
def _construct_tours_numba(pheromone, eta, alpha, beta, n_ants, n_cities):
    tours = np.empty((n_ants, n_cities), dtype=np.int64)

    for k in range(n_ants):
        visited = np.zeros(n_cities, dtype=np.bool_)
        visited[0] = True
        tours[k, 0] = 0
        current = 0

        for step in range(1, n_cities):
            total = 0.0
            for j in range(n_cities):
                if not visited[j]:
                    total += (pheromone[current, j] ** alpha) * (eta[current, j] ** beta)

            next_city = -1

            if total > 1e-300:
                r = np.random.random() * total
                cum = 0.0
                for j in range(n_cities):
                    if not visited[j]:
                        cum += (pheromone[current, j] ** alpha) * (eta[current, j] ** beta)
                        if cum >= r:
                            next_city = j
                            break

            if next_city == -1:
                # fallback (total suy biến hoặc sai số làm tròn): chọn ngẫu nhiên đều
                cnt = 0
                for j in range(n_cities):
                    if not visited[j]:
                        cnt += 1
                pick = np.random.randint(0, cnt)
                idx = 0
                for j in range(n_cities):
                    if not visited[j]:
                        if idx == pick:
                            next_city = j
                            break
                        idx += 1

            tours[k, step] = next_city
            visited[next_city] = True
            current = next_city

    return tours


# ── Bồi đắp pheromone cho toàn bộ đàn kiến (JIT) ─────────────
@njit(cache=True)
def _deposit_pheromone_numba(pheromone, tours, costs, Q):
    n_ants, n_cities = tours.shape
    for k in range(n_ants):
        cost = costs[k]
        if cost <= 0.0:
            continue
        deposit = Q / cost
        for idx in range(n_cities - 1):
            i = tours[k, idx]
            j = tours[k, idx + 1]
            pheromone[i, j] += deposit
            pheromone[j, i] += deposit
        i = tours[k, n_cities - 1]
        j = tours[k, 0]
        pheromone[i, j] += deposit
        pheromone[j, i] += deposit


class Aco_routing:
    """
    Ant Colony Optimization cho TSP 3D (đường đi của AUV qua các cluster head).

    - Mỗi kiến xây dựng 1 tour bắt đầu từ depot (index 0), chọn thành phố kế
      tiếp theo xác suất tỉ lệ với  (pheromone^alpha) * (heuristic^beta)
    - heuristic = 1 / thời_gian_di_chuyển  (thời gian càng nhỏ càng hấp dẫn)
    - Sau mỗi vòng lặp: bay hơi pheromone (evaporation) rồi các kiến bồi đắp
      pheromone tỉ lệ nghịch với chi phí tour
    - init_swarm (shared population, giống GA/PSO) được dùng để "mồi" thêm
      pheromone ban đầu, giúp so sánh công bằng giữa các thuật toán khi cùng
      chạy trên 1 shared population.
    """

    @staticmethod
    def aco_tsp(
        coords, v_f=1.2, v_AUV=3.0,
        n_ants=30, max_iter=100,
        alpha=1.0, beta=3.0,
        rho=0.5, Q=100.0,
        init_swarm=None,
        seed=None,
        verbose=True,
    ):
        """
        Args:
            coords: array (n, 3) — [BS, CH1, CH2, ...]
            v_f, v_AUV: tham số vật lý AUV
            n_ants: số lượng kiến mỗi vòng lặp
            max_iter: số vòng lặp
            alpha: trọng số ảnh hưởng pheromone
            beta: trọng số ảnh hưởng heuristic (nghịch đảo thời gian)
            rho: hệ số bay hơi pheromone (0..1)
            Q: hằng số bồi đắp pheromone
            init_swarm: list[list[int]] — shared population dùng để mồi pheromone
            seed: random seed (numpy global seed, ảnh hưởng cả numba njit RNG)
            verbose: in log

        Returns:
            (best_path, best_time, iter_stats)
        """
        coords = np.asarray(coords, dtype=np.float64)
        n = len(coords)

        if n < 2:
            return [0], 0.0, [{
                "step": -1, "best_time": 0.0, "avg_time": 0.0,
                "std_time": 0.0, "worst_time": 0.0,
            }]

        if n == 2:
            T = build_time_matrix(coords, v_f, v_AUV)
            t = float(tour_time_from_T(np.array([0, 1], dtype=np.int64), T))
            return [0, 1], t, [{
                "step": -1, "best_time": t, "avg_time": t,
                "std_time": 0.0, "worst_time": t,
            }]

        if seed is not None:
            np.random.seed(int(seed))

        # Ma trận thời gian (đóng vai trò ma trận "khoảng cách"), tái sử dụng
        # hàm njit sẵn có trong physics.py — evaluate cost cực nhanh
        T = build_time_matrix(coords, v_f, v_AUV)

        eps = 1e-9
        eta = 1.0 / (T + eps)          # heuristic: thời gian càng nhỏ càng hấp dẫn
        np.fill_diagonal(eta, 0.0)

        pheromone = np.ones((n, n), dtype=np.float64)

        # ── Mồi pheromone từ shared population (nếu có) ──────
        if init_swarm:
            for route in init_swarm:
                r = list(route)
                if len(r) != n or r[0] != 0:
                    continue
                cost = float(tour_time_from_T(np.array(r, dtype=np.int64), T))
                if cost <= 0:
                    continue
                deposit = Q / cost
                for k in range(len(r) - 1):
                    i, j = r[k], r[k + 1]
                    pheromone[i, j] += deposit
                    pheromone[j, i] += deposit
                i, j = r[-1], r[0]
                pheromone[i, j] += deposit
                pheromone[j, i] += deposit

        best_path = None
        best_cost = float("inf")
        iter_stats = []

        for it in range(max_iter):
            tours = _construct_tours_numba(pheromone, eta, float(alpha), float(beta), n_ants, n)

            costs = np.empty(n_ants, dtype=np.float64)
            for k in range(n_ants):
                costs[k] = tour_time_from_T(tours[k], T)

            best_idx = int(np.argmin(costs))
            if costs[best_idx] < best_cost:
                best_cost = float(costs[best_idx])
                best_path = tours[best_idx].copy()

            # bay hơi (vectorized, không cần JIT vì chỉ O(n^2) mỗi vòng)
            pheromone *= (1.0 - rho)
            np.clip(pheromone, 1e-6, None, out=pheromone)

            # bồi đắp pheromone theo chất lượng mỗi tour
            _deposit_pheromone_numba(pheromone, tours, costs, float(Q))

            iter_stats.append({
                "step": int(it),
                "best_time": float(np.min(costs)),
                "avg_time": float(np.mean(costs)),
                "std_time": float(np.std(costs)),
                "worst_time": float(np.max(costs)),
                "global_best_time": float(best_cost),
            })

            if verbose and it % 50 == 0:
                print(f"    [ACO Iter {it:3d}]: Best time = {best_cost:.4f}")

        if verbose:
            print(f"    [ACO Iter {max_iter:3d}]: Final Best time = {best_cost:.4f}")

        return best_path.tolist(), float(best_cost), iter_stats

    # ── Wrapper nhiều outer-loop, cùng interface với PSO ────
    @staticmethod
    def multi_aco_tsp(
        coords, v_f=1.2, v_AUV=3.0,
        n_outer=1, verbose=True,
        init_swarm=None,
        **kwargs,
    ):
        prev_path, prev_cost = None, float("inf")
        best_stats = None

        for outer in range(1, n_outer + 1):
            if verbose:
                print(f"   [ACO Outer loop {outer}/{n_outer}]")

            path, cost, stats = Aco_routing.aco_tsp(
                coords,
                v_f=v_f,
                v_AUV=v_AUV,
                init_swarm=init_swarm,
                verbose=verbose,
                **kwargs,
            )

            if cost < prev_cost:
                prev_path, prev_cost = path, float(cost)
                best_stats = stats

        return prev_path, float(prev_cost), best_stats