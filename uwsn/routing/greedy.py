# ===========================================================
# routing/greedy.py — Thuật toán Greedy TSP
# ===========================================================

import numpy as np

from ..physics import travel_time


class Greedy:
    @staticmethod
    def greedy_tsp(coords, v_f=1.2, v_AUV=3.0):
        """
        Greedy nearest-neighbor TSP.

        Args:
            coords: array (n, 3) - đã bao gồm BS ở index 0
            v_f: float - vận tốc dòng chảy
            v_AUV: float - vận tốc AUV

        Returns:
            (path, best_time) — path bắt đầu và kết thúc bằng 0
        """
        coords = np.asarray(coords, dtype=np.float64)
        n = len(coords)
        if n < 2:
            return list(range(n)), 0.0

        remaining = set(range(1, n))
        path = [0]
        current = 0
        total_time = 0.0

        while remaining:
            best_next = None
            best_dt = float("inf")

            for nxt in remaining:
                dt = travel_time(
                    np.array([current, nxt], dtype=np.int64), coords, v_f, v_AUV
                )
                if dt < best_dt:
                    best_dt = dt
                    best_next = nxt

            path.append(best_next)
            total_time += best_dt
            remaining.remove(best_next)
            current = best_next

        # Quay về BS
        total_time += travel_time(
            np.array([current, 0], dtype=np.int64), coords, v_f, v_AUV
        )
        path.append(0)

        return path, float(total_time)
