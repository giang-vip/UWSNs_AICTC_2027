# ===========================================================
# routing/pso_basic.py — PSO basic cho TSP 3D
# ===========================================================

import random
import numpy as np

from ..physics import travel_time
from ._base import get_swap_sequence, apply_velocity, validate_and_clone_swarm


class Pso_routing:
    """
    PSO basic for 3D TSP routing.
    Sử dụng hàm chung từ _base.py thay vì duplicate.
    """

    @staticmethod
    def pso_tsp_3d_time(
        coords, v_f=1.0, v_AUV=3.0,
        n_particles=50, max_iter=200,
        w=0.5, c1=0.25, c2=0.25,
        init_gbest=None,
        init_swarm=None,
        verbose=True
    ):
        n_cities = len(coords)
        if n_cities < 2:
            return [0], 0.0, [{"step": 0, "best_time": 0.0, "avg_time": 0.0, "std_time": 0.0, "worst_time": 0.0}]
        if n_cities == 2:
            t = travel_time(np.array([0, 1]), coords, v_f, v_AUV)
            return [0, 1], float(t), [{"step": 0, "best_time": float(t), "avg_time": float(t), "std_time": 0.0, "worst_time": float(t)}]

        # INIT SWARM
        cities = list(range(1, n_cities))

        if init_swarm is not None:
            swarm = validate_and_clone_swarm(init_swarm, n_cities)
            n_particles = len(swarm)
        else:
            swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]

        velocities = [[] for _ in range(n_particles)]

        # Evaluate initial swarm
        costs = [travel_time(np.array(p), coords, v_f, v_AUV) for p in swarm]
        pbest = [p.copy() for p in swarm]
        pbest_cost = costs.copy()

        # gbest selection
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = init_gbest.copy()
            gbest_cost = travel_time(np.array(gbest), coords, v_f, v_AUV)
        else:
            best_idx = int(np.argmin(pbest_cost))
            gbest = pbest[best_idx].copy()
            gbest_cost = float(pbest_cost[best_idx])

        iter_stats = []
        iter_stats.append({
            "step": -1,
            "best_time": float(gbest_cost),
            "avg_time": float(np.mean(costs)),
            "std_time": float(np.std(costs)),
            "worst_time": float(np.max(costs)),
        })

        for t in range(max_iter):
            inertia = 0.7 - 0.5 * (t / max_iter)

            for i in range(n_particles):
                xi, vi = swarm[i], velocities[i]
                v_new = vi[:int(inertia * len(vi))]

                if random.random() < c1:
                    seq_pb = get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))

                if random.random() < c2:
                    seq_gb = get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))

                new_x = apply_velocity(xi, v_new)
                new_cost = travel_time(np.array(new_x), coords, v_f, v_AUV)

                swarm[i], velocities[i] = new_x, v_new
                costs[i] = new_cost

                if new_cost < pbest_cost[i]:
                    pbest[i], pbest_cost[i] = new_x, new_cost
                    if new_cost < gbest_cost:
                        gbest, gbest_cost = new_x, float(new_cost)

            iter_stats.append({
                "step": int(t),
                "best_time": float(gbest_cost),
                "avg_time": float(np.mean(costs)),
                "std_time": float(np.std(costs)),
                "worst_time": float(np.max(costs)),
            })

            if verbose and t % 50 == 0:
                print(f"    [PSO Iter {t:3d}]: Best time = {gbest_cost:.4f}")

        if verbose:
            print(f"    [PSO Iter {max_iter:3d}]: Final Best time = {gbest_cost:.4f}")

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
                print(f"   [PSO Outer loop {outer}/{n_outer}]")

            gbest, cost, stats = Pso_routing.pso_tsp_3d_time(
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
