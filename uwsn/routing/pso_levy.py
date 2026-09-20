# ===========================================================
# routing/pso_levy.py — PSO with Lévy Flight cho TSP 3D
# ===========================================================

import random
import numpy as np

from ..physics import travel_time
from ._base import get_swap_sequence, apply_velocity, validate_and_clone_swarm


class Pso_levy_flight:
    """
    PSO with Lévy Flight for 3D TSP routing.
    Sử dụng hàm chung từ _base.py thay vì duplicate.
    """

    @staticmethod
    def pso_tsp_3d_time_levy(
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
                "std_time": 0.0, "worst_time": 0.0, "levy_p": 0.0
            }]
            return [0], 0.0, stats

        if n_cities == 2:
            path = np.array([0, 1], dtype=np.int64)
            cost = float(travel_time(path, coords, v_f, v_AUV))
            stats = [{
                "step": 0, "best_time": cost, "avg_time": cost,
                "std_time": 0.0, "worst_time": cost, "levy_p": 0.0
            }]
            return path.tolist(), cost, stats

        cities = list(range(1, n_cities))

        # Init swarm
        if init_swarm is not None:
            swarm = validate_and_clone_swarm(init_swarm, n_cities)
            n_particles = len(swarm)
        else:
            swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]

        velocities = [[] for _ in range(n_particles)]

        # pbest
        pbest = [p.copy() for p in swarm]
        pbest_cost = []
        for p in swarm:
            p_arr = np.array(p, dtype=np.int64)
            pbest_cost.append(travel_time(p_arr, coords, v_f, v_AUV))

        costs = pbest_cost.copy()

        # gbest
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = list(init_gbest)
            gbest_cost = float(travel_time(np.array(gbest, np.int64), coords, v_f, v_AUV))
        else:
            idx = int(np.argmin(pbest_cost))
            gbest = pbest[idx].copy()
            gbest_cost = float(pbest_cost[idx])

        # ── Lévy params ──
        p_levy = 0.15
        beta = 1.5
        max_levy_swaps = 4
        max_velocity_len = n_cities

        iter_stats = []
        iter_stats.append({
            "step": -1,
            "best_time": float(gbest_cost),
            "avg_time": float(np.mean(costs)),
            "std_time": float(np.std(costs)),
            "worst_time": float(np.max(costs)),
            "levy_p": float(p_levy),
        })

        for t in range(max_iter):
            inertia = 0.7 - 0.5 * (t / max_iter)

            for i in range(n_particles):
                xi = swarm[i]
                vi = velocities[i]

                # inertia
                keep = int(inertia * len(vi))
                v_new = vi[:keep]

                # pbest
                if random.random() < c1:
                    seq_pb = get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))

                # gbest
                if random.random() < c2:
                    seq_gb = get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))

                # ── Lévy flight ──
                if random.random() < p_levy:
                    u = random.gauss(0.0, 1.0)
                    v = random.gauss(0.0, 1.0)
                    denom = abs(v) if abs(v) > 1e-12 else 1e-12
                    step = abs(u) / (denom ** (1.0 / beta))
                    k = min(max_levy_swaps, max(1, int(step)))
                    for _ in range(k):
                        a, b = random.sample(range(1, n_cities), 2)
                        v_new.append((a, b))

                # prevent too-long velocity
                if len(v_new) > max_velocity_len:
                    v_new = random.sample(v_new, max_velocity_len)

                # apply
                new_x = apply_velocity(xi, v_new)

                # safety check
                if new_x is None or len(new_x) != n_cities:
                    continue
                if len(set(new_x)) != n_cities or new_x[0] != 0:
                    continue

                new_cost = travel_time(np.array(new_x, dtype=np.int64), coords, v_f, v_AUV)

                swarm[i] = new_x
                velocities[i] = v_new
                costs[i] = new_cost

                if new_cost < pbest_cost[i]:
                    pbest[i] = new_x.copy()
                    pbest_cost[i] = new_cost

                    if new_cost < gbest_cost:
                        gbest = new_x.copy()
                        gbest_cost = float(new_cost)

            iter_stats.append({
                "step": int(t),
                "best_time": float(gbest_cost),
                "avg_time": float(np.mean(costs)),
                "std_time": float(np.std(costs)),
                "worst_time": float(np.max(costs)),
                "levy_p": float(p_levy),
            })

            if verbose and t % 50 == 0:
                print(f"    [PSO-Levy Iter {t:3d}]: Best time = {gbest_cost:.4f}")

        if verbose:
            print(f"    [PSO-Levy Iter {max_iter:3d}]: Final Best time = {gbest_cost:.4f}")

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
                print(f"   [PSO-Levy Outer loop {outer}/{n_outer}]")

            gbest, cost, stats = Pso_levy_flight.pso_tsp_3d_time_levy(
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
