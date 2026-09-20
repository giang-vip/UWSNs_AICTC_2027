# ===========================================================
# routing/pso_adaptive.py — PSO with Adaptive Noise cho TSP 3D
# ===========================================================

import random
import numpy as np

from ..physics import travel_time
from ._base import get_swap_sequence, apply_velocity, validate_and_clone_swarm


class Pso_adaptive_noise:
    """
    PSO with Adaptive Noise for 3D TSP routing.
    Sử dụng hàm chung từ _base.py thay vì duplicate.
    """

    @staticmethod
    def pso_tsp_3d_time_adaptive(
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

        if n_cities == 2:
            tcost = float(travel_time(np.array([0, 1]), coords, v_f, v_AUV))
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

        costs = [travel_time(np.array(p), coords, v_f, v_AUV) for p in swarm]
        pbest = [p.copy() for p in swarm]
        pbest_cost = costs.copy()

        # gbest
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = init_gbest.copy()
            gbest_cost = float(travel_time(np.array(gbest), coords, v_f, v_AUV))
        else:
            best_idx = int(np.argmin(pbest_cost))
            gbest = pbest[best_idx].copy()
            gbest_cost = float(pbest_cost[best_idx])

        # ── ADAPTIVE NOISE SETUP ──
        no_improve = 0
        last_best = gbest_cost
        p_noise = 0.1
        max_noise_swaps = 2

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

                # pbest influence
                if random.random() < c1:
                    seq_pb = get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))

                # gbest influence
                if random.random() < c2:
                    seq_gb = get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))

                # ── ADAPTIVE NOISE ──
                if random.random() < p_noise:
                    k = random.randint(1, max_noise_swaps)
                    for _ in range(k):
                        i1, i2 = random.sample(range(1, n_cities), 2)
                        v_new.append((i1, i2))

                new_x = apply_velocity(xi, v_new)
                new_cost = travel_time(np.array(new_x), coords, v_f, v_AUV)

                swarm[i], velocities[i] = new_x, v_new
                costs[i] = new_cost

                # update pbest/gbest
                if new_cost < pbest_cost[i]:
                    pbest[i], pbest_cost[i] = new_x, new_cost
                    if new_cost < gbest_cost:
                        gbest, gbest_cost = new_x, float(new_cost)

            # ── ADAPTIVE NOISE UPDATE ──
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
                print(f"    [PSO-Adaptive Iter {t:3d}]: Best time = {gbest_cost:.4f}, Noise = {p_noise:.3f}")

        if verbose:
            print(f"    [PSO-Adaptive Iter {max_iter:3d}]: Final Best time = {gbest_cost:.4f}")

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
                print(f"   [PSO-Adaptive Outer loop {outer}/{n_outer}]")

            gbest, cost, stats = Pso_adaptive_noise.pso_tsp_3d_time_adaptive(
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
