# ===========================================================
# routing/ga.py — Thuật toán GA cho TSP trên cluster heads
# ===========================================================

import random
import numpy as np

from ..physics import build_time_matrix, tour_time_from_T


class ClusterTSP_GA:
    """
    GA chạy trên TSP instance:
    - coords = center_coords = [BS, CH1, CH2, ...]
    - index_to_ch để map index -> cluster_head id
    - accept init_population for shared initial population
    """

    def __init__(self, clusters, center_coords, ga_params=None):
        self.clusters = clusters

        self.coords_np = np.array(center_coords, dtype=np.float64)
        self.n = int(self.coords_np.shape[0])

        # map index -> cluster_head (index 0 = depot)
        sorted_keys = sorted(clusters.keys(), key=lambda x: int(x))
        self.index_to_ch = [None]  # index 0 = depot
        for k in sorted_keys:
            self.index_to_ch.append(clusters[k].get("cluster_head", None))

        defaults = {
            "pop_size": 50,
            "generations": 200,
            "crossover_rate": 0.8,
            "mutation_rate": 0.2,
            "elitism_k": 3,
            "tournament_size": 3,
            "crossover_type": "OX",
            "mutation_type": "inversion",
            "local_search": False,
            "v_f": 1.2,
            "v_AUV": 3.0,
            "verbose": False,
        }
        if ga_params:
            defaults.update(ga_params)
        self.params = defaults

        # Precompute time matrix
        self.time_mat = build_time_matrix(
            self.coords_np, self.params["v_f"], self.params["v_AUV"]
        )

    # ── Individual representation ───────────────────────────
    def create_individual(self):
        if self.n < 2:
            return np.array([0], dtype=np.int64)
        seq = np.arange(1, self.n, dtype=np.int64)
        np.random.shuffle(seq)
        ind = np.empty(self.n, dtype=np.int64)
        ind[0] = 0
        ind[1:] = seq
        return ind

    def _normalize_route(self, route):
        arr = np.array(route, dtype=np.int64).ravel()
        if arr.shape[0] != self.n:
            raise ValueError(f"init route length {arr.shape[0]} != n={self.n}")

        seen = np.zeros(self.n, dtype=np.int8)
        for x in arr:
            if x < 0 or x >= self.n:
                raise ValueError(f"init route has invalid node index: {int(x)}")
            if seen[x] == 1:
                raise ValueError(f"init route has duplicate node: {int(x)}")
            seen[x] = 1

        if seen[0] != 1:
            raise ValueError("init route must include depot 0")

        # rotate so that 0 is at position 0
        if arr[0] != 0:
            pos0 = 0
            for i in range(arr.shape[0]):
                if arr[i] == 0:
                    pos0 = i
                    break
            arr = np.concatenate((arr[pos0:], arr[:pos0]))

        return arr.astype(np.int64)

    def create_population(self, init_population=None):
        pop_size = int(self.params["pop_size"])
        if init_population is None:
            return [self.create_individual() for _ in range(pop_size)]

        pop = [self._normalize_route(r) for r in init_population]
        if len(pop) < pop_size:
            pop += [self.create_individual() for _ in range(pop_size - len(pop))]
        else:
            pop = pop[:pop_size]
        return pop

    # ── Fast evaluation ─────────────────────────────────────
    def evaluate_population(self, pop):
        m = len(pop)
        times = np.empty(m, dtype=np.float64)
        for i in range(m):
            times[i] = tour_time_from_T(pop[i], self.time_mat)
        fitnesses = 1.0 / (times + 1e-9)
        return times, fitnesses

    def tournament_selection_idx(self, fitnesses):
        ts = self.params["tournament_size"]
        best_i = random.randrange(len(fitnesses))
        best_f = fitnesses[best_i]
        for _ in range(ts - 1):
            j = random.randrange(len(fitnesses))
            fj = fitnesses[j]
            if fj > best_f:
                best_f = fj
                best_i = j
        return best_i

    # ── Operators (OX + inversion) ──────────────────────────
    def order_crossover(self, p1, p2):
        if p1.shape[0] <= 2:
            return p1.copy(), p2.copy()

        sub1 = p1[1:]
        sub2 = p2[1:]
        L = sub1.shape[0]
        if L < 2:
            return p1.copy(), p2.copy()

        a, b = sorted(random.sample(range(L), 2))
        c1 = np.full(L, -1, dtype=np.int64)
        c2 = np.full(L, -1, dtype=np.int64)

        c1[a:b] = sub1[a:b]
        c2[a:b] = sub2[a:b]

        ptr = b
        for x in np.concatenate((sub2[b:], sub2[:b])):
            found = False
            for t in range(L):
                if c1[t] == x:
                    found = True
                    break
            if not found:
                c1[ptr % L] = x
                ptr += 1

        ptr = b
        for x in np.concatenate((sub1[b:], sub1[:b])):
            found = False
            for t in range(L):
                if c2[t] == x:
                    found = True
                    break
            if not found:
                c2[ptr % L] = x
                ptr += 1

        child1 = np.empty_like(p1)
        child2 = np.empty_like(p2)
        child1[0] = 0
        child2[0] = 0
        child1[1:] = c1
        child2[1:] = c2
        return child1, child2

    def inversion_mutation(self, ind):
        if ind.shape[0] <= 3:
            return ind
        i, j = sorted(random.sample(range(1, ind.shape[0]), 2))
        ind[i : j + 1] = ind[i : j + 1][::-1]
        return ind

    # ── Evolve ──────────────────────────────────────────────
    def evolve(self, init_population=None):
        pop = self.create_population(init_population=init_population)

        times, fitnesses = self.evaluate_population(pop)
        best_idx = int(np.argmax(fitnesses))
        best = pop[best_idx].copy()
        best_time = float(times[best_idx])

        history = [best_time]
        gen_stats = [{
            "step": -1,
            "best_time": float(np.min(times)),
            "avg_time": float(np.mean(times)),
            "std_time": float(np.std(times)),
            "worst_time": float(np.max(times)),
            "global_best_time": float(best_time),
        }]

        for gen in range(self.params["generations"]):
            elitism_k = min(self.params["elitism_k"], len(pop))
            elite_idx = np.argsort(fitnesses)[-elitism_k:]
            new_pop = [pop[i].copy() for i in elite_idx]

            while len(new_pop) < self.params["pop_size"]:
                i1 = self.tournament_selection_idx(fitnesses)
                i2 = self.tournament_selection_idx(fitnesses)
                p1 = pop[i1]
                p2 = pop[i2]

                if random.random() < self.params["crossover_rate"]:
                    c1, c2 = self.order_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                if random.random() < self.params["mutation_rate"]:
                    c1 = self.inversion_mutation(c1)
                if random.random() < self.params["mutation_rate"]:
                    c2 = self.inversion_mutation(c2)

                new_pop.append(c1)
                if len(new_pop) < self.params["pop_size"]:
                    new_pop.append(c2)

            pop = new_pop[: self.params["pop_size"]]

            times, fitnesses = self.evaluate_population(pop)
            best_gen_idx = int(np.argmax(fitnesses))
            gen_best_time = float(times[best_gen_idx])

            if gen_best_time < best_time:
                best_time = gen_best_time
                best = pop[best_gen_idx].copy()

            history.append(best_time)
            gen_stats.append({
                "step": int(gen),
                "best_time": float(np.min(times)),
                "avg_time": float(np.mean(times)),
                "std_time": float(np.std(times)),
                "worst_time": float(np.max(times)),
                "global_best_time": float(best_time),
            })

            if self.params.get("verbose", False) and (gen % max(1, self.params["generations"] // 10) == 0):
                print(
                    f"    [GA Gen {gen:3d}] best(pop)={np.min(times):.4f} "
                    f"avg={np.mean(times):.4f} std={np.std(times):.4f} "
                    f"global_best={best_time:.4f}"
                )

        mapped_path = ["O" if idx == 0 else self.index_to_ch[int(idx)] for idx in best.tolist()]
        return best.tolist(), mapped_path, float(best_time), history, gen_stats
