# ===========================================================
# clustering.py — Phân cụm KMeans với constraints cho UWSN
# ===========================================================

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist


class Clustering:
    def __init__(self, space_size=400, r_sen=50, max_cluster_size=20, min_cluster_size=5):
        self.space_size = space_size
        self.r_sen = r_sen
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size

    # ── 1. ƯỚC TÍNH K TỐI ƯU ──────────────────────────────
    def estimate_optimal_k(self, nodes, base_station=(200, 200, 400)):
        N = len(nodes)
        if N == 0:
            return 0

        base_pos = np.array(base_station, dtype=float)
        distances = np.linalg.norm(nodes - base_pos, axis=1)

        d_tobs = float(np.mean(distances)) if len(distances) > 0 else 1.0
        d_tobs = max(d_tobs, 1e-9)

        k_optimal = np.sqrt(N * self.space_size / (np.pi * d_tobs))
        k_optimal = max(2, int(np.round(k_optimal)))

        k_min = int(np.ceil(N / self.max_cluster_size))
        k_optimal = max(k_optimal, k_min)
        return k_optimal

    # ── 2. KIỂM TRA TÍNH HỢP LỆ ────────────────────────────
    def check_cluster_validity(self, cluster_nodes):
        size = len(cluster_nodes)

        if size < self.min_cluster_size or size > self.max_cluster_size:
            return False, 0.0, size

        if size > 1:
            distances = pdist(cluster_nodes)
            max_dist = float(np.max(distances)) if len(distances) > 0 else 0.0
            if max_dist > self.r_sen:
                return False, max_dist, size
            return True, max_dist, size

        return True, 0.0, size

    # ── 3. TÁCH CỤM KHÔNG HỢP LỆ ───────────────────────────
    def split_invalid_cluster(self, cluster_nodes, cluster_ids):
        if len(cluster_nodes) < 2:
            return [(cluster_nodes, cluster_ids)]

        kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
        labels = kmeans.fit_predict(cluster_nodes)

        sub_clusters = []
        for i in range(2):
            sub_nodes = cluster_nodes[labels == i]
            sub_ids = [cluster_ids[j] for j in range(len(cluster_ids)) if labels[j] == i]
            if len(sub_nodes) > 0:
                sub_clusters.append((sub_nodes, sub_ids))
        return sub_clusters

    # ── 4. GỘP CỤM NHỎ ─────────────────────────────────────
    def merge_small_clusters(self, clusters_data):

        def max_pairwise_dist(arr):
            if len(arr) <= 1:
                return 0.0
            return float(np.max(pdist(arr)))

        if len(clusters_data) <= 1:
            return clusters_data

        merged = []
        smalls = []

        for nodes, ids in clusters_data:
            if len(nodes) < self.min_cluster_size:
                smalls.append((nodes, ids))
            else:
                merged.append((nodes, ids))

        i = 0
        while i < len(smalls):
            small_nodes, small_ids = smalls[i]

            if len(small_nodes) == 0:
                smalls.pop(i)
                continue

            merged_success = False

            # 1) Thử gộp vào cụm lớn gần nhất
            if len(merged) > 0:
                small_center = np.mean(small_nodes, axis=0)

                dists = []
                for idx, (nodes, ids) in enumerate(merged):
                    if len(nodes) == 0:
                        continue
                    center = np.mean(nodes, axis=0)
                    dists.append((float(np.linalg.norm(small_center - center)), idx))

                dists.sort(key=lambda x: x[0])

                for _, idx in dists:
                    target_nodes, target_ids = merged[idx]

                    if len(target_nodes) + len(small_nodes) > self.max_cluster_size:
                        continue

                    combined_nodes = np.vstack([target_nodes, small_nodes])
                    if max_pairwise_dist(combined_nodes) <= self.r_sen:
                        merged[idx] = (combined_nodes, target_ids + small_ids)
                        merged_success = True
                        break

            if merged_success:
                smalls.pop(i)
                continue

            # 2) Thử gộp với cụm nhỏ khác
            paired = False
            j = i + 1
            while j < len(smalls):
                other_nodes, other_ids = smalls[j]

                if len(other_nodes) == 0:
                    smalls.pop(j)
                    continue

                if len(other_nodes) + len(small_nodes) > self.max_cluster_size:
                    j += 1
                    continue

                combined_nodes = np.vstack([other_nodes, small_nodes])
                if max_pairwise_dist(combined_nodes) <= self.r_sen:
                    merged.append((combined_nodes, other_ids + small_ids))
                    smalls.pop(j)
                    smalls.pop(i)
                    paired = True
                    break

                j += 1

            if paired:
                continue

            # 3) Không gộp được → giữ nguyên
            merged.append((small_nodes, small_ids))
            smalls.pop(i)

        final = [(nodes, ids) for nodes, ids in merged if len(nodes) > 0]
        return final

    # ── 4.5. CÂN BẰNG SỐ LƯỢNG NÚT ────────────────────────
    def balance_clusters(self, clusters):

        def max_pairwise_dist(arr):
            if len(arr) <= 1:
                return 0.0
            return float(np.max(pdist(arr)))

        if len(clusters) <= 1:
            return clusters

        improved = True
        while improved:
            improved = False

            sizes = [len(nodes) for nodes, _ in clusters]
            max_idx = int(np.argmax(sizes))
            min_idx = int(np.argmin(sizes))

            if sizes[max_idx] - sizes[min_idx] <= 1:
                break

            big_nodes, big_ids = clusters[max_idx]
            small_nodes, small_ids = clusters[min_idx]

            moved = False
            for i in range(len(big_nodes)):
                candidate_node = big_nodes[i].reshape(1, -1)
                candidate_id = big_ids[i]

                if len(small_nodes) + 1 > self.max_cluster_size:
                    continue

                new_small = np.vstack([small_nodes, candidate_node])
                if max_pairwise_dist(new_small) > self.r_sen:
                    continue

                # move
                clusters[min_idx] = (new_small, small_ids + [candidate_id])

                new_big_nodes = np.delete(big_nodes, i, axis=0)
                new_big_ids = big_ids[:i] + big_ids[i + 1:]
                clusters[max_idx] = (new_big_nodes, new_big_ids)

                moved = True
                improved = True
                break

            if not moved:
                break

        clusters = [(n, ids) for (n, ids) in clusters if len(n) > 0]
        return clusters

    # ── 5. PHÂN CỤM CHÍNH ───────────────────────────────────
    def cluster_with_constraints(self, nodes, node_ids, k=None, max_iterations=10):

        if len(nodes) == 0:
            return []

        if k is None:
            k = self.estimate_optimal_k(nodes)

        k = max(1, int(k))
        k = min(k, len(nodes))

        print(f"Bắt đầu phân cụm với k={k}")

        kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = kmeans.fit_predict(nodes)

        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        iteration = 0
        while iteration < max_iterations:
            valid_clusters = []
            invalid_clusters = []

            for i in range(k):
                cluster_nodes = nodes[labels == i]
                cluster_ids = [node_ids[j] for j in range(len(node_ids)) if labels[j] == i]

                if len(cluster_nodes) == 0:
                    continue

                is_valid, max_dist, size = self.check_cluster_validity(cluster_nodes)

                if is_valid:
                    valid_clusters.append((cluster_nodes, cluster_ids))
                else:
                    invalid_clusters.append((cluster_nodes, cluster_ids))

            if len(invalid_clusters) == 0:
                print("  → Tất cả cụm hợp lệ!")
                break

            for cluster_nodes, cluster_ids in invalid_clusters:
                sub_clusters = self.split_invalid_cluster(cluster_nodes, cluster_ids)
                valid_clusters.extend(sub_clusters)

            k = len(valid_clusters)
            labels = np.zeros(len(nodes), dtype=int)

            for cluster_idx, (_, c_ids) in enumerate(valid_clusters):
                for nid in c_ids:
                    labels[id_to_idx[nid]] = cluster_idx

            iteration += 1

        final_clusters = self.merge_small_clusters(valid_clusters)
        final_clusters = self.balance_clusters(final_clusters)

        print(f"Số lượng cụm CUỐI CÙNG: {len(final_clusters)}")
        return final_clusters

    # ── 6. CHỌN CLUSTER HEAD ────────────────────────────────
    def choose_cluster_head(self, cluster_nodes, cluster_ids, node_data=None):
        if len(cluster_ids) == 0:
            return None

        if node_data:
            max_energy = -1
            ch_id = cluster_ids[0]
            for nid in cluster_ids:
                if nid in node_data and 'residual_energy' in node_data[nid]:
                    energy = node_data[nid]['residual_energy']
                    if energy > max_energy:
                        max_energy = energy
                        ch_id = nid
            return ch_id

        # fallback: gần tâm cụm nhất
        center = np.mean(cluster_nodes, axis=0)
        distances = np.linalg.norm(cluster_nodes - center, axis=1)
        return cluster_ids[int(np.argmin(distances))]

    # ── 7. METRICS ──────────────────────────────────────────
    def calculate_metrics(self, clusters_data):
        metrics = {
            'num_clusters': len(clusters_data),
            'avg_cluster_size': 0,
            'min_cluster_size': float('inf'),
            'max_cluster_size': 0,
            'avg_intra_distance': 0,
            'max_intra_distance': 0,
            'balance_score': 0
        }

        sizes = []
        intra_dists = []

        for nodes, ids in clusters_data:
            size = len(nodes)
            sizes.append(size)

            metrics['min_cluster_size'] = min(metrics['min_cluster_size'], size)
            metrics['max_cluster_size'] = max(metrics['max_cluster_size'], size)

            if size > 1:
                distances = pdist(nodes)
                intra_dists.append(np.mean(distances))
                metrics['max_intra_distance'] = max(metrics['max_intra_distance'], np.max(distances))

        metrics['avg_cluster_size'] = np.mean(sizes)
        metrics['avg_intra_distance'] = np.mean(intra_dists) if intra_dists else 0

        cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
        metrics['balance_score'] = 1 / (1 + cv)

        return metrics

    # ── PHÂN LẠI CỤM ───────────────────────────────────────
    @staticmethod
    def recluster(all_nodes, node_positions, clustering_instance, r_sen=60, max_size=25, min_size=15):
        ids = sorted(list(all_nodes.keys()))
        if len(ids) == 0:
            return {}
        coords = np.array([node_positions[nid] for nid in ids])
        clustering_instance.r_sen = r_sen
        clustering_instance.max_cluster_size = max_size
        clustering_instance.min_cluster_size = min_size
        clusters_data = clustering_instance.cluster_with_constraints(coords, ids)
        clusters = {}
        for i, (cluster_nodes, cluster_ids) in enumerate(clusters_data):
            center = np.mean(cluster_nodes, axis=0).tolist()
            ch = clustering_instance.choose_cluster_head(cluster_nodes, cluster_ids, all_nodes)
            clusters[i] = {'nodes': cluster_ids, 'center': center, 'cluster_head': ch}
        return clusters

    # ── Chọn lại cụm trưởng ────────────────────────────────
    @staticmethod
    def reselect_cluster_heads(clusters, all_nodes):
        """
        Chỉ chọn lại cluster head cho các cụm hiện tại dựa trên năng lượng.
        Không phân cụm lại.
        """
        for cid, cinfo in clusters.items():
            cluster_ids = cinfo['nodes']
            max_energy = -1
            new_ch = cluster_ids[0]
            for nid in cluster_ids:
                if nid in all_nodes and 'residual_energy' in all_nodes[nid]:
                    energy = all_nodes[nid]['residual_energy']
                    if energy > max_energy:
                        max_energy = energy
                        new_ch = nid
            clusters[cid]['cluster_head'] = new_ch
        return clusters

    # ── Loại bỏ node chết ──────────────────────────────────
    @staticmethod
    def remove_dead_nodes(all_nodes, clusters):
        """
        Loại bỏ các node đã hết năng lượng và cập nhật lại clusters.

        Returns:
            (new_clusters, dead) — clusters còn node sống & list node đã chết
        """
        dead = [nid for nid, info in list(all_nodes.items()) if info['residual_energy'] <= 0]
        for nid in dead:
            del all_nodes[nid]

        new_clusters = {}
        for cid, cinfo in clusters.items():
            alive_nodes = [nid for nid in cinfo.get('nodes', []) if nid in all_nodes]
            if alive_nodes:
                new_c = dict(cinfo)
                new_c['nodes'] = alive_nodes
                new_clusters[cid] = new_c

        return new_clusters, dead
