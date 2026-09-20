import numpy as np
import random
import math
import json
import os
from datetime import datetime
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from numba import njit
import copy

class Clustering:
    def __init__(self, space_size=400, r_sen=50, max_cluster_size=20, min_cluster_size=5):
        self.space_size = space_size
        self.r_sen = r_sen
        self.max_cluster_size = max_cluster_size
        self.min_cluster_size = min_cluster_size

    #  1. ƯỚC TÍNH K TỐI ƯU
    def estimate_optimal_k(self, nodes, base_station=(200, 200, 400)):
        N = len(nodes)
        if N == 0:
            return 0

        base_pos = np.array(base_station, dtype=float)
        distances = np.linalg.norm(nodes - base_pos, axis=1)

        # tránh mean(empty)
        d_tobs = float(np.mean(distances)) if len(distances) > 0 else 1.0
        d_tobs = max(d_tobs, 1e-9)

        k_optimal = np.sqrt(N * self.space_size / (np.pi * d_tobs))
        k_optimal = max(2, int(np.round(k_optimal)))

        k_min = int(np.ceil(N / self.max_cluster_size))
        k_optimal = max(k_optimal, k_min)
        return k_optimal

    #  2. KIỂM TRA TÍNH HỢP LỆ
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

        # size == 1: về validity theo “constraint” thì không sai về r_sen,
        # nhưng thường sẽ bị đưa vào nhóm smalls để merge (vì < min_cluster_size)
        return True, 0.0, size

    #  3. TÁCH CỤM KHÔNG HỢP LỆ
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

    #  4. GỘP CỤM NHỎ (KHÔNG TẠO CỤM RỖNG)
    def merge_small_clusters(self, clusters_data):

        def max_pairwise_dist(arr):
            if len(arr) <= 1:
                return 0.0
            return float(np.max(pdist(arr)))

        if len(clusters_data) <= 1:
            return clusters_data

        merged = []
        smalls = []

        # tách
        for nodes, ids in clusters_data:
            if len(nodes) < self.min_cluster_size:
                smalls.append((nodes, ids))
            else:
                merged.append((nodes, ids))

        # gộp: dùng while + pop để “xóa thật”, không để placeholder rỗng
        i = 0
        while i < len(smalls):
            small_nodes, small_ids = smalls[i]

            # phòng hờ (không nên xảy ra, nhưng để an toàn)
            if len(small_nodes) == 0:
                smalls.pop(i)
                continue

            merged_success = False

            # 1) thử gộp vào cụm lớn gần nhất
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
                smalls.pop(i)   # xóa cụm nhỏ đã được gộp
                continue

            # 2) nếu không gộp được vào cụm lớn -> thử gộp với cụm nhỏ khác
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
                    # xóa 2 cụm nhỏ khỏi list
                    smalls.pop(j)
                    smalls.pop(i)
                    paired = True
                    break

                j += 1

            if paired:
                continue

            # 3) không gộp được với ai -> giữ nguyên như 1 cụm riêng (không xóa)
            merged.append((small_nodes, small_ids))
            smalls.pop(i)

        # final: lọc cụm rỗng (phòng hờ)
        final = [(nodes, ids) for nodes, ids in merged if len(nodes) > 0]
        return final

    #  4.5. CÂN BẰNG SỐ LƯỢNG NÚT
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

        # phòng hờ: loại cụm rỗng nếu có (trường hợp cực hiếm)
        clusters = [(n, ids) for (n, ids) in clusters if len(n) > 0]
        return clusters

    #  5. PHÂN CỤM CHÍNH
    def cluster_with_constraints(self, nodes, node_ids, k=None, max_iterations=10):

        if len(nodes) == 0:
            return []

        if k is None:
            k = self.estimate_optimal_k(nodes)

        k = max(1, int(k))
        k = min(k, len(nodes))  # KMeans không cho k > số điểm

        print(f"Bắt đầu phân cụm với k={k}")

        kmeans = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = kmeans.fit_predict(nodes)

        # map để tránh node_ids.index() O(N^2)
        id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        iteration = 0
        while iteration < max_iterations:
            # print(f"  Vòng lặp {iteration + 1}/{max_iterations}")

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

    #  6. CHỌN CLUSTER HEAD
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

    #  7. METRICS
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

    # PHÂN LẠI CỤM
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

    # Chọn lại cụm trưởng
    def reselect_cluster_heads(clusters, all_nodes):
        """
        Chỉ chọn lại cluster head cho các cụm hiện tại dựa trên năng lượng.
        Không phân cụm lại.
        """
        for cid, cinfo in clusters.items():
            cluster_ids = cinfo['nodes']
            # Tìm node có năng lượng cao nhất
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
        
    def remove_dead_nodes(all_nodes, clusters):
        """
        Loại bỏ các node đã hết năng lượng và cập nhật lại clusters.
        
        Returns:
        - new_clusters: Dictionary các cluster còn node sống
        - dead: List các node_id đã chết
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

    
@njit
def compute_vs(p1, p2, v_f, v_AUV):
    x1, y1, z1 = p1
    x2, y2, z2 = p2

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    L = math.sqrt(dx*dx + dy*dy + dz*dz)
    if L == 0.0:
        return v_AUV

    # Heave velocity v_f is along +z (as in the paper)
    # beta = angle(L, +z) => cos(beta) = (L · z_hat) / |L| = dz / L
    cosb = dz / L
    if cosb > 1.0:
        cosb = 1.0
    elif cosb < -1.0:
        cosb = -1.0

    # v_s = v_f*cos(beta) + sqrt(v_AUV^2 - v_f^2 * sin^2(beta))
    # sin^2(beta) = 1 - cos^2(beta)
    sin2 = 1.0 - cosb*cosb
    sqrt_term = v_AUV*v_AUV - v_f*v_f * sin2

    # Infeasible to keep resultant velocity exactly along L -> avoid NaN
    if sqrt_term < 0.0:
        sqrt_term = 0.0

    v_s = v_f*cosb + math.sqrt(sqrt_term)

    # avoid zero/negative speed for time computation
    if v_s < 1e-9:
        v_s = 1e-9

    return v_s

@njit
def travel_time(path, coords, v_f, v_AUV):
    """
    Tính tổng thời gian di chuyển theo đường đi cho trước
        
    Args:
        path: list/array - danh sách các index của các điểm theo thứ tự
        coords: array - mảng tọa độ các điểm (n x 3)
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
        i2 = path[i+1]

        p1 = coords[i1]
        p2 = coords[i2]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        d = math.sqrt(dx*dx + dy*dy + dz*dz)

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
    d = math.sqrt(dx*dx + dy*dy + dz*dz)

    v_s = compute_vs(p1, p2, v_f, v_AUV)
    if v_s < 1e-9:
        v_s = 1e-9

    total_time += d / v_s

    return total_time

class Computing:
    # @staticmethod
    # @njit
    # def compute_vs(p1, p2, v_f, v_AUV):
    #     x1, y1, z1 = p1
    #     x2, y2, z2 = p2

    #     dx = x2 - x1
    #     dy = y2 - y1
    #     dz = z2 - z1

    #     L = math.sqrt(dx*dx + dy*dy + dz*dz)
    #     if L == 0.0:
    #         return v_AUV

    #     # Heave velocity v_f is along +z (as in the paper)
    #     # beta = angle(L, +z) => cos(beta) = (L · z_hat) / |L| = dz / L
    #     cosb = dz / L
    #     if cosb > 1.0:
    #         cosb = 1.0
    #     elif cosb < -1.0:
    #         cosb = -1.0

    #     # v_s = v_f*cos(beta) + sqrt(v_AUV^2 - v_f^2 * sin^2(beta))
    #     # sin^2(beta) = 1 - cos^2(beta)
    #     sin2 = 1.0 - cosb*cosb
    #     sqrt_term = v_AUV*v_AUV - v_f*v_f * sin2

    #     # Infeasible to keep resultant velocity exactly along L -> avoid NaN
    #     if sqrt_term < 0.0:
    #         sqrt_term = 0.0

    #     v_s = v_f*cosb + math.sqrt(sqrt_term)

    #     # avoid zero/negative speed for time computation
    #     if v_s < 1e-9:
    #         v_s = 1e-9

    #     return v_s
    
    # @staticmethod
    # @njit
    # def travel_time(path, coords, v_f, v_AUV):
    #     """
    #     Tính tổng thời gian di chuyển theo đường đi cho trước
        
    #     Args:
    #         path: list/array - danh sách các index của các điểm theo thứ tự
    #         coords: array - mảng tọa độ các điểm (n x 3)
    #         v_f: float - vận tốc dòng chảy
    #         v_AUV: float - vận tốc của AUV
            
    #     Returns:
    #         float - tổng thời gian di chuyển (bao gồm quay về điểm xuất phát)
    #     """
    #     total_time = 0.0
    #     n = len(path)
    #     if n <= 1:
    #         return 0.0

    #     # Các cạnh trong chu trình
    #     for i in range(n - 1):
    #         i1 = path[i]
    #         i2 = path[i+1]

    #         p1 = coords[i1]
    #         p2 = coords[i2]

    #         dx = p2[0] - p1[0]
    #         dy = p2[1] - p1[1]
    #         dz = p2[2] - p1[2]
    #         d = math.sqrt(dx*dx + dy*dy + dz*dz)

    #         v_s = compute_vs(p1, p2, v_f, v_AUV)
    #         if v_s < 1e-9:
    #             v_s = 1e-9

    #         total_time += d / v_s

    #     # Quay về điểm bắt đầu
    #     p1 = coords[path[-1]]
    #     p2 = coords[path[0]]

    #     dx = p2[0] - p1[0]
    #     dy = p2[1] - p1[1]
    #     dz = p2[2] - p1[2]
    #     d = math.sqrt(dx*dx + dy*dy + dz*dz)

    #     v_s = compute_vs(p1, p2, v_f, v_AUV)
    #     if v_s < 1e-9:
    #         v_s = 1e-9

    #     total_time += d / v_s

    #     return total_time
    
    def energy_member(best_time, d):
        G, L = 100, 1024
        P_r, P_idle = 0.8e-3, 0.1e-3
        DR = 4000
        E_ELEC = 50e-9      # J/bit
        EPS_FS = 10e-12     # J/bit/m^2

        # Time to transmit
        T_tx = G * L / DR

        # Energy terms (THEO ĐÚNG CÔNG THỨC TRONG ẢNH)
        E_tx = G * L * E_ELEC + G * L * EPS_FS * (d ** 2)
        E_rx = P_r * T_tx
        E_idle = (best_time - 2* T_tx) * P_idle
        E_total = E_tx + E_rx + E_idle

        return E_total, E_tx, E_rx
    
    def energy_cluster_head(best_time, n_members):
        G, L = 100, 1024
        P_t, P_idle = 1.6e-3, 0.1e-3
        DR, DR_i = 4000, 10000
        E_ELEC = 50e-9      # J/bit
        EPS_FS = 10e-12     # J/bit/m^2
        E_AGG = 5e-9  # J/bit

        # Time
        T_rx = G * L * n_members / DR
        T_tx = G * L * n_members / DR_i

        # Energy terms (THEO ĐÚNG CÔNG THỨC TRONG ẢNH)
        E_rx = G * L * n_members * E_ELEC + G * L * n_members * E_AGG
        E_tx = P_t * T_tx
        E_idle = (best_time - T_rx - T_tx) * P_idle
    
        return E_rx + E_tx + E_idle

    def update_energy(all_nodes, node_positions, clusters, best_time):
        
        for cid in clusters:
            cluster = clusters[cid]
            ch = cluster['cluster_head']
            nodes = cluster['nodes']

            # Nếu CH đã chết thì bỏ qua cụm
            if ch not in all_nodes:
                continue

            ch_pos = node_positions[ch]

            # MEMBER NODES
            n_members = 0

            for nid in nodes:
                if nid == ch:
                    continue
                if nid not in all_nodes:
                    continue

                d = math.dist(node_positions[nid], ch_pos)
                E_total, E_tx, E_rx = Computing.energy_member(best_time, d)
                if all_nodes[nid]['residual_energy'] < (E_tx + E_rx):
                    all_nodes[nid]['residual_energy'] = 0.0
                    continue
                all_nodes[nid]['residual_energy'] -= E_total
                all_nodes[nid]['residual_energy'] = max(
                    all_nodes[nid]['residual_energy'], 0.0
                )

                n_members += 1

            # CLUSTER HEAD
            E_ch = Computing.energy_cluster_head(best_time, n_members)

            all_nodes[ch]['residual_energy'] -= E_ch
            if all_nodes[ch]['residual_energy'] < 0:
                all_nodes[ch]['residual_energy'] = 0.0
                continue
            all_nodes[ch]['residual_energy'] = max(
                all_nodes[ch]['residual_energy'], 0.0
            )
import random
import math
import numpy as np
from numba import njit


@njit
def build_time_matrix(coords, v_f, v_AUV):
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
    n = path.shape[0]
    if n <= 1:
        return 0.0

    s = 0.0
    for k in range(n - 1):
        s += T[path[k], path[k + 1]]
    s += T[path[n - 1], path[0]]
    return s


class ClusterTSP_GA:
    """
    GA chạy trên cùng TSP instance với PSO:
    - coords = center_coords = [BS, CH1, CH2, ...] (y như PSO)
    - index_to_ch để map index -> cluster_head id
    - accept init_population for shared initial population
    """

    def __init__(self, clusters, center_coords, ga_params=None):
        self.clusters = clusters

        # center_coords must be same as PSO uses: build_center_coords(clusters, node_positions, bs)
        self.coords_np = np.array(center_coords, dtype=np.float64)
        self.n = int(self.coords_np.shape[0])

        # map index -> cluster_head (index 0 is depot)
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

        # Precompute time matrix ONCE using SAME coords as PSO
        self.time_mat = build_time_matrix(
            self.coords_np, self.params["v_f"], self.params["v_AUV"]
        )

    # Individual representation
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

    # Fast evaluation
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

    # Operators (OX + inversion)
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
class Greedy:
    @staticmethod
    def greedy_tsp(coords, v_f=1.2, v_AUV=3.0):
        """
        coords: array (n,3) đã bao gồm BS ở index 0 (GIỐNG PSO/GA)
        return: (path, best_time) với path bắt đầu 0 và kết thúc 0
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
                dt = travel_time(np.array([current, nxt], dtype=np.int64), coords, v_f, v_AUV)
                if dt < best_dt:
                    best_dt = dt
                    best_next = nxt

            path.append(best_next)
            total_time += best_dt
            remaining.remove(best_next)
            current = best_next

        # quay về BS
        total_time += travel_time(np.array([current, 0], dtype=np.int64), coords, v_f, v_AUV)
        path.append(0)

        return path, float(total_time)

class Pso_routing:
    """
    PSO basic for 3D TSP routing
    - UPDATED: accept init_swarm (shared initial population)
    - Does NOT change PSO update logic (only initialization)
    """

    def get_swap_sequence(A, B):
        seq = []
        temp = A.copy()
        for i in range(1, len(A)):
            if temp[i] != B[i]:
                try:
                    j = temp.index(B[i])
                    seq.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
                except ValueError:
                    pass
        return seq

    def apply_velocity(position, velocity):
        pos = position.copy()
        for i, j in velocity:
            if i > 0 and j > 0 and i < len(pos) and j < len(pos):
                pos[i], pos[j] = pos[j], pos[i]
        return pos

    @staticmethod
    def _validate_and_clone_swarm(init_swarm, n_cities, n_particles_expected=None):
        """
        Validate init_swarm routes:
        - each route length == n_cities
        - contains all nodes exactly once
        - depot 0 exists and forced at index 0 (by rotation)
        Returns: swarm(list[list[int]])
        """
        if init_swarm is None:
            return None

        swarm = []
        for route in init_swarm:
            r = list(route)
            if len(r) != n_cities:
                raise ValueError(f"init_swarm route length {len(r)} != n_cities={n_cities}")

            # check range + duplicates
            seen = [0] * n_cities
            for x in r:
                if x < 0 or x >= n_cities:
                    raise ValueError(f"init_swarm has invalid node index: {x}")
                if seen[x] == 1:
                    raise ValueError(f"init_swarm has duplicate node: {x}")
                seen[x] = 1
            if seen[0] != 1:
                raise ValueError("init_swarm route must include depot 0")

            # rotate so that depot 0 is at position 0 (keep order)
            if r[0] != 0:
                p0 = r.index(0)
                r = r[p0:] + r[:p0]

            swarm.append(r)

        # optional: if caller expects a certain particle count
        if n_particles_expected is not None and len(swarm) != n_particles_expected:
            # we DON'T force equality; multi can decide. Here just a guard if you want strict mode.
            pass

        return swarm

    def pso_tsp_3d_time(
        coords, v_f=1.0, v_AUV=3.0,
        n_particles=50, max_iter=200,
        w=0.5, c1=0.25, c2=0.25,
        init_gbest=None,
        init_swarm=None,          # NEW
        verbose=True
    ):
        n_cities = len(coords)
        if n_cities < 2:
            return [0], 0.0, [{"step": 0, "best_time": 0.0, "avg_time": 0.0, "std_time": 0.0, "worst_time": 0.0}]
        if n_cities == 2:
            t = travel_time(np.array([0, 1]), coords, v_f, v_AUV)
            return [0, 1], float(t), [{"step": 0, "best_time": float(t), "avg_time": float(t), "std_time": 0.0, "worst_time": float(t)}]

        # INIT SWARM (UPDATED)
        cities = list(range(1, n_cities))

        if init_swarm is not None:
            swarm = Pso_routing._validate_and_clone_swarm(init_swarm, n_cities)
            # important: sync particle count with provided swarm
            n_particles = len(swarm)
        else:
            swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]

        velocities = [[] for _ in range(n_particles)]

        # Evaluate initial swarm
        costs = [travel_time(np.array(p), coords, v_f, v_AUV) for p in swarm]
        pbest = [p.copy() for p in swarm]
        pbest_cost = costs.copy()

        # ---- gbest selection (UNCHANGED LOGIC) ----
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = init_gbest.copy()
            gbest_cost = travel_time(np.array(gbest), coords, v_f, v_AUV)
        else:
            best_idx = int(np.argmin(pbest_cost))
            gbest = pbest[best_idx].copy()
            gbest_cost = float(pbest_cost[best_idx])

        iter_stats = []
        # log init (t = -1)
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
                    seq_pb = Pso_routing.get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))

                if random.random() < c2:
                    seq_gb = Pso_routing.get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))

                new_x = Pso_routing.apply_velocity(xi, v_new)
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

    def multi_pso_tsp(
        coords, v_f=1.2, v_AUV=3.0,
        n_outer=5, verbose=True,
        init_swarm=None,          # NEW
        **kwargs
    ):
        prev_gbest, prev_cost = None, float("inf")
        best_stats = None

        for outer in range(1, n_outer + 1):
            if verbose:
                print(f"   [PSO Outer loop {outer}/{n_outer}]")

            # If you want outer loops to start from the SAME init_swarm each time,
            # pass init_swarm directly. This keeps your multi-outer behavior consistent.
            gbest, cost, stats = Pso_routing.pso_tsp_3d_time(
                coords,
                v_f=v_f,
                v_AUV=v_AUV,
                init_gbest=prev_gbest,
                init_swarm=init_swarm,     # forward
                verbose=verbose,
                **kwargs,
            )

            if cost < prev_cost:
                prev_gbest, prev_cost = gbest, float(cost)
                best_stats = stats

        return prev_gbest, float(prev_cost), best_stats

class Pso_adaptive_noise:
    """
    PSO with Adaptive Noise for 3D TSP routing
    - UPDATED: accept init_swarm (shared initial population)
    - Does NOT change PSO/Noise update logic (only initialization)
    """

    def get_swap_sequence(A, B):
        seq = []
        temp = A.copy()
        for i in range(1, len(A)):
            if temp[i] != B[i]:
                try:
                    j = temp.index(B[i])
                    seq.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
                except ValueError:
                    pass
        return seq

    def apply_velocity(position, velocity):
        pos = position.copy()
        for i, j in velocity:
            if i > 0 and j > 0 and i < len(pos) and j < len(pos):
                pos[i], pos[j] = pos[j], pos[i]
        return pos

    @staticmethod
    def _validate_and_clone_swarm(init_swarm, n_cities):
        """
        Validate init_swarm routes:
        - each route length == n_cities
        - contains all nodes exactly once
        - depot 0 exists and is rotated to index 0
        Returns: swarm(list[list[int]])
        """
        if init_swarm is None:
            return None

        swarm = []
        for route in init_swarm:
            r = list(route)
            if len(r) != n_cities:
                raise ValueError(f"init_swarm route length {len(r)} != n_cities={n_cities}")

            seen = [0] * n_cities
            for x in r:
                if x < 0 or x >= n_cities:
                    raise ValueError(f"init_swarm has invalid node index: {x}")
                if seen[x] == 1:
                    raise ValueError(f"init_swarm has duplicate node: {x}")
                seen[x] = 1
            if seen[0] != 1:
                raise ValueError("init_swarm route must include depot 0")

            # rotate so depot 0 is at position 0
            if r[0] != 0:
                p0 = r.index(0)
                r = r[p0:] + r[:p0]

            swarm.append(r)

        return swarm

    def pso_tsp_3d_time_adaptive(
        coords, v_f=1.0, v_AUV=3.0,
        n_particles=50, max_iter=200,
        w=0.5, c1=0.25, c2=0.25,
        init_gbest=None,
        init_swarm=None,       # NEW
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

        # INIT SWARM (UPDATED)
        cities = list(range(1, n_cities))
        if init_swarm is not None:
            swarm = Pso_adaptive_noise._validate_and_clone_swarm(init_swarm, n_cities)
            n_particles = len(swarm)  # sync
        else:
            swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]

        velocities = [[] for _ in range(n_particles)]

        costs = [travel_time(np.array(p), coords, v_f, v_AUV) for p in swarm]
        pbest = [p.copy() for p in swarm]
        pbest_cost = costs.copy()

        # Xác định gbest trước (UNCHANGED)
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = init_gbest.copy()
            gbest_cost = float(travel_time(np.array(gbest), coords, v_f, v_AUV))
        else:
            best_idx = int(np.argmin(pbest_cost))
            gbest = pbest[best_idx].copy()
            gbest_cost = float(pbest_cost[best_idx])

        # ================= ADAPTIVE NOISE SETUP =================
        no_improve = 0
        last_best = gbest_cost
        p_noise = 0.1
        max_noise_swaps = 2
        # =======================================================

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
                    seq_pb = Pso_adaptive_noise.get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))

                # gbest influence
                if random.random() < c2:
                    seq_gb = Pso_adaptive_noise.get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))

                # ================= ADAPTIVE NOISE =================
                if random.random() < p_noise:
                    k = random.randint(1, max_noise_swaps)
                    for _ in range(k):
                        i1, i2 = random.sample(range(1, n_cities), 2)
                        v_new.append((i1, i2))
                # =================================================

                new_x = Pso_adaptive_noise.apply_velocity(xi, v_new)
                new_cost = travel_time(np.array(new_x), coords, v_f, v_AUV)

                swarm[i], velocities[i] = new_x, v_new
                costs[i] = new_cost  # giữ đúng như bạn (avg/std đúng)

                # update pbest/gbest
                if new_cost < pbest_cost[i]:
                    pbest[i], pbest_cost[i] = new_x, new_cost
                    if new_cost < gbest_cost:
                        gbest, gbest_cost = new_x, float(new_cost)

            # ============ ADAPTIVE NOISE UPDATE ============
            if gbest_cost >= last_best - 1e-6:
                no_improve += 1
            else:
                no_improve = 0
                last_best = gbest_cost

            if no_improve > 15:
                p_noise = min(0.4, p_noise * 1.3)
            else:
                p_noise = max(0.05, p_noise * 0.95)
            # ===============================================

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

    def multi_pso_tsp(
        coords, v_f=1.2, v_AUV=3.0,
        n_outer=5, verbose=True,
        init_swarm=None,        # NEW
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
                init_swarm=init_swarm,   # forward
                verbose=verbose,
                **kwargs,
            )

            if cost < prev_cost:
                prev_gbest, prev_cost = gbest, float(cost)
                best_stats = stats

        return prev_gbest, float(prev_cost), best_stats

class Pso_levy_flight:
    """
    PSO with Lévy Flight for 3D TSP routing
    - UPDATED: accept init_swarm (shared initial population)
    - Does NOT change PSO/Lévy update logic (only initialization)
    """

    def get_swap_sequence(A, B):
        seq = []
        temp = A.copy()
        for i in range(1, len(A)):
            if temp[i] != B[i]:
                try:
                    j = temp.index(B[i])
                    seq.append((i, j))
                    temp[i], temp[j] = temp[j], temp[i]
                except ValueError:
                    pass
        return seq

    def apply_velocity(position, velocity):
        pos = position.copy()
        for i, j in velocity:
            if i > 0 and j > 0 and i < len(pos) and j < len(pos):
                pos[i], pos[j] = pos[j], pos[i]
        return pos

    @staticmethod
    def _validate_and_clone_swarm(init_swarm, n_cities):
        """
        Validate init_swarm routes:
        - each route length == n_cities
        - contains all nodes exactly once
        - depot 0 exists and is rotated to index 0
        Returns: swarm(list[list[int]])
        """
        if init_swarm is None:
            return None

        swarm = []
        for route in init_swarm:
            r = list(route)
            if len(r) != n_cities:
                raise ValueError(f"init_swarm route length {len(r)} != n_cities={n_cities}")

            seen = [0] * n_cities
            for x in r:
                if x < 0 or x >= n_cities:
                    raise ValueError(f"init_swarm has invalid node index: {x}")
                if seen[x] == 1:
                    raise ValueError(f"init_swarm has duplicate node: {x}")
                seen[x] = 1
            if seen[0] != 1:
                raise ValueError("init_swarm route must include depot 0")

            # rotate so depot 0 is at position 0
            if r[0] != 0:
                p0 = r.index(0)
                r = r[p0:] + r[:p0]

            swarm.append(r)

        return swarm

    def pso_tsp_3d_time_levy(
        coords, v_f=1.0, v_AUV=3.0,
        n_particles=50, max_iter=200,
        w=0.5, c1=0.25, c2=0.25,
        init_gbest=None,
        init_swarm=None,         # NEW
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

        # -------- init swarm (UPDATED) --------
        if init_swarm is not None:
            swarm = Pso_levy_flight._validate_and_clone_swarm(init_swarm, n_cities)
            n_particles = len(swarm)  # sync particles with provided swarm
        else:
            swarm = [[0] + random.sample(cities, len(cities)) for _ in range(n_particles)]

        velocities = [[] for _ in range(n_particles)]

        # pbest
        pbest = [p.copy() for p in swarm]
        pbest_cost = []
        for p in swarm:
            p_arr = np.array(p, dtype=np.int64)
            pbest_cost.append(travel_time(p_arr, coords, v_f, v_AUV))

        # costs (current swarm costs) to compute avg/std
        costs = pbest_cost.copy()

        # -------- gbest (UNCHANGED) --------
        if init_gbest is not None and len(init_gbest) == n_cities:
            gbest = list(init_gbest)
            gbest_cost = float(travel_time(np.array(gbest, np.int64), coords, v_f, v_AUV))
        else:
            idx = int(np.argmin(pbest_cost))
            gbest = pbest[idx].copy()
            gbest_cost = float(pbest_cost[idx])

        # -------- Lévy params (UNCHANGED) --------
        p_levy = 0.15
        beta = 1.5
        max_levy_swaps = 4
        max_velocity_len = n_cities

        iter_stats = []
        # log init (t=-1)
        iter_stats.append({
            "step": -1,
            "best_time": float(gbest_cost),
            "avg_time": float(np.mean(costs)),
            "std_time": float(np.std(costs)),
            "worst_time": float(np.max(costs)),
            "levy_p": float(p_levy),
        })

        # --------------------------------
        for t in range(max_iter):
            inertia = 0.7 - 0.5 * (t / max_iter)

            for i in range(n_particles):
                xi = swarm[i]
                vi = velocities[i]

                # ----- inertia -----
                keep = int(inertia * len(vi))
                v_new = vi[:keep]

                # ----- pbest -----
                if random.random() < c1:
                    seq_pb = Pso_levy_flight.get_swap_sequence(xi, pbest[i])
                    if seq_pb:
                        v_new += random.sample(seq_pb, min(2, len(seq_pb)))

                # ----- gbest -----
                if random.random() < c2:
                    seq_gb = Pso_levy_flight.get_swap_sequence(xi, gbest)
                    if seq_gb:
                        v_new += random.sample(seq_gb, min(2, len(seq_gb)))

                # ----- Lévy flight -----
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

                # ----- apply -----
                new_x = Pso_levy_flight.apply_velocity(xi, v_new)

                # safety (UNCHANGED)
                if new_x is None or len(new_x) != n_cities:
                    continue
                if len(set(new_x)) != n_cities or new_x[0] != 0:
                    continue

                new_cost = travel_time(np.array(new_x, dtype=np.int64), coords, v_f, v_AUV)

                swarm[i] = new_x
                velocities[i] = v_new
                costs[i] = new_cost  # keep for correct std/avg

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

    def multi_pso_tsp(
        coords, v_f=1.2, v_AUV=3.0,
        n_outer=5, verbose=True,
        init_swarm=None,         # NEW
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
                init_swarm=init_swarm,   # forward
                verbose=verbose,
                **kwargs,
            )

            if cost < prev_cost:
                prev_gbest, prev_cost = gbest, float(cost)
                best_stats = stats

        return prev_gbest, float(prev_cost), best_stats
# ==========================================================
# RUNNER (MAIN) - FULL VERSION WITH SHARED POPULATION
# Works with your UPDATED classes:
#   - ClusterTSP_GA(clusters, center_coords, ga_params=...).evolve(init_population=...)
#   - Greedy.greedy_tsp(center_coords,...)
#   - Pso_routing / Pso_adaptive_noise / Pso_levy_flight: multi_pso_tsp(..., init_swarm=...)
# Does NOT change the simulation logic (energy update / recluster / logging)
# Only adds: shared initial population (shared_pop) per phase
# ==========================================================

# assumes you already have:
# - travel_time
# - Computing.update_energy(...)
# - Clustering class with: cluster_with_constraints, choose_cluster_head,
#   reselect_cluster_heads, remove_dead_nodes, recluster
# - your UPDATED algorithm classes in the notebook:
#   ClusterTSP_GA, Greedy, Pso_routing, Pso_adaptive_noise, Pso_levy_flight


def jsonl_append(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def compute_total_energy(all_nodes):
    return float(sum(all_nodes[n]["residual_energy"] for n in all_nodes))


def build_center_coords(clusters, node_positions, bs=(200, 200, 400)):
    sorted_keys = sorted(clusters.keys())
    centers = [tuple(bs)]
    for k in sorted_keys:
        ch = clusters[k]["cluster_head"]
        centers.append(tuple(node_positions[ch]))
    return np.array(centers, dtype=np.float64)


def make_shared_population(n_cities, pop_size, seed=0):
    """
    Shared population/swarm for GA & PSO:
      - list[list[int]]
      - each route length = n_cities
      - contains 0..n_cities-1 exactly once
      - depot 0 at index 0
    """
    rng = np.random.default_rng(int(seed))
    base = np.arange(1, n_cities, dtype=np.int64)

    pop = []
    for _ in range(int(pop_size)):
        perm = base.copy()
        rng.shuffle(perm)
        pop.append([0] + perm.tolist())
    return pop


def solve_route(algorithm, center_coords, v_f, v_AUV, clusters=None, phase="default", shared_pop=None):
    """
    Normalize output to: (current_path, current_time)
    shared_pop is used by GA(init_population) and PSO*(init_swarm)
    """

    if algorithm == "Greedy":
        path, t = Greedy.greedy_tsp(center_coords, v_f=v_f, v_AUV=v_AUV)
        return path, float(t)

    if algorithm == "GA":
        if clusters is None:
            raise ValueError("GA cần tham số clusters")

        ga_solver = ClusterTSP_GA(
            clusters,
            center_coords,  # GA class mới cần center_coords
            ga_params={
                "pop_size": 50 if phase == "init" else 30,  # keep consistent with PSO particles per phase
                "generations": 200 if phase == "init" else 80,
                "v_f": v_f,
                "v_AUV": v_AUV,
                "verbose": False,
            },
        )

        best, mapped_path, best_time, history, gen_stats = ga_solver.evolve(init_population=shared_pop)
        return mapped_path, float(best_time)

    if algorithm == "PSO":
        path, cost, stats = Pso_routing.multi_pso_tsp(
            center_coords,
            v_f=v_f,
            v_AUV=v_AUV,
            n_outer=1,
            n_particles=50 if phase == "init" else 30,
            max_iter=200 if phase == "init" else 80,
            init_swarm=shared_pop,  # shared
            verbose=False,
        )
        return path, float(cost)

    if algorithm == "PSO1":
        path, cost, stats = Pso_adaptive_noise.multi_pso_tsp(
            center_coords,
            v_f=v_f,
            v_AUV=v_AUV,
            n_outer=1,
            n_particles=50 if phase == "init" else 30,
            max_iter=200 if phase == "init" else 80,
            init_swarm=shared_pop,  # shared
            verbose=False,
        )
        return path, float(cost)

    if algorithm == "PSO2":
        path, cost, stats = Pso_levy_flight.multi_pso_tsp(
            center_coords,
            v_f=v_f,
            v_AUV=v_AUV,
            n_outer=1,
            n_particles=50 if phase == "init" else 30,
            max_iter=200 if phase == "init" else 80,
            init_swarm=shared_pop,  # ✅ shared
            verbose=False,
        )
        return path, float(cost)

    raise ValueError(f"Unknown algorithm: {algorithm}")


def main():
    input_folder = "/kaggle/input/nodes-data"
    output_main_folder = "/kaggle/working/results_history"
    os.makedirs(output_main_folder, exist_ok=True)

    if not os.path.exists(input_folder):
        print(f"Lỗi: Thư mục {input_folder} không tồn tại!")
        return

    files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    if len(files) == 0:
        print(f"Không tìm thấy file dữ liệu nào trong {input_folder}")
        return

    INITIAL_ENERGY = 100.0
    v_f = 1.2
    v_AUV = 3.0
    R_SEN = 60
    MAX_SIZE = 25
    MIN_SIZE = 10

    algorithms = ["PSO", "Greedy", "GA", "PSO1", "PSO2"]
    global_results = {alg: {} for alg in algorithms}

    clustering = Clustering(space_size=400, r_sen=R_SEN, max_cluster_size=MAX_SIZE, min_cluster_size=MIN_SIZE)

    for filename in sorted(files):
        input_path = os.path.join(input_folder, filename)
        base_name = filename.replace(".json", "")
        file_output_folder = os.path.join(output_main_folder, f"folder_{base_name}")
        os.makedirs(file_output_folder, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"=== Đang xử lý file: {filename} ===")
        print(f"{'='*80}")

        try:
            with open(input_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f" Lỗi đọc file {filename}: {e}")
            continue

        node_positions = {}
        initial_nodes = {}

        if isinstance(data, list):
            for node in data:
                nid = node["id"]
                initial_nodes[nid] = {
                    "initial_energy": node.get("initial_energy", INITIAL_ENERGY),
                    "residual_energy": node.get("residual_energy", INITIAL_ENERGY),
                }
                node_positions[nid] = (node["x"], node["y"], node["z"])
        else:
            print(f"Cấu trúc file {filename} không được hỗ trợ")
            continue

        total_nodes = len(initial_nodes)
        print(f"Tổng số node: {total_nodes}")

        ids = sorted(list(initial_nodes.keys()))
        coords = np.array([node_positions[nid] for nid in ids], dtype=np.float64)
        clusters_data = clustering.cluster_with_constraints(coords, ids)

        initial_clusters = {}
        for i, (cluster_nodes, cluster_ids) in enumerate(clusters_data):
            center = np.mean(cluster_nodes, axis=0).tolist()
            ch = clustering.choose_cluster_head(cluster_nodes, cluster_ids, initial_nodes)
            initial_clusters[i] = {"nodes": cluster_ids, "center": center, "cluster_head": ch}

        center_coords_init = build_center_coords(initial_clusters, node_positions, bs=(200, 200, 400))

        file_results = {}

        for algorithm in algorithms:
            print(f"\n{'='*60}")
            print(f"Chạy thuật toán: {algorithm}")
            print(f"{'='*60}")

            all_nodes = copy.deepcopy(initial_nodes)
            clusters = copy.deepcopy(initial_clusters)

            # -----------------------------
            # INIT route with SHARED population
            # -----------------------------
            n_cities_init = int(len(center_coords_init))
            pop_size_init = 50  # match phase init: PSO particles=50; GA pop_size will also be 50
            shared_pop_init = make_shared_population(n_cities_init, pop_size_init, seed=123)

            try:
                current_path, current_time = solve_route(
                    algorithm,
                    center_coords_init,
                    v_f,
                    v_AUV,
                    clusters=clusters,
                    phase="init",
                    shared_pop=shared_pop_init,
                )
            except Exception as e:
                print(f" Lỗi route ban đầu ({algorithm}): {e}")
                continue

            print(f"   Đường đi ban đầu: {current_time:.4f}s")

            # File log theo cycle
            cycle_log_path = os.path.join(file_output_folder, f"{algorithm}_cycle_log.jsonl")
            with open(cycle_log_path, "w", encoding="utf-8") as f:
                f.write("")

            cycle = 0
            total_energy_consumed = 0.0

            while True:
                cycle += 1

                alive_ratio = (len(all_nodes) / total_nodes) if total_nodes > 0 else 0.0
                if alive_ratio <= 0.1:
                    print(f" Dừng ở cycle {cycle}: {alive_ratio*100:.2f}% node còn sống")
                    break

                if cycle % 50 == 0:
                    print(f"   Cycle {cycle}: {len(all_nodes)}/{total_nodes} nodes alive")

                # --- Update energy ---
                energy_before = compute_total_energy(all_nodes)
                Computing.update_energy(all_nodes, node_positions, clusters, current_time)
                energy_after = compute_total_energy(all_nodes)

                energy_consumed_cycle = energy_before - energy_after
                total_energy_consumed += energy_consumed_cycle
                energy_remaining_total = energy_after

                # --- Reselect CH ---
                clusters = Clustering.reselect_cluster_heads(clusters, all_nodes)

                # --- Reroute (shared population for this phase) ---
                center_coords = build_center_coords(clusters, node_positions, bs=(200, 200, 400))

                n_cities_loop = int(len(center_coords))
                pop_size_loop = 30  # match phase loop: PSO particles=30; GA pop_size will also be 30
                shared_pop_loop = make_shared_population(n_cities_loop, pop_size_loop, seed=123 + cycle)

                try:
                    current_path, current_time = solve_route(
                        algorithm,
                        center_coords,
                        v_f,
                        v_AUV,
                        clusters=clusters,
                        phase="loop",
                        shared_pop=shared_pop_loop,
                    )
                except Exception as e:
                    print(f" Lỗi route cycle {cycle} ({algorithm}): {e}")
                    break

                # --- Remove dead nodes ---
                clusters, dead_nodes = Clustering.remove_dead_nodes(all_nodes, clusters)

                # Log cycle (alive sau khi loại node chết)
                alive_nodes_now = len(all_nodes)
                alive_ratio_now = alive_nodes_now / total_nodes if total_nodes > 0 else 0.0

                jsonl_append(
                    cycle_log_path,
                    {
                        "filename": filename,
                        "algorithm": algorithm,
                        "cycle": cycle,
                        "alive_nodes": int(alive_nodes_now),
                        "alive_ratio": round(float(alive_ratio_now), 6),
                        "current_time": float(current_time),
                        "energy_consumed_cycle": round(float(energy_consumed_cycle), 6),
                        "energy_consumed_total": round(float(total_energy_consumed), 6),
                        "energy_remaining_total": round(float(energy_remaining_total), 6),
                        "dead_nodes_in_cycle": int(len(dead_nodes)) if dead_nodes else 0,
                    },
                )

                # --- Reclustering nếu có node chết ---
                if dead_nodes:
                    if alive_ratio_now <= 0.1:
                        break

                    clusters = Clustering.recluster(all_nodes, node_positions, clustering, R_SEN, MAX_SIZE, MIN_SIZE)
                    if len(clusters) == 0:
                        break

                    center_coords = build_center_coords(clusters, node_positions, bs=(200, 200, 400))

                    n_cities_re = int(len(center_coords))
                    pop_size_re = 30
                    shared_pop_re = make_shared_population(n_cities_re, pop_size_re, seed=999 + cycle)

                    try:
                        current_path, current_time = solve_route(
                            algorithm,
                            center_coords,
                            v_f,
                            v_AUV,
                            clusters=clusters,
                            phase="recluster",
                            shared_pop=shared_pop_re,
                        )
                    except Exception as e:
                        print(f" Lỗi route sau recluster ({algorithm}): {e}")
                        break

            final_alive = len(all_nodes)
            final_alive_ratio = (final_alive / total_nodes) if total_nodes > 0 else 0.0

            result_data = {
                "filename": filename,
                "algorithm": algorithm,
                "initial_nodes": int(total_nodes),
                "cycles_completed": int(cycle - 1),
                "final_alive_nodes": int(final_alive),
                "final_alive_ratio": round(float(final_alive_ratio), 4),
                "total_energy_consumed": round(float(total_energy_consumed), 4),
                "cycle_log_file": os.path.basename(cycle_log_path),
            }

            file_results[algorithm] = result_data
            alg_file = os.path.join(file_output_folder, f"{algorithm}_result.json")
            with open(alg_file, "w") as f:
                json.dump(result_data, f, indent=4, ensure_ascii=False)

            global_results[algorithm][filename] = result_data

            print(
                f" {algorithm} xong: {cycle-1} cycles, {total_energy_consumed:.2f}J, "
                f"{final_alive}/{total_nodes} nodes sống"
            )
            print(f"Cycle log: {cycle_log_path}")

        file_summary_json = os.path.join(file_output_folder, f"summary_{base_name}.json")
        with open(file_summary_json, "w") as f:
            json.dump(file_results, f, indent=4, ensure_ascii=False)
        print(f"Đã lưu summary cho {filename}: {file_summary_json}")

    summary_data = {"algorithms": algorithms, "detailed_results": global_results}
    summary_json = os.path.join(output_main_folder, "summary_all_results.json")
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    print(f"\n HOÀN THÀNH! Tổng hợp: {summary_json}")


if __name__ == "__main__":
    main()