# ===========================================================
# runner.py — Simulation loop chính: solve_route() & main()
# ===========================================================

import os
import json
import copy
import numpy as np
import subprocess  # Thêm dòng này để gọi lệnh hệ thống
import sys         # Thêm dòng này để lấy đúng môi trường Python đang chạy

from . import config
from .clustering import Clustering
from .energy import EnergyModel
from .routing import Greedy, ClusterTSP_GA, Pso_routing, Pso_adaptive_noise, Pso_levy_flight, Aco_routing, Pso_2opt
from .utils import jsonl_append, compute_total_energy, build_center_coords, make_shared_population


def solve_route(algorithm, center_coords, v_f, v_AUV, clusters=None, phase="default", shared_pop=None):
    """
    Chạy thuật toán routing và trả về (path, time).

    Args:
        algorithm: str - tên thuật toán ("Greedy", "GA", "PSO", "PSO1", "PSO2")
        center_coords: np.ndarray - tọa độ [BS, CH1, CH2, ...]
        v_f: float - vận tốc dòng chảy
        v_AUV: float - vận tốc AUV
        clusters: dict - thông tin clusters (cần cho GA)
        phase: str - "init", "loop", hoặc "recluster"
        shared_pop: list - shared population cho GA/PSO

    Returns:
        (path, time)
    """
    if algorithm == "Greedy":
        path, t = Greedy.greedy_tsp(center_coords, v_f=v_f, v_AUV=v_AUV)
        return path, float(t), None

    if algorithm == "GA":
        if clusters is None:
            raise ValueError("GA cần tham số clusters")

        ga_solver = ClusterTSP_GA(
            clusters,
            center_coords,
            ga_params={
                "pop_size": 50 if phase == "init" else 30,
                "generations": 200 if phase == "init" else 80,
                "v_f": v_f,
                "v_AUV": v_AUV,
                "verbose": False,
            },
        )

        best, mapped_path, best_time, history, gen_stats = ga_solver.evolve(init_population=shared_pop)
        return mapped_path, float(best_time), gen_stats

    if algorithm == "PSO":
        path, cost, stats = Pso_routing.multi_pso_tsp(
            center_coords,
            v_f=v_f,
            v_AUV=v_AUV,
            n_outer=1,
            n_particles=50 if phase == "init" else 30,
            max_iter=200 if phase == "init" else 80,
            init_swarm=shared_pop,
            verbose=False,
        )
        return path, float(cost), stats

    if algorithm == "PSO1":
        path, cost, stats = Pso_adaptive_noise.multi_pso_tsp(
            center_coords,
            v_f=v_f,
            v_AUV=v_AUV,
            n_outer=1,
            n_particles=50 if phase == "init" else 30,
            max_iter=200 if phase == "init" else 80,
            init_swarm=shared_pop,
            verbose=False,
        )
        return path, float(cost), stats

    if algorithm == "PSO2":
        path, cost, stats = Pso_levy_flight.multi_pso_tsp(
            center_coords,
            v_f=v_f,
            v_AUV=v_AUV,
            n_outer=1,
            n_particles=50 if phase == "init" else 30,
            max_iter=200 if phase == "init" else 80,
            init_swarm=shared_pop,
            verbose=False,
        )
        return path, float(cost), stats


# THÊM ĐOẠN NÀY CHO THUẬT TOÁN MỚI
    if algorithm == "PSO3":
        path, cost, stats = Pso_2opt.multi_pso_tsp(
            center_coords,
            v_f=v_f,
            v_AUV=v_AUV,
            n_outer=1,
            n_particles=50 if phase == "init" else 30,
            max_iter=200 if phase == "init" else 80,
            init_swarm=shared_pop,
            verbose=False,
        )
        return path, float(cost), stats
    
# thêm aco
    if algorithm == "ACO":
            path, cost, stats = Aco_routing.multi_aco_tsp(
                center_coords,
                v_f=v_f,
                v_AUV=v_AUV,
                n_outer=1,
                n_ants=50 if phase == "init" else 30,
                max_iter=200 if phase == "init" else 80,
                # alpha=config.ACO_ALPHA,
                # beta=config.ACO_BETA,
                # rho=config.ACO_RHO,
                # Q=config.ACO_Q,
                init_swarm=shared_pop,
                verbose=False,
            )
            return path, float(cost), stats

    raise ValueError(f"Unknown algorithm: {algorithm}")

def main(input_file=None, output_folder=None):
    # input_folder = "/kaggle/input/nodes-data"
    # output_main_folder = "/kaggle/working/results_history"
    #
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. Nhận trực tiếp đường dẫn file từ batch_runner (tránh bị lặp thư mục)
    if input_file is not None:
        input_path = os.path.abspath(input_file)
        if not os.path.exists(input_path):
            print(f"Không tìm thấy file: {input_path}")
            return
        filename = os.path.basename(input_path)
        files = [filename]
        input_folder = os.path.dirname(input_path)
    else:
        input_folder = os.path.join(BASE_DIR, "data", "nodes-data")
        files = [f for f in os.listdir(input_folder) if f.endswith(".json")] if os.path.exists(input_folder) else []

    # 2. Định nghĩa thư mục đầu ra
    if output_folder is not None:
        output_main_folder = output_folder
    else:
        output_main_folder = os.path.join(BASE_DIR, "results")

    os.makedirs(output_main_folder, exist_ok=True)

    # 

    if not os.path.exists(input_folder):
        print(f"Lỗi: Thư mục {input_folder} không tồn tại!")
        return

    # files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    if input_file is not None:
        files = [os.path.basename(input_file)]
    else:
        files = [f for f in os.listdir(input_folder) if f.endswith(".json")]


    for f in files:
        path = os.path.join(input_folder, f)
        if not os.path.exists(path):
            print(f"Không tìm thấy file: {path}")
            return


    
    if len(files) == 0:
        print(f"Không tìm thấy file dữ liệu nào trong {input_folder}")
        return

    INITIAL_ENERGY = config.INITIAL_ENERGY
    v_f = config.V_F
    v_AUV = config.V_AUV
    R_SEN = config.R_SEN
    MAX_SIZE = config.MAX_CLUSTER_SIZE
    MIN_SIZE = config.MIN_CLUSTER_SIZE

    algorithms = config.ALGORITHMS
    global_results = {alg: {} for alg in algorithms}

    clustering = Clustering(
        space_size=config.SPACE_SIZE,
        r_sen=R_SEN,
        max_cluster_size=MAX_SIZE,
        min_cluster_size=MIN_SIZE,
    )

    for filename in sorted(files):
        if input_file is not None:
            input_path = os.path.abspath(input_file)
        else:
            input_path = os.path.join(input_folder, filename)

        base_name = filename.replace(".json", "")
        
        # Nếu chạy batch thì lưu thẳng vào output_folder, nếu không thì tạo folder mới
        if output_folder is not None:
            file_output_folder = output_folder
        else:
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

        bs = config.BASE_STATION
        center_coords_init = build_center_coords(initial_clusters, node_positions, bs=bs)

        file_results = {}

        for algorithm in algorithms:
            print(f"\n{'='*60}")
            print(f"Chạy thuật toán: {algorithm}")
            print(f"{'='*60}")

            all_nodes = copy.deepcopy(initial_nodes)
            clusters = copy.deepcopy(initial_clusters)

            # INIT route with SHARED population
            n_cities_init = int(len(center_coords_init))
            pop_size_init = 50
            shared_pop_init = make_shared_population(n_cities_init, pop_size_init, seed=123)

            try:
                current_path, current_time, init_stats = solve_route(
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

            if init_stats is not None:
                convergence_path = os.path.join(
                    file_output_folder, f"{algorithm}_init_convergence.json"
                )
                with open(convergence_path, "w", encoding="utf-8") as cf:
                    json.dump(init_stats, cf, ensure_ascii=False)

            # File log theo cycle
            cycle_log_path = os.path.join(file_output_folder, f"{algorithm}_cycle_log.jsonl")
            with open(cycle_log_path, "w", encoding="utf-8") as f:
                f.write("")

            cycle = 0
            total_energy_consumed = 0.0
            total_packets_expected = 0
            total_packets_delivered = 0

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
                # EnergyModel.update_energy(all_nodes, node_positions, clusters, current_time)
                packets_expected_cycle, packets_delivered_cycle = EnergyModel.update_energy(
                    all_nodes, node_positions, clusters, current_time
                )
                energy_after = compute_total_energy(all_nodes)

                energy_consumed_cycle = energy_before - energy_after
                total_energy_consumed += energy_consumed_cycle
                energy_remaining_total = energy_after

                total_packets_expected += packets_expected_cycle
                total_packets_delivered += packets_delivered_cycle
                pdr_cycle = (
                    packets_delivered_cycle / packets_expected_cycle
                    if packets_expected_cycle > 0 else 1.0
                )

                # --- Reselect CH ---
                clusters = Clustering.reselect_cluster_heads(clusters, all_nodes)

                # --- Reroute ---
                center_coords = build_center_coords(clusters, node_positions, bs=bs)

                n_cities_loop = int(len(center_coords))
                pop_size_loop = 30
                shared_pop_loop = make_shared_population(n_cities_loop, pop_size_loop, seed=123 + cycle)

                try:
                    current_path, current_time, _loop_stats = solve_route(
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
                        "packets_expected_cycle": int(packets_expected_cycle),
                        "packets_delivered_cycle": int(packets_delivered_cycle),
                        "pdr_cycle": round(float(pdr_cycle), 6),
                    },
                )

                # --- Reclustering nếu có node chết ---
                if dead_nodes:
                    if alive_ratio_now <= 0.1:
                        break

                    clusters = Clustering.recluster(all_nodes, node_positions, clustering, R_SEN, MAX_SIZE, MIN_SIZE)
                    if len(clusters) == 0:
                        break

                    center_coords = build_center_coords(clusters, node_positions, bs=bs)

                    n_cities_re = int(len(center_coords))
                    pop_size_re = 30
                    shared_pop_re = make_shared_population(n_cities_re, pop_size_re, seed=999 + cycle)

                    try:
                        current_path, current_time, _re_stats = solve_route(
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
            # final_alive_ratio = (final_alive / total_nodes) if total_nodes > 0 else 0.0

            final_alive_ratio = (final_alive / total_nodes) if total_nodes > 0 else 0.0
            overall_pdr = (
                total_packets_delivered / total_packets_expected
                if total_packets_expected > 0 else 1.0
            )

            result_data = {
                "filename": filename,
                "algorithm": algorithm,
                "initial_nodes": int(total_nodes),
                "cycles_completed": int(cycle - 1),
                "final_alive_nodes": int(final_alive),
                "final_alive_ratio": round(float(final_alive_ratio), 4),
                "total_energy_consumed": round(float(total_energy_consumed), 4),
                 "total_packets_expected": int(total_packets_expected),
                "total_packets_delivered": int(total_packets_delivered),
                "overall_pdr": round(float(overall_pdr), 6),
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

        # ========================================================
        # THÊM ĐOẠN NÀY: TỰ ĐỘNG GỌI PLOT_RESULTS.PY
        # ========================================================
        print(f"\n---> Đang tự động vẽ biểu đồ cho kịch bản {filename}...")
        try:
            # sys.executable đảm bảo dùng đúng môi trường Python hiện tại
            subprocess.run(
                [sys.executable, "plot_results.py", "--folder", file_output_folder],
                check=True
            )
        except Exception as e:
            print(f" Lỗi khi tự động vẽ biểu đồ: {e}")
            # ========================================================

        # --- Kết thúc vòng lặp for filename in sorted(files): ---

    summary_data = {"algorithms": algorithms, "detailed_results": global_results}
    summary_json = os.path.join(output_main_folder, "summary_all_results.json")
    with open(summary_json, "w") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    print(f"\n HOÀN THÀNH! Tổng hợp: {summary_json}")
