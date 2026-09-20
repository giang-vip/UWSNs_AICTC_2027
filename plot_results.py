# ===========================================================
# plot_results.py — Vẽ biểu đồ minh họa/so sánh kết quả các
# thuật toán routing (Greedy, GA, PSO, PSO1, PSO2, ACO)
#
# Đọc trực tiếp các file mà uwsn/runner.py đã sinh ra trong
# results/folder_<ten_file>/ :
#   - <algorithm>_cycle_log.jsonl         (log từng cycle)
#   - <algorithm>_result.json             (tổng kết 1 thuật toán)
#   - <algorithm>_init_convergence.json   (lịch sử hội tụ GA/PSO/ACO)
#   - summary_<ten_file>.json             (tổng hợp mọi thuật toán)
#
# Cách dùng:
#   python plot_results.py --folder results/folder_test_small
#   python plot_results.py --folder results/folder_test_small --out charts/
# ===========================================================

import os
import json
import argparse
import glob

import matplotlib.pyplot as plt

# Màu cố định cho từng thuật toán để đồng nhất giữa các biểu đồ
ALGO_COLORS = {
    "Greedy": "#7f7f7f",
    "GA": "#2ca02c",
    "PSO": "#1f77b4",
    "PSO1": "#9467bd",
    "PSO2": "#8c564b",
    "ACO": "#d62728",
    "PSO3": "#ff7f0e",
}


def _color(algo):
    return ALGO_COLORS.get(algo, None)


def load_cycle_logs(folder):
    """Đọc tất cả *_cycle_log.jsonl trong folder -> {algorithm: [dict,...]}"""
    data = {}
    for path in sorted(glob.glob(os.path.join(folder, "*_cycle_log.jsonl"))):
        algo = os.path.basename(path).replace("_cycle_log.jsonl", "")
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if rows:
            data[algo] = rows
    return data


def load_results(folder):
    """Đọc tất cả *_result.json -> {algorithm: dict}"""
    data = {}
    for path in sorted(glob.glob(os.path.join(folder, "*_result.json"))):
        algo = os.path.basename(path).replace("_result.json", "")
        with open(path, "r", encoding="utf-8") as f:
            data[algo] = json.load(f)
    return data


def load_convergence(folder):
    """Đọc tất cả *_init_convergence.json -> {algorithm: [dict,...]}"""
    data = {}
    for path in sorted(glob.glob(os.path.join(folder, "*_init_convergence.json"))):
        algo = os.path.basename(path).replace("_init_convergence.json", "")
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if rows:
            data[algo] = rows
    return data


# ── 1. Network lifetime: so nut con song theo cycle ────────
def plot_network_lifetime(cycle_logs, out_path):
    plt.figure(figsize=(8, 5))
    for algo, rows in cycle_logs.items():
        x = [r["cycle"] for r in rows]
        y = [r["alive_nodes"] for r in rows]
        plt.plot(x, y, label=algo, color=_color(algo), linewidth=1.8)

    plt.xlabel("Cycle (vòng thu thập dữ liệu)")
    plt.ylabel("Số node còn sống")
    plt.title("Network Lifetime — Số node sống theo thời gian")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu: {out_path}")


# ── 2. Năng lượng tiêu thụ tích lũy theo cycle ──────────────
def plot_energy_consumed(cycle_logs, out_path):
    plt.figure(figsize=(8, 5))
    for algo, rows in cycle_logs.items():
        x = [r["cycle"] for r in rows]
        y = [r["energy_consumed_total"] for r in rows]
        plt.plot(x, y, label=algo, color=_color(algo), linewidth=1.8)

    plt.xlabel("Cycle")
    plt.ylabel("Tổng năng lượng đã tiêu thụ (J)")
    plt.title("Năng lượng tiêu thụ tích lũy theo thời gian")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu: {out_path}")


# ── 3. So sánh số cycle sống được (bar chart) ───────────────
def plot_cycles_completed_bar(results, out_path):
    algos = list(results.keys())
    values = [results[a]["cycles_completed"] for a in algos]
    colors = [_color(a) for a in algos]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(algos, values, color=colors)
    plt.ylabel("Số cycle hoàn thành (tuổi thọ mạng)")
    plt.title("So sánh tuổi thọ mạng giữa các thuật toán")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v}",
                  ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu: {out_path}")


# ── 4. So sánh tổng năng lượng tiêu thụ tới lúc mạng chết ───
def plot_total_energy_bar(results, out_path):
    algos = list(results.keys())
    values = [results[a]["total_energy_consumed"] for a in algos]
    colors = [_color(a) for a in algos]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(algos, values, color=colors)
    plt.ylabel("Tổng năng lượng tiêu thụ (J)")
    plt.title("So sánh tổng năng lượng tiêu thụ tới khi mạng chết")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                  ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu: {out_path}")


# ── 5. Đường hội tụ của các thuật toán metaheuristic ────────
def plot_convergence(convergence, out_path):
    has_data = any(len(rows) > 0 for rows in convergence.values())
    if not has_data:
        print("Không có dữ liệu convergence (chỉ Greedy?), bỏ qua biểu đồ này.")
        return

    plt.figure(figsize=(8, 5))
    for algo, rows in convergence.items():
        x = [r["step"] for r in rows]
        # Lấy giá trị tốt nhất LŨY KẾ (running min) thay vì best của riêng
        # từng vòng lặp — đây mới là định nghĩa đúng của convergence curve.
        raw = [r["best_time"] for r in rows]
        y = []
        running_best = float("inf")
        for v in raw:
            running_best = min(running_best, v)
            y.append(running_best)
        plt.plot(x, y, label=algo, color=_color(algo), linewidth=1.8)

    plt.xlabel("Iteration / Generation")
    plt.ylabel("Best route time (s)")
    plt.title("Đường hội tụ (Convergence Curve) — lần route đầu tiên")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu: {out_path}")

# ── 6. PDR theo thời gian ───────────────────────────────────
def plot_pdr_over_time(cycle_logs, out_path):
    has_pdr = any("pdr_cycle" in rows[0] for rows in cycle_logs.values() if rows)
    if not has_pdr:
        print("Log cũ chưa có 'pdr_cycle' — bỏ qua biểu đồ PDR.")
        return

    plt.figure(figsize=(8, 5))
    for algo, rows in cycle_logs.items():
        if "pdr_cycle" not in rows[0]:
            continue
        x = [r["cycle"] for r in rows]
        y = [r["pdr_cycle"] for r in rows]
        plt.plot(x, y, label=algo, color=_color(algo), linewidth=1.2, alpha=0.8)

    plt.xlabel("Cycle")
    plt.ylabel("PDR (Packet Delivery Ratio)")
    plt.ylim(-0.02, 1.05)
    plt.title("PDR theo thời gian")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu: {out_path}")


# ── 7. So sánh PDR trung bình (bar chart) ───────────────────
def plot_overall_pdr_bar(results, out_path):
    if not results or "overall_pdr" not in next(iter(results.values())):
        print("*_result.json chưa có 'overall_pdr' — bỏ qua bar chart PDR.")
        return

    algos = list(results.keys())
    values = [results[a]["overall_pdr"] for a in algos]
    colors = [_color(a) for a in algos]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(algos, values, color=colors)
    plt.ylabel("PDR trung bình toàn vòng đời mạng")
    plt.ylim(0, 1.05)
    plt.title("So sánh PDR tổng thể giữa các thuật toán")
    plt.grid(axis="y", alpha=0.3)

    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                  ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Đã lưu: {out_path}")



def main():
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ so sánh kết quả UWSN routing")
    parser.add_argument("--folder", required=True,
                         help="Thư mục kết quả, vd: results/folder_test_small")
    parser.add_argument("--out", default=None,
                         help="Thư mục lưu ảnh (mặc định: <folder>/charts)")
    args = parser.parse_args()

    folder = args.folder
    out_dir = args.out or os.path.join(folder, "charts")
    os.makedirs(out_dir, exist_ok=True)

    cycle_logs = load_cycle_logs(folder)
    results = load_results(folder)
    convergence = load_convergence(folder)

    if not cycle_logs:
        print(f"Không tìm thấy *_cycle_log.jsonl trong {folder}. Đã chạy runner.main() chưa?")
        return

    plot_network_lifetime(cycle_logs, os.path.join(out_dir, "network_lifetime.png"))
    plot_energy_consumed(cycle_logs, os.path.join(out_dir, "energy_consumed.png"))
    plot_pdr_over_time(cycle_logs, os.path.join(out_dir, "pdr_over_time.png"))

    if results:
        plot_cycles_completed_bar(results, os.path.join(out_dir, "cycles_completed_bar.png"))
        plot_total_energy_bar(results, os.path.join(out_dir, "total_energy_bar.png"))
        plot_overall_pdr_bar(results, os.path.join(out_dir, "overall_pdr_bar.png"))
    else:
        print("Chưa có *_result.json (thuật toán chưa chạy xong) — bỏ qua 3 bar chart.")

    if convergence:
        plot_convergence(convergence, os.path.join(out_dir, "convergence.png"))

    print(f"\nHoàn tất! Xem ảnh trong: {out_dir}")


if __name__ == "__main__":
    main()