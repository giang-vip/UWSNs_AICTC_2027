# ===========================================================
# config.py — Hằng số & tham số mặc định cho mô phỏng UWSN
# ===========================================================

# ---- Vị trí trạm gốc (Base Station) ----
# BASE_STATION = (200, 200, 400)
# Đặt ở chính giữa mặt nước (X=1000, Y=1000, Z=1500) để tối ưu năng lượng truyền tải
BASE_STATION = (1000, 1000, 1500)

# ---- Tham số vật lý AUV ----
V_F = 1.2          # Vận tốc dòng chảy (m/s)
V_AUV = 3.0        # Vận tốc AUV (m/s)

# ---- Tham số phân cụm ----
# SPACE_SIZE = 400
# R_SEN = 60          # Bán kính cảm biến (m)
# MAX_CLUSTER_SIZE = 25
# MIN_CLUSTER_SIZE = 10


SPACE_SIZE = 2000      # Cập nhật theo kích thước X, Y tối đa của file dữ liệu
R_SEN = 450            # Tăng mạnh bán kính để bao quát không gian rộng
MAX_CLUSTER_SIZE = 50  # Ép K-means gom cụm to, giảm tổng số cụm xuống
MIN_CLUSTER_SIZE = 15

# ---- Năng lượng ----
# INITIAL_ENERGY = 100.0   # J
INITIAL_ENERGY = 1000.0  # Tăng mức pin khởi tạo do khoảng cách truyền xa (J)

# Tham số năng lượng member
MEMBER_G = 100           # Số gói tin
MEMBER_L = 1024          # Kích thước gói (bit)
MEMBER_P_R = 0.8e-3      # Công suất nhận (W)
MEMBER_P_IDLE = 0.1e-3   # Công suất idle (W)
MEMBER_DR = 4000          # Data rate (bps)
E_ELEC = 50e-9            # J/bit
EPS_FS = 10e-12           # J/bit/m^2

# Tham số năng lượng cluster head
CH_G = 100
CH_L = 1024
CH_P_T = 1.6e-3          # Công suất truyền CH (W)
CH_P_IDLE = 0.1e-3
CH_DR = 4000              # Data rate intra-cluster
CH_DR_I = 10000           # Data rate CH -> AUV
CH_E_AGG = 5e-9           # J/bit (aggregation)

# ---- Danh sách thuật toán ----
ALGORITHMS = ["ACO", "Greedy", "GA"]


