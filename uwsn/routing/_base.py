# ===========================================================
# routing/_base.py — Hàm dùng chung cho các biến thể PSO
#
# Trước đây code này lặp lại 3 lần trong:
#   Pso_routing, Pso_adaptive_noise, Pso_levy_flight
# ===========================================================


def get_swap_sequence(A, B):
    """
    Tính chuỗi swap biến đổi hoán vị A thành B.
    Bỏ qua vị trí 0 (depot).
    """
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
    """
    Áp dụng velocity (danh sách swap) lên position.
    Bỏ qua các swap chạm vào depot (index 0).
    """
    pos = position.copy()
    for i, j in velocity:
        if i > 0 and j > 0 and i < len(pos) and j < len(pos):
            pos[i], pos[j] = pos[j], pos[i]
    return pos


def validate_and_clone_swarm(init_swarm, n_cities):
    """
    Validate và clone init_swarm:
    - Mỗi route có length == n_cities
    - Chứa tất cả node 0..n_cities-1 đúng 1 lần
    - Depot 0 được xoay về vị trí đầu

    Returns:
        list[list[int]] — swarm đã validate
    """
    if init_swarm is None:
        return None

    swarm = []
    for route in init_swarm:
        r = list(route)
        if len(r) != n_cities:
            raise ValueError(
                f"init_swarm route length {len(r)} != n_cities={n_cities}"
            )

        seen = [0] * n_cities
        for x in r:
            if x < 0 or x >= n_cities:
                raise ValueError(f"init_swarm has invalid node index: {x}")
            if seen[x] == 1:
                raise ValueError(f"init_swarm has duplicate node: {x}")
            seen[x] = 1
        if seen[0] != 1:
            raise ValueError("init_swarm route must include depot 0")

        # Xoay để depot 0 ở vị trí đầu
        if r[0] != 0:
            p0 = r.index(0)
            r = r[p0:] + r[:p0]

        swarm.append(r)

    return swarm
