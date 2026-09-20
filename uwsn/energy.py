# ===========================================================
# energy.py — Mô hình năng lượng cho member node & cluster head
# ===========================================================

import math

from . import config


class EnergyModel:
    """Tính toán và cập nhật năng lượng cho mạng UWSN."""

    @staticmethod
    def energy_member(best_time, d):
        """
        Tính năng lượng tiêu thụ của một member node trong 1 cycle.

        Args:
            best_time: float - thời gian AUV di chuyển qua cluster
            d: float - khoảng cách từ member đến cluster head

        Returns:
            (E_total, E_tx, E_rx)
        """
        G = config.MEMBER_G
        L = config.MEMBER_L
        P_r = config.MEMBER_P_R
        P_idle = config.MEMBER_P_IDLE
        DR = config.MEMBER_DR
        E_ELEC = config.E_ELEC
        EPS_FS = config.EPS_FS

        # Time to transmit
        T_tx = G * L / DR

        # Energy terms
        E_tx = G * L * E_ELEC + G * L * EPS_FS * (d ** 2)
        E_rx = P_r * T_tx
        E_idle = (best_time - 2 * T_tx) * P_idle
        E_total = E_tx + E_rx + E_idle

        return E_total, E_tx, E_rx

    @staticmethod
    def energy_cluster_head(best_time, n_members):
        """
        Tính năng lượng tiêu thụ của cluster head trong 1 cycle.

        Args:
            best_time: float - thời gian AUV di chuyển qua cluster
            n_members: int - số member trong cluster

        Returns:
            float - tổng năng lượng tiêu thụ
        """
        G = config.CH_G
        L = config.CH_L
        P_t = config.CH_P_T
        P_idle = config.CH_P_IDLE
        DR = config.CH_DR
        DR_i = config.CH_DR_I
        E_ELEC = config.E_ELEC
        E_AGG = config.CH_E_AGG

        # Time
        T_rx = G * L * n_members / DR
        T_tx = G * L * n_members / DR_i

        # Energy terms
        E_rx = G * L * n_members * E_ELEC + G * L * n_members * E_AGG
        E_tx = P_t * T_tx
        E_idle = (best_time - T_rx - T_tx) * P_idle

        return E_rx + E_tx + E_idle

    @staticmethod
    def update_energy(all_nodes, node_positions, clusters, best_time):
        """
        Cập nhật năng lượng cho tất cả node trong mạng sau 1 cycle.
        Returns:
                    (packets_expected, packets_delivered) — dùng để tính PDR.
        
        Args:
            all_nodes: dict - thông tin tất cả node (sẽ bị thay đổi in-place)
            node_positions: dict - vị trí các node
            clusters: dict - thông tin các cluster
            best_time: float - thời gian AUV di chuyển
        """

        packets_expected = 0
        packets_delivered = 0

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
            members_delivered_to_ch = 0

            for nid in nodes:
                if nid == ch:
                    continue
                if nid not in all_nodes:
                    continue

                packets_expected += config.MEMBER_G

                d = math.dist(node_positions[nid], ch_pos)
                E_total, E_tx, E_rx = EnergyModel.energy_member(best_time, d)
                if all_nodes[nid]['residual_energy'] < (E_tx + E_rx):
                    all_nodes[nid]['residual_energy'] = 0.0
                    continue
                all_nodes[nid]['residual_energy'] -= E_total
                all_nodes[nid]['residual_energy'] = max(
                    all_nodes[nid]['residual_energy'], 0.0
                )

                n_members += 1
                members_delivered_to_ch += config.MEMBER_G

            # CLUSTER HEAD
            E_ch = EnergyModel.energy_cluster_head(best_time, n_members)

            all_nodes[ch]['residual_energy'] -= E_ch
            if all_nodes[ch]['residual_energy'] < 0:
                all_nodes[ch]['residual_energy'] = 0.0
                continue
            all_nodes[ch]['residual_energy'] = max(
                all_nodes[ch]['residual_energy'], 0.0
            )

            packets_delivered += members_delivered_to_ch

        return packets_expected, packets_delivered
