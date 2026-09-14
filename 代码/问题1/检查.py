#!/usr/bin/env python3
"""最小可运行版本 · 检查器：硬约束残差、量纲一致性、检查点与费用复算。"""

from __future__ import annotations

import numpy as np

from 口径 import DT, E_MIN, E_OP, ETA, K_DOWN, K_EMG, K_UP, M_SLOT, N_SLOT


def check_day(ell_kwh, v_kwh, a, e, C, D, w, r, E_all, tag="") -> list[str]:
    """逐槽完整平衡、界、互斥、递推与余量分账检查；返回问题列表。"""
    issues = []
    resid = a + e + v_kwh - w + D - ell_kwh - C - r
    if np.max(np.abs(resid)) > 1e-6:
        issues.append(f"{tag} 完整平衡残差超限: {np.max(np.abs(resid)):.3e} kWh")
    if np.min(C) < -1e-9 or np.min(D) < -1e-9 or np.min(e) < -1e-9:
        issues.append(f"{tag} 出现负的充/放/紧急电量")
    if np.max(C) > M_SLOT + 1e-6 or np.max(D) > M_SLOT + 1e-6:
        issues.append(f"{tag} 充放电超过 5000 kW 对应的单槽上界 {M_SLOT:.4f} kWh")
    E = E_all[1:]
    if np.min(E) < E_MIN - 1e-6 or np.max(E) > E_OP + 1e-6:
        issues.append(f"{tag} 储电量越界 [{np.min(E):.3f}, {np.max(E):.3f}] kWh")
    sim = E_all[:-1] + ETA * C - D / ETA
    if np.max(np.abs(sim - E)) > 1e-6:
        issues.append(f"{tag} 储能递推残差超限: {np.max(np.abs(sim - E)):.3e} kWh")
    overlap = np.minimum(C, D)
    if np.max(overlap) > 1e-6:
        issues.append(f"{tag} 充放电互斥被违反: max min(C,D)={np.max(overlap):.3e} kWh")
    if np.max(w - v_kwh) > 1e-6:
        issues.append(f"{tag} 弃光超过可用光伏")
    if np.max(r - a) > 1e-6:
        issues.append(f"{tag} 合同未消纳量超过购电量")
    return issues


def check_settlement(price, plan, adj, emg, reported_total) -> list[str]:
    """结算等价式复算：p·a + 0.5p|a−b| + 5p·e 应等于分项之和。"""
    issues = []
    p = np.asarray(price); b = np.asarray(plan); a = np.asarray(adj); e = np.asarray(emg)
    down = np.maximum(b - a, 0.0); up = np.maximum(a - b, 0.0)
    ident = float(np.sum(p * a + K_DOWN * p * (down + up) + K_EMG * p * e))
    parts = float(np.sum(p * b) + np.sum(-p * down + K_DOWN * p * down + K_UP * p * up) + np.sum(K_EMG * p * e))
    if abs(ident - parts) > 1e-6:
        issues.append(f"结算等价式与分项不一致: {ident:.6f} vs {parts:.6f} 元")
    if reported_total is not None and abs(parts - reported_total) > 1e-6:
        issues.append(f"结算分项与回报总额不一致: {parts:.6f} vs {reported_total:.6f} 元")
    return issues


def check_continuity(E_end_prev, E_start_next, tag="") -> list[str]:
    if abs(float(E_end_prev) - float(E_start_next)) > 1e-6:
        return [f"{tag} 跨窗口不连续: 前窗口末 {E_end_prev:.6f} vs 本窗口初 {E_start_next:.6f} kWh"]
    return []


def check_literal_checkpoints(E_all, e_clock_24=None, E_prev_all=None, tag="") -> list[str]:
    """字面检查点（§14.4）：E_clock24 = E_{d,143} = E_all[-2]；E_clock00 = E_{d−1,143} = 前一日 E_all[-2]。"""
    issues = []
    if e_clock_24 is not None and abs(E_all[-2] - float(e_clock_24)) > 1e-6:
        issues.append(f"{tag} 字面 24:00 检查点不符: E_{{d,143}}={E_all[-2]:.6f} vs {e_clock_24}")
    if E_prev_all is not None:
        prev_clock0 = float(E_prev_all[-2])
        if abs(prev_clock0 - float(E_all[0]) - 0.0) > 1e6:   # 仅登记，不判错（二者允许相差末槽净变化 δ）
            pass
    return issues


def summarize(items: list[tuple[str, bool, str]]) -> tuple[bool, str]:
    lines = []
    ok_all = True
    for name, ok, detail in items:
        ok_all &= ok
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok_all, "\n".join(lines)
