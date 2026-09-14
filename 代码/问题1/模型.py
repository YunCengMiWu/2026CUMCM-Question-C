#!/usr/bin/env python3
"""最小可运行版本 · 模型层：单日 MILP（方案丙）＋执行层（§14.6）。

变量（每槽 13 个，索引见 NAMES；另有 1 个窗口初态 E0）：
  a 执行/调整购电量, e 紧急购电量, C 充电量, D 放电量, E 槽末储电量,
  w 弃光, r 合同未消纳, u=|a−b|, np 净缺额正部, nm 负部, y(0-1), z(0-1), b 计划购电量
约束：执行一致的完整平衡（§14.5）＋储能递推＋容量/功率界＋互斥＋净额正负互斥。
目标：q1 = Σp·a；q2 = Σ(p·b+5p·e)；plan_adj = Σp·b；joint = Σ(p·a+0.5p·u+5p·e)（= §14.7 总成本）。
"""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from 口径 import DT, E_MIN, E_OP, ETA, K_DOWN, K_EMG, K_UP, M_SLOT, N_SLOT

NAMES = ["a", "e", "C", "D", "E", "w", "r", "u", "np", "nm", "y", "z", "b"]
IDX = {n: k for k, n in enumerate(NAMES)}
NV = len(NAMES)
I_E0 = N_SLOT * NV          # 窗口初态（0:10 的储电量）


def _var(t: int, name: str) -> int:
    return t * NV + IDX[name]


def solve_day(price, load_kw, pv_kw, E_start=None, mode="q1", b_fixed=None,
              E_clock_24=None, time_limit=60.0, mip_rel_gap=1e-6) -> dict:
    """求解单日模型。

    price/load_kw/pv_kw：长度 144（元/kWh、kW、kW）；E_start：窗口初态（kWh），None 表示自由；
    mode：q1（问题1，禁紧急、日循环＋字面锚定）、q2（问题2 计划=执行、允许紧急、终端自由）、
          plan_adj（问题3 的 0:00 计划，b=a）、joint（问题3 的调整，b 可固定、a 可调）。
    """
    ell = np.asarray(load_kw, dtype=float) * DT          # kWh
    v = np.asarray(pv_kw, dtype=float) * DT              # kWh
    p = np.asarray(price, dtype=float)                   # 元/kWh
    abar = ell + M_SLOT                                  # 购电量数值界（§14.5）

    lb = np.zeros(N_SLOT * NV + 1)
    ub = np.zeros(N_SLOT * NV + 1)
    c = np.zeros(N_SLOT * NV + 1)
    integrality = np.zeros(N_SLOT * NV + 1)

    for t in range(N_SLOT):
        lb[_var(t, "a")], ub[_var(t, "a")] = 0.0, abar[t]
        lb[_var(t, "e")], ub[_var(t, "e")] = 0.0, ell[t]
        lb[_var(t, "C")], ub[_var(t, "C")] = 0.0, M_SLOT
        lb[_var(t, "D")], ub[_var(t, "D")] = 0.0, M_SLOT
        lb[_var(t, "E")], ub[_var(t, "E")] = E_MIN, E_OP
        lb[_var(t, "w")], ub[_var(t, "w")] = 0.0, v[t]
        lb[_var(t, "r")], ub[_var(t, "r")] = 0.0, abar[t]
        lb[_var(t, "u")], ub[_var(t, "u")] = 0.0, abar[t]
        lb[_var(t, "np")], ub[_var(t, "np")] = 0.0, ell[t]
        lb[_var(t, "nm")], ub[_var(t, "nm")] = 0.0, v[t] + abar[t]
        lb[_var(t, "y")], ub[_var(t, "y")] = 0.0, 1.0
        lb[_var(t, "z")], ub[_var(t, "z")] = 0.0, 1.0
        lb[_var(t, "b")], ub[_var(t, "b")] = 0.0, abar[t]
        integrality[_var(t, "y")] = 1
        integrality[_var(t, "z")] = 1
    lb[I_E0], ub[I_E0] = E_MIN, E_OP

    rows, lows, highs = [], [], []

    def add(coefs, lo, hi):
        rows.append(coefs)
        lows.append(lo)
        highs.append(hi)

    for t in range(N_SLOT):
        one = np.zeros(N_SLOT * NV + 1)
        # (1) np − nm + a = ℓ − v
        one[_var(t, "np")] = 1.0
        one[_var(t, "nm")] = -1.0
        one[_var(t, "a")] = 1.0
        add(one, ell[t] - v[t], ell[t] - v[t])
        # (2) e − np + D = 0
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "e")] = 1.0
        one[_var(t, "np")] = -1.0
        one[_var(t, "D")] = 1.0
        add(one, 0.0, 0.0)
        # (3) w + r − nm + C = 0
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "w")] = 1.0
        one[_var(t, "r")] = 1.0
        one[_var(t, "nm")] = -1.0
        one[_var(t, "C")] = 1.0
        add(one, 0.0, 0.0)
        # (4) np − U⁺ y ≤ 0
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "np")] = 1.0
        one[_var(t, "y")] = -ell[t]
        add(one, -np.inf, 0.0)
        # (5) nm + U⁻ y ≤ U⁻
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "nm")] = 1.0
        one[_var(t, "y")] = v[t] + abar[t]
        add(one, -np.inf, v[t] + abar[t])
        # (6) C − M z ≤ 0 ; (7) D + M z ≤ M
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "C")] = 1.0
        one[_var(t, "z")] = -M_SLOT
        add(one, -np.inf, 0.0)
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "D")] = 1.0
        one[_var(t, "z")] = M_SLOT
        add(one, -np.inf, M_SLOT)
        # (8) 储能递推
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "E")] = 1.0
        one[_var(t, "C")] = -ETA
        one[_var(t, "D")] = 1.0 / ETA
        if t == 0:
            one[I_E0] = -1.0
        else:
            one[_var(t - 1, "E")] = -1.0
        add(one, 0.0, 0.0)
        # (9) r − a ≤ 0
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "r")] = 1.0
        one[_var(t, "a")] = -1.0
        add(one, -np.inf, 0.0)
        # (12) u ≥ a − b ; u ≥ b − a
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "u")] = 1.0
        one[_var(t, "a")] = -1.0
        one[_var(t, "b")] = 1.0
        add(one, 0.0, np.inf)
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "u")] = 1.0
        one[_var(t, "a")] = 1.0
        one[_var(t, "b")] = -1.0
        add(one, 0.0, np.inf)

    # b = a（计划即执行；问题3 的调整阶段 b 固定）
    for t in range(N_SLOT):
        one = np.zeros(N_SLOT * NV + 1)
        one[_var(t, "b")] = 1.0
        one[_var(t, "a")] = -1.0
        if mode in ("q1", "q2", "plan_adj") or b_fixed is not None:
            add(one, 0.0, 0.0)
        if b_fixed is not None:
            lb[_var(t, "b")] = ub[_var(t, "b")] = float(b_fixed[t])

    # 目标
    for t in range(N_SLOT):
        if mode == "q1":
            c[_var(t, "a")] = p[t]
        elif mode == "q2":
            c[_var(t, "b")] = p[t]
            c[_var(t, "e")] = K_EMG * p[t]
        elif mode == "plan_adj":
            c[_var(t, "b")] = p[t]
            c[_var(t, "e")] = K_EMG * p[t]
        elif mode == "joint":
            c[_var(t, "a")] = p[t]
            c[_var(t, "u")] = K_DOWN * p[t]
            c[_var(t, "e")] = K_EMG * p[t]
        else:
            raise ValueError(f"未知 mode: {mode}")

    if mode == "q1":
        for t in range(N_SLOT):
            lb[_var(t, "e")] = ub[_var(t, "e")] = 0.0   # 问题1 无紧急购电机制
        if E_clock_24 is not None:                        # 字面 24:00 锚定（E_{d,143}）
            lb[_var(N_SLOT - 2, "E")] = ub[_var(N_SLOT - 2, "E")] = float(E_clock_24)
        one = np.zeros(N_SLOT * NV + 1)                   # 日循环 E_{d,144} = E_{d,0}
        one[_var(N_SLOT - 1, "E")] = 1.0
        one[I_E0] = -1.0
        add(one, 0.0, 0.0)

    if E_start is not None:
        lb[I_E0] = ub[I_E0] = float(E_start)

    t0 = time.perf_counter()
    res = milp(c=c, constraints=[LinearConstraint(np.array(rows), np.array(lows), np.array(highs))],
               integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "presolve": True})
    wall = time.perf_counter() - t0

    x = res.x if res.x is not None else None
    out = {"status": int(res.status), "message": str(res.message), "wall_s": wall,
           "objective": float(res.fun) if res.fun is not None else None,
           "mip_gap": float(getattr(res, "mip_gap", np.nan)) if res.x is not None else None,
           "node_count": int(getattr(res, "mip_node_count", getattr(res, "node_count", -1)) or -1),
           "x": x, "mode": mode}
    if x is None:
        out["ok"] = False
        return out
    out["ok"] = int(res.status) == 0
    for name in NAMES:
        out[name] = np.array([x[_var(t, name)] for t in range(N_SLOT)])
    out["E0"] = float(x[I_E0])
    out["E_all"] = np.concatenate([[out["E0"]], out["E"]])   # E_0..E_144（窗_end）
    return out


def execute_day(plan, target_C, target_D, load_kw, pv_kw, E_start) -> dict:
    """§14.6 执行层：给定锁定计划与储能目标，用实际量逐槽执行安全修正。"""
    ell = np.asarray(load_kw, dtype=float) * DT
    v = np.asarray(pv_kw, dtype=float) * DT
    a = np.asarray(plan, dtype=float)
    C_bar = np.asarray(target_C, dtype=float)
    D_bar = np.asarray(target_D, dtype=float)
    C = np.zeros(N_SLOT); D = np.zeros(N_SLOT); e = np.zeros(N_SLOT)
    w = np.zeros(N_SLOT); r = np.zeros(N_SLOT); E = np.zeros(N_SLOT + 1)
    E[0] = float(E_start)
    for t in range(N_SLOT):
        n = ell[t] - v[t] - a[t]
        n_pos, n_neg = max(n, 0.0), max(-n, 0.0)
        C[t] = min(C_bar[t], M_SLOT, (E_OP - E[t]) / ETA, n_neg)
        D[t] = min(D_bar[t], M_SLOT, ETA * (E[t] - E_MIN), n_pos)
        C[t] = max(C[t], 0.0); D[t] = max(D[t], 0.0)
        e[t] = max(n_pos - D[t], 0.0)
        s = max(n_neg - C[t], 0.0)
        w[t] = min(v[t], s)
        r[t] = s - w[t]
        E[t + 1] = E[t] + ETA * C[t] - D[t] / ETA
    return {"a": a, "C": C, "D": D, "e": e, "w": w, "r": r, "E": E}
