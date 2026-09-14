# -*- coding: utf-8 -*-
"""
核心库：数据读取 / 滚动预测 / 整日残差场景 / 两阶段随机LP内核 / 实时调度 / 结算
新思路路线：预测 → 残差场景 → 两阶段随机MILP(实际为LP) → 5倍紧急购电；MPC+调整成本；波动电价。
所有模型为 LP（HiGHS, scipy.optimize.linprog）。
"""
import os
import numpy as np
import pandas as pd
import openpyxl
from scipy import sparse
from scipy.optimize import linprog

# 路径：默认由本文件位置推导（本文件在 03_代码/，其上一级即项目根），不再写死绝对路径。
# 2026-09-11 修复：原值 r"C:\Users\my\Desktop\MILP+两阶段+MPC" 是搬迁前的旧机器路径，项目移到
# C:\Users\John\Desktop\26题\MILP+两阶段+MPC 后不存在，导致所有 p*.py 直接 FileNotFoundError。
# 三个环境变量为可选覆盖，便于把流水线重跑到临时目录、不覆盖已交付的 results/ 与检验日志：
#   MICROGRID_ROOT / MICROGRID_RES / MICROGRID_LOG
_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("MICROGRID_ROOT", os.path.dirname(_HERE))
import sys
sys.path.insert(0, os.path.join(ROOT, ".runtime"))
SRC = os.path.join(ROOT, "00_题目与原始数据", "C题", "附件")
RES = os.environ.get("MICROGRID_RES", os.path.join(ROOT, "04_结果与检验", "results"))
LOGD = os.environ.get("MICROGRID_LOG", os.path.join(ROOT, "04_结果与检验", "检验日志"))
SEED = 2025
DT = 1.0 / 6.0            # 10min in hours
ETA_C = 0.9               # charge efficiency (one-way)
ETA_D = 0.9               # discharge efficiency (one-way)
E_MIN, E_MAX, E_INIT = 1200.0, 10800.0, 6000.0
P_MAX_KWH = 5000.0 * DT   # max charge/discharge energy per slot (kWh)
BIG = 1e7
# 吞吐平局价；实际执行另按同SOC消去配对吞吐，核验互斥。
TP_PRICE = 1e-6

# ---------------------------------------------------------------- 数据读取
def _norm_label(v):
    if isinstance(v, str):
        return v.strip()
    try:
        return f"{v.hour:02d}:{v.minute:02d}"
    except Exception:
        return str(v)

def load_all():
    d = {}
    a1 = pd.read_excel(os.path.join(SRC, "附件1.xlsx"))
    d["p_fix"] = a1["电价"].to_numpy(float)          # (144,)
    d["L1"] = a1["小区负载"].to_numpy(float)
    d["G1hat"] = a1["光伏发电预测功率"].to_numpy(float)
    a2L = pd.read_excel(os.path.join(SRC, "附件2.xlsx"), sheet_name="小区负载")
    a2G = pd.read_excel(os.path.join(SRC, "附件2.xlsx"), sheet_name="光伏发电实际功率")
    d["dates"] = pd.to_datetime(a2L.iloc[:, 0]).dt.date.to_numpy()
    d["L"] = a2L.iloc[:, 1:].to_numpy(float)         # (365,144) kW
    d["G"] = a2G.iloc[:, 1:].to_numpy(float)
    a3 = pd.read_excel(os.path.join(SRC, "附件3.xlsx"))
    a3[a3.columns[0]] = a3[a3.columns[0]].ffill()
    d["a3"] = a3
    a4 = pd.read_excel(os.path.join(SRC, "附件4.xlsx"))
    d["P_var"] = a4.iloc[:, 1:].to_numpy(float)      # (365,144)
    # 附件3 索引: (day_idx, release_hour) -> np.array(24)
    tmap = {"0:00": 0, "6:00": 6, "12:00": 12, "18:00": 18}
    fcs = {}
    d0 = pd.Timestamp(d["dates"][0])
    for _, r in a3.iterrows():
        di = int((pd.Timestamp(r[a3.columns[0]]) - d0).days)
        h0 = tmap[str(r[a3.columns[1]]).strip()]
        fcs[(di, h0)] = r.iloc[2:26].to_numpy(float)
    d["fc3"] = fcs
    return d

def slot_start_min(j):
    return 10 * j               # label is interval END: 0:10 -> [0:00,0:10)

def fmt_min(m):
    """分钟 -> 标签; 1440 -> '24:00'"""
    m = int(round(m))
    if m >= 1440:
        return "24:00" if m == 1440 else f"{m//60}:{m%60:02d}"
    return f"{m//60}:{m%60:02d}"

def fc3_step_day(fcs, di, release_h, slots=None, method="hold"):
    """当前发布行展开到自然日剩余槽；小时预报为前一小时的保持值。"""
    fv = np.asarray(fcs[(di, release_h)])
    out = np.zeros(144)
    covered = np.arange(144) >= release_h * 6
    lead = (np.arange(144)[covered] + 1) / 6 - release_h
    if method == "hold":
        out[covered] = fv[np.ceil(lead).astype(int)-1]
    elif method == "linear":
        out[covered] = np.interp(lead, np.arange(1, 25), fv)
    else:
        raise ValueError(method)
    return out, covered

# ---------------------------------------------------------------- 预测层
FEATURE_LAGS = (1, 2, 7)

def build_features(X, d_idx_list):
    """X: (365,144) 原始量; 返回 (rows, feat) 列表。特征严格只用 d 之前的数据。"""
    rows = []
    for di in d_idx_list:
        if di < 7:
            continue
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=di)
        dow, doy = date.dayofweek, date.dayofyear
        for t in range(144):
            hour = slot_start_min(t) // 60
            f = [t / 144.0,
                 np.sin(2 * np.pi * t / 144), np.cos(2 * np.pi * t / 144),
                 np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                 dow / 6.0, doy / 366.0,
                 X[di - 1, t], X[di - 2, t], X[di - 7, t],
                 X[di - 7:di, t].mean(), X[di - 7:di, t].std()]
            rows.append((di, t, f))
    return rows

def rolling_forecast(X, refit_every=7, start=14):
    """LightGBM 扩展窗口滚动预测（每 refit_every 天重训，start 为首个预测日）。
    返回 F: (365,144)，不可预测天为 nan。样本外。"""
    import lightgbm as lgb
    F = np.full_like(X, np.nan)
    cache = {}
    for di in range(start, 365):
        rid = (di - 7) // refit_every
        if rid not in cache:
            train_rows = build_features(X, list(range(7, di)))
            A = np.array([r[2] for r in train_rows]); y = np.array([X[r[0], r[1]] for r in train_rows])
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                  min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
                                  random_state=SEED, verbose=-1, n_jobs=2)
            m.fit(A, y)
            cache = {rid: m}
        # 特征仅用 di 之前数据
        date = pd.Timestamp("2025-01-01") + pd.Timedelta(days=di)
        dow, doy = date.dayofweek, date.dayofyear
        feats = []
        for t in range(144):
            hour = slot_start_min(t) // 60
            feats.append([t / 144.0,
                          np.sin(2 * np.pi * t / 144), np.cos(2 * np.pi * t / 144),
                          np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
                          dow / 6.0, doy / 366.0,
                          X[di - 1, t], X[di - 2, t], X[di - 7, t],
                          X[di - 7:di, t].mean(), X[di - 7:di, t].std()])
        F[di] = cache[rid].predict(np.array(feats))
    return F

def baseline_mean7(X):
    F = np.full_like(X, np.nan)
    for di in range(7, 365):
        F[di] = X[di - 7:di].mean(axis=0)     # 前7个同槽均值（不含当日）
    return F

# ---------------------------------------------------------------- LP 内核
def solve_plan_lp(p, Ls, Gs, E0, q_old=None, lam=0.0, alpha=0.95,
                  q_fixed=None, E_term=None, eps_soc=0.0, tp=0.0, v_term=0.0):
    """两阶段随机计划 LP。
    p: (T,) 电价; Ls,Gs: (S,T) 场景功率 kW; E0: 期初储量;
    q_old: (T,) 调整基准(Q3)，None=无调整费用(Q2);
    lam>0: 加入 CVaR_alpha 项（对场景日紧急购电成本）。
    q_fixed: (T,) 固定购电量（实时调度复用本函数）。
    eps_soc>0: 末槽储电数值平局奖励（元/kWh，逐场景，语义冻结同 v3），在并列解中偏向"不把储能放空"。
    v_term>0: 跨日库存经济续存价值（元/kWh 内部电量），按等权期望归一注入：
              每个场景末槽系数为 -(eps_soc + v_term/S)，与紧急项 5p/S 同口径，
              保证该参数含义不随情景数 S 变化（q23_janwarm_v4，修订协议 §13）。
    tp>0: 充放电吞吐平局价（元/kWh），在并列解中排除"同槽同时充放"伪解。
    返回 dict(q, qEM(S,T), c(S,T), d(S,T), E(S,T), obj, status, dqp, dqm)"""
    T = len(p); S = Ls.shape[0]
    adj = q_old is not None
    n = T + 4 * S * T
    i_q = 0
    def i_blk(s, k, t):  # k:0 qEM,1 c,2 d,3 E
        return T + s * 4 * T + k * T + t
    if adj:
        i_dqp = T + 4 * S * T; i_dqm = i_dqp + T; n += 2 * T
    if lam > 0:
        i_eta = n; i_u = n + 1; n += 1 + S
    c_vec = np.zeros(n)
    ub_l = np.zeros(n); ub_u = np.zeros(n)
    # 目标与边界
    # At an update the original plan has already been paid for.  Only the
    # incremental adjustment bill is decision-dependent; charging p*q again
    # would count the replacement energy twice.
    c_vec[i_q:i_q + T] = 0.0 if adj else p
    ub_u[i_q:i_q + T] = BIG if q_fixed is None else 0.0
    if q_fixed is not None:
        ub_l[i_q:i_q + T] = q_fixed
        ub_u[i_q:i_q + T] = q_fixed
    for s in range(S):
        for k in range(4):
            sl = slice(i_blk(s, k, 0), i_blk(s, k, 0) + T)
            if k == 0:
                c_vec[sl] = 5.0 * p / S
                ub_u[sl] = BIG
            elif k in (1, 2):
                ub_u[sl] = P_MAX_KWH
                if tp > 0:
                    c_vec[sl] = tp                    # 吞吐平局价：排除"同槽同时充放"伪解
            else:
                ub_l[sl] = E_MIN; ub_u[sl] = E_MAX
                if v_term < 0:
                    raise ValueError("v_term 必须非负")
                # eps_soc：逐场景数值平局（v3 冻结语义）；v_term/S：经济续存价值等权期望
                coef = eps_soc + (v_term / S)
                if coef > 0:
                    c_vec[sl][-1] = -coef          # 末槽奖励，消除退化/为日末库存定价
    if adj:
        c_vec[i_dqp:i_dqp + T] = 1.5 * p
        c_vec[i_dqm:i_dqm + T] = -0.5 * p
        ub_u[i_dqp:i_dqp + T] = BIG; ub_u[i_dqm:i_dqm + T] = BIG
    if lam > 0:
        c_vec[i_eta] = lam; c_vec[i_u:i_u + S] = lam / (S * (1 - alpha))
        ub_l[i_eta] = -BIG; ub_u[i_eta] = BIG; ub_u[i_u:i_u + S] = BIG
    # 等式: 储能递推 (S*T)
    rows_e, cols_e, vals_e, be = [], [], [], []
    r = 0
    for s in range(S):
        for t in range(T):
            def add(col, val):
                rows_e.append(r); cols_e.append(col); vals_e.append(val)
            add(i_blk(s, 3, t), 1.0)
            add(i_blk(s, 1, t), -ETA_C)
            add(i_blk(s, 2, t), 1.0 / ETA_D)
            if t > 0:
                add(i_blk(s, 3, t - 1), -1.0)
                be.append(0.0)
            else:
                be.append(E0)
            r += 1
    if adj:
        for t in range(T):
            rows_e += [r, r, r]
            cols_e += [i_q + t, i_dqp + t, i_dqm + t]
            vals_e += [1.0, -1.0, 1.0]
            be.append(q_old[t]); r += 1
    if E_term is not None:
        for s in range(S):
            rows_e.append(r); cols_e.append(i_blk(s, 3, T - 1)); vals_e.append(1.0)
            be.append(float(E_term)); r += 1
    A_eq = sparse.coo_matrix((vals_e, (rows_e, cols_e)), shape=(r, n)).tocsr()
    b_eq = np.array(be)
    # 不等式: 供能 (S*T): -q -qEM -d +c <= (G-L)*DT
    rows_u, cols_u, vals_u, bu = [], [], [], []
    r = 0
    for s in range(S):
        for t in range(T):
            rows_u += [r, r, r, r]
            cols_u += [i_q + t, i_blk(s, 0, t), i_blk(s, 2, t), i_blk(s, 1, t)]
            vals_u += [-1.0, -1.0, -1.0, 1.0]
            bu.append((Gs[s, t] - Ls[s, t]) * DT); r += 1
    if lam > 0:
        # C_s - eta - u_s <= 0 ; 1/(1-alpha) occurs in objective only.
        for s in range(S):
            for t in range(T):
                rows_u.append(r); cols_u.append(i_blk(s, 0, t)); vals_u.append(5.0 * p[t])
            rows_u.append(r); cols_u.append(i_eta); vals_u.append(-1.0)
            rows_u.append(r); cols_u.append(i_u + s); vals_u.append(-1.0)
            bu.append(0.0); r += 1
    A_ub = sparse.coo_matrix((vals_u, (rows_u, cols_u)), shape=(r, n)).tocsr()
    b_ub = np.array(bu)
    res = linprog(c_vec, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.stack([ub_l, ub_u], axis=1), method="highs")
    out = {"status": res.status, "msg": res.message, "obj": res.fun if res.success else np.nan}
    if res.success:
        x = res.x
        out["q"] = x[i_q:i_q + T].copy()
        out["qEM"] = np.stack([x[i_blk(s, 0, 0):i_blk(s, 0, 0) + T] for s in range(S)])
        out["c"] = np.stack([x[i_blk(s, 1, 0):i_blk(s, 1, 0) + T] for s in range(S)])
        out["d"] = np.stack([x[i_blk(s, 2, 0):i_blk(s, 2, 0) + T] for s in range(S)])
        out["E"] = np.stack([x[i_blk(s, 3, 0):i_blk(s, 3, 0) + T] for s in range(S)])
        if adj:
            out["dqp"] = x[i_dqp:i_dqp + T].copy(); out["dqm"] = x[i_dqm:i_dqm + T].copy()
    return out

def hindsight_dispatch(p, L, G, E0, q, *, eps_soc=1e-3, tp=TP_PRICE, v_term=0.0):
    """事后完美信息调度，仅用于诊断；禁止用于正式策略回测。"""
    T = len(p); S = 1
    return solve_plan_lp(p, L[None, :], G[None, :], E0, q_fixed=q,
                         eps_soc=eps_soc, tp=tp, v_term=v_term)


def realtime_dispatch(p, L, G, E0, q, *, forecast_L=None, forecast_G=None,
                      eps_soc=1e-3, tp=TP_PRICE, v_term=0.0):
    """Causal 10-minute receding-horizon execution with a fixed market plan.

    Only the current measured L/G replace their forecasts. Future actual values
    are never passed into the optimization. Callers must supply forecasts made
    using information available before execution. All strategies use this rule.
    """
    if forecast_L is None or forecast_G is None:
        raise ValueError("实时执行必须显式提供当时可得预测，禁止用未来实际曲线代替")
    T = len(p)
    out = {k: np.zeros((1, T)) for k in ("c", "d", "E", "qEM")}
    state = float(E0)
    for t in range(T):
        r = causal_step(p[t:], q[t:], state, L[t], G[t],
                        np.asarray(forecast_L)[t:], np.asarray(forecast_G)[t:],
                        eps_soc=eps_soc, tp=tp, v_term=v_term)
        for k in out:
            out[k][0, t] = r[k]
        state = r["E"]
    out.update(status=0, msg="causal execution", q=np.asarray(q).copy())
    return out


def causal_step(p, q, E0, current_L, current_G, forecast_L, forecast_G, *,
                eps_soc=1e-3, tp=TP_PRICE, v_term=0.0):
    """Execute one slot. Interface deliberately accepts no future observations."""
    lf = np.asarray(forecast_L, dtype=float).copy()
    gf = np.asarray(forecast_G, dtype=float).copy()
    lf[0], gf[0] = current_L, current_G
    r = solve_plan_lp(np.asarray(p), lf[None, :], gf[None, :], E0,
                      q_fixed=np.asarray(q), eps_soc=eps_soc, tp=tp, v_term=v_term)
    if r["status"] != 0:
        raise RuntimeError(r["msg"])
    # Remove any solver-level simultaneous throughput exactly, preserving SOC.
    # Supply weakly increases; surplus is spilled, as allowed by the model.
    c, d = float(r["c"][0, 0]), float(r["d"][0, 0])
    paired = min(c, d / (ETA_C * ETA_D))
    c -= paired
    d -= paired * ETA_C * ETA_D
    em = max(0.0, (current_L-current_G)*DT + c-d-float(q[0]))
    return dict(c=c, d=d, E=float(E0+ETA_C*c-d/ETA_D), qEM=em)


# ---------------------------------------------------------------- 场景
def sample_scenarios(res_pool, hatL, hatG, S, rng):
    """res_pool: list of (resL(144), resG(144))；整日配对抽样。"""
    n = len(res_pool)
    idx = rng.choice(n, size=min(S, n), replace=n < S)
    Ls = np.zeros((len(idx), 144)); Gs = np.zeros((len(idx), 144))
    for k, i in enumerate(idx):
        rL, rG = res_pool[i]
        Ls[k] = np.maximum(0.0, hatL + rL)
        Gs[k] = np.maximum(0.0, hatG + rG)
    return Ls, Gs

# ---------------------------------------------------------------- 结算
def merge_emergency_intervals(qEM, thr=1e-6):
    """qEM: (144,) -> [(start_min, end_min, qty)]"""
    ivs = []
    j = 0
    while j < 144:
        if qEM[j] > thr:
            k = j; qty = 0.0
            while k < 144 and qEM[k] > thr:
                qty += qEM[k]; k += 1
            ivs.append((slot_start_min(j), slot_start_min(k - 1) + 10, qty))
            j = k
        else:
            j += 1
    return ivs

def settle_q2(p, q, qEM):
    """Q2 全天费用 = 计划 p·q + 紧急 5p·qEM"""
    return float(np.sum(p * q) + np.sum(5 * p * qEM))

def settle_q3(p, q0, updates):
    """Q3: updates = [(tau_slot, q_new(T'), dqp, dqm)] 顺序（相对上一确认计划）。
    返回 (总计划相关费用, 调整费用合计, 最终执行曲线 qa)"""
    qa = q0.copy(); adj_fee = 0.0
    for tau, qn, dqp, dqm in updates:
        T = len(qn)
        adj_fee += float(np.sum(1.5 * p[tau:tau + T] * dqp - 0.5 * p[tau:tau + T] * dqm))
        qa[tau:tau + T] = qn
    plan_fee = float(np.sum(p * q0))
    return plan_fee + adj_fee, adj_fee, qa
