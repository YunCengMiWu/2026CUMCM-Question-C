# -*- coding: utf-8 -*-
"""Q2/Q3 fixed-price causal rerun, protocol q23_janwarm_v4 (v_term arm driver).

This driver is deliberately separate from p90_reproduce.py.  It starts every
path at 2025-01-01 00:00 with E=6000 kWh, keeps January as a visible warm-up,
and writes only to the explicitly supplied case directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from p10_core import (  # noqa: E402
    DT, E_INIT, E_MAX, E_MIN, ETA_C, ETA_D, P_MAX_KWH, TP_PRICE,
    causal_step, fc3_step_day, fmt_min, load_all, solve_plan_lp,
)
from p15_forecasts import (  # noqa: E402
    baseline_mean7, rolling_forecast, select_past_only,
)

PROTOCOL = "q23_janwarm_v4"
COMBOS = [(), (6,), (12,), (18,), (6, 12), (6, 18), (12, 18), (6, 12, 18)]
EPS_GRID = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
CALIBRATION_STOP = 90  # [2025-01-01, 2025-04-01): Jan warm-up + Feb/Mar calibration


def key(combo):
    return "+".join(map(str, combo)) or "none"


def phase_of(di):
    if di < 31:
        return "init"
    if di < 90:
        return "calibration"
    return "eval"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hashes():
    files = [HERE / n for n in ("p10_core.py", "p15_forecasts.py", "p95_q23_v4.py")]
    return {str(p): sha256(p) for p in files}


def input_hashes():
    ad = ROOT / "00_题目与原始数据" / "C题" / "附件"
    return {str(p): sha256(p) for p in sorted(ad.glob("附件[1-4].xlsx"))}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def make_forecasts(out_dir, reuse=False):
    """Build or safely reuse a cache only when protocol and hashes match."""
    out_dir = Path(out_dir)
    fdir = out_dir / "forecast"
    fdir.mkdir(parents=True, exist_ok=True)
    cache = fdir / "预测缓存.npz"
    meta_path = fdir / "预测缓存_meta.json"
    expected = dict(protocol=PROTOCOL, training_start=14,
                    input_hashes=input_hashes(), code_hashes=code_hashes(),
                    early_policy="lag1_for_di_lt_14", seed=2025)
    if reuse and cache.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(k) == expected.get(k) for k in expected):
            with np.load(cache) as z:
                if all(k in z for k in ("FL", "FG", "protocol")):
                    if str(z["protocol"]) == PROTOCOL and np.isfinite(z["FL"][:14]).all() and np.isfinite(z["FG"][:14]).all():
                        return str(cache), meta

    data = load_all()
    meta = expected.copy()
    saved = {}
    selections = {}
    metrics = {}
    for prefix, actual in (("L", data["L"]), ("G", data["G"])):
        lag = np.empty_like(actual)
        lag[0] = actual[0]
        lag[1:] = actual[:-1]
        m7 = baseline_mean7(actual)
        m7[:7] = lag[:7]
        lgb = rolling_forecast(actual, start=14)
        lgb[:14] = lag[:14]
        candidates = {"lgb": lgb, "lag1": lag, "m7": m7}
        forecast, selected = select_past_only(candidates, actual, start=14)
        saved["F" + prefix] = forecast
        selections[prefix] = selected
        for name, arr in candidates.items():
            saved["F" + prefix + "_" + name] = arr
        for name, arr in dict(candidates, online=forecast).items():
            err = arr[31:] - actual[31:]
            metrics[prefix + "_" + name + "_eval"] = dict(
                MAE=float(np.abs(err).mean()),
                RMSE=float(np.sqrt((err ** 2).mean())),
                bias=float(err.mean()),
            )
    saved["protocol"] = np.array(PROTOCOL)
    np.savez_compressed(cache, **saved)
    pd.DataFrame(dict(日期=data["dates"], 负荷模型=selections["L"], 光伏模型=selections["G"])).to_csv(
        fdir / "逐日预测选型.csv", index=False, encoding="utf-8-sig")
    meta.update(dict(metrics=metrics, created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    write_json(meta_path, meta)
    return str(cache), meta


def load_forecasts(path):
    with np.load(path) as z:
        if str(z["protocol"]) != PROTOCOL:
            raise ValueError("预测缓存协议不匹配")
        f = {k: z[k].copy() for k in ("FL", "FG")}
    if not np.isfinite(f["FL"][:14]).all() or not np.isfinite(f["FG"][:14]).all():
        raise ValueError("1月早期预测不完整，不能进入冷启动")
    return f


def scenarios(data, forecasts, di, release_h, S, seed, use_fc3, method):
    """Causal paired residual scenarios; no completed residual day means one scene."""
    lf = np.asarray(forecasts["FL"][di], dtype=float)
    if use_fc3:
        gf = fc3_step_day(data["fc3"], di, release_h, method=method)[0]
    else:
        gf = np.asarray(forecasts["FG"][di], dtype=float)
    hist = np.arange(14, di, dtype=int)
    if len(hist) == 0:
        return lf[None, :], gf[None, :], [], lf, gf, 0
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(di), int(release_h)]))
    days = rng.choice(hist, size=int(S), replace=len(hist) < int(S))
    rl = data["L"][days] - forecasts["FL"][days]
    if use_fc3:
        rg = np.stack([
            data["G"][d] - fc3_step_day(data["fc3"], int(d), release_h, method=method)[0]
            for d in days
        ])
    else:
        rg = data["G"][days] - forecasts["FG"][days]
    return np.maximum(0.0, lf[None, :] + rl), np.maximum(0.0, gf[None, :] + rg), days.tolist(), lf, gf, len(hist)


def cold_day(data, di, p, E0):
    L = data["L"][di]
    G = data["G"][di]
    qem = np.maximum(0.0, (L - G) * DT)
    z = np.zeros(144, dtype=float)
    return dict(
        日期=str(data["dates"][di]), di=di, q=z.copy(), q0=z.copy(), qc=z.copy(),
        c=z.copy(), d=z.copy(), E=np.full(144, float(E0)), qEM=qem,
        E0=float(E0), Eend=float(E0), updates=[], scenario_days={"0": []},
        scenario_pool_counts={"0": 0}, planned_emergency_qty=0.0,
        planned_emergency_fee=0.0, 计划费=0.0, 调整费=0.0,
        紧急费=float(np.sum(5.0 * p * qem)), 总费用=float(np.sum(5.0 * p * qem)),
        阶段=phase_of(di), cold_start=True,
    )


def simulate_day(data, forecasts, di, E0, mode="fixed", model="B", combo=None,
                 S=10, seed=2025, lam=.3, method="hold", eps_soc=1e-3,
                 tp=TP_PRICE, v_term=0.0, update_mode="FM"):
    if mode != "fixed":
        raise ValueError("q23_janwarm_v3 只允许固定电价")
    p = np.asarray(data["p_fix"], dtype=float)
    if di == 0:
        return cold_day(data, di, p, E0)
    use_fc3 = combo is not None
    ls, gs, days, lf, gf, pool_n = scenarios(data, forecasts, di, 0, S, seed, use_fc3, method)
    if model == "A":
        ls, gs = lf[None, :], gf[None, :]
    plan = solve_plan_lp(p, ls, gs, E0, lam=lam if model == "C" else 0.0,
                         eps_soc=eps_soc, tp=tp, v_term=v_term)
    if plan["status"] != 0:
        raise RuntimeError(plan["msg"])
    q0 = plan["q"].copy()
    qc = q0.copy()
    state = float(E0)
    arrays = {k: np.zeros(144, dtype=float) for k in ("c", "d", "E", "qEM")}
    updates = []
    draws = {"0": list(days)}
    pools = {"0": int(pool_n)}
    expected_qty = float(np.mean(plan["qEM"].sum(axis=1)))
    expected_fee = float(np.mean(np.sum(5.0 * p[None, :] * plan["qEM"], axis=1)))
    adj = 0.0
    for t in range(144):
        hour = t // 6
        if use_fc3 and t > 0 and t % 6 == 0 and hour in combo:
            ls, gs, days, lf, gf, pool_n = scenarios(data, forecasts, di, hour, S, seed, True, method)
            draws[str(hour)] = list(days)
            pools[str(hour)] = int(pool_n)
            if update_mode == "F":
                # D4 的 F 轨：只用新发布的附件3预报刷新日内预测（执行器视界随之更新），
                # 但**锁定日前购电计划 q0**，不做市场购电调整，故调整费为 0、qc 不变。
                updates.append(dict(
                    tau=t, hour=hour, old=qc[t:].copy(), new=qc[t:].copy(), fee=0.0,
                    E0=float(state), scenario_days=list(days), scenario_pool_n=int(pool_n),
                    expected_emergency_qty=0.0, expected_emergency_fee=0.0, forecast_only=True,
                ))
            else:
                old = qc[t:].copy()
                update = solve_plan_lp(p[t:], ls[:, t:], gs[:, t:], state, q_old=old,
                                       eps_soc=eps_soc, tp=tp, v_term=v_term)
                if update["status"] != 0:
                    raise RuntimeError(update["msg"])
                qc[t:] = update["q"]
                plus = np.maximum(qc[t:] - old, 0.0)
                minus = np.maximum(old - qc[t:], 0.0)
                fee = float(p[t:] @ (1.5 * plus - .5 * minus))
                adj += fee
                updates.append(dict(
                    tau=t, hour=hour, old=old, new=qc[t:].copy(), fee=fee,
                    E0=float(state), scenario_days=list(days), scenario_pool_n=int(pool_n),
                    expected_emergency_qty=float(np.mean(update["qEM"].sum(axis=1))),
                    expected_emergency_fee=float(np.mean(np.sum(5.0 * p[t:][None, :] * update["qEM"], axis=1))),
                ))
        step = causal_step(p[t:], qc[t:], state, data["L"][di, t], data["G"][di, t],
                           lf[t:], gf[t:], eps_soc=eps_soc, tp=tp, v_term=v_term)
        for k in arrays:
            arrays[k][t] = step[k]
        state = float(step["E"])
    pf = float(p @ q0)
    ef = float(5.0 * p @ arrays["qEM"])
    return dict(
        日期=str(data["dates"][di]), di=di, q=q0, q0=q0, qc=qc, **arrays,
        E0=float(E0), Eend=float(state), updates=updates,
        scenario_days=draws, scenario_pool_counts=pools,
        planned_emergency_qty=expected_qty, planned_emergency_fee=expected_fee,
        计划费=pf, 调整费=adj, 紧急费=ef, 总费用=pf + adj + ef,
        阶段=phase_of(di), cold_start=False,
    )


def summary(recs, start=None, stop=None):
    rows = recs[slice(start, stop)] if start is not None or stop is not None else recs
    return dict(
        天数=len(rows), 计划购电费=float(sum(r["计划费"] for r in rows)),
        调整费=float(sum(r["调整费"] for r in rows)),
        紧急购电费=float(sum(r["紧急费"] for r in rows)),
        紧急购电量=float(sum(np.sum(r["qEM"]) for r in rows)),
        总费用=float(sum(r["总费用"] for r in rows)),
        初SOC=float(rows[0]["E0"]) if rows else None,
        末SOC=float(rows[-1]["Eend"]) if rows else None,
    )


def worker_run_path(task):
    label, cfg = task
    data = load_all()
    forecasts = load_forecasts(cfg["forecast_cache"])
    start, stop = int(cfg["start"]), int(cfg["stop"])
    state = float(cfg.get("initial_state", E_INIT))
    recs = []
    begin = time.time()
    cp = Path(cfg["checkpoint_dir"]) / (label.replace("+", "plus") + ".json")
    for di in range(start, stop):
        rec = simulate_day(data, forecasts, di, state, model=cfg["model"],
                           combo=cfg.get("combo"), S=cfg["S"], seed=cfg["seed"],
                           lam=cfg["lam"], method=cfg["method"],
                           eps_soc=cfg["eps_soc"], tp=cfg["tp"], v_term=cfg["v_term"],
                           update_mode=cfg.get("update_mode", "FM"))
        state = rec["Eend"]
        recs.append(rec)
        write_json(cp, dict(protocol=PROTOCOL, label=label, last_complete_date=rec["日期"],
                            last_di=di, E=float(state), shadow_total=float(sum(x["总费用"] for x in recs)),
                            parameters={k: cfg[k] for k in ("model", "S", "seed", "eps_soc", "tp", "v_term", "method")},
                            complete=di == stop - 1))
    return label, recs, time.time() - begin


def daily_frame(recs, label):
    return pd.DataFrame([
        dict(日期=r["日期"], 模型=label, 计划购电费=r["计划费"], 调整费=r["调整费"],
             紧急购电费=r["紧急费"], 紧急购电量=float(np.sum(r["qEM"])),
             总费用=r["总费用"], 末SOC=r["Eend"], 阶段=r["阶段"],
             计划紧急购电量=r["planned_emergency_qty"], 计划紧急购电费=r["planned_emergency_fee"])
        for r in recs
    ])


def save_pickle(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_paths(out_dir, forecast_cache, labels, cfg, workers=2, resume=False):
    out_dir = Path(out_dir)
    path_dir = out_dir / "paths"
    cp_dir = out_dir / "checkpoints"
    path_dir.mkdir(parents=True, exist_ok=True)
    cp_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    loaded = {}
    for label, model, combo in labels:
        path_file = path_dir / (label.replace("+", "plus") + ".pkl")
        if resume and path_file.exists():
            with open(path_file, "rb") as f:
                obj = pickle.load(f)
            if obj.get("protocol") == PROTOCOL and obj.get("config", {}).get("eps_soc") == cfg["eps_soc"] and obj.get("config", {}).get("v_term") == cfg.get("v_term"):
                loaded[label] = obj["recs"]
                continue
        child = dict(cfg, forecast_cache=str(forecast_cache), checkpoint_dir=str(cp_dir),
                     model=model, combo=list(combo) if combo is not None else None)
        tasks.append((label, child))
    durations = {}
    if tasks:
        with ProcessPoolExecutor(max_workers=max(1, min(int(workers), 2))) as pool:
            future_map = {pool.submit(worker_run_path, t): t[0] for t in tasks}
            for future in as_completed(future_map):
                label, recs, elapsed = future.result()
                loaded[label] = recs
                durations[label] = float(elapsed)
                save_pickle(path_dir / (label.replace("+", "plus") + ".pkl"),
                            dict(protocol=PROTOCOL, label=label, config=cfg, recs=recs,
                                 elapsed_seconds=elapsed))
                write_json(out_dir / "progress.json", dict(protocol=PROTOCOL, completed_paths=sorted(loaded),
                                                           durations=durations, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    for label, _, _ in labels:
        if label not in loaded:
            raise RuntimeError("路径未返回: " + label)
    return loaded, durations


def deploy_from_shadow(data, forecasts, shadows, cfg):
    ordered = [key(c) for c in COMBOS]
    n = len(next(iter(shadows.values())))
    cumulative = {k: 0.0 for k in ordered}
    state = E_INIT
    deployed = []
    for i in range(n):
        di = shadows[ordered[0]][i]["di"]
        selected = "none" if di == 0 else min(ordered, key=lambda k: cumulative[k])
        combo = COMBOS[ordered.index(selected)]
        rec = simulate_day(data, forecasts, di, state, combo=combo, S=cfg["S"], seed=cfg["seed"],
                           method=cfg["method"], eps_soc=cfg["eps_soc"], tp=cfg["tp"],
                           v_term=cfg["v_term"], update_mode=cfg.get("update_mode", "FM"))
        rec["selected_combo"] = selected
        rec["selection_history_end"] = di - 1
        deployed.append(rec)
        state = rec["Eend"]
        for k in ordered:
            cumulative[k] += shadows[k][i]["总费用"]
    return deployed, cumulative


def write_result_workbooks(out_dir, q2_b, q3_deployed):
    """Fill only result2/result3 in the isolated output directory."""
    import openpyxl
    tpl = ROOT / "00_题目与原始数据" / "C题" / "附件" / "附件5"
    out_dir = Path(out_dir)
    dates = [d.strftime("%Y/%m/%d") for d in pd.date_range("2025-02-01", "2025-12-31")]
    buckets = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
    labels = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]

    def fill_plan(ws, recs, field, include_adj=False):
        for j in range(144):
            ws.cell(1, 2 + j, f"{fmt_min(j * 10)}-{fmt_min((j + 1) * 10)}")
        for i, (dt, r) in enumerate(zip(dates, recs)):
            q = np.asarray(r[field])
            ws.cell(2 + i, 1, dt)
            for j in range(144):
                ws.cell(2 + i, 2 + j, round(float(q[j]), 2))
            ws.cell(2 + i, 146, round(float(q.sum()), 2))
            fee = r["计划费"] + r["调整费"] if include_adj else r["计划费"]
            ws.cell(2 + i, 147, round(float(fee), 2))

    def fill_storage(ws, recs):
        for row in range(2, ws.max_row + 1):
            for col in range(1, 7):
                ws.cell(row, col).value = None
        row = 2
        for dt, r in zip(dates, recs):
            for bi, (a, b) in enumerate(buckets):
                ws.cell(row, 1, dt if bi == 0 else None)
                ws.cell(row, 2, labels[bi])
                ws.cell(row, 3, round(float(np.sum(r["c"][a:b])), 2))
                ws.cell(row, 4, round(float(np.sum(r["d"][a:b])), 2))
                if bi == 0:
                    ws.cell(row, 5, "0:00"); ws.cell(row, 6, round(float(r["E0"]), 2))
                if bi == 1:
                    ws.cell(row, 5, "24:00"); ws.cell(row, 6, round(float(r["Eend"]), 2))
                row += 1

    def fill_emergency(ws, recs):
        for row in range(2, ws.max_row + 1):
            for col in range(1, 4):
                ws.cell(row, col).value = None
        row = 2
        for dt, r in zip(dates, recs):
            qem = np.asarray(r["qEM"]); t = 0; intervals = []
            while t < 144:
                if qem[t] <= 1e-6:
                    t += 1; continue
                a = t; amount = 0.0
                while t < 144 and qem[t] > 1e-6:
                    amount += float(qem[t]); t += 1
                intervals.append((a, t, amount))
            if not intervals:
                ws.cell(row, 1, dt); row += 1
            else:
                for a, b, amount in intervals:
                    ws.cell(row, 1, dt); ws.cell(row, 2, f"{fmt_min(a * 10)}-{fmt_min(b * 10)}")
                    ws.cell(row, 3, round(amount, 2)); row += 1

    eval_q2 = q2_b[31:]
    wb = openpyxl.load_workbook(tpl / "result2.xlsx")
    fill_plan(wb["计划购电量"], eval_q2, "q0")
    fill_storage(wb["充放电量"], eval_q2)
    fill_emergency(wb["紧急购电量"], eval_q2)
    wb.save(out_dir / "result2.xlsx")

    eval_q3 = q3_deployed[31:]
    wb = openpyxl.load_workbook(tpl / "result3.xlsx")
    fill_plan(wb["计划购电量"], eval_q3, "q0")
    fill_plan(wb["调整购电量"], eval_q3, "qc", include_adj=True)
    fill_storage(wb["充放电量"], eval_q3)
    fill_emergency(wb["紧急购电量"], eval_q3)
    wb.save(out_dir / "result3.xlsx")


def run_main(args, out_dir, forecast_cache, metadata):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(start=0, stop=365, S=args.S, seed=args.seed, lam=args.lam,
               method=args.method, eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, initial_state=E_INIT)
    q2_labels = [("A", "A", None), ("B", "B", None), ("C", "C", None)]
    q2, d2 = run_paths(out_dir, forecast_cache, q2_labels, cfg, args.workers, args.resume)
    q3_labels = [(key(c), "B", c) for c in COMBOS]
    q3, d3 = run_paths(out_dir, forecast_cache, q3_labels, cfg, args.workers, args.resume)
    data = load_all(); forecasts = load_forecasts(forecast_cache)
    deployed, cumulative = deploy_from_shadow(data, forecasts, q3, cfg)
    save_pickle(out_dir / "q2_best.pkl", dict(protocol=PROTOCOL, best="B", recs=q2["B"], q2_all=q2, config=cfg))
    save_pickle(out_dir / "q3_daystore.pkl", dict(protocol=PROTOCOL, best_key="历史选型执行策略",
                                                    store=deployed, all_stores=q3,
                                                    results={k: summary(v) for k, v in q3.items()} |
                                                            {"历史选型执行策略": summary(deployed)}, config=cfg))
    q2_df = pd.concat([daily_frame(q2[k], k) for k in ("A", "B", "C")], ignore_index=True)
    q2_df.to_csv(out_dir / "q2_逐日费用.csv", index=False, encoding="utf-8-sig")
    q2_df.groupby("模型")[["计划购电费", "调整费", "紧急购电费", "紧急购电量", "总费用"]].sum().to_csv(
        out_dir / "q2_汇总.csv", encoding="utf-8-sig")
    q3_df = pd.concat([daily_frame(q3[k], k) for k in [key(c) for c in COMBOS]] +
                      [daily_frame(deployed, "历史选型执行策略")], ignore_index=True)
    q3_df.to_csv(out_dir / "q3_逐日费用.csv", index=False, encoding="utf-8-sig")
    combo_rows = [{"组合": k, **summary(q3[k])} for k in [key(c) for c in COMBOS]]
    combo_rows.append({"组合": "历史选型执行策略", **summary(deployed)})
    pd.DataFrame(combo_rows).to_csv(out_dir / "q3_组合对照.csv", index=False, encoding="utf-8-sig")
    write_result_workbooks(out_dir, q2["B"], deployed)
    write_json(out_dir / "main_metadata.json", dict(metadata, protocol=PROTOCOL, config=cfg,
                                                     q2_durations=d2, q3_durations=d3,
                                                     q2_summary={k: summary(v) for k, v in q2.items()},
                                                     q3_summary={k: summary(v) for k, v in q3.items()},
                                                     deployed_summary=summary(deployed),
                                                     selection_cumulative=cumulative))
    return q2, q3, deployed


def run_calibration(args, out_dir, forecast_cache, metadata):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for eps in EPS_GRID:
        cfg = dict(start=0, stop=CALIBRATION_STOP, S=args.S, seed=args.seed, lam=args.lam,
                   method=args.method, eps_soc=eps, tp=args.tp, v_term=0.0, initial_state=E_INIT)
        labels = [("A", "A", None), ("B", "B", None)]
        run_dir = out_dir / ("eps_" + (str(eps).replace(".", "p") if eps else "0"))
        paths, durations = run_paths(run_dir, forecast_cache, labels, cfg, args.workers, args.resume)
        b = summary(paths["B"], start=31, stop=90)
        a = summary(paths["A"], start=31, stop=90)
        rows.append(dict(eps_soc=eps, B_2_3_total=b["总费用"], A_2_3_total=a["总费用"],
                         B_minus_A=b["总费用"] - a["总费用"], elapsed_seconds=sum(durations.values())))
    frame = pd.DataFrame(rows)
    best_val = min(float(x) for x in frame["B_2_3_total"])
    tied = frame[np.abs(frame["B_2_3_total"] - best_val) <= .01]
    chosen = min([float(x) for x in tied["eps_soc"]])
    result = dict(protocol=PROTOCOL, rule="2-3月B总账单；并列0.01元取较小eps；看结果前冻结",
                  chosen_eps=chosen, candidates=rows, metadata=metadata)
    frame.to_csv(out_dir / "eps_calibration.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "eps_calibration.json", result)
    return result


def run_smoke(args, out_dir, forecast_cache, metadata):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(); forecasts = load_forecasts(forecast_cache)
    checks = []

    def record(name, ok, detail):
        checks.append(dict(name=name, status="PASS" if ok else "FAIL", detail=detail))
        if not ok:
            raise AssertionError(name + ": " + detail)

    # 1/1-1/3: cold start, empty scenario pool and continuity.
    state = E_INIT
    for di in range(3):
        r = simulate_day(data, forecasts, di, state, model="B", S=args.S, seed=args.seed,
                         eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)
        if di == 0:
            ok = r["cold_start"] and r["q0"].sum() == 0 and r["Eend"] == state
        else:
            ok = (not r["cold_start"]) and r["scenario_pool_counts"]["0"] == 0
        record("cold_" + str(di), ok,
               "cold_start=" + str(r["cold_start"]) + ", pool=" +
               str(r["scenario_pool_counts"]["0"]) + ", emergency=" + str(float(r["qEM"].sum())))
        state = r["Eend"]
    # 1/14-1/16 boundary: no negative index and residual pool starts on 1/15.
    r14 = simulate_day(data, forecasts, 13, E_INIT, combo=(6,), S=args.S, seed=args.seed,
                       eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)
    r15 = simulate_day(data, forecasts, 14, r14["Eend"], combo=(6,), S=args.S, seed=args.seed,
                       eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)
    r16 = simulate_day(data, forecasts, 15, r15["Eend"], combo=(6,), S=args.S, seed=args.seed,
                       eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)
    record("residual_boundary", r14["scenario_pool_counts"]["0"] == 0 and r15["scenario_pool_counts"]["0"] == 0 and
           r16["scenario_pool_counts"]["0"] == 1, "pool counts 1/14,1/15,1/16=" +
           str([r14["scenario_pool_counts"]["0"], r15["scenario_pool_counts"]["0"], r16["scenario_pool_counts"]["0"]]))
    # Full January plus Feb 1-2 cross-month continuity.
    state = E_INIT; jan = []
    for di in range(33):
        r = simulate_day(data, forecasts, di, state, combo=(6,), S=args.S, seed=args.seed,
                         eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)
        jan.append(r); state = r["Eend"]
    record("cross_month", jan[31]["E0"] == jan[30]["Eend"] and jan[32]["E0"] == jan[31]["Eend"],
           "Jan31 Eend=" + str(jan[30]["Eend"]) + ", Feb1 E0=" + str(jan[31]["E0"]))
    # A single Q3 day must realize 6/12/18 updates and retain pre-update slots.
    r = simulate_day(data, forecasts, 31, E_INIT, combo=(6, 12, 18), S=args.S, seed=args.seed,
                     eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)
    record("q3_updates", [u["tau"] for u in r["updates"]] == [36, 72, 108] and
           np.allclose(r["qc"][:36], r["q0"][:36]),
           "taus=" + str([u["tau"] for u in r["updates"]]))
    # Timing sample, three continuous days per one Q2 and one Q3 path.
    t0 = time.time(); state = E_INIT
    for di in range(31, 34):
        state = simulate_day(data, forecasts, di, state, model="B", S=args.S, seed=args.seed,
                             eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)["Eend"]
    q2_seconds = time.time() - t0
    t0 = time.time(); state = E_INIT
    for di in range(31, 34):
        state = simulate_day(data, forecasts, di, state, combo=(6, 12, 18), S=args.S,
                             seed=args.seed, eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term, method=args.method)["Eend"]
    q3_seconds = time.time() - t0
    timing = dict(q2_3day_seconds=q2_seconds, q3_3day_seconds=q3_seconds,
                  estimate_note="仅用于估算，不替代全量实测")
    write_json(out_dir / "timing_3days.json", timing)
    write_json(out_dir / "smoke_result.json", dict(metadata=metadata, protocol=PROTOCOL,
                                                    checks=checks, timing=timing))
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["smoke", "main", "calibration"], required=True)
    ap.add_argument("--case-id", default="q23_janwarm_v4")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--eps-soc", type=float, default=1e-3)
    ap.add_argument("--v-term", type=float, default=0.0)
    ap.add_argument("--tp", type=float, default=TP_PRICE)
    ap.add_argument("--lam", type=float, default=.3)
    ap.add_argument("--method", choices=["hold", "linear"], default="hold")
    ap.add_argument("--reuse-forecast", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    forecast_cache, forecast_meta = make_forecasts(out_dir.parent if args.phase != "main" else out_dir,
                                                   reuse=args.reuse_forecast)
    metadata = dict(case_id=args.case_id, protocol=PROTOCOL, project_root=str(ROOT),
                    phase=args.phase, forecast_cache=forecast_cache,
                    input_hashes=input_hashes(), code_hashes=code_hashes(),
                    parameters=dict(S=args.S, seed=args.seed, eps_soc=args.eps_soc, tp=args.tp, v_term=args.v_term,
                                    lam=args.lam, method=args.method), forecast_meta=forecast_meta)
    write_json(out_dir / "case_config.json", metadata)
    if args.phase == "smoke":
        run_smoke(args, out_dir, forecast_cache, metadata)
    elif args.phase == "calibration":
        run_calibration(args, out_dir, forecast_cache, metadata)
    else:
        run_main(args, out_dir, forecast_cache, metadata)
    print(json.dumps(dict(status="PASS", phase=args.phase, output=str(out_dir)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
