# -*- coding: utf-8 -*-
"""问题4 补做批次：v 台阶探针与 D4 式拆分的**可复现汇总脚本**（补落盘）。

背景：2026-09-12 晚首次生成 `v台阶_Q4B.json` / `d4_Q4B.json` 时用的是临时内联脚本，
未落盘（复现性缺口）。本脚本依据保留的路径 pkl 重算并覆盖这两个 JSON，
用于验证既有数值与后续复现。

用法：
  python p108_q4_vstep_d4.py [--q4-dir <重跑_Q4_B_20260912>] [--write]

口径：结算价 = 附件4 真实价；评估窗口 = 2025-02-01—12-31（334 天，即 recs[31:]）；
      v 台阶四轨与 d4_F 轨均为 365 天因果冷启动运行。
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p10_core import load_all  # noqa: E402

ROOT = HERE.parent
Q4_DEFAULT = ROOT / "04_结果与检验" / "重跑_Q4_B_20260912"
E_MIN = 1200.0
TOL = 0.0          # 触底判定容差（kWh）；0 = 严格「日初储能 ≤ 1200」，
                   # 这与 2026-09-12 首次生成的 v台阶_Q4B.json 的 bottom 定义一致（56/26/5/4/1/0）


def daily(recs, P):
    """逐日汇总（recs 为 365 天记录，取 [31:] 为 334 天评估窗口）。"""
    ev = recs[31:]
    cost = float(sum(r["总费用"] for r in ev))
    emg = float(sum(5.0 * float(P[int(r["di"])] @ np.asarray(r["qEM"], float)) for r in ev))
    adj = float(sum(r["调整费"] for r in ev))
    E0s = np.array([float(r["E0"]) for r in ev])
    # 「触底天数」= 日初储能 ≤ E_MIN(+TOL) 的天数：即当天的电是「昨天没留下电」开始的
    bottom = int(sum(1 for e in E0s if e <= E_MIN + TOL))
    return dict(days=len(ev), cost334=cost, emg334=emg, adj334=adj,
                E0_mean=float(E0s.mean()), E0_med=float(np.median(E0s)),
                bottom=bottom, eend=float(recs[-1]["Eend"]))


def load_pkl(p):
    with open(p, "rb") as fh:
        o = pickle.load(fh)
    return o["recs"] if isinstance(o, dict) and "recs" in o else o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4-dir", default=str(Q4_DEFAULT))
    ap.add_argument("--write", action="store_true", help="覆盖写回两个 JSON（默认只打印对比）")
    a = ap.parse_args()
    Q4 = Path(a.q4_dir)
    P = np.asarray(load_all()["P_var"], float)

    # ---- v 台阶（v=0 行取自 B_v0 格；0.35–0.50 取自 probe_* 轨；滚动行取自正式格 B_v）
    steps = [("v=0（对照）", 0.0, Q4 / "B_v0" / "paths" / "B.pkl"),
             ("v=0.35", 0.35, Q4 / "probe_v035" / "paths" / "B.pkl"),
             ("v=0.44", 0.44, Q4 / "probe_v044" / "paths" / "B.pkl"),
             ("v=0.46", 0.46, Q4 / "probe_v046" / "paths" / "B.pkl"),
             ("v=0.50", 0.50, Q4 / "probe_v050" / "paths" / "B.pkl"),
             ("滚动 v（正式，均值0.4800）", None, Q4 / "B_v" / "paths" / "B.pkl")]
    rows = []
    for label, v, p in steps:
        if not p.exists():
            print(f"[缺] {label}: {p}")
            continue
        recs = load_pkl(p)
        d = daily(recs, P)
        rows.append(dict(label=label, v=v, **d))
    print("=== v 台阶（334 天，元）===")
    for r in rows:
        print(f"  {r['label']:<26} v={str(r['v']):>5}  总={r['cost334']:>15,.2f}  "
              f"紧急费={r['emg334']:>13,.2f}  E0均值={r['E0_mean']:>8,.1f}  "
              f"触底天数={r['bottom']:>3}  年末储能={r['eend']:>10,.2f}")

    # 校验：v=0 行应等于 B_v0 格 Q4-2 B 轨的 334 天总额；滚动行应等于 B_v 格 Q4-2 B 轨
    def q42B(cell):
        import pandas as pd
        d = pd.read_csv(Q4 / cell / "q42_逐日费用.csv")
        d["日期"] = d["日期"].astype(str)
        return float(d[~d["日期"].str.startswith("2025-01") & (d["模型"] == "B")]["总费用"].sum())

    try:
        zc, rc = q42B("B_v0"), q42B("B_v")
        print(f"\n  交叉核对：B_v0 格 Q4-2 B 334 = {zc:,.2f}；B_v 格 Q4-2 B 334 = {rc:,.2f}")
        print(f"            v=0 轨（本脚本）      = {rows[0]['cost334']:,.2f}   差 = {rows[0]['cost334'] - zc:+,.2f}")
        print(f"            滚动 v 轨（本脚本）   = {rows[-1]['cost334']:,.2f}   差 = {rows[-1]['cost334'] - rc:+,.2f}")
    except Exception as e:                                   # noqa: BLE001
        print("  交叉核对跳过:", e)

    # ---- D4 式拆分（N 取 B_v 格 q43 none；F 取 d4_F 轨；FM 取 B_v 格 q43 6+12+18）
    fm = Q4 / "B_v" / "paths" / "6plus12plus18.pkl"
    ff = Q4 / "d4_F" / "paths" / "6plus12plus18.pkl"
    d4 = {}
    try:
        import pandas as pd
        d = pd.read_csv(Q4 / "B_v" / "q43_逐日费用.csv")
        d["日期"] = d["日期"].astype(str)
        d = d[~d["日期"].str.startswith("2025-01")]
        d4["N"] = float(d[d["模型"] == "none"]["总费用"].sum())
    except Exception as e:                                   # noqa: BLE001
        print("  N 读取失败:", e)
    if ff.exists():
        d4["F"] = daily(load_pkl(ff), P)["cost334"]
    if fm.exists():
        d4["FM"] = daily(load_pkl(fm), P)["cost334"]
    if {"N", "F", "FM"} <= set(d4):
        d4["N_minus_F"] = d4["N"] - d4["F"]
        d4["F_minus_FM"] = d4["F"] - d4["FM"]
        d4["N_minus_FM"] = d4["N"] - d4["FM"]
        print("\n=== D4 式拆分（B_v，334 天，元）===")
        print(f"  N  不更新           = {d4['N']:>15,.2f}")
        print(f"  F  仅刷新预报       = {d4['F']:>15,.2f}   N−F  = {d4['N_minus_F']:>+13,.2f}")
        print(f"  FM 刷新＋购电调整   = {d4['FM']:>15,.2f}   F−FM = {d4['F_minus_FM']:>+13,.2f}")
        print(f"                                        N−FM = {d4['N_minus_FM']:>+13,.2f}")

    if a.write:
        (Q4 / "v台阶_Q4B.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        (Q4 / "d4_Q4B.json").write_text(json.dumps(d4, ensure_ascii=False, indent=1), encoding="utf-8")
        print("\n已写回 v台阶_Q4B.json / d4_Q4B.json")


if __name__ == "__main__":
    main()
