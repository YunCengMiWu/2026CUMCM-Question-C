# -*- coding: utf-8 -*-
"""问题4 口径B：按被比较路径**真实 2/1 初态**计算全期 PI（价格完全信息）上界与缺口。

口径说明：PI 是"价格与负荷/光伏均完全预知"的理想化连续优化，**用附件4 真实价**；
口径B 的可执行策略用价格预测，故 GAP 同时包含价格预测误差 —— 用于量化"信息价值 + 算法/滚动近似"，
不是 MILP 求解 gap。终端自由、物理约束与滚动策略一致。
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p10_core import load_all, solve_plan_lp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4-dir", required=True)
    ap.add_argument("--cells", default="A_v0,A_v,B_v0,B_v")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    q4 = Path(a.q4_dir)
    data = load_all()
    sl = slice(31, 365)                                     # 2-01—12-31，334 天
    L = np.asarray(data["L"][sl], float).reshape(-1)
    G = np.asarray(data["G"][sl], float).reshape(-1)
    P = np.asarray(data["P_var"][sl], float).reshape(-1)     # 真实波动电价（完全信息）
    cache, rows = {}, []
    for cell in a.cells.split(","):
        c = q4 / cell
        if not (c / "main_metadata.json").exists():
            print("跳过未完成格:", cell); continue
        with open(c / "q42_best.pkl", "rb") as f:
            q42 = pickle.load(f)
        with open(c / "q43_daystore.pkl", "rb") as f:
            q43 = pickle.load(f)
        tracks = [("Q4-2_B", q42["recs"])] + [(f"Q4-3_{k}", v) for k, v in q43["all_stores"].items()] \
                 + [("Q4-3_部署", q43["store"])]
        for name, recs in tracks:
            E0 = float(recs[31]["E0"])
            key = f"{E0:.10f}"
            if key not in cache:
                r = solve_plan_lp(P, L[None, :], G[None, :], E0, eps_soc=0.0, tp=0.0)
                if r["status"] != 0:
                    raise RuntimeError(f"{cell}/{name}: PI 求解失败 {r['msg']}")
                cache[key] = float(r["obj"])
            actual = float(sum(r["总费用"] for r in recs[31:]))
            rows.append(dict(cell=cell, path=name, E0_feb1=E0, actual_eval_cost=actual,
                             pi_cost=cache[key], gap=actual - cache[key],
                             gap_pct=(actual - cache[key]) / actual * 100))
    out = dict(protocol="q4_scenarioB_v1", scope="2025-02-01--12-31（334 天）",
               note=("PI = 价格与负荷/光伏完全预知的连续优化（真实附件4 电价、自由终端、相同物理约束）；"
                     "口径B 策略用价格预测 ⇒ GAP 含价格预测误差，是「信息＋算法＋滚动近似」的综合差距"),
               unique_pi_solves=len(cache), paths=rows)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'格':<8}{'路径':<14}{'2/1初态':>10}{'实际(元)':>15}{'PI(元)':>15}{'缺口(元)':>14}{'缺口%':>8}")
    for r in rows:
        print(f"{r['cell']:<8}{r['path']:<14}{r['E0_feb1']:>10,.1f}{r['actual_eval_cost']:>15,.2f}"
              f"{r['pi_cost']:>15,.2f}{r['gap']:>14,.2f}{r['gap_pct']:>7.2f}%")
    print("\n唯一 PI 求解数:", len(cache))


if __name__ == "__main__":
    main()
