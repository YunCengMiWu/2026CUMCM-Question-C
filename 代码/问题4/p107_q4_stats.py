# -*- coding: utf-8 -*-
"""问题4 口径B：配对统计（7 日分块 bootstrap），补齐此前只报总额的缺口。

覆盖：
  ① 各格组内：Q4-2 B−A、C−B；Q4-3 部署−none、各固定组合−none
  ② 跨格：B_v−A_v（Q4-2 B 与 Q4-3 部署）、B_v−B_v0（跨日价值）、A_v−A_v0
  ③ 价格轨：hist7（主轨）vs lag1 / fix / shape（Q4-2 B）
  ④ 刷新：可刷新B vs 严格B（Q4-3 部署）
全部在同日期配对（2-01—12-31，334 天）。
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def block_bootstrap(x, reps=2000, block=7, seed=2025):
    x = np.asarray(x, float); n = len(x); rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(reps, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)) % n
    return np.mean(x[idx.reshape(reps, -1)[:, :n]], axis=1)


def stat(x):
    x = np.asarray(x, float); b = block_bootstrap(x)
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)),
                q025=float(np.quantile(x, .025)), q975=float(np.quantile(x, .975)),
                bootstrap_q025=float(np.quantile(b, .025)),
                bootstrap_q975=float(np.quantile(b, .975)),
                positive_days=int((x > 0).sum()), negative_days=int((x < 0).sum()),
                significant=bool(np.quantile(b, .025) > 0 or np.quantile(b, .975) < 0))


def load_daily(case: Path, which="q42"):
    df = pd.read_csv(case / f"{which}_逐日费用.csv")
    df["日期"] = df["日期"].astype(str)
    df = df[~df["日期"].str.startswith("2025-01")]
    return {k: v.sort_values("日期").reset_index(drop=True) for k, v in df.groupby("模型")}


def load_single(pkl: Path):
    with open(pkl, "rb") as f:
        recs = pickle.load(f)["recs"]
    df = pd.DataFrame([dict(日期=r["日期"], 总费用=r["总费用"]) for r in recs])
    return df[~df["日期"].str.startswith("2025-01")].sort_values("日期").reset_index(drop=True)


def diff(dfa, ka, dfb, kb, col="总费用"):
    A = dfa[ka].sort_values("日期").reset_index(drop=True)
    B = dfb[kb].sort_values("日期").reset_index(drop=True)
    assert (A["日期"] == B["日期"]).all(), "日期未对齐"
    return (B[col].to_numpy() - A[col].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    q4 = Path(a.q4_dir)
    out = dict(protocol="q4_scenarioB_v1", scope="2025-02-01—12-31（334 天）",
               bootstrap=dict(reps=2000, block_days=7, seed=2025),
               note="全部为同日期配对；正 = 前者更贵；账单一律真实价", comparisons={})
    cells = {}
    for c in ("A_v0", "A_v", "B_v0", "B_v", "sens_refresh"):
        p = q4 / c
        if (p / "q42_逐日费用.csv").exists():
            cells[c] = (load_daily(p, "q42"), load_daily(p, "q43"))

    # ① 组内
    for c, (q2, q3) in cells.items():
        out["comparisons"][f"{c}_Q4-2_B_minus_A"] = stat(diff(q2, "A", q2, "B"))
        out["comparisons"][f"{c}_Q4-2_C_minus_B"] = stat(diff(q2, "B", q2, "C"))
        out["comparisons"][f"{c}_Q4-3_部署_minus_none"] = stat(
            diff(q3, "none", q3, "历史选型执行策略"))
        for k in ("6", "12", "18", "6+12", "6+18", "12+18", "6+12+18"):
            out["comparisons"][f"{c}_Q4-3_{k}_minus_none"] = stat(diff(q3, "none", q3, k))

    # ② 跨格
    if "A_v" in cells and "B_v" in cells:
        out["comparisons"]["跨格_B_v_minus_A_v_Q4-2B"] = stat(diff(cells["A_v"][0], "B", cells["B_v"][0], "B"))
        out["comparisons"]["跨格_B_v_minus_A_v_Q4-3部署"] = stat(
            diff(cells["A_v"][1], "历史选型执行策略", cells["B_v"][1], "历史选型执行策略"))
    if "B_v0" in cells and "B_v" in cells:
        out["comparisons"]["跨格_B_v_minus_B_v0_Q4-2B"] = stat(diff(cells["B_v0"][0], "B", cells["B_v"][0], "B"))
        out["comparisons"]["跨格_B_v_minus_B_v0_Q4-3部署"] = stat(
            diff(cells["B_v0"][1], "历史选型执行策略", cells["B_v"][1], "历史选型执行策略"))
    if "A_v0" in cells and "A_v" in cells:
        out["comparisons"]["跨格_A_v_minus_A_v0_Q4-2B"] = stat(diff(cells["A_v0"][0], "B", cells["A_v"][0], "B"))

    # ③ 价格轨（单路径格）
    base = cells.get("B_v", (None, None))[0]
    if base is not None:
        bref = base["B"].sort_values("日期").reset_index(drop=True)
        for name in ("sens_track_lag1", "sens_track_fix", "sens_track_shape"):
            p = q4 / name / "paths" / "B.pkl"
            if not p.exists():
                continue
            s = load_single(p)
            d = s["总费用"].to_numpy() - bref["总费用"].to_numpy()
            out["comparisons"][f"价格轨_{name.replace('sens_track_','')}_minus_hist7"] = stat(d)

    # ④ 刷新
    if "sens_refresh" in cells and "B_v" in cells:
        out["comparisons"]["刷新_可刷新B_minus_严格B_Q4-3部署"] = stat(
            diff(cells["B_v"][1], "历史选型执行策略", cells["sens_refresh"][1], "历史选型执行策略"))

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    sig = sum(1 for v in out["comparisons"].values() if v["significant"])
    print(f"比较数 {len(out['comparisons'])}，显著 {sig}，非显著 {len(out['comparisons'])-sig}\n")
    for k, v in out["comparisons"].items():
        mark = "显著" if v["significant"] else "不显著"
        print(f"  {k:<46} 均值={v['mean']:>11,.2f} CI=[{v['bootstrap_q025']:>10,.1f},{v['bootstrap_q975']:>10,.1f}] "
              f"负/正={v['negative_days']}/{v['positive_days']} {mark}")


if __name__ == "__main__":
    main()
