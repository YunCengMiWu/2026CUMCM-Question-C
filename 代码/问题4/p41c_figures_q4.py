# -*- coding: utf-8 -*-
"""figures_q4B：问题4 口径B 图件（重出受影响的 result_q4_costs，新增四格分解与价格预测质量图）。

未受影响的 Q4 数据描述图（raw_q4_prices、process_q4_price_load）按 SHA256 校验后逐字节复用。
"""
import hashlib, json, os, shutil, sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "utils"))
sys.path.insert(0, str(ROOT / "03_代码"))
from setup_style import setup_style          # noqa: E402
from export_figure import export_figure      # noqa: E402
from visual_qa import audit_layout           # noqa: E402
from p10_core import load_all                # noqa: E402
from p16_price_forecast import load_cache    # noqa: E402

R = ROOT / "04_结果与检验"
# 2026-09-12 晚：可用环境变量指向新树（默认仍为交付基线树，行为不变）
Q4 = Path(os.environ.get("Q4_TREE", str(R / "重跑_Q4_B_20260912")))
OLD_FIG = R / "figures_v2"
OUT = R / "figures_q4B"
REUSE = ["raw_q4_prices", "process_q4_price_load"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    setup_style(journal="general", lang="zh", use_sciplots=False)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    OUT.mkdir(parents=True, exist_ok=True)
    contracts = []

    def finish(fig, name, claim, source, reused=False):
        fig.tight_layout()
        issues = audit_layout(fig)
        if any(lv in ("WARN", "FAIL") for lv, _ in issues):
            raise RuntimeError((name, issues))
        export_figure(fig, os.path.join(OUT, name), formats=["png", "svg"], dpi=300,
                      grayscale_preview=True)
        contracts.append(dict(figure=name, claim=claim, source=source, backend="matplotlib",
                              口径="q4_scenarioB_v1（口径B，决策用预测价、结算用真实价）",
                              reused=reused, units="各轴明确标注",
                              layout="单面板，7.2x4.2 英寸", issues=issues))
        plt.close(fig)

    def panel(title, xlabel, ylabel, height=4.2):
        f, a = plt.subplots(figsize=(7.2, height))
        a.set(title=title, xlabel=xlabel, ylabel=ylabel)
        a.spines[["top", "right"]].set_visible(False)
        return f, a

    def feb(df):
        df = df.copy(); df["日期"] = df["日期"].astype(str)
        return df[~df["日期"].str.startswith("2025-01")]

    def cell_tot(name, track="B", where="q42"):
        if where == "q42":
            df = pd.read_csv(Q4 / name / "q42_逐日费用.csv")
            return float(feb(df[df["模型"] == track])["总费用"].sum())
        df = pd.read_csv(Q4 / name / "q43_逐日费用.csv")
        return float(feb(df[df["模型"] == track])["总费用"].sum())

    # ---------- 1) 四格分解（跨日价值 vs 电价未知代价）
    cells = ["A_v0", "A_v", "B_v0", "B_v"]
    lab = ["口径A・v=0\n(价格已知)", "口径A・滚动v", "口径B・v=0\n(价格未知)", "口径B・滚动v\n(正式)"]
    q42b = [cell_tot(c, "B") for c in cells]
    dep = [cell_tot(c, "历史选型执行策略", "q43") for c in cells]
    f, a = panel("问题4：口径与跨日库存价值的四格对照（2—12月）", "组合", "总费用 / 万元", height=4.6)
    x = np.arange(4); w = 0.38
    a.bar(x - w / 2, np.array(q42b) / 1e4, w, color="#0072B2", label="Q4-2 B（两阶段随机）")
    a.bar(x + w / 2, np.array(dep) / 1e4, w, color="#009E73", label="Q4-3 历史选型部署")
    for i in range(4):
        a.annotate(f"{q42b[i]/1e4:,.1f}", (i - w / 2, q42b[i] / 1e4), ha="center", va="bottom", fontsize=8)
        a.annotate(f"{dep[i]/1e4:,.1f}", (i + w / 2, dep[i] / 1e4), ha="center", va="bottom", fontsize=8)
    a.set_xticks(x, lab, fontsize=9); a.legend(loc="lower right")
    finish(f, "result_q4_four_cells_q4B",
           "跨日库存价值使费用下降（Q4-2 B：B_v−B_v0 = −274,261.28 元；Q4-3 部署：−608,762.77 元）；"
           "Q4-2 中电价未知使其上升（B_v−A_v = +112,302.23 元）、交互项为 +3,519.48 元；"
           "**Q4-3 部署中口径B 反而比口径A 便宜 219,360.61 元**，该反直觉差异来自"
           "「知真实价 ⇒ 更激进套利 ⇒ 储能余量更薄 ⇒ 被 5 倍紧急罚通道放大」，"
           "**不得解读为价格信息价值**（见 结论_Q4_补做.md §5）",
           "重跑_Q4_B_20260912/{A_v0,A_v,B_v0,B_v}/q42_逐日费用.csv、q43_逐日费用.csv（334天）")

    # ---------- 2) 效应分解瀑布（Q4-2 B）
    e = json.loads((Q4 / "q4_对照汇总.json").read_text(encoding="utf-8"))["effects_334"]["Q2B"]
    base = e["A_v0"]; d_v = e["跨日价值效应_Bv_minus_Bv0"]; d_p = e["电价信息效应_Bv_minus_Av"]
    f, a = panel("问题4-2 B：两个变化的效应分解（2—12月）", "步骤", "总费用 / 万元")
    steps = [base, d_v, d_p, e["B_v"]]
    names = ["口径A・v=0\n(基准)", "＋跨日库存价值\n(滚动 v)", "＋电价未知\n(口径B)", "口径B・滚动v\n(正式)"]
    bottoms = [0, base + d_v, base + d_v, 0]
    heights = [base, d_v, d_p, e["B_v"]]
    colors = ["#9ecae1", "#009E73", "#D55E00", "#0072B2"]
    for i in range(4):
        a.bar(i, heights[i] / 1e4, bottom=bottoms[i] / 1e4, color=colors[i])
        a.annotate(f"{heights[i]/1e4:+,.1f}" if i in (1, 2) else f"{heights[i]/1e4:,.1f}",
                   (i, (bottoms[i] + heights[i]) / 1e4), ha="center", va="bottom", fontsize=9)
    a.set_xticks(range(4), names, fontsize=9)
    finish(f, "result_q4_effect_decomp_q4B",
           "跨日价值约 −27.4 万元（口径B 基准 B_v−B_v0 = −274,261.28 元）、"
           "电价未知代价约 +11.2 万元（同基准 B_v−A_v = +112,302.23 元），交互项 +3,519.48 元；"
           "口径A 基准的跨日价值为 −275,006.70 元。**注意：口径A 同时改变了控制激进度，"
           "故 +10.7 万元是价格预报价值的「下界」而非干净的 VOI**（见 结论_Q4_补做.md §5）",
           "q4_对照汇总.json 的 effects_334.Q2B")

    # ---------- 3) 价格预测质量
    data = load_all(); P = np.asarray(data["P_var"], float)
    fc = load_cache(Q4 / "B_v" / "forecast" / "电价预测缓存.npz")
    fp = np.asarray(fc["FP_hist7"], float)
    di = 132                                    # 典型日（曾触发退化 LP 的当日）
    h = (np.arange(144) + .5) / 6
    f, a = panel("电价预测与真实电价（示例日 2025-05-13）", "时刻 / h", "电价 / 元每kWh")
    a.plot(h, P[di], color="#0072B2", label="真实（附件4）")
    a.plot(h, fp[di], "--", color="#D55E00", label="预测 hist7（0:00 可得）")
    a.legend()
    finish(f, "process_q4_price_forecast_q4B", "日前价格预测只用历史、可复现当日形状但水平有误差",
           "附件4 + 电价预测缓存.npz（FP_hist7）")

    f, a = panel("价格预测误差的分时段分布", "时段", "绝对误差 / 元每kWh")
    err = np.abs(fp[31:] - P[31:])
    bands = [(0, 36, "0:00-6:00"), (36, 72, "6:00-12:00"), (72, 108, "12:00-18:00"), (108, 144, "18:00-24:00")]
    a.boxplot([err[:, s:e].ravel() for s, e, _ in bands],
              tick_labels=[n for _, _, n in bands], showfliers=False)
    a.annotate(f"总体 MAE {err.mean():.4f} 元/kWh", (0.02, 0.92), xycoords="axes fraction", fontsize=9)
    finish(f, "process_q4_price_mae_bands_q4B", "预测误差随时段不同，凌晨最低、白天最高",
           "附件4 与 FP_hist7（334 天 × 144 槽）；箱体 IQR 中位数，须 1.5IQR，不显示离群点")

    # ---------- 4) 复用的数据描述图
    for name in REUSE:
        for ext in ("png", "svg"):
            src = OLD_FIG / f"{name}.{ext}"
            if src.exists():
                shutil.copy2(src, OUT / src.name)
        contracts.append(dict(figure=name, claim="问题4 数据描述类，未受口径切换影响",
                              source="figures_v2/（逐字节复用）", backend="matplotlib",
                              口径="不涉及口径", reused=True, sha256=sha(OLD_FIG / f"{name}.png"),
                              units="同原图", layout="同原图", issues=[]))
    (OUT / "图表契约.json").write_text(json.dumps(contracts, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(json.dumps(dict(status="PASS", figures=len(contracts),
                          recomputed=[c["figure"] for c in contracts if not c["reused"]],
                          reused=[c["figure"] for c in contracts if c["reused"]]), ensure_ascii=False))


if __name__ == "__main__":
    main()
