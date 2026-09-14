# -*- coding: utf-8 -*-
"""问题4 口径B：四个指定日期的表1/表2/表3 生成器（Q4-2 B 与 Q4-3 部署）。

来源：口径B 正式格（默认 重跑_Q4_B_20260912/B_v）的 q42_best.pkl 与 q43_daystore.pkl。
口径：决策用价格预测、**账单一律用附件4 真实价**（与 p103 审计一致）。
输出：results/四个指定日期_表123_q4B.json、检验日志/四个指定日期_表123_q4B.md，
      并把 q42_* 条目合并进 results/指定日期表格.json（带 _口径 标注）。
内置：与 result4-2.xlsx / result4-3.xlsx 逐格交叉校验。
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p10_core import fmt_min, load_all, merge_emergency_intervals  # noqa: E402

DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]
BUCKETS = [(0, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 144)]
BLABEL = ["0:00-4:00", "4:00-8:00", "8:00-12:00", "12:00-16:00", "16:00-20:00", "20:00-24:00"]
KOUJING = "q4_scenarioB_v1（口径B：0:00 不知当日电价；决策用预测价、结算用真实价；滚动跨日价值）"
# 2026-09-12 晚：内核版本标注（问题4 已按 eps 归一化修正后的内核重跑；问题2/3 按用户裁定未重跑）
KERN_Q4 = ("eps归一化修正后：p10_core.py b3242ad925c82d63…"
           "（末槽系数 (eps_soc+v_term)/S，有效日末系数 0.482548）")
KERN_Q23 = ("eps归一化修正前：p10_core.py 0a2cef2e2b4428bb…"
            "（末槽系数 eps_soc+v_term/S，有效日末系数 0.491548）")


def slot_label(j):
    return f"{fmt_min(j * 10)}-{fmt_min((j + 1) * 10)}"


def build(case: Path):
    P = np.asarray(load_all()["P_var"], float)
    with open(case / "q42_best.pkl", "rb") as f:
        q42 = pickle.load(f)
    with open(case / "q43_daystore.pkl", "rb") as f:
        q43 = pickle.load(f)
    assert q42["protocol"] == q43["protocol"] == "q4_scenarioB_v1", "协议不符，拒绝生成"
    out, old = {}, {}
    for tag, recs, is_q3 in (("q42", q42["recs"], False), ("q43", q43["store"], True)):
        for ds in DATES:
            di = (np.datetime64(ds) - np.datetime64("2025-01-01")).astype(int)
            r = recs[di]
            assert str(r["日期"]) == ds, f"{tag} {ds} 索引错位: {r['日期']}"
            p = P[di]                                        # 结算价 = 真实价
            q0 = np.asarray(r["q0"], float); qc = np.asarray(r.get("qc", r["q0"]), float)
            c = np.asarray(r["c"], float); d = np.asarray(r["d"], float)
            qEM = np.asarray(r["qEM"], float)
            plan_fee = float(p @ q0); adj = float(r["调整费"]); emg = float(5 * p @ qEM)
            out[f"{tag}_{ds}"] = {
                "日期": ds, "阶段": r["阶段"], "口径": KOUJING,
                "轨道": "问题4-2 B（两阶段随机，价格未知）" if not is_q3 else "问题4-3 历史选型部署",
                "当日真实电价": {"均值": round(float(p.mean()), 4), "最小": round(float(p.min()), 4),
                            "最大": round(float(p.max()), 4)},
                "价格预测MAE": round(float(r.get("price_mae", float("nan"))), 4),
                "滚动v_term": round(float(r.get("v_term_used", 0.0)), 6),
                "表1_计划购电量_kWh": {slot_label(j): round(float(q0[j]), 4) for j in range(144)},
                "表2_充放电量_kWh": {BLABEL[i]: [round(float(c[a:b].sum()), 4), round(float(d[a:b].sum()), 4)]
                                     for i, (a, b) in enumerate(BUCKETS)},
                "表3_紧急购电量_kWh": [[f"{fmt_min(a)}-{fmt_min(b)}", round(float(v), 4)]
                                       for a, b, v in merge_emergency_intervals(qEM)],
                "储电量": {"0:00": round(float(r["E0"]), 4), "24:00": round(float(r["Eend"]), 4)},
                "汇总": {"计划购电量": round(float(q0.sum()), 4), "计划购电费": round(plan_fee, 4),
                        "紧急购电量": round(float(qEM.sum()), 4), "紧急购电费": round(emg, 4),
                        "调整费": round(adj, 4), "总费用": round(plan_fee + adj + emg, 4)},
            }
            if is_q3:
                out[f"{tag}_{ds}"]["调整购电量_kWh"] = {slot_label(j): round(float(qc[j]), 4) for j in range(144)}
                out[f"{tag}_{ds}"]["更新事件"] = [dict(tau=int(u["tau"]), hour=int(u["hour"]),
                                                     fee=round(float(u["fee"]), 4))
                                                for u in r.get("updates", [])]
            old[f"{tag}_{ds}"] = {
                "_口径": KOUJING,
                "全天购电量": round(float(q0.sum() + qEM.sum()), 1),
                "计划购电费": round(plan_fee, 1), "调整费": round(adj, 1),
                "紧急购电费": round(emg, 1), "总费用": round(plan_fee + adj + emg, 1),
                "紧急购电区间": [[f"{fmt_min(a)}-{fmt_min(b)}", round(float(v), 1)]
                                 for a, b, v in merge_emergency_intervals(qEM)],
                "充放电六段": [[round(float(c[a:b].sum()), 1), round(float(d[a:b].sum()), 1)]
                               for a, b in BUCKETS],
                "E0": round(float(r["E0"]), 1), "E24": round(float(r["Eend"]), 1),
            }
    return out, old


def cross_check(case: Path, out: dict):
    res = {}
    for fname, tag, adj_sheet in (("result4-2.xlsx", "q42", None), ("result4-3.xlsx", "q43", "调整购电量")):
        wb = openpyxl.load_workbook(case / fname, data_only=True)
        worst = 0.0
        for sheet, field in ((("计划购电量"), "表1_计划购电量_kWh"), (adj_sheet, "调整购电量_kWh")):
            if sheet is None:
                continue
            ws = wb[sheet]
            for ds in DATES:
                i = (np.datetime64(ds) - np.datetime64("2025-02-01")).astype(int)
                vals = [ws.cell(i + 2, 2 + j).value for j in range(144)]
                exp = [out[f"{tag}_{ds}"][field][slot_label(j)] for j in range(144)]
                worst = max(worst, max(abs(float(a) - b) for a, b in zip(vals, exp)))
        wb.close()
        res[fname] = dict(max_abs_diff=worst, tol=0.0051, status="PASS" if worst <= 0.0051 else "FAIL")
    return res


def to_md(out: dict) -> str:
    L = ["# 问题4（口径B）四个指定日期表1/表2/表3", "", f"> 口径：{KOUJING}",
         "> 指定日期 = 2025.3.20、2025.6.21、2025.9.23、2025.12.21；账单一律按当日**真实**附件4 电价结算。", ""]
    for key, e in out.items():
        s = e["汇总"]
        L += [f"## {key}（{e['轨道']}，阶段 {e['阶段']}）", "",
              f"- 当日真实电价：均值 {e['当日真实电价']['均值']}，范围 {e['当日真实电价']['最小']}–{e['当日真实电价']['最大']} 元/kWh"
              f"；价格预测 MAE {e['价格预测MAE']}；滚动 v_term {e['滚动v_term']}",
              f"- **全天**：计划购电量 {s['计划购电量']:,.4f} kWh / {s['计划购电费']:,.4f} 元"
              f"；紧急购电量 {s['紧急购电量']:,.4f} kWh / {s['紧急购电费']:,.4f} 元"
              + (f"；调整费 {s['调整费']:,.4f} 元" if s["调整费"] else "")
              + f"；**总费用 {s['总费用']:,.4f} 元**",
              f"- 储电量：0:00 {e['储电量']['0:00']:,.4f} kWh → 24:00 {e['储电量']['24:00']:,.4f} kWh", "",
              "### 表1 计划购电量（kWh）", "", "| 时段 | 购电量 | 时段 | 购电量 |", "|---|---|---|---|"]
        items = list(e["表1_计划购电量_kWh"].items())
        for i in range(0, 144, 2):
            L.append(f"| {items[i][0]} | {items[i][1]:,.4f} | {items[i+1][0]} | {items[i+1][1]:,.4f} |")
        L += ["", "### 表2 充放电量（六段）与储电量", "", "| 时段 | 充电量 | 放电量 |", "|---|---|---|"]
        for k, (cc, dd) in e["表2_充放电量_kWh"].items():
            L.append(f"| {k} | {cc:,.4f} | {dd:,.4f} |")
        L += [f"| 0:00 储电量 | {e['储电量']['0:00']:,.4f} | 24:00 储电量 | {e['储电量']['24:00']:,.4f} |", "",
              "### 表3 紧急购电量（区间）", ""]
        if e["表3_紧急购电量_kWh"]:
            L += ["| 时段 | 紧急购电量 |", "|---|---|"]
            for a, v in e["表3_紧急购电量_kWh"]:
                L.append(f"| {a} | {v:,.4f} |")
        else:
            L.append("（当日无紧急购电）")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", default=str(HERE.parent / "04_结果与检验" /
                                             "重跑_Q4_B_20260912" / "B_v"))
    ap.add_argument("--res-dir", default=str(HERE.parent / "04_结果与检验" / "results"))
    ap.add_argument("--log-dir", default=str(HERE.parent / "04_结果与检验" / "检验日志"))
    a = ap.parse_args()
    case, res, logd = Path(a.case_dir), Path(a.res_dir), Path(a.log_dir)
    out, old = build(case)
    checks = cross_check(case, out)
    assert all(v["status"] == "PASS" for v in checks.values()), f"工作簿交叉校验失败: {checks}"
    (res / "四个指定日期_表123_q4B.json").write_text(
        json.dumps(dict(protocol="q4_scenarioB_v1", 口径=KOUJING, 内核=KERN_Q4, dates=DATES,
                        xlsx_cross_check=checks, tables=out), ensure_ascii=False, indent=1),
        encoding="utf-8")
    (logd / "四个指定日期_表123_q4B.md").write_text(to_md(out), encoding="utf-8")
    # 合并进 指定日期表格.json（替换 q42_*，保留 q2_*/q3_*/旧 q42 备份在 备份目录）
    p = res / "指定日期表格.json"
    merged = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    for k, v in old.items():
        merged[k] = v
    for k, v in list(merged.items()):
        if isinstance(v, dict):
            v["_内核"] = KERN_Q4 if k.startswith("q4") else KERN_Q23
    (res / "指定日期表格.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(dict(status="PASS", entries=len(out), xlsx_cross_check=checks,
                          keys=sorted(merged.keys())), ensure_ascii=False))


if __name__ == "__main__":
    main()
