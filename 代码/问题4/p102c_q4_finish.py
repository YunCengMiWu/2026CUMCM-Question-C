# -*- coding: utf-8 -*-
"""问题4 口径B：单臂合成器（读取 paths/*.pkl，执行部署策略并写出全部产物）。"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p102_q4_core import (COMBOS, E_INIT, PROTOCOL, code_hashes, daily_frame,  # noqa: E402
                          deploy_from_shadow, input_hashes, key, load_all,
                          load_loadpv_forecasts, save_pickle, summary, write_json,
                          write_result_workbooks)
from p16_price_forecast import load_cache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--track", default="hist7")
    ap.add_argument("--refresh-mode", choices=["none", "hourly"], default="none")
    ap.add_argument("--v-mode", choices=["rolling", "zero"], default="rolling")
    ap.add_argument("--price-source", choices=["forecast", "actual"], default="forecast")
    ap.add_argument("--v-const", type=float, default=None)
    a = ap.parse_args()
    case = Path(a.case_dir)

    def load_path(label):
        p = case / "paths" / (label.replace("+", "plus") + ".pkl")
        if not p.exists():
            raise FileNotFoundError(p)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        assert obj["protocol"] == PROTOCOL and obj["label"] == label
        return obj

    q2_labels = ["A", "B", "C"]
    q3_labels = [key(c) for c in COMBOS]
    q2, q3, d2, d3 = {}, {}, {}, {}
    for lb in q2_labels:
        o = load_path(lb); q2[lb] = o["recs"]; d2[lb] = float(o["elapsed_seconds"])
    for lb in q3_labels:
        o = load_path(lb); q3[lb] = o["recs"]; d3[lb] = float(o["elapsed_seconds"])

    cfg = dict(track=a.track, refresh_mode=a.refresh_mode, v_mode=a.v_mode,
               price_source=a.price_source, v_const=a.v_const, S=10, seed=2025,
               eps_soc=1e-3, tp=1e-6, lam=.3, method="hold")
    data = load_all()
    forecasts = load_loadpv_forecasts(case / "forecast" / "负荷光伏预测缓存.npz")
    price_fc = load_cache(case / "forecast" / "电价预测缓存.npz")
    deployed, cumulative = deploy_from_shadow(data, forecasts, price_fc, q3, cfg)

    save_pickle(case / "q42_best.pkl", dict(protocol=PROTOCOL, best="B", recs=q2["B"],
                                            q2_all=q2, config=cfg))
    save_pickle(case / "q43_daystore.pkl", dict(protocol=PROTOCOL, best_key="历史选型执行策略",
                                                store=deployed, all_stores=q3,
                                                results={k: summary(v) for k, v in q3.items()} |
                                                        {"历史选型执行策略": summary(deployed)},
                                                config=cfg))
    q2_df = pd.concat([daily_frame(q2[k], k) for k in q2_labels], ignore_index=True)
    q2_df.to_csv(case / "q42_逐日费用.csv", index=False, encoding="utf-8-sig")
    q2_df.groupby("模型")[["计划购电费", "调整费", "紧急购电费", "紧急购电量", "总费用"]].sum().to_csv(
        case / "q42_汇总.csv", encoding="utf-8-sig")
    q3_df = pd.concat([daily_frame(q3[k], k) for k in q3_labels] +
                      [daily_frame(deployed, "历史选型执行策略")], ignore_index=True)
    q3_df.to_csv(case / "q43_逐日费用.csv", index=False, encoding="utf-8-sig")
    rows = [{"组合": k, **summary(q3[k])} for k in q3_labels]
    rows.append({"组合": "历史选型执行策略", **summary(deployed)})
    pd.DataFrame(rows).to_csv(case / "q43_组合对照.csv", index=False, encoding="utf-8-sig")
    write_result_workbooks(case, q2["B"], deployed)

    pmeta = json.loads((case / "forecast" / "电价预测缓存_meta.json").read_text(encoding="utf-8"))
    lmeta = json.loads((case / "forecast" / "负荷光伏预测缓存_meta.json").read_text(encoding="utf-8"))
    metadata = dict(case_id=a.case_id, protocol=PROTOCOL, project_root=str(HERE.parent),
                    phase="main", 口径="B（0:00 不知道当日电价；决策用预测价、结算用真实价）",
                    price_track=a.track, refresh_mode=a.refresh_mode, v_mode=a.v_mode,
                    price_source=a.price_source, v_const=a.v_const,
                    input_hashes=input_hashes(), code_hashes=code_hashes(),
                    price_forecast_meta=dict(chosen_track=pmeta.get("chosen_track"),
                                             metrics=pmeta.get("metrics"),
                                             audit_no_future=pmeta.get("audit_no_future")),
                    loadpv_forecast_meta=lmeta,
                    parameters=dict(S=10, seed=2025, eps_soc=1e-3, tp=1e-6, lam=.3, method="hold"))
    write_json(case / "main_metadata.json", dict(metadata, config=cfg, q42_durations=d2,
                                                 q43_durations=d3,
                                                 q42_summary={k: summary(v) for k, v in q2.items()},
                                                 q43_summary={k: summary(v) for k, v in q3.items()},
                                                 deployed_summary=summary(deployed),
                                                 selection_cumulative=cumulative))
    print(json.dumps(dict(status="PASS", case=str(case)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
