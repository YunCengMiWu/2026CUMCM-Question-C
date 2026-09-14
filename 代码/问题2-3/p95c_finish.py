# -*- coding: utf-8 -*-
"""v4 单臂合成器：读取 paths/*.pkl（11 条），执行部署策略并写出全部主轨产物。
等价于 p95 run_main 的合成尾段（无进程池）。"""
import argparse, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p95_q23_v4 import (COMBOS, E_INIT, PROTOCOL, daily_frame, deploy_from_shadow,  # noqa: E402
                        input_hashes, code_hashes, key, load_all, load_forecasts,
                        save_pickle, summary, write_json, write_result_workbooks)
import pickle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--v-term", type=float, required=True)
    ap.add_argument("--S", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--method", choices=["hold", "linear"], default="hold")
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
    q2, d2, q3, d3 = {}, {}, {}, {}
    for lb in q2_labels:
        obj = load_path(lb)
        q2[lb] = obj["recs"]; d2[lb] = float(obj["elapsed_seconds"])
    for lb in q3_labels:
        obj = load_path(lb)
        q3[lb] = obj["recs"]; d3[lb] = float(obj["elapsed_seconds"])

    cfg = dict(start=0, stop=365, S=int(a.S), seed=int(a.seed), lam=0.3, method=a.method,
               eps_soc=1e-3, tp=1e-6, v_term=float(a.v_term), initial_state=E_INIT)
    data = load_all()
    forecasts = load_forecasts(str(case / "forecast" / "预测缓存.npz"))
    deployed, cumulative = deploy_from_shadow(data, forecasts, q3, cfg)

    save_pickle(case / "q2_best.pkl", dict(protocol=PROTOCOL, best="B", recs=q2["B"],
                                           q2_all=q2, config=cfg))
    save_pickle(case / "q3_daystore.pkl", dict(protocol=PROTOCOL, best_key="历史选型执行策略",
                                               store=deployed, all_stores=q3,
                                               results={k: summary(v) for k, v in q3.items()} |
                                                       {"历史选型执行策略": summary(deployed)},
                                               config=cfg))
    q2_df = pd.concat([daily_frame(q2[k], k) for k in q2_labels], ignore_index=True)
    q2_df.to_csv(case / "q2_逐日费用.csv", index=False, encoding="utf-8-sig")
    q2_df.groupby("模型")[["计划购电费", "调整费", "紧急购电费", "紧急购电量", "总费用"]].sum().to_csv(
        case / "q2_汇总.csv", encoding="utf-8-sig")
    q3_df = pd.concat([daily_frame(q3[k], k) for k in q3_labels] +
                      [daily_frame(deployed, "历史选型执行策略")], ignore_index=True)
    q3_df.to_csv(case / "q3_逐日费用.csv", index=False, encoding="utf-8-sig")
    combo_rows = [{"组合": k, **summary(q3[k])} for k in q3_labels]
    combo_rows.append({"组合": "历史选型执行策略", **summary(deployed)})
    pd.DataFrame(combo_rows).to_csv(case / "q3_组合对照.csv", index=False, encoding="utf-8-sig")
    write_result_workbooks(case, q2["B"], deployed)

    fmeta = json.loads((case / "forecast" / "预测缓存_meta.json").read_text(encoding="utf-8"))
    metadata = dict(case_id=a.case_id, protocol=PROTOCOL, project_root=str(HERE.parent),
                    phase="main", forecast_cache=str(case / "forecast" / "预测缓存.npz"),
                    input_hashes=input_hashes(), code_hashes=code_hashes(),
                    parameters=dict(S=int(a.S), seed=int(a.seed), eps_soc=1e-3, tp=1e-6, lam=0.3,
                                    method=a.method, v_term=float(a.v_term)),
                    forecast_meta=fmeta)
    write_json(case / "main_metadata.json", dict(metadata, config=cfg,
                                                 q2_durations=d2, q3_durations=d3,
                                                 q2_summary={k: summary(v) for k, v in q2.items()},
                                                 q3_summary={k: summary(v) for k, v in q3.items()},
                                                 deployed_summary=summary(deployed),
                                                 selection_cumulative=cumulative))
    print(json.dumps(dict(status="PASS", case=str(case)), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
