"""逐日历史选型：模型及残差均只使用当日之前信息。"""
import os, json
import numpy as np
import pandas as pd
from p10_core import *

def select_past_only(candidates, actual, start=14):
    names = list(candidates)
    out = np.zeros_like(actual)
    selected = []
    for di in range(len(actual)):
        if di <= start:
            winner = "lag1"
        else:
            winner = min(names, key=lambda n: float(np.mean(
                np.abs(candidates[n][start:di]-actual[start:di]))))
        out[di] = candidates[winner][di]
        selected.append(winner)
    return out, selected

def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(LOGD, exist_ok=True)
    data = load_all(); saved = {}; metrics = {}; selections = {}
    for prefix, actual in (("L",data["L"]),("G",data["G"])):
        print("rolling forecast", prefix, flush=True)
        lag = np.empty_like(actual); lag[1:] = actual[:-1]; lag[0] = 0
        m7 = baseline_mean7(actual); m7[:7] = lag[:7]
        lgb = rolling_forecast(actual); lgb[:14] = lag[:14]
        candidates = dict(lgb=lgb, lag1=lag, m7=m7)
        forecast, selections[prefix] = select_past_only(candidates, actual)
        saved["F"+prefix] = forecast
        for name, array in candidates.items(): saved["F"+prefix+"_"+name] = array
        for name, array in dict(candidates, online=forecast).items():
            for period, idx in (("334天因果回测",slice(31,365)),("5至12月",slice(120,365))):
                err = array[idx]-actual[idx]
                metrics[prefix+"_"+name+"_"+period] = dict(
                    MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err**2).mean())),
                    bias=float(err.mean()))
    saved["protocol"] = np.array("end_label_causal_v2")
    np.savez_compressed(os.path.join(RES,"预测缓存.npz"), **saved)
    pd.DataFrame(dict(日期=data["dates"],负荷模型=selections["L"],光伏模型=selections["G"])).to_csv(
        os.path.join(RES,"逐日预测选型.csv"),index=False,encoding="utf-8-sig")
    with open(os.path.join(RES,"预测模型选择.json"),"w",encoding="utf-8") as f:
        json.dump(dict(protocol="end_label_causal_v2", selection="每日仅依据此前样本外MAE",
                       metrics=metrics),f,ensure_ascii=False,indent=2)
    print("forecast DONE",flush=True)

if __name__ == "__main__": main()
