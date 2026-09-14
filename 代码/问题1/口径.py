#!/usr/bin/env python3
"""最小可运行版本 · 口径与数据层（方案丙）。

- 槽口径：第 j 槽（j=1..144）= [10j, 10j+10) 分钟，即 0:10→次日 0:10；标签＝槽起点。
- 单位：功率 kW、电量 kWh、电价 元/kWh、Δt=1/6 h；电量 = 功率 × Δt。
- 只读附件；不改原始数据。
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from openpyxl import load_workbook

ROOT = Path(r"C:\Users\John\Desktop\26题")
ATT = ROOT / "00_题目与原始数据" / "C题" / "附件"
DT = 1.0 / 6.0                 # h
ETA = 0.9                      # 单向效率（D4 裁定）
E_MIN, E_OP = 1200.0, 10800.0  # kWh（附录1 运行区间）
E_START_1JAN = 6000.0          # kWh（附录1，2025-01-01 0:00）
P_MAX = 5000.0                 # kW
M_SLOT = P_MAX * DT            # 833.3333 kWh（单槽充放电量上界）
N_SLOT = 144
K_EMG, K_DOWN, K_UP = 5.0, 0.5, 1.5   # 紧急 5 倍、下调 0.5 倍、上调 1.5 倍

RESULT_DAYS = 334              # 2025-02-01—12-31


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def label_minutes(value) -> int:
    """时间标签 → 距当日 0:00 的分钟数；'0:00+1' → 1440（附录2 定义）。"""
    if isinstance(value, datetime.time):
        return value.hour * 60 + value.minute
    text = str(value).strip()
    if "+1" in text:
        return 1440
    hh, mm, *_ = text.split(":")
    return int(hh) * 60 + int(mm)


def read_rows(path: Path, sheet=0):
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
    rows = [tuple(r) for r in ws.iter_rows(values_only=True)]
    while rows and not any(v is not None for v in rows[-1]):
        rows.pop()
    wb.close()
    return rows


def load_attachment1() -> dict:
    """典型日：电价/负载/光伏预测（144 槽，槽起点标签 00:10…0:00+1）。"""
    rows = read_rows(ATT / "附件1.xlsx", 0)
    labels = [label_minutes(r[0]) for r in rows[1:]]
    assert labels == [10 * (j + 1) for j in range(N_SLOT)], "附件1 槽标签不符方案丙"
    return {
        "slot_minutes": labels,
        "price": np.array([float(r[1]) for r in rows[1:]]),
        "load_kw": np.array([float(r[2]) for r in rows[1:]]),
        "pv_kw": np.array([float(r[3]) for r in rows[1:]]),
    }


def load_attachment2() -> dict:
    load_rows = read_rows(ATT / "附件2.xlsx", "小区负载")
    pv_rows = read_rows(ATT / "附件2.xlsx", "光伏发电实际功率")
    dates = [r[0].date() for r in load_rows[1:]]
    return {
        "dates": dates,
        "index": {d: i for i, d in enumerate(dates)},
        "load_kw": np.array([[float(v) for v in r[1:]] for r in load_rows[1:]]),
        "pv_kw": np.array([[float(v) for v in r[1:]] for r in pv_rows[1:]]),
    }


def load_attachment3() -> dict:
    """(日期, 发布时刻) → 24 个整点预报（kW），h=1..24 指向 发布时刻+h 整点。"""
    rows = read_rows(ATT / "附件3.xlsx", 0)[1:]
    out, cur = {}, None
    for r in rows:
        if r[0] not in (None, ""):
            y, m, d = str(r[0]).split("-")
            cur = datetime.date(int(y), int(m), int(d))
        out[(cur, str(r[1]))] = np.array([float(v) for v in r[2:]])
    return out


def load_attachment4() -> dict:
    rows = read_rows(ATT / "附件4.xlsx", 0)
    dates = [r[0].date() for r in rows[1:]]
    return {
        "dates": dates,
        "index": {d: i for i, d in enumerate(dates)},
        "price": np.array([[float(v) for v in r[1:]] for r in rows[1:]]),
    }


def forecast_from_forecast3(fc: dict, day: datetime.date, slot_minutes: list[int]) -> np.ndarray:
    """把附件3 的整点预报摊到 144 槽（方案丙：槽起点 t 取满足 起点 ≤ t < 起点+60 的整点值，保持法）。

    发布时刻 0:00 的 h=1..24 指向当日 1:00…次日 0:00；槽 j 的起点为 10j 分钟。
    """
    base = fc[(day, "0:00")]
    out = np.zeros(N_SLOT)
    for j, start in enumerate(slot_minutes):
        if start <= 0:
            continue
        H = (start // 60) * 60           # §14.4 保持法：取"起点所在小时区间的左端整点"（floor，不是 ceil）
        h = H // 60                      # 该整点相对 0:00 发布的预见期（小时）
        out[j] = base[h - 1] if 1 <= h <= 24 else base[0]   # h=0（[0:10,1:00)）退回 1:00 值（已声明的最小配置简化）
    return out


def persistence_from_actual(prev_row: np.ndarray) -> np.ndarray:
    """预报驱动分支的最小规则：最近已完整观测的前一日同槽值（S5 用同星期均值，正式配置再启用）。"""
    return prev_row.copy()


def cost(price_kwh: np.ndarray, plan: np.ndarray, adj: np.ndarray | None = None,
         emg: np.ndarray | None = None) -> dict:
    """§14.7 结算：f_p(b,a)=p·a+0.5p|a−b|（含紧急 5p·e）；返回分项与总额（元）。"""
    plan = np.asarray(plan, dtype=float)
    adj = plan if adj is None else np.asarray(adj, dtype=float)
    emg = np.zeros_like(plan) if emg is None else np.asarray(emg, dtype=float)
    down = np.maximum(plan - adj, 0.0)
    up = np.maximum(adj - plan, 0.0)
    c_plan = float(np.sum(price_kwh * plan))
    c_net_adj = float(np.sum(-price_kwh * down + K_DOWN * price_kwh * down + K_UP * price_kwh * up))
    c_emg = float(np.sum(K_EMG * price_kwh * emg))
    return {
        "plan": c_plan, "net_adj": c_net_adj, "emg": c_emg,
        "total": c_plan + c_net_adj + c_emg,
        "down_kwh": float(down.sum()), "up_kwh": float(up.sum()), "emg_kwh": float(emg.sum()),
    }


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
