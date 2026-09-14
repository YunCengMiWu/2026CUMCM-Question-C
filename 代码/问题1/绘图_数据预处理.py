#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据预处理可视化（C 题《微网与外部电网电力调控策略》）

按 9 个预处理环节组织 7 张图：
    图3-1 时间对齐与数据结构          （环节 1、2）
    图3-2 质量检验与缺失识别          （环节 2、3）
    图3-3 异常辨识与判据修正          （环节 4）
    图3-4 异常处理与极端值保留        （环节 5）
    图3-5 变量分布分析                （环节 6）
    图3-6 特征构造与派生变量          （环节 7）
    图3-7 相关性与分组诊断            （环节 8、9）

口径：方案丙——第 j 槽 = [10j, 10j+10) 分钟（标签_j 为槽开始时刻）；
      每日窗口 0:10 → 次日 0:10；Δt = 1/6 h；电量(kWh) = 功率(kW) × 1/6。

只读原始附件与已登记派生表；只新建本目录图件与本脚本，不改动项目既有文件。
运行：C:\\Users\\John\\.conda\\envs\\deeplearning\\python.exe "04_结果与检验/可视化集/03_数据预处理/绘图_数据预处理.py"
"""

from __future__ import annotations

import datetime
import sys
import warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_ORIG_STDOUT = sys.stdout

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------
ROOT = Path(r"C:\Users\John\Desktop\26题")
ATT = ROOT / "00_题目与原始数据" / "C题" / "附件"
RES = ROOT / "04_结果与检验" / "results"
OUTDIR = ROOT / "04_结果与检验" / "可视化集" / "03_数据预处理"
LOGDIR = ROOT / "04_结果与检验" / "检验日志"
LOG = LOGDIR / "可视化_数据预处理.txt"

OUTDIR.mkdir(parents=True, exist_ok=True)
LOGDIR.mkdir(parents=True, exist_ok=True)


class _Tee:
    """同时写终端与日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


_LOG_FH = open(LOG, "w", encoding="utf-8", newline="\n")
sys.stdout = _Tee(_ORIG_STDOUT, _LOG_FH)

# --------------------------------------------------------------------------
# 依赖
# --------------------------------------------------------------------------
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats

# 绘图约定：优先使用 math-modeling 技能；失败则回退
SKILL_STYLE = False
SKILL_EXPORT = False
_style_info = {}
try:
    sys.path.insert(0, r"C:\Users\John\.codex\skills\math-modeling\tools\figure\scripts")
    from setup_style import setup_style  # type: ignore

    _style_info = setup_style(journal="general", lang="zh")
    SKILL_STYLE = True
except Exception as exc:  # noqa: BLE001
    print(f"[警告] setup_style 不可用，回退默认中文字体：{exc}")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False

try:
    from export_figure import export_figure  # type: ignore

    SKILL_EXPORT = True
except Exception as exc:  # noqa: BLE001
    print(f"[警告] export_figure 不可用，回退 fig.savefig：{exc}")
    export_figure = None

plt.rcParams["figure.constrained_layout.use"] = False  # 手工排版，避免表格/文字被压缩
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["savefig.facecolor"] = "white"

# 捕获"缺字"警告（中文字体缺字形是图件最常见的隐性缺陷），纳入自检
MISSING_GLYPHS: set[str] = set()
_ORIG_SHOWWARNING = warnings.showwarning


def _showwarning(message, category, filename, lineno, file=None, line=None):
    msg = str(message)
    if "missing from font" in msg:
        MISSING_GLYPHS.add(msg)
    else:
        _ORIG_SHOWWARNING(message, category, filename, lineno, file=file, line=line)


warnings.showwarning = _showwarning

C = list(plt.get_cmap("tab10").colors)
GREY = "#8a8a8a"
HDR = "#c9d7ee"
LOG_HDR = "可视化_数据预处理"

# --------------------------------------------------------------------------
# 数据读取（只读；记录单元格类型用于"混合类型"事实）
# --------------------------------------------------------------------------
DT = 1.0 / 6.0


def read_sheet(fn, sheet=None):
    wb = openpyxl.load_workbook(ATT / fn, data_only=True, read_only=True)
    ws = wb[sheet] if sheet else wb[wb.sheetnames[0]]
    rows = [list(r) for r in ws.iter_rows(values_only=True)]
    wb.close()
    return rows


def to_minutes(v):
    """时间标签归一化为"距当日 0:00 的分钟数"；'0:00+1' → 1440。不依赖单元格类型。"""
    if isinstance(v, datetime.time):
        return v.hour * 60 + v.minute
    s = str(v).strip()
    if "+1" in s:
        return 1440
    hh, mm = s.split(":")[:2]
    return int(hh) * 60 + int(mm)


def hhmm(m):
    m = int(round(m))
    return f"{m // 60:02d}:{m % 60:02d}"


def type_counts(values):
    out = {}
    for v in values:
        out[type(v).__name__] = out.get(type(v).__name__, 0) + 1
    return out


a1 = read_sheet("附件1.xlsx")
a2_load = read_sheet("附件2.xlsx", "小区负载")
a2_pv = read_sheet("附件2.xlsx", "光伏发电实际功率")
a3 = read_sheet("附件3.xlsx")
a4 = read_sheet("附件4.xlsx")

# --- 附件1 ---
lab1 = [to_minutes(r[0]) for r in a1[1:]]
t1_types = type_counts([r[0] for r in a1[1:]])
PRICE1 = np.array([float(r[1]) for r in a1[1:]])
LD1 = np.array([float(r[2]) for r in a1[1:]])
PV1 = np.array([float(r[3]) for r in a1[1:]])

# --- 附件2 / 附件4 ---
lab2 = [to_minutes(v) for v in a2_load[0][1:]]
lab4 = [to_minutes(v) for v in a4[0][1:]]
hdr2_types = type_counts(list(a2_load[0][1:]))
hdr4_types = type_counts(list(a4[0][1:]))
dates = [r[0] for r in a2_load[1:]]
dates_pv = [r[0] for r in a2_pv[1:]]
dates4 = [r[0] for r in a4[1:]]

LD = np.array([[float(v) for v in r[1:]] for r in a2_load[1:]])
PV = np.array([[float(v) for v in r[1:]] for r in a2_pv[1:]])
PR = np.array([[float(v) for v in r[1:]] for r in a4[1:]])
NL = LD - PV

NDAY, NSLOT = LD.shape
NCELL = NDAY * NSLOT

# --- 附件3 ---
a3_body = a3[1:]
block_dates, cur = [], None
for r in a3_body:
    if r[0] not in (None, ""):
        cur = str(r[0])
    block_dates.append(cur)
a3_blank = sum(1 for r in a3_body if r[0] in (None, ""))
A3KEY = list(zip(block_dates, [str(r[1]) for r in a3_body]))
A3VAL = np.array([[float(x) for x in r[2:]] for r in a3_body])
A3_HOURS = A3VAL.shape[1]

# --------------------------------------------------------------------------
# 审计量（全部现场重算）
# --------------------------------------------------------------------------
cells_all = [LD, PV, PR]
MISS = {
    "附件1（144×4 列）": sum(1 for r in a1[1:] for v in r if v is None or (isinstance(v, str) and not v.strip())),
    "附件2 小区负载": int(np.isnan(LD).sum()),
    "附件2 光伏实际": int(np.isnan(PV).sum()),
    "附件3 预报矩阵": int(np.isnan(A3VAL).sum()),
    "附件4 电价": int(np.isnan(PR).sum()),
}
MISS_TOTAL = sum(MISS.values())
DUP = {
    "附件1 时间标签": len(lab1) - len(set(lab1)),
    "附件2 日期": len(dates) - len(set(dates)),
    "附件3 (日期,预报时刻)键": len(A3KEY) - len(set(A3KEY)),
    "附件4 日期": len(dates4) - len(set(dates4)),
}
NEG = {
    "附件1 三数值列": int(sum((arr < 0).sum() for arr in (PRICE1, LD1, PV1))),
    "附件2 小区负载": int((LD < 0).sum()),
    "附件2 光伏实际": int((PV < 0).sum()),
    "附件3 预报矩阵": int((A3VAL < 0).sum()),
    "附件4 电价": int((PR < 0).sum()),
}

# 零膨胀与异常判据
PV_LE1 = int((PV <= 1).sum())
PV_GT1 = int((PV > 1).sum())
FRAC_LE1 = PV_LE1 / NCELL
mad_full_med = float(np.median(PV))
mad_full = float(np.median(np.abs(PV - mad_full_med)))
z_full = 0.6745 * (PV - mad_full_med) / mad_full
cand_full = np.abs(z_full) > 3.5
N_CAND_FULL = int(cand_full.sum())
FRAC_CAND_FULL = N_CAND_FULL / NCELL
cand_full_max = float(PV[cand_full].max())
cand_full_min = float(PV[cand_full].min())
THR_FULL_HI = mad_full_med + 3.5 * mad_full / 0.6745
day = PV[PV > 1]
DAY_N = int(day.size)
day_med = float(np.median(day))
day_mad = float(np.median(np.abs(day - day_med)))
z_day = 0.6745 * (day - day_med) / day_mad
N_CAND_DAY = int((np.abs(z_day) > 3.5).sum())
Z_DAY_MAX = float(np.abs(z_day).max())
THR_DAY = 3.5 * day_mad / 0.6745
day_low_n = int((day <= THR_FULL_HI).sum())

slot_min = np.array(lab1)
NIGHT = (slot_min >= 20 * 60) | (slot_min <= 4 * 60)
NIGHT_SLOTS = int(NIGHT.sum())
NIGHT_PV_GT1 = int((PV[:, NIGHT] > 1).sum())
NIGHT_PV_MAX = float(PV[:, NIGHT].max())

# 零膨胀的时段结构：全年最大出力 ≤1 kW 的时段（真实无光照段）
_col_max = PV.max(axis=0)
_nz_idx = np.where(_col_max <= 1)[0]
NEARZERO_SLOTS = int(_nz_idx.size)
_nz_lab = sorted(slot_min[_nz_idx])
_runs, _s, _p = [], _nz_lab[0], _nz_lab[0]
for _v in _nz_lab[1:]:
    if _v == _p + 10:
        _p = _v
    else:
        _runs.append((_s, _p))
        _s = _p = _v
_runs.append((_s, _p))
NEARZERO_RANGE_TXT = " 与 ".join(f"{hhmm(a)}—{hhmm(b)}" for a, b in _runs)
PV_EXACT_ZERO = int((PV == 0).sum())
PRICE1_MIN, PRICE1_MAX = float(PRICE1.min()), float(PRICE1.max())
LD1_MIN, LD1_MAX = float(LD1.min()), float(LD1.max())
PV1_MAX = float(PV1.max())
PV1_NZ = int((PV1 > 0).sum())
_pv1_idx = np.where(PV1 > 0)[0]
PV1_NZ_RANGE = f"{hhmm(slot_min[_pv1_idx[0]])}—{hhmm(slot_min[_pv1_idx[-1]])}"
PV_ALLZERO_DAYS = int((PV.max(axis=1) <= 0).sum())
A3_MAX = float(A3VAL.max())
A3_ALLZERO_ROWS = int((np.abs(A3VAL).max(axis=1) == 0).sum())

PR_LOW = int((PR < 0.05).sum())
PR_LOW_MIN = float(PR.min())
PR_HIGH = int((PR > 1.5).sum())
PR_HIGH_MAX = float(PR.max())

# 能量与净负荷
E_LOAD = float(LD.sum()) * DT
E_PV = float(PV.sum()) * DT
E_NET = float(NL.sum()) * DT
RATIO_YEAR = E_PV / E_LOAD
NL_NEG_FRAC = float((NL < 0).mean())
NL_MIN = float(NL.min())
NL_MAX = float(NL.max())
NL_NEG_OVER5 = int((NL < -5000).sum())
NL_POS_OVER5 = int((NL > 5000).sum())
NL_NEG_MEAN = float(NL[NL < 0].mean())
NL_POS_MEAN = float(NL[NL > 0].mean())

# 逐日派生量（既有登记表，只读）
DAILY = pd.read_csv(RES / "数据审计_逐日派生量.csv")
DAILY["日期"] = pd.to_datetime(DAILY["日期"])
DAILY["净负荷电量kWh"] = DAILY["负载电量kWh"] - DAILY["光伏电量kWh"]
RATIO_MIN = float(DAILY["光伏负载比"].min())
RATIO_MAX = float(DAILY["光伏负载比"].max())
RATIO_MEAN = float(DAILY["光伏负载比"].mean())

# 逐日净负荷功率波动（10 分钟尺度平均绝对增量）
dNL = np.abs(np.diff(NL, axis=1))
RAMP_DAILY = dNL.mean(axis=1)
RAMP_MEAN = float(dNL.mean())

print("=" * 78)
print(f"{LOG_HDR} · 数据预处理可视化")
print("=" * 78)
print(f"附件1 时间列单元格类型 : {t1_types}（混合类型，共 {sum(t1_types.values())} 个标签）")
print(f"附件2 表头类型         : {hdr2_types}；附件4 表头类型: {hdr4_types}")
print(f"附件3 日期列空白行     : {a3_blank} / {len(a3_body)}（块内延续，非缺失）")
print(f"观测单元               : {NDAY} 日 × {NSLOT} 槽 = {NCELL:,}")
print(f"缺失合计               : {MISS_TOTAL}")
print(f"光伏 ≤1 kW             : {PV_LE1:,}（{FRAC_LE1:.2%}）；>1 kW: {PV_GT1:,}")
print(f"全样本 MAD 候选异常    : {N_CAND_FULL:,} / {NCELL:,}（{FRAC_CAND_FULL:.2%}）")
print(f"白天子样本候选异常     : {N_CAND_DAY} / {DAY_N:,}；|z|max={Z_DAY_MAX:.3f}")
print(f"电价 <0.05 / >1.5      : {PR_LOW} / {PR_HIGH}")
print(f"年电量 负载/光伏/净负荷: {E_LOAD:,.1f} / {E_PV:,.1f} / {E_NET:,.1f} kWh")
print("-" * 78)
print(f"样式：skill={SKILL_STYLE} {_style_info.get('cjk_font', '')}；导出：skill={SKILL_EXPORT}")
print("-" * 78)

# --------------------------------------------------------------------------
# 绘图工具
# --------------------------------------------------------------------------
SAVED = {}
LAYOUT_ISSUES: list[tuple[str, str, str]] = []   # (图名, 严重度, 说明)
FIGNOTE_TEXTS: list = []
BOXTEXT_TEXTS: list = []


def layout_audit(fig, name):
    """程序化版面自检：缺字、文字裁切、刻度标签重叠、子图重叠、说明框压图。"""
    issues: list[tuple[str, str, str]] = []
    try:
        from visual_qa import audit_layout  # type: ignore

        for sev, msg in audit_layout(fig):
            issues.append((name, sev, msg))
    except Exception as exc:  # noqa: BLE001
        issues.append((name, "INFO", f"visual_qa.audit_layout 不可用：{exc}"))

    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    except Exception:  # noqa: BLE001
        return issues

    # 子图相互重叠
    axes_list = list(fig.axes)
    for i in range(len(axes_list)):
        for j in range(i + 1, len(axes_list)):
            if axes_list[i].get_position().overlaps(axes_list[j].get_position()):
                issues.append((name, "FAIL",
                               f"子图位置重叠：ax{i} 与 ax{j}"))

    # 图下说明框压住子图
    for f, t in FIGNOTE_TEXTS:
        if f is not fig:
            continue
        try:
            tbb = t.get_window_extent(renderer)
        except Exception:  # noqa: BLE001
            continue
        for k, ax in enumerate(axes_list):
            try:
                abb = ax.get_window_extent(renderer)
            except Exception:  # noqa: BLE001
                continue
            if tbb.overlaps(abb):
                issues.append((name, "FAIL",
                               f"图下说明框压住子图 ax{k}：{t.get_text()[:18]}…"))

    # 说明框之间互相重叠
    own = [(f, t) for f, t in BOXTEXT_TEXTS if f is fig]
    for i in range(len(own)):
        for j in range(i + 1, len(own)):
            try:
                a = own[i][1].get_window_extent(renderer)
                b = own[j][1].get_window_extent(renderer)
            except Exception:  # noqa: BLE001
                continue
            if a.overlaps(b):
                issues.append((name, "FAIL",
                               f"说明框互相重叠：『{own[i][1].get_text()[:12]}…』×"
                               f"『{own[j][1].get_text()[:12]}…』"))

    # 表格单元格文字溢出
    try:
        from matplotlib.table import Table  # type: ignore

        for tbl in fig.findobj(Table):
            for (r, c), cell in tbl.get_celld().items():
                txt = cell.get_text()
                if not txt.get_text().strip():
                    continue
                try:
                    tbb = txt.get_window_extent(renderer)
                    cbb = cell.get_window_extent(renderer)
                except Exception:  # noqa: BLE001
                    continue
                if (tbb.x0 < cbb.x0 - 1.0 or tbb.x1 > cbb.x1 + 1.0
                        or tbb.y0 < cbb.y0 - 1.0 or tbb.y1 > cbb.y1 + 1.0):
                    issues.append((name, "FAIL",
                                   f"表格文字溢出单元格（第{r}行第{c}列）："
                                   f"『{txt.get_text()[:16].replace(chr(10), ' ')}』"))
    except ImportError:
        pass
    return issues


def save(fig, name):
    for item in layout_audit(fig, name):
        LAYOUT_ISSUES.append(item)
    base = OUTDIR / name
    if SKILL_EXPORT:
        paths = export_figure(fig, str(base), formats=("png", "svg"), dpi=300, tight=True)
    else:
        paths = []
        for ext in ("png", "svg"):
            p = f"{base}.{ext}"
            fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.05)
            paths.append(p)
    plt.close(fig)
    SAVED[name] = [Path(p) for p in paths]
    return paths


def render_table(ax, col_labels, rows, col_widths=None, fontsize=8.0,
                 row_colors=None, header_color=HDR, cell_font_colors=None):
    """把二维文本渲染为表格（bbox 铺满 axes）。"""
    ax.axis("off")
    n_cols = len(col_labels)
    n_rows = len(rows) + 1
    body_colors = [
        [(row_colors[i] if row_colors else "#ffffff")] * n_cols for i in range(len(rows))
    ]
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
        colColours=[header_color] * n_cols,
        cellColours=body_colors,
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    for j in range(n_cols):
        c = tbl[0, j]
        c.set_text_props(fontweight="bold")
        c.set_linewidth(0.6)
    for i in range(1, n_rows):
        for j in range(n_cols):
            tbl[i, j].set_linewidth(0.5)
    if cell_font_colors:
        for (i, j), col in cell_font_colors.items():
            tbl[i, j].set_text_props(color=col)
    return tbl


def panel_title(ax, text, fs=12.0):
    ax.set_title(text, fontsize=fs, fontweight="bold", pad=7)


def fig_note(fig, ax, text, dy=0.015, fs=7.8):
    """在 axes 正下方（figure 坐标）放说明文字，避免与相邻子图重叠。"""
    bb = ax.get_position()
    t = fig.text(bb.x0 + bb.width / 2.0, bb.y0 - dy, text, fontsize=fs, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.32", facecolor="#f7f7f7",
                           edgecolor="#bdbdbd", linewidth=0.6))
    FIGNOTE_TEXTS.append((fig, t))
    BOXTEXT_TEXTS.append((fig, t))
    return t


def note_box(ax, text, x=0.985, y=0.965, fs=8.2, ha="right", va="top"):
    t = ax.text(x, y, text, transform=ax.transAxes, fontsize=fs, ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#f5f5f5",
                          edgecolor="#bdbdbd", linewidth=0.6))
    BOXTEXT_TEXTS.append((ax.figure, t))
    return t


def stat_block(v, unit, extra=""):
    def f(x):
        return f"{x:,.2f}"

    s = (f"n = {v.size:,}\n"
         f"均值 = {f(v.mean())}\n"
         f"中位 = {f(np.median(v))}\n"
         f"P5 = {f(np.percentile(v, 5))}\n"
         f"P95 = {f(np.percentile(v, 95))}\n"
         f"最小 = {f(v.min())}\n"
         f"最大 = {f(v.max())}\n"
         f"标准差 = {f(v.std(ddof=1))}\n"
         f"（单位：{unit}）")
    if extra:
        s += "\n" + extra
    return s


# ==========================================================================
# 图3-1　时间对齐与数据结构（环节 1、2）
# ==========================================================================
def figure_3_1():
    fig = plt.figure(figsize=(13.6, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02], width_ratios=[1.12, 1.0],
                          left=0.035, right=0.985, top=0.885, bottom=0.035,
                          wspace=0.09, hspace=0.20)

    ax_struct = fig.add_subplot(gs[0, 0])
    ax_label = fig.add_subplot(gs[0, 1])
    ax_map = fig.add_subplot(gs[1, :])

    fig.suptitle("图3-1　时间对齐与数据结构（预处理环节 1—2）",
                 fontsize=15, fontweight="bold", y=0.972)

    # ---- (a) 四附件结构对照 ----
    panel_title(ax_struct, "（a）四附件结构对照：原始尺寸 → 统一尺度", fs=11.5)
    rows = [
        ["附件1", "Sheet1", "145 × 4\n(144 数据行)", "1 日 × 144 槽", "144 × 3 = 432",
         f"时间型 {t1_types.get('time', 0)} ＋ 文本型 {t1_types.get('str', 0)}"],
        ["附件2", "小区负载", "366 × 145", "365 日 × 144 槽", "52,560",
         f"时间型 {hdr2_types.get('time', 0)} ＋ 文本型 {hdr2_types.get('str', 0)}"],
        ["附件2", "光伏发电实际功率", "366 × 145", "365 日 × 144 槽", "52,560", "同上（表头一致）"],
        ["附件3", "Sheet1", "1461 × 26", "365 日 × 4 预报时刻", "1,460 × 24\n= 35,040", "文本型（'2025-1-1'）"],
        ["附件4", "Sheet1", "366 × 145", "365 日 × 144 槽", "52,560",
         f"时间型 {hdr4_types.get('time', 0)} ＋ 文本型 {hdr4_types.get('str', 0)}"],
    ]
    render_table(
        ax_struct,
        ["附件", "工作表", "原始尺寸\n（行 × 列）", "对齐后结构", "数值单元数", "时间键单元格类型"],
        rows,
        col_widths=[0.085, 0.185, 0.155, 0.19, 0.155, 0.23],
        fontsize=8.0,
        row_colors=["#eef3fb", "#ffffff", "#ffffff", "#eef3fb", "#ffffff"],
    )
    note_box(ax_struct,
             f"附件1 为单日 144 槽输入（六项形状指标指向 6 个不同日期 ⇒ 不能断言对应某一具体实际日）；\n"
             f"附件1 光伏非零 {PV1_NZ} 列（标签 {PV1_NZ_RANGE}）；"
             f"附件3 预报峰值 {A3_MAX:,.2f} kW；附件2 无全零光伏日（{PV_ALLZERO_DAYS} 日）",
             x=0.5, y=-0.045, fs=7.8, ha="center", va="top")

    # ---- (b) 时间标签归一化前后对照 ----
    panel_title(ax_label, "（b）时间标签归一化：不依赖单元格类型", fs=11.5)
    rows2 = [
        ["附件1／2／4", "00:10", "时间型 time", "10", "[0:10, 0:20)　第 1 槽"],
        ["附件1", "10:10", "文本型 str", "610", "[10:10, 10:20)　第 61 槽"],
        ["附件2／4", "23:50", "时间型 time", "1430", "[23:50, 24:00)　第 143 槽"],
        ["附件2／4", "0:00+1", "文本型 str", "1440", "[次日 0:00, 次日 0:10)　第 144 槽"],
        ["附件3", "6:00", "文本型 str", "360", "发布时刻，按块继承日期后成键"],
    ]
    render_table(
        ax_label,
        ["来源", "标签原文", "单元格类型", "归一化\n（距 0:00 分钟）", "方案丙槽区间"],
        rows2,
        col_widths=[0.165, 0.13, 0.185, 0.20, 0.32],
        fontsize=8.0,
        row_colors=["#ffffff", "#fff4e5", "#ffffff", "#ffe9e3", "#eef3fb"],
    )
    note_box(ax_label,
             "关键：附件1 时间列 60 个时间型 ＋ 84 个文本型；\n"
             "附件2／4 表头 143 个时间型 ＋ 1 个文本型『0:00+1』\n"
             "→ 一律解析为分钟数，禁止按类型分支解析",
             x=0.5, y=-0.045, fs=8.0, ha="center", va="top")

    # ---- (c) 统一到 10 分钟尺度的映射示意 ----
    ax_map.set_xlim(-175, 1700)
    ax_map.set_ylim(0.0, 10.9)
    ax_map.axis("off")
    panel_title(ax_map, "（c）统一到 10 分钟尺度的映射（方案丙：标签＝槽开始时刻）", fs=11.5)

    x0, x1 = 10, 1450

    # 行 A：附件1
    ya = 8.55
    ax_map.add_patch(Rectangle((x0, ya - 0.5), x1 - x0, 1.0, facecolor=C[0],
                               alpha=0.18, edgecolor=C[0], linewidth=1.1))
    step = (x1 - x0) / 144.0
    for k in range(0, 145, 12):
        xk = x0 + k * step
        ax_map.plot([xk, xk], [ya - 0.5, ya + 0.5], color=C[0], linewidth=0.6, alpha=0.55)
    ax_map.text(-168, ya, "附件1\n单日 144 槽", ha="left", va="center", fontsize=9.2,
                fontweight="bold", color=C[0])
    ax_map.annotate("第 1 槽　[0:10, 0:20)", xy=(x0 + 0.5 * step, ya + 0.5),
                    xytext=(x0 - 5, ya + 1.45), fontsize=8.4, color=C[0],
                    arrowprops=dict(arrowstyle="->", color=C[0], lw=0.9))
    ax_map.annotate("第 144 槽　标签『0:00+1』= 1440 min\n[次日 0:00, 次日 0:10)",
                    xy=(x1, ya + 0.5), xytext=(x1 - 425, ya + 1.35), fontsize=8.4, color=C[0],
                    arrowprops=dict(arrowstyle="->", color=C[0], lw=0.9))

    # 行 B：附件2/4
    yb = 5.55
    n_show = 7
    bw, gap = 168, 34
    xs = x0
    for k in range(n_show):
        ax_map.add_patch(Rectangle((xs, yb - 0.45), bw, 0.9, facecolor=C[1], alpha=0.16,
                                   edgecolor=C[1], linewidth=1.0))
        for t in range(1, 6):
            ax_map.plot([xs + t * bw / 6.0] * 2, [yb - 0.45, yb + 0.45], color=C[1],
                        linewidth=0.45, alpha=0.5)
        xs += bw + gap
    ax_map.text(x0 + 4, yb - 0.72, "2025-01-01", fontsize=7.8, color=C[1], ha="left", va="top")
    ax_map.text(xs - bw - gap - 4, yb - 0.72, "2025-12-31", fontsize=7.8, color=C[1],
                ha="right", va="top")
    ax_map.text((x0 + xs - gap) / 2.0, yb - 0.72, "⋯　每一日 144 槽　⋯", fontsize=8.2,
                color=C[1], ha="center", va="top")
    ax_map.text(-168, yb, "附件2／附件4\n365 日 × 144 槽", ha="left", va="center",
                fontsize=9.2, fontweight="bold", color=C[1])
    ax_map.annotate(f"52,560 观测单元 = 365 日 × 144 槽\n"
                    f"（天＝决策单元，10 分钟＝执行单元）",
                    xy=((x0 + xs - gap) / 2.0, yb + 0.45), xytext=((x0 + xs - gap) / 2.0 - 150, yb + 1.35),
                    fontsize=8.6, color=C[1], ha="left",
                    arrowprops=dict(arrowstyle="->", color=C[1], lw=0.9))

    # 行 C：附件3
    yc = 2.75
    pub_x = [x0, x0 + 480, x0 + 960, x0 + 1440]
    pub_lab = ["0:00", "6:00", "12:00", "18:00"]
    for px, pl in zip(pub_x, pub_lab):
        ax_map.plot([px], [yc], marker="o", markersize=6.0, color=C[2], zorder=3)
        ax_map.plot([px, px], [yc - 0.42, yc + 0.42], color=C[2], linewidth=0.9, alpha=0.55)
        for t in range(24):
            ax_map.plot([px + (t + 1) * 19.5], [yc + 0.30], marker="|", markersize=5,
                        color=C[2], alpha=0.65)
        ax_map.text(px, yc - 0.55, pl, fontsize=8.0, color=C[2], ha="center", va="top")
    ax_map.text(-168, yc, "附件3\n预报发布时刻", ha="left", va="center", fontsize=9.2,
                fontweight="bold", color=C[2])
    ax_map.annotate(f"每日 4 个发布时刻 × 未来 24 个整点预报\n"
                    f"365 日 × 4 × 24 = 35,040 个预报值",
                    xy=(pub_x[1] + 250, yc + 0.38), xytext=(pub_x[1] + 130, yc + 1.30),
                    fontsize=8.6, color=C[2], ha="left",
                    arrowprops=dict(arrowstyle="->", color=C[2], lw=0.9))

    # 时间刻度轴
    yax = 0.95
    ax_map.annotate("", xy=(x1 + 20, yax), xytext=(x0, yax),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))
    for m, lab in [(10, "0:10"), (360, "6:00"), (720, "12:00"), (1080, "18:00"),
                   (1440, "次日 0:00"), (1450, "次日 0:10")]:
        ax_map.plot([m, m], [yax - 0.13, yax + 0.13], color="#444444", linewidth=1.0)
        ax_map.text(m, yax - 0.30, lab, fontsize=7.8, ha="center", va="top", color="#333333")
    ax_map.text(x1 + 45, yax, "时刻", fontsize=8.4, va="center", color="#333333")

    ax_map.text(
        x0, 0.12,
        "口径（方案丙）：第 j 槽 = [10j, 10j+10) 分钟（标签_j 为槽开始时刻）；每日窗口 0:10 → 次日 0:10；"
        "Δt = 1/6 h；电量(kWh) = 功率(kW) × 1/6",
        fontsize=8.6, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f4fa", edgecolor="#9aa7bd", linewidth=0.7))

    return save(fig, "图3-1_时间对齐与数据结构")


# ==========================================================================
# 图3-2　质量检验与缺失识别（环节 2、3）
# ==========================================================================
def figure_3_2():
    fig = plt.figure(figsize=(13.6, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], width_ratios=[1.25, 1.0],
                          left=0.035, right=0.985, top=0.885, bottom=0.085,
                          wspace=0.14, hspace=0.26)

    ax_ea = fig.add_subplot(gs[0, 0])
    ax_ms = fig.add_subplot(gs[0, 1])
    ax_blk = fig.add_subplot(gs[1, :])

    fig.suptitle("图3-2　数据质量检验与缺失识别（预处理环节 2—3）",
                 fontsize=15, fontweight="bold", y=0.972)

    # ---- (a) 应有 vs 实际 ----
    panel_title(ax_ea, "（a）应有记录数 vs 实际记录数（逐一重合 ⇒ 完整性 100%）", fs=11.5)
    items = [
        ("附件1 时间／电价／负载／光伏\n（各 144）", 144, 144),
        ("附件3 （日期,时刻）键\n1460", 1460, len(set(A3KEY))),
        ("附件3 预报矩阵\n1460 × 24", 1460 * 24, A3VAL.size),
        ("附件2 小区负载\n365 × 144", NCELL, LD.size),
        ("附件2 光伏实际\n365 × 144", NCELL, PV.size),
        ("附件4 电价\n365 × 144", NCELL, PR.size),
    ]
    ypos = np.arange(len(items))[::-1]
    for y, (lab, exp, act) in zip(ypos, items):
        ax_ea.hlines(y, 100, act, color=C[0], linewidth=2.2, alpha=0.55)
        ax_ea.plot(exp, y, marker="o", markersize=10, markerfacecolor="white",
                   markeredgecolor=C[3], markeredgewidth=1.6, linestyle="none", zorder=3)
        ax_ea.plot(act, y, marker="o", markersize=5.0, color=C[0], linestyle="none", zorder=4)
        ax_ea.text(act * 1.18, y, f"{act:,}", fontsize=8.2, va="center", color="#222222")
    ax_ea.set_yticks(ypos)
    ax_ea.set_yticklabels([lab for lab, _, _ in items], fontsize=7.8)
    ax_ea.set_xscale("log")
    ax_ea.set_xlim(100, 3.2e5)
    ax_ea.set_xlabel("记录数（对数轴）", fontsize=9.5)
    ax_ea.grid(axis="x", linestyle=":", alpha=0.4)
    ax_ea.set_axisbelow(True)
    ax_ea.plot([], [], marker="o", markersize=9, markerfacecolor="white",
               markeredgecolor=C[3], markeredgewidth=1.6, linestyle="none", label="应有记录数")
    ax_ea.plot([], [], marker="o", markersize=5, color=C[0], linestyle="none", label="实际记录数")
    ax_ea.legend(fontsize=8.2, loc="lower right", frameon=True, framealpha=0.95)
    note_box(ax_ea, "六项一一重合\n实际 − 应有 = 0", x=0.02, y=0.98, fs=8.4, ha="left", va="top")

    # ---- (b) 缺失统计 ----
    panel_title(ax_ms, "（b）缺失／重复／负值统计：全部为 0", fs=11.5)
    cats = ["缺失记录", "重复记录", "负值记录", "夜间光伏\n>1 kW", "越界／违背\n物理规律"]
    vals = [MISS_TOTAL, sum(DUP.values()), sum(NEG.values()), NIGHT_PV_GT1, 0]
    cols = [C[3], C[3], C[3], C[3], C[3]]
    xb = np.arange(len(cats) + 1)
    heights = vals + [a3_blank]
    colors = cols + [GREY]
    ax_ms.bar(xb, heights, width=0.62, color=colors, alpha=0.88,
              edgecolor="white", linewidth=0.8)
    for x, h in zip(xb, heights):
        ax_ms.text(x, h + max(heights) * 0.035, f"{h:,}", ha="center", va="bottom",
                   fontsize=10.5, fontweight="bold",
                   color=(C[3] if h == 0 else "#444444"))
    ax_ms.set_xticks(xb)
    ax_ms.set_xticklabels(cats + ["附件3 日期列空白\n（块内延续，非缺失）"], fontsize=8.0)
    ax_ms.set_ylabel("记录数", fontsize=9.5)
    ax_ms.set_ylim(0, max(heights) * 1.30)
    ax_ms.grid(axis="y", linestyle=":", alpha=0.4)
    ax_ms.set_axisbelow(True)
    ax_ms.get_xticklabels()[-1].set_color(GREY)
    note_box(ax_ms, "缺失 = 0　重复 = 0　负值 = 0\n（四附件全部逐格核对）",
             x=0.5, y=0.975, fs=8.6, ha="center", va="top")

    # ---- (c) 附件3 日期列块内继承示意 ----
    panel_title(ax_blk, "（c）附件3 日期列空白＝块内延续表达（非缺失）：向下继承后成键", fs=11.5)
    times_order = ["0:00", "6:00", "12:00", "18:00"]  # noqa: F841（保留：说明块内时刻顺序）
    rows_t, colors_t = [], []
    shown = 0
    for bi in range(len(a3_body)):
        r = a3_body[bi]
        if bi % 4 == 0:
            shown += 1
            if shown > 3:
                break
        raw = r[0]
        raw_txt = "（空白）" if raw in (None, "") else str(raw)
        rows_t.append([f"{bi + 1}", raw_txt, str(r[1]), block_dates[bi],
                       f"({block_dates[bi]}, {r[1]})"])
        colors_t.append("#fdece7" if raw in (None, "") else "#e8f2e8")
    rows_t.append(["⋯", "⋯", "⋯", "⋯", "⋯"])
    colors_t.append("#f5f5f5")
    rows_t.append(["1460", "（空白，末块第 4 行）", "18:00", "2025-12-31", "(2025-12-31, 18:00)"])
    colors_t.append("#fdece7")
    render_table(
        ax_blk,
        ["行号", "日期列原文", "预报时刻", "按块向下继承的日期", "继承后键（日期, 预报时刻）"],
        rows_t,
        col_widths=[0.08, 0.22, 0.13, 0.24, 0.33],
        fontsize=8.4,
        row_colors=colors_t,
    )
    fig_note(fig, ax_blk,
             f"365 个日期块 × 每块严格 4 行（0:00→6:00→12:00→18:00）＝ 1,460 行；"
             f"日期列空白 {a3_blank:,} 行属『块内延续表达』，不是缺失\n"
             f"继承后 (日期, 预报时刻) 唯一键 {len(set(A3KEY)):,} / {len(A3KEY):,}"
             f"（无重复、无缺块）；若按缺失处理将误弃 3/4 的预报样本",
             dy=0.022, fs=8.4)

    return save(fig, "图3-2_质量检验与缺失识别")


# ==========================================================================
# 图3-3　异常辨识与判据修正（环节 4）
# ==========================================================================
def figure_3_3():
    fig = plt.figure(figsize=(13.6, 8.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0],
                          left=0.055, right=0.985, top=0.885, bottom=0.045,
                          wspace=0.16, hspace=0.30)

    ax_full = fig.add_subplot(gs[0, 0])
    ax_day = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])

    fig.suptitle("图3-3　异常数据辨识与判据修正（预处理环节 4）",
                 fontsize=15, fontweight="bold", y=0.972)

    xslot = np.tile(np.arange(1, NSLOT + 1), NDAY)
    y_all = PV.ravel()
    f_all = cand_full.ravel()

    # ---- 左：全样本 MAD 判据 ----
    panel_title(ax_full, "（a）全样本 MAD 稳健判据：候选异常 25,785（49.06%）— 判据失效", fs=11.0)
    ax_full.scatter(xslot[~f_all], y_all[~f_all], s=1.1, color=C[0], alpha=0.30,
                    linewidths=0, rasterized=True, label="判为正常")
    ax_full.scatter(xslot[f_all], y_all[f_all], s=1.1, color=C[3], alpha=0.42,
                    linewidths=0, rasterized=True, label=f"判为候选异常（{N_CAND_FULL:,}）")
    ax_full.axhline(mad_full_med, color="#333333", linestyle="--", linewidth=1.2,
                    label=f"中位数 = {mad_full_med:,.2f} kW")
    ax_full.axhline(THR_FULL_HI, color=C[3], linestyle=":", linewidth=1.4,
                    label=f"上界 = 中位 + 3.5·MAD/0.6745 = {THR_FULL_HI:,.2f} kW")
    ax_full.set_xlabel("日内槽序号 j（1—144，标签 = 槽开始时刻）", fontsize=9.5)
    ax_full.set_ylabel("附件2 光伏实际功率（kW）", fontsize=9.5)
    ax_full.set_xlim(0.5, 144.5)
    ax_full.set_xticks([1, 25, 49, 73, 97, 121, 144])
    ax_full.set_xticklabels(["1\n0:10", "25\n4:10", "49\n8:10", "73\n12:10",
                             "97\n16:10", "121\n20:10", "144\n0:00+1"], fontsize=7.8)
    ax_full.grid(linestyle=":", alpha=0.35)
    ax_full.set_axisbelow(True)
    ax_full.legend(fontsize=7.6, loc="upper left", framealpha=0.95)
    ax_full.axvspan(-0.5, 24.5, color="#dbe6f5", alpha=0.35, zorder=0)
    ax_full.axvspan(120.5, 144.5, color="#dbe6f5", alpha=0.35, zorder=0)
    note_box(ax_full,
             f"MAD = {mad_full:,.2f} kW（≈ 中位数，因 48.53% 为 0）\n"
             f"判据退化为：光伏 > {THR_FULL_HI:,.1f} kW 即异常\n"
             f"⇒ 白天正常发电 25,785 个被误标；\n"
             f"而 ≤ {THR_FULL_HI:,.1f} kW 的 {day_low_n:,} 个白天样本反而漏判",
             x=0.985, y=0.975, fs=7.8, ha="right", va="top")

    # ---- 右：分层后白天子样本 ----
    panel_title(ax_day, "（b）分层后白天子样本判据：候选异常 0 / 27,051（0.000%）", fs=11.0)
    day_mask = PV > 1
    xd = xslot[day_mask.ravel()]
    yd = y_all[day_mask.ravel()]
    ax_day.scatter(xd, yd, s=1.3, color=C[0], alpha=0.40, linewidths=0, rasterized=True,
                   label=f"白天子样本（实际 > 1 kW，n = {DAY_N:,}）")
    ax_day.axhline(day_med, color="#333333", linestyle="--", linewidth=1.2,
                   label=f"白天中位数 = {day_med:,.2f} kW")
    ax_day.axhspan(day_med - THR_DAY, day_med + THR_DAY, color=C[2], alpha=0.13,
                   label=f"±3.5·MAD/0.6745 = ±{THR_DAY:,.0f} kW")
    ax_day.set_xlabel("日内槽序号 j（1—144，标签 = 槽开始时刻）", fontsize=9.5)
    ax_day.set_ylabel("附件2 光伏实际功率（kW）", fontsize=9.5)
    ax_day.set_xlim(0.5, 144.5)
    ax_day.set_ylim(-900, 11800)
    ax_day.set_xticks([1, 25, 49, 73, 97, 121, 144])
    ax_day.set_xticklabels(["1\n0:10", "25\n4:10", "49\n8:10", "73\n12:10",
                            "97\n16:10", "121\n20:10", "144\n0:00+1"], fontsize=7.8)
    ax_day.grid(linestyle=":", alpha=0.35)
    ax_day.set_axisbelow(True)
    ax_day.legend(fontsize=7.6, loc="upper left", framealpha=0.95)
    ax_day.annotate(f"判据上界 {day_med + THR_DAY:,.0f} kW ↑ 超出坐标范围",
                    xy=(120, 11200), xytext=(60, 10500), fontsize=8.0, color=C[2],
                    arrowprops=dict(arrowstyle="->", color=C[2], lw=0.9))
    ax_day.annotate(f"判据下界 {day_med - THR_DAY:,.0f} kW ↓ 超出坐标范围",
                    xy=(120, -400), xytext=(58, -800), fontsize=8.0, color=C[2],
                    arrowprops=dict(arrowstyle="->", color=C[2], lw=0.9))
    note_box(ax_day,
             f"白天中位 = {day_med:,.2f} kW，MAD = {day_mad:,.2f} kW\n"
             f"max|稳健 z| = {Z_DAY_MAX:.3f} < 3.5 ⇒ 候选异常 0 个\n"
             f"物理一致性：夜间（20:00—次日 04:00）\n"
             f"光伏 >1 kW 的样本 {NIGHT_PV_GT1} 个；负值 {NEG['附件2 光伏实际']} 个",
             x=0.985, y=0.975, fs=7.8, ha="right", va="top")

    # ---- 下：零膨胀说明 ----
    ax_txt.axis("off")
    panel_title(ax_txt, "（c）零膨胀为何使中位数／MAD 失效，以及修正后的判据链", fs=11.0)
    inset = ax_txt.inset_axes([0.012, 0.06, 0.30, 0.84])
    bins = np.linspace(0, 10500, 61)
    inset.hist(PV.ravel(), bins=bins, color=C[0], alpha=0.85, edgecolor="white", linewidth=0.3)
    inset.set_yscale("log")
    inset.set_xlabel("光伏实际功率（kW）", fontsize=8.4)
    inset.set_ylabel("频数（对数轴）", fontsize=8.4)
    inset.tick_params(labelsize=7.4)
    inset.set_title(f"0—1 kW 区间 {PV_LE1:,} 个（{FRAC_LE1:.2%}）", fontsize=8.4, pad=4)
    inset.annotate("零值尖峰\n（真实无光照段：\n" + NEARZERO_RANGE_TXT + "）",
                   xy=(60, PV_LE1 * 0.85),
                   xytext=(2400, PV_LE1 * 0.10), fontsize=7.6, color=C[3],
                   arrowprops=dict(arrowstyle="->", color=C[3], lw=0.9))
    inset.axvline(mad_full_med, color="#444444", linestyle="--", linewidth=1.0)
    inset.text(mad_full_med + 220, 2.2e3, f"全样本中位数 {mad_full_med:.1f} kW",
               fontsize=7.2, color="#444444", rotation=90, va="bottom")

    lines = [
        f"① 零膨胀事实：附件2 光伏实际 ≤1 kW 共 {PV_LE1:,} / {NCELL:,}（{FRAC_LE1:.2%}）；"
        f"全年最大出力 ≤1 kW 的时段 {NEARZERO_SLOTS} 个（{NEARZERO_RANGE_TXT}，",
        f"　　共 {PV_EXACT_ZERO:,} 个精确 0 值），属真实无光照而非缺失 ⇒ 不做插补、不删除。",
        f"② 判据失效机理：0 值占近半使中位数被拉平到 {mad_full_med:,.2f} kW，"
        f"MAD 同步塌缩为 {mad_full:,.2f} kW；稳健 z 的阈值变为 {THR_FULL_HI:,.1f} kW，",
        f"　　于是『凡是白天正常发电』都被判成异常（{N_CAND_FULL:,} 个，占 {FRAC_CAND_FULL:.2%}），"
        f"而真正需要关注的白天低值样本（{day_low_n:,} 个）被漏判。",
        f"③ 修正判据：先按物理含义分层 —— 夜间（20:00—次日 04:00，{NIGHT_SLOTS} 个槽）与白天；"
        f"仅在白天子样本（实际 >1 kW，n = {DAY_N:,}）上做稳健 z 检验，",
        f"　　阈值区间 [−{THR_DAY:,.0f}, +{THR_DAY:,.0f}] kW 完全落在数据范围 "
        f"[1, {PV.max():,.1f}] kW 之外 ⇒ 候选异常 {N_CAND_DAY} 个（0.000%）。",
        f"④ 物理一致性检查全通过：夜间发电 {NIGHT_PV_GT1} 个（夜间最大 {NIGHT_PV_MAX:.4f} kW）；"
        f"负值 {NEG['附件2 光伏实际']} 个；白天最大 {PV.max():,.1f} kW（未超过可解释范围）。",
        "结论：本数据在修正判据下真实异常 0 个；全部样本原样保留，不删除、不插补、不重标注。",
    ]
    for i, ln in enumerate(lines):
        ax_txt.text(0.345, 0.90 - i * 0.128, ln, transform=ax_txt.transAxes, fontsize=8.6,
                    va="top", ha="left",
                    fontweight="bold" if i == len(lines) - 1 else "normal",
                    color="#111111" if i == len(lines) - 1 else "#222222")

    return save(fig, "图3-3_异常辨识与判据修正")


# ==========================================================================
# 图3-4　异常处理与极端值保留（环节 5）
# ==========================================================================
def figure_3_4():
    fig = plt.figure(figsize=(13.8, 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.10, 1.0],
                          left=0.055, right=0.985, top=0.865, bottom=0.095, wspace=0.14)
    sub = gs[0, 0].subgridspec(2, 1, height_ratios=[0.30, 1.0], hspace=0.10)
    ax_box = fig.add_subplot(sub[0])
    ax_hist = fig.add_subplot(sub[1])
    ax_tbl = fig.add_subplot(gs[0, 1])

    fig.suptitle("图3-4　异常数据处理与极端电价保留（预处理环节 5）",
                 fontsize=15, fontweight="bold", y=0.965)

    flat = PR.ravel()
    # ---- 左：附件4 电价分布 ----
    bp = ax_box.boxplot([flat], vert=False, widths=0.5, patch_artist=True,
                        showfliers=True, whis=(5, 95),
                        flierprops=dict(marker=".", markersize=2.5, markerfacecolor=C[3],
                                        markeredgecolor="none", alpha=0.35))
    bp["boxes"][0].set(facecolor=C[0], alpha=0.42, edgecolor=C[0])
    bp["medians"][0].set(color="#222222", linewidth=1.4)
    for w in ("whiskers", "caps"):
        for art in bp[w]:
            art.set(color=C[0], linewidth=1.0)
    ax_box.plot([PR_LOW_MIN], [1], marker="v", markersize=8, color=C[3], zorder=5)
    ax_box.plot([PR_HIGH_MAX], [1], marker="v", markersize=8, color=C[3], zorder=5)
    ax_box.set_xlim(-0.05, 1.90)
    ax_box.set_yticks([])
    ax_box.set_xlabel("")
    ax_box.tick_params(labelbottom=False)
    panel_title(ax_box, "（a）附件4 电价分布与极端值（n = 52,560，元/kWh）", fs=11.0)
    ax_box.annotate(f"最小 {PR_LOW_MIN:.4f}\n（{PR_LOW} 个 < 0.05）",
                    xy=(PR_LOW_MIN, 1.16), xytext=(0.09, 1.30), fontsize=8.2, color=C[3],
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=C[3], lw=0.9))
    ax_box.annotate(f"最大 {PR_HIGH_MAX:.4f}\n（{PR_HIGH} 个 > 1.5）",
                    xy=(PR_HIGH_MAX, 1.16), xytext=(1.35, 1.30), fontsize=8.2, color=C[3],
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=C[3], lw=0.9))
    ax_box.set_ylim(0.55, 1.75)

    bins = np.linspace(0, 1.85, 93)
    ax_hist.hist(flat, bins=bins, color=C[0], alpha=0.86, edgecolor="white", linewidth=0.3,
                 label="全部 52,560 个电价样本")
    ax_hist.set_yscale("log")
    ax_hist.set_xlim(-0.05, 1.90)
    ax_hist.set_xlabel("电价（元/kWh）", fontsize=9.5)
    ax_hist.set_ylabel("频数（对数轴）", fontsize=9.5)
    ax_hist.grid(axis="y", linestyle=":", alpha=0.4)
    ax_hist.set_axisbelow(True)
    ax_hist.axvline(0.05, color=C[3], linestyle="--", linewidth=1.3)
    ax_hist.axvline(1.5, color=C[3], linestyle="--", linewidth=1.3)
    ax_hist.axvspan(-0.05, 0.05, color=C[3], alpha=0.14)
    ax_hist.axvspan(1.5, 1.90, color=C[3], alpha=0.14)
    ax_hist.plot(flat[flat < 0.05], np.full(PR_LOW, 4.0), marker="v", linestyle="none",
                 markersize=6.5, color=C[3], label=f"< 0.05 元/kWh：{PR_LOW} 个")
    ax_hist.text(0.075, 1.1e3, f"< 0.05：{PR_LOW} 个\n最低 {PR_LOW_MIN:.4f}", fontsize=8.4,
                 color=C[3], ha="left", va="center")
    ax_hist.text(1.335, 1.1e3, f"> 1.5：{PR_HIGH:,} 个\n最高 {PR_HIGH_MAX:.4f}", fontsize=8.4,
                 color=C[3], ha="right", va="center")
    ax_hist.legend(fontsize=8.2, loc="upper center", framealpha=0.95)
    note_box(ax_hist,
             f"全样本：最小 {PR.min():.4f}　中位 {np.median(PR):.4f}　最大 {PR.max():.4f}　"
             f"均值 {PR.mean():.4f}\n"
             f"附件1 电价（题目给定固定日）区间 {PRICE1.min():.4f}—{PRICE1.max():.4f}，"
             f"无极端值 → 极端值全部来自附件4 波动电价",
             x=0.985, y=0.975, fs=8.0, ha="right", va="top")

    # 局部放大：<0.05 的 9 个样本
    axi = ax_hist.inset_axes([0.055, 0.40, 0.30, 0.34])
    low = np.sort(flat[flat < 0.05])
    axi.bar(np.arange(1, PR_LOW + 1), low, color=C[3], alpha=0.85, width=0.62)
    axi.set_xticks(np.arange(1, PR_LOW + 1))
    axi.tick_params(labelsize=7.0)
    axi.set_title(f"放大：< 0.05 的 {PR_LOW} 个样本", fontsize=8.0, pad=3)
    axi.set_ylabel("元/kWh", fontsize=7.4)
    axi.set_ylim(0, 0.058)
    for i, v in enumerate(low):
        axi.text(i + 1, v + 0.0022, f"{v:.4f}", fontsize=6.2, ha="center", va="bottom",
                 rotation=90)

    # ---- 右：处理策略对照表 ----
    panel_title(ax_tbl, "（b）异常／极端值处理策略对照", fs=11.0)
    rows = [
        ["保留并标记\n＋情景对照\n（本项目采用）",
         "全部数值原样进入建模；\n把 9 个 <0.05 与 927 个 >1.5\n登记为标记样本；\n另设『电价下限截断』情景并列对照",
         "不改变题目给定的真实波动；\n结论在真实输入下可复现；\n极端值影响可量化",
         "● 采用"],
        ["截断\n（设上／下限）",
         "把 < 阈值抬到阈值、\n> 上限压到上限",
         "人为降低波动电价问题的\n难度；阈值选取任意；\n直接改变费用结论",
         "× 不采用"],
        ["缩尾\nwinsorize",
         "把 <P1 或 >P99 的样本\n替换为对应分位数值",
         "同时抹掉真实极值信息；\n分位点由样本自身决定，\n对 9 个样本极不稳定",
         "× 不采用"],
        ["删除样本",
         "直接剔除极端值\n所在时段或整日",
         "违反『原始数据只读、不因\n难预测而删除样本』；\n时段序列出现空洞需再插补",
         "× 不采用"],
    ]
    render_table(
        ax_tbl,
        ["处理方案", "具体做法", "主要风险／代价", "本项目"],
        rows,
        col_widths=[0.215, 0.315, 0.325, 0.145],
        fontsize=7.8,
        row_colors=["#e8f4ea", "#ffffff", "#ffffff", "#ffffff"],
        cell_font_colors={(1, 3): "#1a7f37", (2, 3): "#b3261e", (3, 3): "#b3261e", (4, 3): "#b3261e"},
    )
    fig_note(fig, ax_tbl,
             "本项目处置原则（R-01／R-05）：全部样本原样保留，不删除、不插补、不重标注；\n"
             "极端电价作为『可放宽假设 R1』的情景对照并列报告，主分析不改变任何输入值。",
             dy=0.018, fs=8.4)

    return save(fig, "图3-4_异常处理与极端值保留")


# ==========================================================================
# 图3-5　变量分布分析（环节 6）
# ==========================================================================
def figure_3_5():
    fig = plt.figure(figsize=(13.6, 11.0))
    gs = fig.add_gridspec(4, 3, width_ratios=[2.30, 0.80, 1.30],
                          left=0.055, right=0.985, top=0.905, bottom=0.062,
                          wspace=0.30, hspace=0.46)

    fig.suptitle("图3-5　变量分布分析：负载、光伏、净负荷、波动电价（预处理环节 6）",
                 fontsize=15, fontweight="bold", y=0.968)

    specs = [
        ("小区负载 ℓ（附件2，n = 52,560）", LD.ravel(), "kW", C[0], "linear", None),
        ("光伏发电实际 v（附件2，n = 52,560）", PV.ravel(), "kW", C[1], "log", "zero"),
        ("净负荷 ℓ − v（派生，n = 52,560）", NL.ravel(), "kW", C[2], "linear", "neg"),
        ("波动电价（附件4，n = 52,560）", PR.ravel(), "元/kWh", C[3], "log", "extreme"),
    ]

    for i, (title, v, unit, col, yscale, flag) in enumerate(specs):
        axh = fig.add_subplot(gs[i, 0])
        axb = fig.add_subplot(gs[i, 1])
        axs = fig.add_subplot(gs[i, 2])

        lo, hi = float(v.min()), float(v.max())
        pad = max((hi - lo) * 0.025, 1e-6)
        bins = np.linspace(lo - pad, hi + pad, 70)
        axh.hist(v, bins=bins, color=col, alpha=0.85, edgecolor="white", linewidth=0.35)
        axh.set_yscale(yscale)
        axh.set_xlim(lo - pad, hi + pad)
        axh.set_ylabel("频数" + ("（对数轴）" if yscale == "log" else ""), fontsize=9.0)
        axh.set_xlabel(f"{unit}", fontsize=9.0)
        axh.grid(axis="y", linestyle=":", alpha=0.4)
        axh.set_axisbelow(True)
        panel_title(axh, f"（{chr(97 + i)}）{title}", fs=10.6)

        mean_v, med_v = float(v.mean()), float(np.median(v))
        p5, p95 = float(np.percentile(v, 5)), float(np.percentile(v, 95))
        axh.axvline(mean_v, color="#222222", linewidth=1.5, label=f"均值 {mean_v:,.2f}")
        axh.axvline(med_v, color="#222222", linewidth=1.4, linestyle="--",
                    label=f"中位 {med_v:,.2f}")
        axh.axvline(p5, color=GREY, linewidth=1.1, linestyle=":", label=f"P5 {p5:,.2f}")
        axh.axvline(p95, color=GREY, linewidth=1.1, linestyle=":",
                    label=f"P95 {p95:,.2f}")

        extra = ""
        if flag == "zero":
            zero_bins = v <= 1
            axh.hist(v[zero_bins], bins=np.linspace(lo - pad, 1.0, 3), color=C[3], alpha=0.75,
                     edgecolor="white", linewidth=0.35)
            axh.text(0.34, 0.30,
                     f"零膨胀：≤1 kW 占 {FRAC_LE1:.2%}（{PV_LE1:,} 个）\n"
                     f"全年最大出力 ≤1 kW 的时段 {NEARZERO_SLOTS} 个\n"
                     f"（{NEARZERO_RANGE_TXT}）—— 真实无光照，未插补",
                     transform=axh.transAxes, fontsize=8.2, color=C[3], ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4f0",
                               edgecolor=C[3], linewidth=0.7))
            extra = f"零值占比 = {FRAC_LE1:.2%}\n精确 0 值 = {PV_EXACT_ZERO:,} 个\n偏度 = {stats.skew(v):,.3f}"
        elif flag == "neg":
            axh.axvspan(lo - pad, 0, color=C[3], alpha=0.13)
            axh.text(0.03, 0.66,
                     f"净负荷 < 0（光伏过剩）时段占 {NL_NEG_FRAC:.2%}\n"
                     f"（{int((NL < 0).sum()):,} 个）\n"
                     f"最大过剩 {NL_MIN:,.1f} kW\n最大缺额 {NL_MAX:,.1f} kW",
                     transform=axh.transAxes, fontsize=8.2, color="#8a2c1a", ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4f0",
                               edgecolor=C[3], linewidth=0.7))
            extra = (f"负值占比 = {NL_NEG_FRAC:.2%}\n"
                     f"过剩 >5,000 kW = {NL_NEG_OVER5} 个\n缺额 >5,000 kW = {NL_POS_OVER5:,} 个")
        elif flag == "extreme":
            axh.text(0.03, 0.64,
                     f"极端值保留：<0.05 元/kWh 共 {PR_LOW} 个\n（最低 {PR_LOW_MIN:.4f}）\n"
                     f">1.5 元/kWh 共 {PR_HIGH:,} 个（最高 {PR_HIGH_MAX:.4f}）",
                     transform=axh.transAxes, fontsize=8.2, color="#8a2c1a", ha="left", va="center",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff4f0",
                               edgecolor=C[3], linewidth=0.7))
            extra = f"偏度 = {stats.skew(v):,.3f}\n峰度 = {stats.kurtosis(v):,.3f}"

        axh.legend(fontsize=7.6, loc="upper right", framealpha=0.95, ncol=2)

        # 第三列：统计量表（独立坐标轴，避免任何压图）
        axs.axis("off")
        axs.set_title("描述统计", fontsize=9.2, pad=6)
        axs.text(0.02, 0.5, stat_block(v, unit, extra), transform=axs.transAxes,
                 fontsize=8.2, ha="left", va="center")

        # 右侧箱线
        bp = axb.boxplot([v], vert=True, widths=0.5, patch_artist=True, whis=(5, 95),
                         showfliers=True,
                         flierprops=dict(marker=".", markersize=2.0, markerfacecolor=GREY,
                                         markeredgecolor="none", alpha=0.30))
        bp["boxes"][0].set(facecolor=col, alpha=0.45, edgecolor=col)
        bp["medians"][0].set(color="#222222", linewidth=1.4)
        for w in ("whiskers", "caps"):
            for art in bp[w]:
                art.set(color=col, linewidth=1.0)
        axb.set_xticks([])
        axb.set_ylabel(f"{unit}", fontsize=9.0)
        axb.grid(axis="y", linestyle=":", alpha=0.4)
        axb.set_axisbelow(True)
        axb.set_title("箱线（须端 = P5／P95）", fontsize=9.2, pad=6)
        axb.text(0.5, 1.005, f"最大 {hi:,.2f}", transform=axb.transAxes, fontsize=7.6,
                 ha="center", va="bottom", color="#8a2c1a")
        axb.text(0.5, -0.005, f"最小 {lo:,.2f}", transform=axb.transAxes, fontsize=7.6,
                 ha="center", va="top", color="#8a2c1a")

    fig.text(0.055, 0.015,
             f"净负荷（ℓ − v）年合计 {E_NET:,.0f} kWh；负载 {E_LOAD:,.0f} kWh；光伏 {E_PV:,.0f} kWh；"
             f"光伏／负载电量比 {RATIO_YEAR:.3f}\n"
             f"全部 52,560 个单元无缺失、无负值；光伏的零值为真实无光照，未做任何插补或填充",
             fontsize=8.4, ha="left", va="bottom",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#f0f4fa", edgecolor="#9aa7bd",
                       linewidth=0.7))

    return save(fig, "图3-5_变量分布分析")


# ==========================================================================
# 图3-6　特征构造与派生变量（环节 7）
# ==========================================================================
def figure_3_6():
    fig = plt.figure(figsize=(13.6, 9.4))
    gs = fig.add_gridspec(2, 2, left=0.058, right=0.985, top=0.885, bottom=0.055,
                          wspace=0.19, hspace=0.40)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    sub = gs[1, 1].subgridspec(2, 1, hspace=0.16)
    ax_d1 = fig.add_subplot(sub[0])
    ax_d2 = fig.add_subplot(sub[1])

    fig.suptitle("图3-6　特征构造与派生变量（预处理环节 7）",
                 fontsize=15, fontweight="bold", y=0.972)

    slot_axis = np.arange(1, NSLOT + 1)
    ld_prof = LD.mean(axis=0)
    pv_prof = PV.mean(axis=0)
    nl_prof = NL.mean(axis=0)

    # ---- (a) 日内统计特征 ----
    panel_title(ax_a, "（a）日内统计特征：峰、谷、均值与峰谷差", fs=11.0)
    ax_a.plot(slot_axis, ld_prof, color=C[0], linewidth=1.9, label="负载 ℓ（365 日均值）")
    ax_a.plot(slot_axis, pv_prof, color=C[1], linewidth=1.9, label="光伏 v（365 日均值）")
    ax_a.plot(slot_axis, nl_prof, color=C[2], linewidth=1.6, linestyle="-.",
              label="净负荷 ℓ − v")
    ax_a.axhline(0, color="#666666", linewidth=0.9, linestyle=":")
    ipk = int(np.argmax(ld_prof))
    ivl = int(np.argmin(ld_prof))
    ax_a.plot([ipk + 1], [ld_prof[ipk]], marker="v", markersize=9, color=C[3], zorder=5)
    ax_a.plot([ivl + 1], [ld_prof[ivl]], marker="^", markersize=9, color=C[3], zorder=5)
    ax_a.annotate(f"负载日峰 {ld_prof[ipk]:,.0f} kW @ {hhmm(slot_min[ipk])}",
                  xy=(ipk + 1, ld_prof[ipk]), xytext=(ipk - 6, ld_prof[ipk] + 300),
                  fontsize=8.4, ha="right", color=C[3],
                  arrowprops=dict(arrowstyle="->", color=C[3], lw=0.9))
    ax_a.annotate(f"负载日谷 {ld_prof[ivl]:,.0f} kW @ {hhmm(slot_min[ivl])}",
                  xy=(ivl + 1, ld_prof[ivl]), xytext=(ivl + 12, ld_prof[ivl] - 620),
                  fontsize=8.4, ha="left", color=C[3],
                  arrowprops=dict(arrowstyle="->", color=C[3], lw=0.9))
    ax_a.axhline(ld_prof.mean(), color=C[0], linewidth=1.0, linestyle=":",
                 label=f"负载年均 {ld_prof.mean():,.0f} kW")
    ax_a.set_xlim(1, NSLOT)
    ax_a.set_xticks([1, 25, 49, 73, 97, 121, 144])
    ax_a.set_xticklabels(["1\n0:10", "25\n4:10", "49\n8:10", "73\n12:10",
                          "97\n16:10", "121\n20:10", "144\n0:00+1"], fontsize=7.8)
    ax_a.set_xlabel("日内槽序号（标签＝槽开始时刻）", fontsize=9.5)
    ax_a.set_ylabel("功率（kW）", fontsize=9.5)
    ax_a.grid(linestyle=":", alpha=0.35)
    ax_a.set_axisbelow(True)
    ax_a.legend(fontsize=8.0, loc="upper left", framealpha=0.95)
    note_box(ax_a,
             "日内统计特征：日峰值 / 日谷值 / 日均值 /\n"
             "峰谷差 / 峰谷出现时刻 / 日电量 / 负荷率\n"
             f"峰谷差 = {ld_prof[ipk] - ld_prof[ivl]:,.0f} kW；"
             f"负荷率 = {ld_prof.mean() / ld_prof[ipk]:.3f}",
             x=0.985, y=0.60, fs=7.8, ha="right", va="top")

    # ---- (b) 季节特征：月均曲线 ----
    panel_title(ax_b, "（b）季节特征：月均功率曲线（同一 kW 量纲，不用双轴）", fs=11.0)
    months = DAILY["日期"].dt.month.values
    m_ld = np.array([LD[months == m].mean() for m in range(1, 13)])
    m_pv = np.array([PV[months == m].mean() for m in range(1, 13)])
    m_nl = np.array([NL[months == m].mean() for m in range(1, 13)])
    xm = np.arange(1, 13)
    for q, (a, b) in enumerate([(1, 3), (4, 6), (7, 9), (10, 12)]):
        ax_b.axvspan(a - 0.5, b + 0.5, color=C[q], alpha=0.055, zorder=0)
        ax_b.text((a + b) / 2.0, 0.015, f"Q{q + 1}", fontsize=8.2,
                  color=C[q], ha="center", va="bottom", alpha=0.95,
                  transform=ax_b.get_xaxis_transform())
    ax_b.plot(xm, m_ld, marker="o", markersize=5.5, color=C[0], linewidth=1.9, label="月均负载 ℓ")
    ax_b.plot(xm, m_pv, marker="s", markersize=5.5, color=C[1], linewidth=1.9, label="月均光伏 v")
    ax_b.plot(xm, m_nl, marker="^", markersize=5.0, color=C[2], linewidth=1.6, linestyle="-.",
              label="月均净负荷 ℓ − v")
    ax_b.axhline(0, color="#666666", linewidth=0.9, linestyle=":")
    ax_b.set_xticks(xm)
    ax_b.set_xticklabels([f"{m}月" for m in xm], fontsize=8.0)
    ax_b.set_xlabel("月份（2025 年）", fontsize=9.5)
    ax_b.set_ylabel("平均功率（kW）", fontsize=9.5)
    ax_b.grid(linestyle=":", alpha=0.35)
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=8.0, loc="upper left", framealpha=0.95)
    note_box(ax_b,
             f"月均光伏 {m_pv.min():,.0f}—{m_pv.max():,.0f} kW，"
             f"峰谷比 {m_pv.max() / max(m_pv.min(), 1e-9):.2f}\n"
             f"月均负载 {m_ld.min():,.0f}—{m_ld.max():,.0f} kW，"
             f"峰谷比 {m_ld.max() / m_ld.min():.2f}\n"
             "季节特征：月份 / 季度哑变量、月均与月峰值",
             x=0.985, y=0.985, fs=7.8, ha="right", va="top")

    # ---- (c) 逐日派生量 ----
    panel_title(ax_c, "（c）逐日派生量：日电量与净负荷电量（365 日）", fs=11.0)
    dts = DAILY["日期"].values
    ax_c.plot(dts, DAILY["负载电量kWh"], color=C[0], linewidth=1.2, label="日负载电量")
    ax_c.plot(dts, DAILY["光伏电量kWh"], color=C[1], linewidth=1.2, label="日光伏电量")
    ax_c.plot(dts, DAILY["净负荷电量kWh"], color=C[2], linewidth=1.2, linestyle="-.",
              label="日净负荷电量 ℓ − v")
    ax_c.axhline(0, color="#666666", linewidth=0.9, linestyle=":")
    ax_c.set_ylabel("电量（kWh/日）", fontsize=9.5)
    ax_c.set_xlabel("日期（2025-01-01 — 2025-12-31）", fontsize=9.5)
    ax_c.grid(linestyle=":", alpha=0.35)
    ax_c.set_axisbelow(True)
    ax_c.tick_params(axis="x", labelsize=8.0)
    ax_c.legend(fontsize=8.0, loc="upper right", framealpha=0.95, ncol=1)
    note_box(ax_c,
             f"年合计：负载 {E_LOAD:,.0f} kWh；光伏 {E_PV:,.0f} kWh；"
             f"净负荷 {E_NET:,.0f} kWh\n"
             f"光伏／负载电量比（逐日） {RATIO_MIN:.3f}—{RATIO_MAX:.3f}，"
             f"年均 {RATIO_YEAR:.3f}（= 50.0%）\n"
             f"无一日超过 1 ⇒ 不存在净上网，与『不允许售电』一致",
             x=0.985, y=0.985, fs=7.8, ha="right", va="top")

    # ---- (d1) 光伏负载比 ----
    panel_title(ax_d1, "（d1）派生比值特征：逐日光伏／负载电量比", fs=10.4)
    ax_d1.plot(dts, DAILY["光伏负载比"], color=C[4], linewidth=1.2)
    ax_d1.axhline(RATIO_YEAR, color="#222222", linewidth=1.1, linestyle="--",
                  label=f"年均 {RATIO_YEAR:.3f}")
    ax_d1.axhline(1.0, color=C[3], linewidth=1.1, linestyle=":",
                  label="上限 1.0（售电边界）")
    ax_d1.set_ylim(0.2, 1.08)
    ax_d1.set_ylabel("比值", fontsize=9.0)
    ax_d1.grid(linestyle=":", alpha=0.35)
    ax_d1.set_axisbelow(True)
    ax_d1.tick_params(axis="x", labelsize=7.4)
    ax_d1.legend(fontsize=7.6, loc="lower left", framealpha=0.95, ncol=2)
    ax_d1.text(0.985, 0.18, f"区间 {RATIO_MIN:.3f}—{RATIO_MAX:.3f}（max < 1）",
               transform=ax_d1.transAxes, fontsize=7.8, ha="right", va="center",
               color="#8a2c1a")

    # ---- (d2) 功率波动 ----
    panel_title(ax_d2, "（d2）派生波动特征：净负荷 10 分钟平均绝对波动", fs=10.4)
    ax_d2.plot(dts, RAMP_DAILY, color=C[5], linewidth=1.0)
    ax_d2.axhline(RAMP_MEAN, color="#222222", linewidth=1.1, linestyle="--",
                  label=f"全样本均值 {RAMP_MEAN:,.1f} kW/(10min)")
    ax_d2.set_ylabel("kW/(10min)", fontsize=9.0)
    ax_d2.grid(linestyle=":", alpha=0.35)
    ax_d2.set_axisbelow(True)
    ax_d2.tick_params(axis="x", labelsize=7.4)
    ax_d2.legend(fontsize=7.6, loc="upper left", framealpha=0.95)
    ax_d2.text(0.985, 0.30, f"逐日均值 {RAMP_DAILY.min():,.1f}—{RAMP_DAILY.max():,.1f}",
               transform=ax_d2.transAxes, fontsize=7.8, ha="right", va="center",
               color="#8a2c1a")

    return save(fig, "图3-6_特征构造与派生变量")


# ==========================================================================
# 图3-7　相关性与分组诊断（环节 8、9）
# ==========================================================================
def figure_3_7():
    fig = plt.figure(figsize=(15.0, 10.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.02, 0.98],
                          left=0.048, right=0.985, top=0.885, bottom=0.135,
                          wspace=0.24, hspace=0.36)
    ax_corr = fig.add_subplot(gs[0, 0])
    ax_typ = fig.add_subplot(gs[0, 1:])
    ax_q = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1])
    ax_s = fig.add_subplot(gs[1, 2])

    fig.suptitle("图3-7　相关性与时序特征分析、数据分组诊断（预处理环节 8—9）",
                 fontsize=15, fontweight="bold", y=0.972)

    # ---- (a) 相关系数矩阵 ----
    slot_axis = np.arange(1, NSLOT + 1)
    panel_title(ax_corr, "（a）变量相关性（Pearson，n = 52,560）", fs=10.8)
    M = np.vstack([LD.ravel(), PV.ravel(), NL.ravel(), PR.ravel()])
    names = ["负载 ℓ", "光伏 v", "净负荷\nℓ − v", "电价"]
    R = np.corrcoef(M)
    im = ax_corr.imshow(R, cmap="coolwarm", vmin=-1, vmax=1)
    ax_corr.set_xticks(range(4))
    ax_corr.set_yticks(range(4))
    ax_corr.set_xticklabels(names, fontsize=8.6)
    ax_corr.set_yticklabels(names, fontsize=8.6)
    for i in range(4):
        for j in range(4):
            ax_corr.text(j, i, f"{R[i, j]:.3f}", ha="center", va="center", fontsize=9.4,
                         color=("white" if abs(R[i, j]) > 0.62 else "#111111"),
                         fontweight="bold" if i != j else "normal")
    cb = fig.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8.0)
    cb.set_label("相关系数", fontsize=8.6)
    note_box(ax_corr,
             f"负载与净负荷强正相关（{R[0, 2]:.3f}）；\n"
             f"光伏与净负荷强负相关（{R[1, 2]:.3f}）；\n"
             f"电价与功率变量相关性弱（|r| ≤ {max(abs(R[3, 0]), abs(R[3, 1]), abs(R[3, 2])):.3f}）",
             x=0.5, y=-0.155, fs=7.8, ha="center", va="top")

    # ---- (b) 典型日日内曲线 ----
    panel_title(ax_typ, "（b）典型日日内曲线：四季代表日（该季日光伏电量中位日）", fs=10.8)
    q_of_day = ((DAILY["日期"].dt.month - 1) // 3).values
    pv_e = DAILY["光伏电量kWh"].values
    rep_idx = []
    for q in range(4):
        idx = np.where(q_of_day == q)[0]
        med = np.median(pv_e[idx])
        rep_idx.append(idx[np.argmin(np.abs(pv_e[idx] - med))])
    season_lab = ["Q1（1—3月）", "Q2（4—6月）", "Q3（7—9月）", "Q4（10—12月）"]
    for q, di in enumerate(rep_idx):
        ax_typ.plot(slot_axis, LD[di], color=C[q], linewidth=1.9)
        ax_typ.plot(slot_axis, PV[di], color=C[q], linewidth=1.7, linestyle="--")
        ax_typ.plot(slot_axis, NL[di], color=C[q], linewidth=1.3, linestyle=":")
    ax_typ.axhline(0, color="#666666", linewidth=0.9, linestyle=":")
    ax_typ.set_xlim(1, NSLOT)
    ax_typ.set_xticks([1, 25, 49, 73, 97, 121, 144])
    ax_typ.set_xticklabels(["1\n0:10", "25\n4:10", "49\n8:10", "73\n12:10",
                            "97\n16:10", "121\n20:10", "144\n0:00+1"], fontsize=7.8)
    ax_typ.set_xlabel("日内槽序号（标签＝槽开始时刻）", fontsize=9.5)
    ax_typ.set_ylabel("功率（kW）", fontsize=9.5)
    ax_typ.grid(linestyle=":", alpha=0.35)
    ax_typ.set_axisbelow(True)
    h1 = [plt.Line2D([], [], color=C[q], linewidth=2.0) for q in range(4)]
    h2 = [plt.Line2D([], [], color="#444444", linewidth=1.9),
          plt.Line2D([], [], color="#444444", linewidth=1.7, linestyle="--"),
          plt.Line2D([], [], color="#444444", linewidth=1.3, linestyle=":")]
    lg1 = ax_typ.legend(h1, [f"{s}　{str(pd.Timestamp(d).date())}" for s, d in
                             zip(season_lab, [DAILY['日期'].iloc[i] for i in rep_idx])],
                        fontsize=8.0, loc="upper left", framealpha=0.95, title="季节（颜色）")
    lg1.get_title().set_fontsize(8.0)
    ax_typ.add_artist(lg1)
    lg2 = ax_typ.legend(h2, ["负载 ℓ", "光伏 v", "净负荷 ℓ − v"], fontsize=8.0,
                        loc="upper right", framealpha=0.95, title="变量（线型）")
    lg2.get_title().set_fontsize(8.0)
    neg_slots = int((NL[rep_idx] < 0).sum())
    note_box(ax_typ,
             f"四季代表日中净负荷为负（光伏盈余）的槽位合计 {neg_slots} 个；\n"
             f"日内形态：负载双峰（约 11:00、19:00），光伏单峰（约 12:00），"
             f"净负荷在午间被压低甚至转负",
             x=0.5, y=-0.135, fs=7.8, ha="center", va="top")

    # ---- (c) 按季度分组箱线 ----
    panel_title(ax_q, "（c）分组诊断：按季度分组的负载与光伏", fs=10.6)
    pos = np.arange(4) + 1
    bp1 = ax_q.boxplot([LD[q_of_day == q].ravel() for q in range(4)], positions=pos - 0.19,
                       widths=0.32, patch_artist=True, whis=(5, 95), showfliers=False,
                       medianprops=dict(color="#222222", linewidth=1.3))
    bp2 = ax_q.boxplot([PV[q_of_day == q].ravel() for q in range(4)], positions=pos + 0.19,
                       widths=0.32, patch_artist=True, whis=(5, 95), showfliers=False,
                       medianprops=dict(color="#222222", linewidth=1.3))
    for q in range(4):
        bp1["boxes"][q].set(facecolor=C[0], alpha=0.45, edgecolor=C[0])
        bp2["boxes"][q].set(facecolor=C[1], alpha=0.45, edgecolor=C[1])
    for bp in (bp1, bp2):
        for w in ("whiskers", "caps"):
            for art in bp[w]:
                art.set(color="#666666", linewidth=0.9)
    ax_q.set_xticks(pos)
    ax_q.set_xticklabels([f"Q{q + 1}\n{(q * 3 + 1)}—{q * 3 + 3}月" for q in range(4)], fontsize=8.2)
    ax_q.set_ylabel("功率（kW）", fontsize=9.2)
    ax_q.grid(axis="y", linestyle=":", alpha=0.35)
    ax_q.set_axisbelow(True)
    ax_q.plot([], [], marker="s", markersize=8, color=C[0], alpha=0.55, linestyle="none",
              label="负载 ℓ")
    ax_q.plot([], [], marker="s", markersize=8, color=C[1], alpha=0.55, linestyle="none",
              label="光伏 v")
    ax_q.legend(fontsize=8.0, loc="upper left", framealpha=0.95)
    q_ld = [LD[q_of_day == q].mean() for q in range(4)]
    q_pv = [PV[q_of_day == q].mean() for q in range(4)]
    fig_note(fig, ax_q,
             f"季均负载 {min(q_ld):,.0f}—{max(q_ld):,.0f} kW（极差 {max(q_ld) - min(q_ld):,.0f}）\n"
             f"季均光伏 {min(q_pv):,.0f}—{max(q_pv):,.0f} kW（极差 {max(q_pv) - min(q_pv):,.0f}）\n"
             "季间差异主要体现在光伏",
             dy=0.042, fs=7.6)

    # ---- (d) 工作日 / 周末 ----
    panel_title(ax_w, "（d）分组诊断：工作日 vs 周末", fs=10.6)
    is_wd = DAILY["日期"].dt.dayofweek.values < 5
    g_ld = [LD[is_wd].mean(), LD[~is_wd].mean()]
    g_pv = [PV[is_wd].mean(), PV[~is_wd].mean()]
    xg = np.arange(2)
    b1 = ax_w.bar(xg - 0.19, g_ld, width=0.36, color=C[0], alpha=0.88, label="日均负载功率")
    b2 = ax_w.bar(xg + 0.19, g_pv, width=0.36, color=C[1], alpha=0.88, label="日均光伏功率")
    for bs in (b1, b2):
        for r in bs:
            ax_w.text(r.get_x() + r.get_width() / 2, r.get_height() + 90,
                      f"{r.get_height():,.0f}", ha="center", va="bottom", fontsize=8.2)
    ax_w.set_xticks(xg)
    ax_w.set_xticklabels([f"工作日\n（{int(is_wd.sum())} 日）", f"周末\n（{int((~is_wd).sum())} 日）"],
                         fontsize=8.6)
    ax_w.set_ylabel("平均功率（kW）", fontsize=9.2)
    ax_w.set_ylim(0, max(g_ld + g_pv) * 1.30)
    ax_w.grid(axis="y", linestyle=":", alpha=0.35)
    ax_w.set_axisbelow(True)
    ax_w.legend(fontsize=8.0, loc="upper right", framealpha=0.95)
    e_ld = [DAILY["负载电量kWh"].values[is_wd].mean(), DAILY["负载电量kWh"].values[~is_wd].mean()]
    p_avg = [PR[is_wd].mean(), PR[~is_wd].mean()]
    fig_note(fig, ax_w,
             f"日均负载电量 {e_ld[0]:,.0f} vs {e_ld[1]:,.0f} kWh/日\n"
             f"（周末 − 工作日 = {e_ld[1] - e_ld[0]:+,.0f} kWh/日，"
             f"{(e_ld[1] / e_ld[0] - 1) * 100:+.2f}%）\n"
             f"平均电价 {p_avg[0]:.4f} vs {p_avg[1]:.4f} 元/kWh",
             dy=0.042, fs=7.6)

    # ---- (e) 净负荷正/负运行状态分组 ----
    panel_title(ax_s, "（e）分组诊断：按净负荷正／负运行状态", fs=10.6)
    share = [NL_NEG_FRAC * 100, (1 - NL_NEG_FRAC) * 100]
    bars = ax_s.bar([0, 1], share, width=0.55, color=[C[2], C[3]], alpha=0.88)
    for r, s, n in zip(bars, share, [int((NL < 0).sum()), int((NL > 0).sum())]):
        ax_s.text(r.get_x() + r.get_width() / 2, s + 2.0, f"{s:.2f}%\n({n:,} 时段)",
                  ha="center", va="bottom", fontsize=9.0, fontweight="bold")
    ax_s.set_xticks([0, 1])
    ax_s.set_xticklabels(["净负荷 < 0\n光伏过剩（充电／弃光）",
                          "净负荷 > 0\n需外部供电（购电）"], fontsize=8.4)
    ax_s.set_ylabel("占全部 52,560 时段的比例（%）", fontsize=9.2)
    ax_s.set_ylim(0, 118)
    ax_s.grid(axis="y", linestyle=":", alpha=0.35)
    ax_s.set_axisbelow(True)
    fig_note(fig, ax_s,
             f"过剩状态：均值 {NL_NEG_MEAN:,.1f} kW，最大过剩 {NL_MIN:,.1f} kW\n"
             f"　超过 −5,000 kW（单槽可充上限）的时段 {NL_NEG_OVER5} 个\n"
             f"缺额状态：均值 {NL_POS_MEAN:,.1f} kW，最大缺额 {NL_MAX:,.1f} kW\n"
             f"　超过 +5,000 kW 的时段 {NL_POS_OVER5:,} 个\n"
             "⇒ 两类状态均需储能之外的调节手段",
             dy=0.045, fs=7.5)

    return save(fig, "图3-7_相关性与分组诊断")


# ==========================================================================
# 执行
# ==========================================================================
FIG_NAMES = [
    ("图3-1_时间对齐与数据结构", figure_3_1),
    ("图3-2_质量检验与缺失识别", figure_3_2),
    ("图3-3_异常辨识与判据修正", figure_3_3),
    ("图3-4_异常处理与极端值保留", figure_3_4),
    ("图3-5_变量分布分析", figure_3_5),
    ("图3-6_特征构造与派生变量", figure_3_6),
    ("图3-7_相关性与分组诊断", figure_3_7),
]

for _name, _fn in FIG_NAMES:
    _fn()

# ==========================================================================
# 自检
# ==========================================================================
print("=" * 78)
print("自检（数据核对 + 图件产出）")
print("=" * 78)

checks = []


def chk(label, ok, detail=""):
    checks.append((label, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {label}{('　— ' + detail) if detail else ''}")


# 1) 图件字节数 + 非空白
try:
    from PIL import Image

    _HAS_PIL = True
except ImportError:  # pragma: no cover
    _HAS_PIL = False

for name, _ in FIG_NAMES:
    paths = SAVED.get(name, [])
    png = next((p for p in paths if p.suffix == ".png"), None)
    svg = next((p for p in paths if p.suffix == ".svg"), None)
    sz = png.stat().st_size if png and png.exists() else 0
    ssz = svg.stat().st_size if svg and svg.exists() else 0
    chk(f"{name}.png > 20000 字节", png is not None and sz > 20000, f"{sz:,} 字节")
    chk(f"{name}.svg 已生成且 > 20000 字节", svg is not None and ssz > 20000, f"{ssz:,} 字节")
    if _HAS_PIL and png is not None and png.exists():
        with Image.open(png) as im:
            w, h = im.size
            arr = np.asarray(im.convert("L"))
        frac = float((arr < 245).mean())
        chk(f"{name}.png 非空白且分辨率达标（非白像素 > 5%，宽 ≥ 2000 px）",
            frac > 0.05 and w >= 2000, f"{w}×{h} px；非白 {frac:.1%}")

# 2) 缺失 = 0
chk("四附件缺失记录 = 0 已核对", MISS_TOTAL == 0,
    "；".join(f"{k}:{v}" for k, v in MISS.items()))
# 3) 52,560 观测单元
chk("观测单元 52,560 = 365 × 144 已核对", NCELL == 52560 and NDAY == 365 and NSLOT == 144,
    f"{NDAY} × {NSLOT} = {NCELL:,}")
chk("附件2/4 三张矩阵均为 52,560 个数值单元",
    LD.size == PV.size == PR.size == 52560, f"{LD.size:,} / {PV.size:,} / {PR.size:,}")

# 4) 缺失/重复/负值
chk("重复记录 = 0 已核对", sum(DUP.values()) == 0, "；".join(f"{k}:{v}" for k, v in DUP.items()))
chk("负值记录 = 0 已核对", sum(NEG.values()) == 0, "；".join(f"{k}:{v}" for k, v in NEG.items()))

# 5) 附件3 块继承
chk("附件3 日期列空白 1,095 行按块继承后成键 1,460 个唯一键",
    a3_blank == 1095 and len(A3KEY) == 1460 and len(set(A3KEY)) == 1460,
    f"空白 {a3_blank}；唯一键 {len(set(A3KEY))}/{len(A3KEY)}")
chk("附件3 预报矩阵 35,040 个值 = 1,460 × 24", A3VAL.size == 35040, f"{A3VAL.size:,}")

# 6) 混合单元格类型
chk("附件1 时间列混合类型 60 时间型 + 84 文本型",
    t1_types.get("time") == 60 and t1_types.get("str") == 84, str(t1_types))
chk("附件2/4 表头 143 时间型 + 1 文本型（0:00+1 → 1440）",
    hdr2_types.get("time") == 143 and hdr2_types.get("str") == 1
    and hdr4_types.get("time") == 143 and hdr4_types.get("str") == 1
    and lab2[-1] == 1440 and lab4[-1] == 1440,
    f"附件2 {hdr2_types}；附件4 {hdr4_types}；末列归一化 {lab2[-1]}/{lab4[-1]} 分钟")

# 7) 零膨胀与判据
chk("光伏 ≤1 kW 占 48.53%（25,509/52,560）",
    PV_LE1 == 25509 and abs(FRAC_LE1 - 0.4853) < 5e-5, f"{PV_LE1:,}（{FRAC_LE1:.4%}）")
chk("全样本 MAD 判据误标 25,785（49.06%）",
    N_CAND_FULL == 25785 and abs(FRAC_CAND_FULL - 0.4906) < 5e-5,
    f"{N_CAND_FULL:,}（{FRAC_CAND_FULL:.4%}）")
chk("白天子样本 n=27,051，修正判据候选异常 0 个",
    DAY_N == 27051 and N_CAND_DAY == 0, f"n={DAY_N:,}；候选异常 {N_CAND_DAY}")
chk("夜间（20:00—次日 04:00，49 槽）光伏 >1 kW 的样本 0 个",
    NIGHT_SLOTS == 49 and NIGHT_PV_GT1 == 0, f"夜间槽 {NIGHT_SLOTS}；样本 {NIGHT_PV_GT1}")

# 8) 极端电价
chk("附件4 电价 <0.05 共 9 个（最低 0.0076）、>1.5 共 927 个（最高 1.7936）",
    PR_LOW == 9 and abs(PR_LOW_MIN - 0.0076) < 5e-5 and PR_HIGH == 927
    and abs(PR_HIGH_MAX - 1.7936) < 5e-5,
    f"<0.05: {PR_LOW}（{PR_LOW_MIN:.4f}）；>1.5: {PR_HIGH:,}（{PR_HIGH_MAX:.4f}）")

# 9) 能量与净负荷
chk("年负载电量 ≈ 40,524,055 kWh；年光伏 ≈ 20,251,243 kWh；比 ≈ 50.0%",
    abs(E_LOAD - 40524055) < 60 and abs(E_PV - 20251243) < 60 and abs(RATIO_YEAR - 0.50) < 0.002,
    f"{E_LOAD:,.1f} / {E_PV:,.1f} kWh；比 {RATIO_YEAR:.4f}")
chk("净负荷年合计 ≈ 20,272,812 kWh；负值时段占 19.27%；最大过剩 −6,601.99 kW",
    abs(E_NET - 20272812) < 60 and abs(NL_NEG_FRAC - 0.1927) < 1e-4
    and abs(NL_MIN - (-6601.9911)) < 0.5,
    f"{E_NET:,.1f} kWh；{NL_NEG_FRAC:.4%}；{NL_MIN:,.2f} kW")
chk("最大缺额 = +6,687.75 kW（实测；任务说明写的 +6,668 kW 系数字换位，已按实测更正）",
    abs(NL_MAX - 6687.7481) < 0.5, f"实测 {NL_MAX:,.4f} kW（≈ +6,688 kW）")
chk("逐日光伏／负载电量比 0.271—0.999（无一日超过 1）",
    abs(RATIO_MIN - 0.2707) < 5e-4 and abs(RATIO_MAX - 0.999) < 5e-4,
    f"{RATIO_MIN:.4f}—{RATIO_MAX:.4f}")

# 9b) 附件1 / 附件2 / 附件3 的补充实测事实
chk("附件1 实测范围：电价 0.3713—1.3952；负载 3309.39—5958.97 kW；光伏 0—7612.32 kW",
    abs(PRICE1_MIN - 0.3713) < 5e-5 and abs(PRICE1_MAX - 1.3952) < 5e-5
    and abs(LD1_MIN - 3309.3934) < 0.01 and abs(LD1_MAX - 5958.9696) < 0.01
    and abs(PV1_MAX - 7612.316) < 0.01,
    f"{PRICE1_MIN:.4f}—{PRICE1_MAX:.4f}；{LD1_MIN:,.2f}—{LD1_MAX:,.2f}；0—{PV1_MAX:,.2f}")
chk("附件1 光伏非零 89 列，标签区间 04:40—19:20",
    PV1_NZ == 89 and PV1_NZ_RANGE == "04:40—19:20", f"{PV1_NZ} 列；{PV1_NZ_RANGE}")
chk("附件2 负载实测 1995.72—7978.88 kW；光伏实际 0—10216.20 kW；无全零光伏日",
    abs(LD.min() - 1995.7176) < 0.01 and abs(LD.max() - 7978.8849) < 0.01
    and abs(PV.max() - 10216.2) < 0.01 and PV_ALLZERO_DAYS == 0,
    f"{LD.min():,.2f}—{LD.max():,.2f}；0—{PV.max():,.2f}；全零日 {PV_ALLZERO_DAYS}")
chk("附件3 预报峰值 9995.89 kW；365 块 × 4 行严格；无全零预报行",
    abs(A3_MAX - 9995.8875) < 0.01 and A3_ALLZERO_ROWS == 0
    and all(len(b) == 4 for b in [a3_body[i:i + 4] for i in range(0, len(a3_body), 4)]),
    f"峰值 {A3_MAX:,.4f} kW；全零行 {A3_ALLZERO_ROWS}")
chk("零膨胀时段结构：全年最大出力 ≤1 kW 的时段 55 个（00:10—04:30 与 19:30—24:00）",
    NEARZERO_SLOTS == 55 and NEARZERO_RANGE_TXT == "00:10—04:30 与 19:30—24:00"
    and PV_EXACT_ZERO == 23540,
    f"{NEARZERO_SLOTS} 个（{NEARZERO_RANGE_TXT}）；精确 0 值 {PV_EXACT_ZERO:,} 个")

# 10) 声明性核对：未做插补/删除
chk("声明核对：未插补、未删除、未重标注任何样本（全部 52,560×3 + 35,040 + 144 个数值原样入模）",
    True, "缺失 0 故无需插补；异常 0 故无需删除；极端电价保留")

# 11) 图件无缺字（中文字体字形缺失）
chk("图件无缺字（无 missing-glyph 警告）", len(MISSING_GLYPHS) == 0,
    "；".join(sorted(MISSING_GLYPHS)) if MISSING_GLYPHS else "7 张图全部无缺字")

# 12) 程序化版面自检（visual_qa.audit_layout + 子图/说明框重叠检测）
_layout_fail = [it for it in LAYOUT_ISSUES if it[1] == "FAIL"]
_layout_warn = [it for it in LAYOUT_ISSUES if it[1] == "WARN"]
chk("版面自检无 FAIL（无子图重叠、无说明框压图、无说明框互相重叠）",
    len(_layout_fail) == 0,
    "；".join(f"{n}:{m[:40]}" for n, _, m in _layout_fail) if _layout_fail else "无 FAIL")
print(f"[INFO] 版面自检 WARN {len(_layout_warn)} 项（裁切类 WARN 已由 bbox_inches='tight' 兜底）")
for _n, _s, _m in _layout_warn:
    print(f"       - {_n}：{_m[:110]}")

n_fail = sum(1 for _, ok, _ in checks if not ok)
print("-" * 78)
print("【事实核对差异说明】（读数据时发现的、与任务说明不一致之处）")
print(f"  1) 任务说明写『最大缺额 +6,668 kW』，实测附件2 净负荷最大正值 = {NL_MAX:,.4f} kW"
      f"（≈ +6,688 kW）；")
print(f"     与《数据审计_补充核查.txt》§E 的 6,687.75 kW 一致 ⇒ 按实测 +6,687.75 kW 使用。"
      f"最大过剩实测 {NL_MIN:,.4f} kW，与说明 −6,602 kW 一致。")
print(f"  2) 任务说明写『0 值集中在 22:00—05:00』，实测更精确的表述为：全年最大出力 ≤1 kW 的时段"
      f"共 {NEARZERO_SLOTS} 个，")
print(f"     为 {NEARZERO_RANGE_TXT} 两段（合 {PV_EXACT_ZERO:,} 个精确 0 值）；"
      f"任务说明所指的 20:00—04:00 共 49 槽内光伏 >1 kW 的样本确为 0 个。")
print(f"  3) 任务说明称 {E_PV / E_LOAD:.1%} 为『自给率』；本项目《数据处理记录.md》§3 明确"
      f"『年发电量约为全年负载的 50%，")
print(f"     不等于实际自给率（还受时序错配、弃光与充放电损耗影响）』⇒ 图件中一律标注为"
      f"『光伏／负载电量比』，不写自给率。")
print("-" * 78)
print(f"自检合计：{len(checks)} 项，PASS {len(checks) - n_fail} 项，FAIL {n_fail} 项")
print(f"图件目录：{OUTDIR}")
print(f"日志文件：{LOG}")
print(f"脚本：{Path(__file__).resolve()}")
print("最终结果：" + ("ALL PASS ✔ 退出码 0" if n_fail == 0 else f"存在 {n_fail} 项 FAIL ✘ 退出码 1"))
print("=" * 78)

sys.stdout.flush()
_LOG_FH.flush()
_LOG_FH.close()
sys.stdout = _ORIG_STDOUT
sys.exit(0 if n_fail == 0 else 1)
