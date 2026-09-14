"""把 fig5/fig6/fig7 三张「四季典型日」折线图合并成一张三联图。

与直接拼贴三张 PNG 不同，这里是用原始数据重画：
  - 三块面板共享 x 轴，只在最底部标一次「时间段（10 min）」；
  - 四季图例只出一次，放在图顶；
  - 输出矢量 PDF/SVG，可直接进 LaTeX。

出两个版本，差别只在纵轴：
  fig5-7_四季典型日合并图          各面板纵轴独立（忠实于原来的 fig5/6/7）
  fig5-7_四季典型日合并图_统一纵轴   光伏与负载共用 (0, 9000) 纵轴 —— 堆叠在一起时
                                  读者必然横向比高度，纵轴不一致会误导对量级的判断

两版都做了两处必要的补充（原 fig5/6/7 没有）：
  - 净负荷面板画一条 0 参考线：典型日里 3/4 天存在光伏过剩时段，
    不标零线就无法判断正负；
  - 纵轴单位写明 kW。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from plot_q23_contour import load_data          # noqa: E402 复用同一套数据读取

OUT = ROOT / 'figures' / '问题23_新增7图'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
})

# 四季典型日与配色（沿用原 fig5/6/7 的 Okabe-Ito 色盲友好配色）
SEASONS = [('2025-03-20', '3月20日'), ('2025-06-21', '6月21日'),
           ('2025-09-23', '9月23日'), ('2025-12-21', '12月21日')]
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7']
PANEL_TITLES = ['（a）四季典型日光伏出力曲线',
                '（b）四季典型日负载曲线',
                '（c）四季典型日净负荷曲线']


def typical_day_rows(d1):
    """返回四个典型日在 365 天矩阵中的行号。"""
    rows = []
    for ds, lab in SEASONS:
        hit = np.flatnonzero(d1.values == np.datetime64(ds))
        assert hit.size == 1, '典型日 %s 在数据中未唯一命中' % ds
        rows.append(int(hit[0]))
    return rows


def build(shared_y, filename, title):
    d1, load, pv, times = load_data()
    net = load - pv
    rows = typical_day_rows(d1)
    x = np.arange(1, 145)

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.8), sharex=True)
    fig.subplots_adjust(top=0.885, bottom=0.072, left=0.105,
                        right=0.975, hspace=0.20)

    for ax, arr, ptitle in zip(axes, [pv, load, net], PANEL_TITLES):
        for r, (_, lab), c in zip(rows, SEASONS, COLORS):
            ax.plot(x, arr[r], lw=1.8, color=c, label=lab)
        ax.set_ylabel('功率 / kW')
        ax.set_title(ptitle, loc='left', pad=6)
        ax.grid(alpha=.25)
        ax.set_xlim(1, 144)
        ax.set_xticks([24, 48, 72, 96, 120, 144])

    if shared_y:
        # 光伏与负载同一纵轴，两者的量级差才读得出来
        for ax in axes[:2]:
            ax.set_ylim(0, 9000)

    # 净负荷必须标零线才能读出正负
    axes[2].axhline(0, color='#555555', lw=0.9, ls=(0, (4, 3)), zorder=0)

    axes[2].set_xlabel('时间段（10 min）')

    handles = [plt.Line2D([0], [0], color=c, lw=2.0, label=lab)
               for (_, lab), c in zip(SEASONS, COLORS)]
    fig.legend(handles=handles, ncol=4, frameon=False,
               loc='upper center', bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(title, y=0.982)

    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / (filename + '.' + ext), bbox_inches='tight')
    plt.close(fig)

    spans = {k: (float(v.min()), float(v.max()))
             for k, v in (('光伏', pv), ('负载', load), ('净负荷', net))}
    print('->', filename)
    for k, (a, b) in spans.items():
        print('     %-4s 四季典型日范围 %9.1f ~ %9.1f kW' % (k, a, b))
    if shared_y:
        print('     纵轴: 光伏/负载 统一 (0, 9000)；净负荷独立')


def main():
    build(False, 'fig5-7_四季典型日合并图',
          '四季典型日光伏、负载与净负荷曲线')
    build(True, 'fig5-7_四季典型日合并图_统一纵轴',
          '四季典型日光伏、负载与净负荷曲线（光伏与负载共用纵轴）')
    print('输出目录:', OUT)


if __name__ == '__main__':
    main()
