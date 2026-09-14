from pathlib import Path
import openpyxl, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
R=Path(__file__).resolve().parent
src=R/'项目'/'00_题目与原始数据'/'Q1_Q2_Q3_Q4_原始附件'/'附件2.xlsx'
w=openpyxl.load_workbook(src,data_only=True,read_only=True)
def read(ws):
    rows=list(ws.values); return np.asarray([r[1:145] for r in rows[1:]],float)
load,pv=read(w.worksheets[0]),read(w.worksheets[1]); net=load-pv
out=R/'figures'/'问题23_新增7图'/'3D遮挡方案对比'/'G_3D加2D双联_修正版'; out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Noto Sans SC','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':9})
def draw(a,name,title,cmap):
    x=np.arange(1,366); y=np.arange(1,145); X,Y=np.meshgrid(x,y,indexing='ij')
    fig=plt.figure(figsize=(12,5.7)); gs=fig.add_gridspec(1,2,left=.045,right=.91,bottom=.13,top=.84,wspace=.22)
    ax=fig.add_subplot(gs[0],projection='3d'); ax.plot_surface(X,Y,a,cmap=cmap,rstride=3,cstride=2,linewidth=0,antialiased=True)
    ax.set_xlabel('日期',labelpad=5); ax.set_ylabel('时间段',labelpad=7); ax.set_zlabel('功率',labelpad=13); ax.zaxis.label.set_rotation(90); ax.view_init(elev=32,azim=-60); ax.set_title('(a) 3D曲面',pad=8); ax.tick_params(pad=1,labelsize=8)
    ax2=fig.add_subplot(gs[1]); cf=ax2.contourf(X,Y,a,levels=18,cmap=cmap); ax2.set_xlabel('日期'); ax2.set_ylabel('时间段'); ax2.set_title('(b) 同一场的2D等高线',pad=8); ax2.set_xticks([1,91,182,274,365],['1月','4月','7月','10月','12月']); ax2.set_yticks([1,25,49,73,97,121,144],['00:00','04:00','08:00','12:00','16:00','20:00','24:00']); ax2.tick_params(labelsize=8); fig.colorbar(cf,ax=ax2,shrink=.82,pad=.03,label='功率 / kW')
    fig.suptitle(title,y=.93); fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight'); fig.savefig(out/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)
draw(pv,'fig2_全年光伏_修正版','全年光伏发电量时空分布','YlOrBr'); draw(load,'fig3_全年负载_修正版','全年负载时空分布','Blues'); draw(net,'fig4_全年净负荷_修正版','全年净负荷时空分布','RdYlBu_r')
print(out)
