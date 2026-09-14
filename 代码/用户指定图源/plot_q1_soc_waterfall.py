from pathlib import Path
import openpyxl, numpy as np, matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','Noto Sans SC','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
p=R/'项目'/'04_结果与检验'/'问题1_结果'/'result1.xlsx'; w=openpyxl.load_workbook(p,data_only=True,read_only=True)
rows=list(w.worksheets[1].values)[1:7]; c=np.array([float(r[1]) for r in rows]); d=np.array([float(r[2]) for r in rows]); eta=.9
delta=eta*c-d/eta; start=6000.; end=start+delta.sum(); cum=np.r_[start,start+np.cumsum(delta)]
labels=['0:00—4:00','4:00—8:00','8:00—12:00','12:00—16:00','16:00—20:00','20:00—24:00']
fig,ax=plt.subplots(figsize=(9,5.2),dpi=160); x=np.arange(8)
ax.bar(0,start,width=.62,color='#2F80ED',edgecolor='white')
for i,v in enumerate(delta):
    bottom=cum[i] if v>=0 else cum[i]+v
    ax.bar(i+1,abs(v),bottom=bottom,width=.62,color=('#E74C3C' if v>=0 else '#2ECC71'),edgecolor='white')
    ax.text(i+1,cum[i+1]+(160 if v>=0 else -260),f'{v:+.1f}',ha='center',va='bottom' if v>=0 else 'top',fontsize=9,color='black')
ax.bar(7,end,width=.62,color='#2F80ED',edgecolor='white')
ax.text(0,start+180,f'{start:.0f}',ha='center',fontsize=9); ax.text(7,end+180,f'{end:.1f}',ha='center',fontsize=9,fontweight='bold')
ax.set_xticks(x,['0:00\n初始']+labels+['24:00\n末端'],rotation=18,fontsize=8); ax.set_ylabel('储电量（kWh）'); ax.set_ylim(0,14000); ax.set_title('问题一六时段储电量变化瀑布图')
ax.axhline(1200,color='#777',ls='--',lw=.9,label='下限1200'); ax.axhline(10800,color='#777',ls='-.',lw=.9,label='上限10800'); ax.grid(axis='y',alpha=.22); ax.legend(frameon=False,ncol=1,loc='center left',bbox_to_anchor=(1.01,.5)); fig.subplots_adjust(bottom=.22,top=.88,right=.82)
out=R/'figures'/'问题1_六时段储电量变化瀑布图.png'; fig.savefig(out,dpi=300,bbox_inches='tight'); fig.savefig(out.with_suffix('.svg'),bbox_inches='tight'); plt.close(fig)
print('delta=',delta.tolist(),'end=',end,'output=',out)
