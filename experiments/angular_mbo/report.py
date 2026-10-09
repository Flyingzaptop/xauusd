"""Self-contained result page and fixed next-candle examples, after test completion only."""
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import *

def candle(ax,x,p,color,alpha=1.,width=.55):
    h,o,c,l=p;ax.plot([x,x],[l,h],color=color,alpha=alpha,lw=1)
    ax.add_patch(Rectangle((x-width/2,min(o,c)),width,max(abs(o-c),.015),facecolor=color,edgecolor=color,alpha=alpha))

def report():
    result=json.loads((OUT/'results.json').read_text());manifest=json.loads((OUT/'manifest.json').read_text())
    z=np.load(OUT/'candles.npz');lp=z['lp'];ts=z['ts'];s0=json.loads((OUT/'scale.json').read_text())['S0']
    datasets={v:np.load(OUT/f'test_{v}.npz') for v in ['m1','gc','mbo']}
    ix=datasets['mbo']['indices']; plots=OUT/'plots';plots.mkdir(exist_ok=True)
    for number,q in enumerate([.1,.3,.5,.7,.9],1):
        pos=int(q*(len(ix)-1));chosen=ix[max(0,pos-10):pos+11]
        rows=np.arange(chosen[0]-5,chosen[-1]+1)
        fig,axs=plt.subplots(3,1,figsize=(15,10),sharex=True,sharey=True)
        for ax,v in zip(axs,['m1','gc','mbo']):
            for j in rows:candle(ax,int(j-rows[0]),np.exp(lp[j]),'#a7afb8',.6,.7)
            slots=np.searchsorted(ix,chosen);angles=datasets[v]['angles'][slots]
            pred=lp[chosen-1][:,COLS]+s0*np.tan(angles*np.pi/2)
            for j,p in zip(chosen,pred):
                candle(ax,int(j-rows[0])+.2,np.exp([p[0],lp[j-1,2],p[1],p[2]]),'#176dcc',.85,.3)
            ax.set_title(v+' | grey: actual; blue: independent next-minute forecast');ax.grid(alpha=.2);ax.set_ylabel('USD')
        ticks=np.linspace(0,len(rows)-1,6,dtype=int);axs[-1].set_xticks(ticks,[str(np.datetime64(int(ts[rows[t]]),'s'))[11:16] for t in ticks])
        fig.suptitle(str(np.datetime64(int(ts[chosen[0]]),'s'))[:10]+' UTC; each prediction uses newly observed past, not a multi-minute rollout')
        fig.tight_layout();fig.savefig(plots/f'example_{number}.png',dpi=130);plt.close(fig)
    keys=['samples','angular_mse','price_mse_bps2','price_mae_usd','close_mse_bps2','close_direction_accuracy','close_movement_ratio','invalid_candle_fraction']
    table='<table><tr><th>Variant</th>'+''.join('<th>'+k+'</th>' for k in keys)+'</tr>'
    for name,m in result['test'].items():
        table+='<tr><td>'+name+'</td>'+''.join('<td>'+format(m[k],'.6g')+'</td>' for k in keys)+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>M1 + raw MBO: causal test</title>
    <style>body{font:16px system-ui;max-width:1500px;margin:36px auto;padding:24px;background:#f8fafc;color:#17243b}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse;font-size:13px}img{width:100%}pre{white-space:pre-wrap}</style>
    <h1>Прогноз следующей M1: прошлые свечи + прошлые MBO</h1>
    <p>Обучение: октябрь 2025 — май 2026; выбор модели: июнь; тест: июль. Все три варианта проверены на одних минутах. Июль ранее исследовался; это не нетронутый исторический период.</p>
    <p>512 M1-токенов и до 2048 отдельных MBO-событий до начала целевой минуты. GC-фьючерсы и XAUUSD — разные источники. Ценовая MSE — в bps² логарифмической цены, MAE — в USD. Показана также доля некорректной геометрии свечей. Торговая доходность не оценивалась.</p>'''+table
    page+='<h2>Разница MBO с контролями: отрицательная лучше</h2><pre>'+html.escape(json.dumps(result['comparisons'],ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>Обучение</h2><pre>'+html.escape(json.dumps(result['training'],ensure_ascii=False,indent=2))+'</pre>'
    page+='<h2>Заранее фиксированные позиции 10%, 30%, 50%, 70%, 90% теста</h2><p>Серый — факт, синий — прогноз. Перед каждым прогнозом предыдущая минута уже известна; это не рекурсивный прогноз всего участка.</p>'
    page+=''.join(f'<img src="plots/example_{n}.png">' for n in range(1,6))
    (OUT/'results.html').write_text(page+'</html>',encoding='utf-8')

if __name__=='__main__':report()
