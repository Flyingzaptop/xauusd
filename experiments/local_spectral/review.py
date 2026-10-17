"""Standalone browser viewer for the saved spectral analysis; no external APIs."""
import argparse
import json
from pathlib import Path
import matplotlib
import numpy as np
from run import relative_db, reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    s=json.loads((a.out/'summary.json').read_text())
    z=np.load(a.out/'coefficients.npz');n=int(z['train_candles']);maps={}
    def add(name,label,period,times,db,note):
        encoded=np.clip(np.nan_to_num((db+12)/30*255,nan=0),0,255).astype(int)
        encoded[~np.isfinite(db)]=256
        maps[name]={'label':label,'period':period.tolist(),'time':times.tolist(),'pixels':encoded.tolist(),'note':note}
    for length in [512,2048]:
        coef=z[f'stft_{length}'];t=z[f'stft_{length}_end'];period=z[f'stft_{length}_period']
        # Hann PSD scale is a constant per window size and cancels in ratios.
        power=abs(coef)**2;db=relative_db(power,reference(power,t,n))
        add(str(length),f'STFT доходностей: окно {length}',period,t,db,
            'Окно использует только прошлые свечи; коэффициент расположен на его правом краю. Цвета нормированы по всему июню для просмотра.')
    t=z['cwt_times'];period=z['cwt_periods'];ref=z['cwt_train_reference_power']
    add('cwt','Вейвлеты доходностей: центрированная карта',period,t,relative_db(abs(z['cwt_returns'])**2,ref),
        'Ретроспективная карта: использует будущих соседей. Свечение перед событием может быть вызвано самим будущим событием.')
    power=abs(z['cwt_logprice'])**2
    add('price','Вейвлеты log-цены: центрированная карта',period,t,relative_db(power,reference(power,t,n)),
        'Ретроспективная карта log-цены со сдвигом среднего; тренд сохранён. Использует будущих соседей.')
    add('delayed','Вейвлеты: момент полной доступности',period,t,relative_db(abs(z['cwt_delayed'])**2,ref),
        'Те же коэффициенты показаны с задержкой ceil(8*scale)+4 свечи. На разных масштабах это разные задержки; актуальная фаза здесь не предсказана.')
    payload={'close':np.round(z['close'],4).tolist(),'date':[str(v) for v in z['timestamp_minutes'].astype('datetime64[m]')],
             'split':n,'event':s['largest_train_event']['index'],'maps':maps,
             'palette':(matplotlib.colormaps['magma'](np.linspace(0,1,256))[:,:3]*255).astype(int).tolist()}
    template='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>STFT и вейвлеты XAUUSD</title>
<style>body{font:16px system-ui;background:#f2f4f8;color:#192438;margin:24px auto;max-width:1150px;padding:0 18px}h1{font-size:25px}.card{background:white;padding:18px;border-radius:12px;margin:14px 0}canvas{width:100%;display:block}select,button{padding:8px;margin:4px;border:1px solid #bdc7d7;border-radius:6px;background:white}input{width:100%}.small{font-size:13px;color:#526176}#note{padding:10px;background:#fff2df;border-radius:8px}a{color:#245cc7}</style>
<h1>STFT и вейвлеты: июнь 2021 + следующая неделя</h1>
<p>Свечи склеены без выходных и вставок. Все показанные цены уже наблюдены. Это исследование структуры, прогнозов здесь нет.</p>
<div class="card"><select id="method"></select><select id="span"><option value="512">512 свечей</option><option value="2048">2048 свечей</option><option value="8192" selected>8192 свечи</option><option value="all">Весь участок</option></select>
<button id="begin">Начало июня</button><button id="shock">Падение 16 июня</button><button id="week">Следующая неделя</button>
<input id="start" type="range" min="0" step="32" value="0"><div id="range" class="small"></div><p id="note"></p></div>
<div class="card"><canvas id="price" width="1080" height="235"></canvas><canvas id="heat" width="1080" height="420"></canvas><p class="small">Яркость — энергия относительно июньской медианы каждой частоты/масштаба (−12…+18 dB). Масштаб цвета общий для методов по этому правилу, абсолютная энергия между строками напрямую не сравнивается. Пустые области — коэффициент ещё недоступен. Розовая линия — конец июня.</p></div>
<p><a href="overview.png">Обзор</a> · <a href="event_detail.png">Крупнейшее падение</a> · <a href="controls.png">Случайные контроли</a> · <a href="REPORT.md">Отчёт</a></p>
<script>const D=__DATA__;const $=x=>document.getElementById(x);for(const [key,m] of Object.entries(D.maps)){const o=document.createElement('option');o.value=key;o.textContent=m.label;$('method').append(o)}
function nearest(a,x){let lo=0,hi=a.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(a[mid]<x)lo=mid+1;else hi=mid}if(lo&&Math.abs(a[lo-1]-x)<Math.abs(a[lo]-x))return lo-1;return lo}
function draw(){const N=D.close.length;const span=$('span').value==='all'?N:Math.min(N,+$('span').value);$('start').max=N-span;let start=Math.min(+$('start').value,N-span);$('start').value=start;const end=start+span;const m=D.maps[$('method').value];$('range').textContent=D.date[start]+' → '+D.date[end-1]+' | '+span+' реальных свечей';$('note').textContent=m.note;
const pc=$('price'),c=pc.getContext('2d'),W=pc.width,H=pc.height;const L=68,R=W-18,T=25,B=H-38;c.clearRect(0,0,W,H);let lo=Infinity,hi=-Infinity;for(let i=start;i<end;i++){lo=Math.min(lo,D.close[i]);hi=Math.max(hi,D.close[i])}const margin=Math.max(.1,(hi-lo)*.06);lo-=margin;hi+=margin;c.strokeStyle='#17283e';c.lineWidth=1;c.beginPath();for(let i=start;i<end;i++){const x=L+(i-start)/(span-1)*(R-L),y=B-(D.close[i]-lo)/(hi-lo)*(B-T);if(i===start)c.moveTo(x,y);else c.lineTo(x,y)}c.stroke();c.fillStyle='#526176';c.font='12px system-ui';for(let j=0;j<4;j++){const v=lo+(hi-lo)*j/3,y=B-j/3*(B-T);c.fillText(v.toFixed(1),8,y+4)}c.fillText('Close ($)',L,16);boundary(c,L,R,T,B,start,span);ticks(c,L,R,H-15,start,span);
const hc=$('heat'),ctx=hc.getContext('2d'),hW=hc.width,hH=hc.height,hL=68,hR=hW-18,hT=28,hB=hH-40;ctx.clearRect(0,0,hW,hH);const width=hR-hL,height=hB-hT,img=ctx.createImageData(width,height);const lp0=Math.log2(m.period[0]),lp1=Math.log2(m.period.at(-1));const rows=[];for(let y=0;y<height;y++)rows.push(nearest(m.period,2**(lp1-(lp1-lp0)*y/(height-1))));const cols=[];for(let x=0;x<width;x++){const t=start+x/(width-1)*(span-1);cols.push(t<m.time[0]? -1:Math.max(0,nearest(m.time,t)))}for(let y=0;y<height;y++){const row=rows[y];for(let x=0;x<width;x++){const col=cols[x],p=col<0?256:m.pixels[row][col],rgb=p===256?[237,240,245]:D.palette[p],i=(y*width+x)*4;img.data[i]=rgb[0];img.data[i+1]=rgb[1];img.data[i+2]=rgb[2];img.data[i+3]=255}}ctx.putImageData(img,hL,hT);ctx.fillStyle='#526176';ctx.font='12px system-ui';ctx.fillText('Период, свечи',hL,16);for(const p of [4,8,16,32,64,128,256,512,1024])if(p>=m.period[0]&&p<=m.period.at(-1)){const y=hB-(Math.log2(p)-lp0)/(lp1-lp0)*(hB-hT);ctx.fillText(p,16,y+4)}boundary(ctx,hL,hR,hT,hB,start,span);ticks(ctx,hL,hR,hH-16,start,span)}
function boundary(c,L,R,T,B,start,span){if(D.split>=start&&D.split<start+span){const x=L+(D.split-start)/(span-1)*(R-L);c.strokeStyle='#e05a8b';c.setLineDash([5,4]);c.beginPath();c.moveTo(x,T);c.lineTo(x,B);c.stroke();c.setLineDash([])}}
function ticks(c,L,R,y,start,span){c.fillStyle='#526176';c.font='11px system-ui';for(let j=0;j<4;j++){const i=Math.min(D.date.length-1,Math.round(start+j/3*(span-1))),x=L+j/3*(R-L);c.textAlign=j===0?'left':j===3?'right':'center';c.fillText(D.date[i].replace('T',' '),x,y)}c.textAlign='left'}
for(const id of ['method','span','start'])$(id).addEventListener('input',draw);$('begin').onclick=()=>{$('start').value=0;draw()};$('shock').onclick=()=>{$('span').value='2048';$('start').max=D.close.length-2048;$('start').value=Math.max(0,D.event-512);draw()};$('week').onclick=()=>{$('span').value='8192';$('start').max=D.close.length-8192;$('start').value=D.split;draw()};draw();</script></html>'''
    (a.out/'review.html').write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'))))
    print('review.html', (a.out/'review.html').stat().st_size)


if __name__=='__main__':main()
