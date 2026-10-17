"""Candle-by-candle viewer: hide every candle after the selected closed candle."""
import argparse
import json
from pathlib import Path
import matplotlib
import numpy as np
from causal_day import db


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    z=np.load(a.out/'features.npz');s=json.loads((a.out/'summary.json').read_text());maps={}
    def add(key,label,period,v,phase=False):
        lo,hi=(-np.pi,np.pi) if phase else (-12,18)
        enc=np.clip(np.nan_to_num((v-lo)/(hi-lo)*255,nan=0),0,255).astype(int);enc[~np.isfinite(v)]=256
        maps[key]={'label':label,'period':period.tolist(),'pixels':enc.tolist(),'phase':phase}
    for length in [128,512]:
        add(str(length),f'STFT доходностей: окно {length}',z[f'stft_{length}_period'],db(z[f'stft_{length}'],z[f'stft_{length}_reference']))
    power=db(z['wavelet_returns'],z['wavelet_return_reference'])
    add('wave','Односторонний Morlet: изменения цены',z['wavelet_periods'],power)
    add('price','Односторонний Morlet: log-цена',z['wavelet_periods'],db(z['wavelet_logprice'],z['wavelet_price_reference']))
    phase=np.angle(z['wavelet_returns']);phase[power<6]=np.nan
    add('phase','Односторонний Morlet: фаза сильных коэффициентов',z['wavelet_periods'],phase,True)
    payload={'date':[str(t) for t in z['timestamp_minutes'].astype('datetime64[m]')],
             'available':[str(t) for t in z['available_timestamp_minutes'].astype('datetime64[m]')],
             'close':np.round(z['close'],4).tolist(),'maps':maps,'event':s['largest_change']['day_index'],
             'palette':{k:(matplotlib.colormaps[k](np.linspace(0,1,256))[:,:3]*255).astype(int).tolist() for k in ['magma','twilight']}}
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Причинные карты: каждая M1 свеча</title>
<style>body{font:16px system-ui;color:#192638;background:#f2f4f8;max-width:1140px;margin:24px auto;padding:0 18px}h1{font-size:25px}.card{background:white;border-radius:12px;padding:18px;margin:14px 0}canvas{display:block;width:100%}select,button{padding:8px;margin:4px;border:1px solid #bdc7d7;border-radius:6px;background:white}input{width:100%}.small{font-size:13px;color:#526176}#status{padding:10px;background:#eef5fc;border-radius:6px}a{color:#245cc7}</style>
<h1>16 июня 2021: STFT и односторонние вейвлеты</h1><p>Каждая реальная M1 свеча. Текущая свеча уже закрыта. Всё после выбранной свечи скрыто; коэффициенты используют только прошлое и её Close. Цветовая шкала зафиксирована до начала дня. Паузы склеены.</p>
<div class="card"><select id="method"></select><select id="span"><option value="all">Вся доступная часть дня</option><option value="128">128 последних свечей</option><option value="512">512 последних свечей</option></select>
<button id="prev">−1 свеча</button><button id="next">+1 свеча</button><button id="before">Перед падением</button><button id="end">Конец дня</button><input id="point" type="range" min="0" step="1"><div id="status"></div></div>
<div class="card"><canvas id="price" width="1080" height="235"></canvas><canvas id="heat" width="1080" height="445"></canvas><p id="legend" class="small"></p></div>
<p><a href="day_overview.png">Весь день</a> · <a href="shock_detail.png">Деталь падения</a> · <a href="day_phase.png">Фазы</a> · <a href="REPORT.md">Отчёт и проверки</a></p>
<script>const D=__DATA__;const $=x=>document.getElementById(x);for(const [key,m] of Object.entries(D.maps)){const o=document.createElement('option');o.value=key;o.textContent=m.label;$('method').append(o)}$('method').value='wave';$('point').max=D.close.length-1;$('point').value=D.close.length-1;
function nearest(a,x){let lo=0,hi=a.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(a[mid]<x)lo=mid+1;else hi=mid}if(lo&&Math.abs(a[lo-1]-x)<Math.abs(a[lo]-x))return lo-1;return lo}
function ticks(c,L,R,y,start,end){c.fillStyle='#526176';c.font='12px system-ui';for(let j=0;j<5;j++){const i=Math.round(start+j/4*(end-start));c.textAlign=j===0?'left':j===4?'right':'center';c.fillText(D.date[i].slice(11),L+j/4*(R-L),y)}c.textAlign='left'}
function draw(){const point=+$('point').value,start=$('span').value==='all'?0:Math.max(0,point-(+$('span').value)+1),count=point-start+1,m=D.maps[$('method').value];$('status').textContent='Свеча '+(point+1)+'/'+D.close.length+' | метка '+D.date[point]+' | доступно после '+D.available[point]+' | Close $'+D.close[point].toFixed(2);$('legend').textContent=m.phase?'Фаза −π…+π. Пустые области: энергия ниже 4× прошлой медианы. Это наблюдённая фаза, не прогноз.':'Яркость −12…+18 dB относительно медианы до начала дня для каждой частоты/масштаба. Периоды указаны в реальных свечах; масштаб Morlet номинальный.';
const pc=$('price'),c=pc.getContext('2d'),W=pc.width,H=pc.height,L=68,R=W-18,T=22,B=H-34;c.clearRect(0,0,W,H);let lo=Infinity,hi=-Infinity;for(let i=start;i<=point;i++){lo=Math.min(lo,D.close[i]);hi=Math.max(hi,D.close[i])}const pad=Math.max(.1,(hi-lo)*.06);lo-=pad;hi+=pad;c.strokeStyle='#17283e';c.lineWidth=1.2;c.beginPath();for(let i=start;i<=point;i++){const x=L+(i-start)/Math.max(count-1,1)*(R-L),y=B-(D.close[i]-lo)/(hi-lo)*(B-T);if(i===start)c.moveTo(x,y);else c.lineTo(x,y)}c.stroke();c.fillStyle='#526176';c.font='12px system-ui';for(let j=0;j<4;j++)c.fillText((lo+j/3*(hi-lo)).toFixed(1),8,B-j/3*(B-T)+4);c.fillText('Close ($), только доступная история',L,14);ticks(c,L,R,H-10,start,point);
const hc=$('heat'),ctx=hc.getContext('2d'),hL=68,hR=hc.width-18,hT=28,hB=hc.height-34,width=hR-hL,height=hB-hT;ctx.clearRect(0,0,hc.width,hc.height);const img=ctx.createImageData(width,height),p0=Math.log2(m.period[0]),p1=Math.log2(m.period.at(-1)),palette=D.palette[m.phase?'twilight':'magma'];for(let y=0;y<height;y++){const row=nearest(m.period,2**(p1-y/(height-1)*(p1-p0)));for(let x=0;x<width;x++){const i=Math.min(point,start+Math.floor(x/(width-1)*Math.max(count-1,0))),v=m.pixels[row][i],rgb=v===256?[238,241,246]:palette[v],at=(y*width+x)*4;img.data[at]=rgb[0];img.data[at+1]=rgb[1];img.data[at+2]=rgb[2];img.data[at+3]=255}}ctx.putImageData(img,hL,hT);ctx.fillStyle='#526176';ctx.font='12px system-ui';ctx.fillText('Период, свечи',hL,16);for(const p of [4,8,16,32,64,128,256,512])if(p>=m.period[0]&&p<=m.period.at(-1)){const y=hB-(Math.log2(p)-p0)/(p1-p0)*(hB-hT);ctx.fillText(p,16,y+4)}ticks(ctx,hL,hR,hc.height-12,start,point)}
for(const id of ['method','span','point'])$(id).addEventListener('input',draw);$('prev').onclick=()=>{$('point').value=Math.max(0,+$('point').value-1);draw()};$('next').onclick=()=>{$('point').value=Math.min(D.close.length-1,+$('point').value+1);draw()};$('before').onclick=()=>{$('point').value=D.event-1;draw()};$('end').onclick=()=>{$('point').value=D.close.length-1;draw()};draw();</script></html>'''
    (a.out/'review.html').write_text(page.replace('__DATA__',json.dumps(payload,separators=(',',':'))))
    print('review.html', (a.out/'review.html').stat().st_size)


if __name__=='__main__':main()
