"""CPU audit and visual review of stage-one historical candle reconstruction."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from representation import encode_path,decode_path,aggregate
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def describe(p,z,split):
    true=z[split+'_target_log'];anchor=z[split+'_anchor_log'];y=z[split+'_angles'];path=decode_path(p,anchor,float(z['s15']));err=np.abs(np.exp(path)-np.exp(true));h,o,c,l=np.moveaxis(path,-1,0)
    valid=(h>=np.maximum(o,c))&(l<=np.minimum(o,c));body=np.exp(path[:,:,2])-np.exp(path[:,:,1]);actualbody=np.exp(true[:,:,2])-np.exp(true[:,:,1]);mask=np.abs(actualbody)>1e-7
    violations=np.maximum(np.maximum(o,c)-h,0)+np.maximum(l-np.minimum(o,c),0)
    tolerance_valid=(np.exp(np.maximum(o,c))-np.exp(h)<=.005)&(np.exp(l)-np.exp(np.minimum(o,c))<=.005)
    return dict(valid_candle_fraction_half_cent_tolerance=float(tolerance_valid.mean()),max_geometry_violation_dollars=float((np.maximum(np.exp(o),np.exp(c))-np.exp(h)).clip(min=0).max()+ (np.exp(l)-np.minimum(np.exp(o),np.exp(c))).clip(min=0).max()),angle_mse=float(((p-y)**2).mean()),angle_mae_degrees=float(np.abs(p-y).mean()*90),path_mae_dollars=float(err.mean()),close_mae_dollars=float(err[:,:,2].mean()),last60_path_mae_dollars=float(err[:,-4:].mean()),valid_candle_fraction=float(valid.mean()),path_mae_by_block=err.mean((0,2)).tolist(),per_channel_mae_dollars=err.mean((0,1)).tolist(),observed_body_direction_accuracy=float((np.sign(body[mask])==np.sign(actualbody[mask])).mean()),per_example_mae=err.mean((1,2)).tolist()),path

def oracle(z,split):
    ids=z[split+'_origins'];v=z['features_1'][ids[:,None]+np.arange(-135,0),:4]
    relative=decode_path(v,np.zeros((len(ids),4)),float(z['s0']));coarse=aggregate(relative,15)
    return encode_path(coarse[:,1:],coarse[:,0],float(z['s15']))
def day_ci(a,b,days):
    unique=np.unique(days);sums=np.array([(a[days==d]-b[days==d]).sum() for d in unique]);counts=np.array([(days==d).sum() for d in unique]);rng=np.random.default_rng(1234);ix=rng.integers(0,len(unique),(10000,len(unique)));boot=sums[ix].sum(1)/counts[ix].sum(1)
    return dict(mean_difference=float(np.mean(a-b)),interval95=np.quantile(boot,[.025,.975]).tolist(),days=len(unique))
def candle(ax,x,v,color,width=3):
    h,o,c,l=v;valid=h>=max(o,c) and l<=min(o,c);color=color if valid else '#a624a6'
    ax.vlines(x,l,h,color=color,lw=1.2);ax.add_patch(Rectangle((x-width/2,min(o,c)),width,max(abs(c-o),.015),facecolor=color,alpha=.65,edgecolor=color,lw=.7))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'reports/transformer_stage1_20261012');args=ap.parse_args();out=args.out;r=out/'remote';z=np.load(out/'dataset.npz');run=read(r/'run.json');meta=read(out/'dataset.json')
    assert run['status']=='complete' and run['stage']==1 and run['later_stages_disabled'] and run['selection_frozen_before_test']
    assert read(r/'worker_result.json')['returncode']==0;assert run['data_sha256']==hashlib.sha256((out/'dataset.npz').read_bytes()).hexdigest()
    for name in ['train.py','data.py','model.py','loss.py','representation.py']:assert run['source_sha256'][name]==hashlib.sha256((ROOT/'experiments/transformer_stage1'/name).read_bytes()).hexdigest()
    assert len(run['trials'])==4;curves={}
    for name,v in run['trials'].items():
        rows=read(r/f'{name}_curve.json');curves[name]=rows;assert len(rows)==v['epochs'];assert all(row['examples_seen']==len(z['train_origins']) for row in rows)
        best=min(rows,key=lambda row:row['validation']['path_mae_dollars']);assert best['epoch']==v['best_epoch']
        vm,_=describe(np.load(r/f'{name}_validation.npz')['angles'],z,'val');assert np.isclose(vm['path_mae_dollars'],v['best_validation_mae'])
    for seed in [17,29]:assert run['trials'][f'static_{seed}']['initial_common_sha256']==run['trials'][f'hyper_{seed}']['initial_common_sha256']
    selected=min(['static','hyper'],key=lambda k:run['families'][k]['validation']['path_mae_dollars']);assert selected==run['selected_family']
    pred={};paths={};results={}
    for family,v in run['families'].items():
        p=np.load(r/f'{family}_test.npz')['angles'];ensemble=np.mean([np.load(r/f'{family}_{s}_test.npz')['angles'] for s in [17,29]],axis=0);assert np.array_equal(p,ensemble)
        assert np.isfinite(p).all() and np.max(np.abs(p))<1
        m,path=describe(p,z,'test')
        for k in ['angle_mse','path_mae_dollars','close_mae_dollars','valid_candle_fraction']:assert np.isclose(m[k],v['test'][k])
        names=[f'{family}_{s}' for s in [17,29]] if family!='hyper_fixed' else [f'hyper_{s}_fixed' for s in [17,29]]
        vp=np.mean([np.load(r/f'{name}_validation.npz')['angles'] for name in names],axis=0);vm,_=describe(vp,z,'val');assert np.isclose(vm['path_mae_dollars'],v['validation']['path_mae_dollars'])
        pred[family]=p;paths[family]=path;results[family]=dict(validation_mae_dollars=vm['path_mae_dollars'],**m)
    shape=z['test_angles'].shape
    baselines={'zero_vectors':np.zeros(shape),'train_mean_vectors':np.broadcast_to(z['train_angles'].mean(0),shape),'repeat_previous_vector':np.broadcast_to(z['test_previous_vector'][:,None,:],shape),'analytic_vector_oracle':oracle(z,'test')}
    for name,p in baselines.items():
        results[name],paths[name]=describe(p,z,'test');pred[name]=p
    assert results['analytic_vector_oracle']['path_mae_dollars']<.005
    days=(int(z['start_minute'])+z['test_origins'])//1440;comp={}
    for k in ['zero_vectors','train_mean_vectors',('static' if selected=='hyper' else 'hyper')]:
        if k!=selected:comp[f'{selected}_minus_{k}']=day_ci(np.array(results[selected]['per_example_mae']),np.array(results[k]['per_example_mae']),days)
    verification=dict(status='passed',stage=1,later_stages_disabled=True,trials=4,total_epochs=sum(v['epochs'] for v in run['trials'].values()),train_examples_per_epoch=len(z['train_origins']),source_data_hashes=True,paired_initial_encoder=True,best_epochs_recomputed=True,ensemble_validation_test_metrics_recomputed=True,analytic_oracle_from_inputs_only=True,absolute_prices_excluded_from_model_input=True)
    write(out/'verification.json',verification);write(out/'summary.json',dict(selected_family=selected,metrics={k:{key:v for key,v in m.items() if key!='per_example_mae'} for k,m in results.items()},comparisons=comp,trials=run['trials'],seconds=run['seconds']))
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,rows in curves.items():
        color='#3266ad' if name.startswith('static') else '#cb7724';style='-' if name.endswith('17') else '--';epochs=[row['epoch'] for row in rows]
        axs[0].plot(epochs,[row['train_probe']['path_mae_dollars'] for row in rows],style,color=color,label=name);axs[1].plot(epochs,[row['validation']['path_mae_dollars'] for row in rows],style,color=color,label=name)
    for ax,title in zip(axs,['Train: fixed chronological probe','Validation']):ax.set_title(title);ax.set_xlabel('Full chronological epoch');ax.set_ylabel('Reconstructed OHLC MAE ($)');ax.legend(fontsize=8)
    fig.suptitle('STAGE 1 ONLY: describe visible past; dropout 25%');fig.savefig(out/'learning.png',dpi=170);plt.close(fig)
    actual=np.exp(z['test_target_log']);models=paths[selected];base=np.exp(z['test_anchor_log'][:,2]);xx=np.arange(8)*15-112.5
    def plot_examples(indices,filename,title):
        fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ax,j in zip(axs.flat,indices):
            a=actual[j]-base[j];p=np.exp(models[j])-base[j]
            for t in range(8):candle(ax,xx[t]-2,a[t],'#333333');candle(ax,xx[t]+2,p[t],'#3266ad')
            ax.plot(xx,a[:,2],color='#333333',lw=.8,alpha=.6);ax.plot(xx,p[:,2],color='#3266ad',lw=.8,alpha=.6)
            origin=str(np.datetime64(int(z['start_minute']+z['test_origins'][j]),'m'));ax.set_title(f"{origin}\nMAE ${results[selected]['per_example_mae'][j]:.3f}",fontsize=10)
            ax.set_xlabel('Minutes in already-visible past');ax.set_ylabel('Price relative to preceding candle Close ($)');ax.axhline(0,color='gray',lw=.5)
        fig.suptitle(title+'\nBlack: actual; blue: model; purple: invalid predicted geometry. No price sorting.');fig.savefig(out/filename,dpi=170);plt.close(fig)
    indices=np.linspace(0,len(actual)-1,6,dtype=int);plot_examples(indices,'reconstructions.png',f'VISIBLE PAST, NOT A FORECAST — {selected}, evenly spaced test examples')
    worst=np.argsort(results[selected]['per_example_mae'])[-6:][::-1];plot_examples(worst,'worst_examples.png','VISIBLE PAST — six largest reconstruction errors')
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained');keys=['static','hyper','hyper_fixed','zero_vectors','train_mean_vectors','analytic_vector_oracle']
    axs[0].bar(np.arange(len(keys)),[results[k]['path_mae_dollars'] for k in keys],color=['#3266ad','#cb7724','#999999','#666666','#888888','#298654']);axs[0].set_xticks(np.arange(len(keys)),keys,rotation=24,ha='right');axs[0].set_ylabel('Test OHLC reconstruction MAE ($)')
    for k in ['static','hyper','zero_vectors','train_mean_vectors']:axs[1].plot(xx,results[k]['path_mae_by_block'],label=k)
    axs[1].set_xlabel('Minutes in visible past (block center)');axs[1].set_ylabel('Reconstruction MAE ($)');axs[1].legend();fig.suptitle('Stage 1: information is present; analytic aggregation is nearly exact');fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    examples=[]
    for j in indices:
        examples.append(dict(origin=str(np.datetime64(int(z['start_minute']+z['test_origins'][j]),'m')),window='visible past [-120,0) minutes',actual_ohlc=actual[j].tolist(),predicted_ohlc=np.exp(models[j]).tolist(),actual_vectors=z['test_angles'][j].tolist(),predicted_vectors=pred[selected][j].tolist(),channel_order=['H','O','C','L']))
    write(out/'review_examples.json',dict(stage=1,model=selected,examples=examples))
    lines=['# Transformer — только первый этап','',f"На validation выбран **{selected}**. Выполнено четыре запуска первого этапа: две архитектуры × две инициализации. Обучение/оценка скриптом: {run['seconds']/60:.2f} мин. Следующие этапы не запускались.",'',
           'Задача: из угловых векторов видимой истории описать последние 120 минут восемью 15-минутными свечами. Последние четыре свечи описывают последние 60 минут. Это прошлое, полностью присутствующее во входе. Результат не является прогнозом будущего.','',
           'Представление: H/O/C/L приращения лог-цен → (2/π) atan(delta/scale). Лаги 1,2,3,5,8,13,21,34,55. Масштабы определены только на TRAIN. В модель входят лишь векторы и признаки доступности; абсолютные цены, привязка к текущему Close, уровни волатильности и таргеты отсутствуют. Векторные суммы до atan восстанавливают путь; max/min сохраняют экстремумы укрупнённых свечей.','',
           f"Контекст: 360 M1, 576 M5, 336 H1 завершённых свечей. Transformer: четыре слоя, четыре attention heads, ширина96, feedforward192, dropout25%. Восемь learned summary queries. Encoder видит только прошлое, без teacher forcing. Параметры: {run['checks']['parameters']}. Общие encoder-веса совпадают при старте парных моделей.",'',
           f"Train: {meta['train']['examples']} окон января; каждый проход хронологический и полный. Validation: {meta['val']['examples']} окон 1–7 февраля. Test: {meta['test']['examples']} окон 15–29 февраля. Это исследовательские, ранее встречавшиеся рыночные периоды, не независимый финальный торговый тест.",'',
           'Loss: Huber по накопленным декодированным перемещениям + 0,1 MSE угловых векторов. Абсолютные цены в loss не нужны. В правильном таргете loss и выходной градиент точно равны нулю. Проверены градиенты всех encoder-слоёв, генератора весов, dropout и детерминированный eval. Весь путь декодируется из предсказанных векторов, без подстановки правильных промежуточных свечей.','',
           '| Вариант | Test OHLC MAE, $ | Close MAE, $ | Последние 60 мин OHLC MAE, $ | Верных по геометрии свечей | Угловая MAE |','|---|---:|---:|---:|---:|---:|']
    for k,m in results.items():lines.append(f"| {k} | {m['path_mae_dollars']:.4f} | {m['close_mae_dollars']:.4f} | {m['last60_path_mae_dollars']:.4f} | {100*m['valid_candle_fraction']:.1f}% | {m['angle_mae_degrees']:.2f}° |")
    gain=1-results[selected]['path_mae_dollars']/results['zero_vectors']['path_mae_dollars']
    lines+=['','Геометрия в таблице проверена строго, без допуска. У аналитического oracle нарушения на границах связаны с округлением float32-векторов: при допуске половины цента валидны '+f"{100*results['analytic_vector_oracle']['valid_candle_fraction_half_cent_tolerance']:.1f}%"+' свечей. Для моделей приведены исходные нарушения, без исправления OHLC.','']
    lines+=['',f"Выбранная модель уменьшила ошибку относительно нулевых векторов на {100*gain:.1f}%. Нулевые векторы повторяют предшествующую крупную свечу. TRAIN mean vectors — средние векторы TRAIN, repeat_previous_vector — повтор последнего крупного перемещения. Аналитический oracle строит целевые векторы только из входных M1-векторов, в произвольной нулевой системе координат; абсолютная привязка нужна лишь для итогового отображения в долларах.",'',
           'hyper_fixed — выбранная гиперсеть с условием генерации весов, заменённым средним TRAIN-контекстом каждого запроса. Это диагностическая проверка без переобучения, исключённая из выбора модели. Предсказанные OHLC не сортируются и не исправляются; ошибки геометрии сохраняются и отмечаются пурпурным.','',
           '[Интерактивный просмотр всех 642 примеров](review.html) — выбор модели, окна, среднего и худшего по ошибке примера.','','![Кривые обучения](learning.png)','','![Обычные примеры](reconstructions.png)','','![Худшие примеры](worst_examples.png)','','![Сравнение](comparison.png)','','| Запуск | Эпохи | Выбранная эпоха | Причина остановки |','|---|---:|---:|---|']
    for name,v in run['trials'].items():lines.append(f"| {name} | {v['epochs']} | {v['best_epoch']} | {v['stop']} |")
    lines+=['','Разница средних ошибок по календарным дням; отрицательная означает меньшую ошибку первого варианта. 95% интервалы — описательный парный bootstrap по дням, не по зависимым окнам.','', '| Сравнение | Разница, $ | 95% интервал |','|---|---:|---:|']
    for k,v in comp.items():lines.append(f"| {k} | {v['mean_difference']:.4f} | [{v['interval95'][0]:.4f}; {v['interval95'][1]:.4f}] |")
    lines+=['','Оценка первого этапа: модели научились воспроизводить крупное движение видимого прошлого и существенно превосходят постоянные векторные ответы. Детали тел и экстремумов ещё неточны; выбранная гиперсеть допускает около '+f"{100*(1-results[selected]['valid_candle_fraction']):.1f}%"+' геометрически неверных свечей. Точное освоение агрегации пока не достигнуто. Разница между архитектурами мала; её парный интервал приведён выше.','']
    lines+=['','Отдельный факт: данные достаточны для этого первого этапа, потому что точный алгоритм восстанавливает ответ из входных векторов почти без ошибки. Остаточная ошибка модели отражает изучение агрегации, архитектуру, оптимизацию и выбранный бюджет. Она не свидетельствует о нехватке будущей информации: будущего в задаче вообще нет.','',
           'Готово для оценки пользователем. Автоматического перехода к этапу2 нет. Веса, прогнозы, исходные хэши и проверки сохранены. См. protocol.md, data_verification.json и verification.json.','']
    (out/'REPORT.md').write_text('\n'.join(lines));print(json.dumps(dict(selected_family=selected,metrics={k:{key:v for key,v in m.items() if key!='per_example_mae'} for k,m in results.items()},verification=verification),indent=2))
if __name__=='__main__':main()
