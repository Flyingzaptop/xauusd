from pathlib import Path
import ast,numpy as np,json
source=Path(__file__).with_name('evaluate.py').read_text();module=ast.parse(source);fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='trades');ns={'np':np,'hs':[15,30,60,120]};exec(compile(ast.Module(body=[fn],type_ignores=[]),'trades-test','exec'),ns)
ns['gt']=np.arange(200)*60;ns['go']=100+np.arange(200)*.01;ns['a']=np.array([0,300,600,900,1200]);p=np.ones((5,4));t,e=ns['trades'](np.arange(5),p,0,0)
assert abs(t[0,3]-.14)<1e-9;assert np.all(t[1:,0]>=t[:-1,1]);assert e==0
ns['gt']=np.r_[np.arange(5),np.arange(100,200)]*60;ns['go']=100+ns['gt']/60*.01;ns['a']=np.array([0]);p=np.ones((1,4));t,_=ns['trades'](np.array([0]),p,0,0);assert t[0,1]==6000
v,_=ns['trades'](np.array([0]),p,0,0,cutoff=1800);assert len(v)==0
ns['go'][ns['gt']>=1800]*=2;v2,_=ns['trades'](np.array([0]),p,0,0,cutoff=1800);assert len(v2)==0
print(json.dumps({'passed':True,'one_minute_extra_entry_delay':True,'known_pnl':True,'no_position_overlap':True,'future_gaps_not_expost_filtered':True,'validation_does_not_use_post_boundary_exit_prices':True}))
