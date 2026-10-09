"""Eight-window memorization test: compare objective effects, never report as validation."""
import sys,time,json
import numpy as np
import pandas as pd
import torch
from core import ROOT,OUT,CHECKPOINT,old,DEVICE,build,batch,output,losses,diagnostic_scores

def main():
    sys.stdout.reconfigure(encoding='utf-8');torch.set_num_threads(4);torch.set_float32_matmul_precision('high')
    folder=OUT/'diagnostic';folder.mkdir(parents=True,exist_ok=True)
    lp,ts,pieces,va,_,info=old.read_data();ck=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    candidates=np.concatenate([np.arange(a+566,b-512-40) for a,b in pieces['train'] if b-512-40>a+566])
    anchors=candidates[np.floor(np.linspace(.1,.9,8)*(len(candidates)-1)).astype(int)]
    lengths=np.array([1,5,10,20,1,5,10,20]);data=old.MaskedData(lp,ck['scale']['S0'],ck['scale']['H'])
    b=batch(data,lp,ts,anchors,lengths);del data
    old.dump(folder/'windows.json',dict(anchors=anchors.tolist(),lengths=lengths.tolist(),split='train',dropout=0,weight_decay=0,lr=1e-4,max_updates=1200))
    results=[]
    for kind in ['A','B']:
        net=build(kind,ck,diagnostic=True);opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=0.)
        started=time.time();rows=[];best=float('inf')
        for step in range(0,1201):
            net.train();opt.zero_grad(set_to_none=True);angles,path=output(net,b,kind);terms=losses(angles,path,b)
            loss=terms['angle'] if kind=='A' else terms['total']
            if step%50==0:
                scores=diagnostic_scores(angles,path,b);row=dict(kind=kind,step=step,seconds=time.time()-started,**scores,**{k:float(v.detach()) for k,v in terms.items()})
                rows.append(row);pd.DataFrame(rows).to_csv(folder/f'{kind}_curve.csv',index=False)
                print(json.dumps(row),flush=True)
                if scores['price_mae_bps']<best:
                    best=scores['price_mae_bps'];torch.save(dict(model=net.state_dict(),step=step,kind=kind,metrics=scores),folder/f'{kind}_best.pt')
                if step>=100 and scores['angle_rmse_degrees']<1 and scores['price_mae_bps']<.05:break
            if step==1200:break
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
        net.eval()
        with torch.no_grad():angles,path=output(net,b,kind)
        np.savez_compressed(folder/f'{kind}_predictions.npz',anchors=anchors,lengths=lengths,angles=angles.cpu().numpy(),
            predicted_relative_logs=path.cpu().numpy(),actual_relative_logs=b['truth'].cpu().numpy())
        results.append(dict(kind=kind,final=rows[-1],best_price_mae_bps=best))
        del net,opt
        if DEVICE=='cuda':torch.cuda.empty_cache()
    old.dump(folder/'results.json',dict(status='complete',memorization_not_generalization=True,results=results))

if __name__=='__main__':main()
