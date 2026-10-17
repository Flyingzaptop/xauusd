"""CPU metrics and predeclared vector-only activity baselines."""
import numpy as np
Q=np.array([.1,.5,.9])
def pinball(p,y):
    e=y[...,None]-p;return np.maximum(Q*e,(Q-1)*e)
def describe(p,z,split):
    y=z[split+'_y'];scale=z['target_scale'];anchor=z[split+'_anchor_price'];error=np.abs(p[...,1]-y)
    dollar=error*scale[None,:,:]*anchor[:,None,None];covered=(y>=p[...,0])&(y<=p[...,2]);width=(p[...,2]-p[...,0])*scale[None,:,:]*anchor[:,None,None]
    return dict(pinball_normalized=float(pinball(p,y).mean()),median_mae_normalized=float(error.mean()),mae_dollars=dollar.mean(0).tolist(),coverage80=covered.mean(0).tolist(),interval_width_dollars=width.mean(0).tolist(),actual_mean_dollars=(z[split+'_relative']*anchor[:,None,None]).mean(0).tolist(),median_bias_dollars=((p[...,1]-y)*scale[None,:,:]*anchor[:,None,None]).mean(0).tolist(),horizon_violation_fraction=float((p[:,1]*scale[None,1,:,None]<p[:,0]*scale[None,0,:,None]-1e-12).mean()))
def baseline_predictions(z,split):
    y=z['train_y'];unconditional=np.moveaxis(np.quantile(y,Q,axis=0),0,-1)
    train_proxy=np.maximum(z['train_past_proxy'],z['proxy_floor']);ratios=np.moveaxis(np.quantile(z['train_relative']/train_proxy,Q,axis=0),0,-1)
    conditional=np.maximum(z[split+'_past_proxy'],z['proxy_floor'])[...,None]*ratios/z['target_scale'][None,:,:,None]
    return dict(train_distribution=np.broadcast_to(unconditional,z[split+'_y'].shape+(3,)).astype('float32'),past_activity=conditional.astype('float32'))
