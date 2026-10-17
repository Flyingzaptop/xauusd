"""Existing angular H/O/C/L displacement representation; aggregate raw paths."""
import numpy as np
COLS=['H','O','C','L'];LAGS=np.array([1,2,3,5,8,13,21,34,55])
def encode(current,previous,scale):return (2/np.pi)*np.arctan((current-previous)/scale)
def decode(angles,previous,scale):return previous+scale*np.tan(np.pi*np.clip(angles,-.999999,.999999)/2)
def aggregate(path,k):
    a=np.asarray(path).reshape(*path.shape[:-2],-1,k,4)
    return np.stack([a[...,0].max(-1),a[...,0,1],a[...,-1,2],a[...,3].min(-1)],-1)
def encode_path(path,previous,scale):
    prev=np.concatenate([previous[...,None,:],path[...,:-1,:]],-2);return encode(path,prev,scale)
def decode_path(angles,previous,scale):
    return previous[...,None,:]+np.cumsum(scale*np.tan(np.pi*np.clip(angles,-.999999,.999999)/2),axis=-2)
