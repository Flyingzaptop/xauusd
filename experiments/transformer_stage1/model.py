"""Past-only Transformer encoder with eight historical summary queries."""
import torch
from torch import nn
class Model(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.fixed_context=None
        self.patches=[2,6,4];self.projections=nn.ModuleList([nn.Linear(37*p,96) for p in self.patches])
        self.position=nn.Parameter(torch.randn(360,96)*.01);self.scale_embed=nn.Parameter(torch.randn(3,96)*.01)
        self.queries=nn.Parameter(torch.randn(8,96)*.02)
        layer=nn.TransformerEncoderLayer(96,4,192,dropout=.25,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,4,enable_nested_tensor=False);self.norm=nn.LayerNorm(96);self.drop=nn.Dropout(.25)
        # Encoder layers start differently rather than cloned identical tensors.
        for block in self.encoder.layers:
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight);nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(block.linear1.weight);nn.init.xavier_uniform_(block.linear2.weight)
        hidden=219 if kind=='static' else 128;self.fc1=nn.Linear(96,hidden);self.fc2=nn.Linear(hidden,64);self.readout=nn.Linear(64,4)
        nn.init.normal_(self.readout.weight,std=.01);nn.init.zeros_(self.readout.bias)
        if kind=='hyper':
            self.hypernet=nn.Sequential(nn.Linear(96,16),nn.GELU(),nn.Linear(16,768))
            nn.init.normal_(self.hypernet[-1].weight,std=.03);nn.init.zeros_(self.hypernet[-1].bias)
        elif kind!='static':raise ValueError(kind)
    def contexts(self,x):
        parts=[];offset=0
        for s,(v,p,projection) in enumerate(zip(x,self.patches,self.projections)):
            part=projection(v.reshape(len(v),-1,37*p));length=part.shape[1]
            parts.append(part+self.position[offset:offset+length]+self.scale_embed[s]);offset+=length
        assert offset==360
        observed=torch.cat(parts,1);query=self.queries[None].expand(len(observed),-1,-1)
        # Bidirectional attention over fully visible past is intentional. No target
        # tokens, prices, future candles, decoder anchor, or teacher forcing.
        return self.norm(self.encoder(torch.cat([observed,query],1))[:,-8:])
    def factors(self,c):
        raw=self.hypernet(c);return raw[...,:256].reshape(-1,64,4),raw[...,256:].reshape(-1,4,128)
    def forward(self,x):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            c=self.contexts(x);h=self.drop(torch.nn.functional.gelu(self.fc1(c)));v=self.fc2(h)
            if self.kind=='hyper':
                condition=c if self.fixed_context is None else self.fixed_context.expand(len(c),-1,-1)
                a,b=self.factors(condition);v=v+(torch.bmm(a,torch.bmm(b,h.reshape(-1,128,1))).squeeze(-1)/2).reshape(len(c),8,64)
            v=self.drop(torch.nn.functional.gelu(v));raw=self.readout(v).float()
        return .999*raw.tanh()
