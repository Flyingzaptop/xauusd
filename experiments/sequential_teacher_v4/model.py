import torch
from torch import nn
class Encoder(nn.Module):
    def __init__(self,features,width=32,depth=2,context=8):
        super().__init__();self.proj=nn.Linear(features,width);self.pos=nn.Parameter(torch.zeros(context,width));self.norm=nn.LayerNorm(width)
        self.blocks=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0.,activation='gelu',batch_first=True,norm_first=True),depth,enable_nested_tensor=False)
    def forward(self,x):return self.norm(self.blocks(self.proj(x)+self.pos[:x.shape[1]])[:,-1])
class Actor(nn.Module):
    def __init__(self,features,width=32,depth=2,context=8):
        super().__init__();self.encoder=Encoder(features,width,depth,context);self.policy=nn.Linear(width,3);self.value=nn.Linear(width,1)
        nn.init.zeros_(self.policy.bias);nn.init.normal_(self.policy.weight,std=.01)
    def forward(self,x,mask=None):
        z=self.encoder(x);logits=self.policy(z)
        if mask is not None:logits=logits.masked_fill(~mask,-1e9)
        return torch.distributions.Categorical(logits=logits),self.value(z).squeeze(-1)
class Teacher(nn.Module):
    """Outputs two Beta score distributions. Scores are NOT probabilities of profit."""
    def __init__(self,features,width=32,depth=2,context=8,future_steps=8):
        super().__init__();self.past=Encoder(features,width,depth,context);self.future=Encoder(2,width,depth,future_steps);self.head=nn.Linear(width*2,4)
        nn.init.normal_(self.head.weight,std=.01);nn.init.zeros_(self.head.bias)
    def forward(self,x,future):
        r=self.head(torch.cat([self.past(x),self.future(future)],-1)).float()
        return torch.distributions.Beta(1+nn.functional.softplus(r[:,:2]),1+nn.functional.softplus(r[:,2:]))
