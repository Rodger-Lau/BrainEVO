pass
import torch
import torch.nn as nn
from models.gwn import gwnet

class PNNGWN(nn.Module):
    pass
    def __init__(self, device, num_nodes, in_dim=3, dropout=.3, max_tasks=64):
        super().__init__(); self.device=device; self.num_nodes=num_nodes; self.dropout=dropout
        self.kw=dict(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=dropout)
        self.columns=nn.ModuleList(); self.lateral=nn.ParameterList(); self.active_task=-1
        self.add_task()
    def add_task(self):
        col=gwnet(**self.kw).to(self.device)
        if len(self.columns):
            for p in self.columns[-1].parameters(): p.requires_grad=False
        self.columns.append(col); self.lateral.append(nn.Parameter(torch.tensor(0.1, device=self.device))); self.active_task=len(self.columns)-1
    def set_task(self, task):
        while len(self.columns)<=task: self.add_task()
        self.active_task=task
    def forward(self,x):
        idx=self.active_task
        _,out,act=self.columns[idx](x)
        if idx:
            with torch.no_grad():
                _,old,_=self.columns[idx-1](x)
            out=out+self.lateral[idx]*old
        return None,out,act

class _Adapter(nn.Module):
    def __init__(self, rank=8):
        super().__init__(); self.down=nn.Linear(1,rank); self.act=nn.GELU(); self.up=nn.Linear(rank,1)
    def forward(self,x): return self.up(self.act(self.down(x)))

class HiDePETGWN(nn.Module):
    pass
    def __init__(self, device, num_nodes, in_dim=3, dropout=.3, rank=8, max_tasks=64):
        super().__init__(); self.device=device; self.dropout=dropout
        self.backbone=gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=dropout).to(device)


        self.adapters=nn.ModuleList(); self.scales=nn.ParameterList(); self.active_task=-1
        self.add_task()
    def add_task(self):
        self.adapters.append(_Adapter(8).to(self.device)); self.scales.append(nn.Parameter(torch.tensor(.1,device=self.device))); self.active_task=len(self.adapters)-1
    def set_task(self, task):
        while len(self.adapters)<=task: self.add_task()
        self.active_task=task
    def forward(self,x):
        _,base,act=self.backbone(x); a=self.adapters[self.active_task](base); return None,base+self.scales[self.active_task]*a,act

def build_baseline(kind, device, num_nodes, in_dim=3, dropout=.3):
    if kind == "pnn": return PNNGWN(device,num_nodes,in_dim,dropout)
    if kind == "hidepet": return HiDePETGWN(device,num_nodes,in_dim,dropout)
    return gwnet(device=device,num_nodes=num_nodes,gcn_bool=True,addaptadj=True,in_dim=in_dim,dropout=dropout)
