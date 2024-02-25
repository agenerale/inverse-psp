import math
import torch
import torchdyn
from torchdyn.core import NeuralODE
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w=512):
        super(ResidualBlock, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, w),
            torch.nn.LayerNorm(w),
            torch.nn.GELU(),
            torch.nn.Linear(w, out_dim),
            )
        
        self.bn = torch.nn.LayerNorm(out_dim)
        self.gelu = torch.nn.GELU()
        
    def forward(self, x, yt):
        in_x = torch.cat([x, yt], -1)
        residual = x
        out = self.net(in_x)
        out += residual
        out = self.bn(out)
        out = self.gelu(out)
        return out
    
class CondTimeEmbed(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w=16):
        super(CondTimeEmbed, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, w),
            torch.nn.LayerNorm(w),
            torch.nn.GELU(),
            torch.nn.Linear(w, out_dim),
            )
        
    def forward(self, y, t):
        in_x = torch.cat([y, t], -1)
        out = self.net(in_x)
        return out   
    
class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, edim, out_dim=None, layers=3, w=512):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = dim
        self.first_layer = torch.nn.Sequential(
            torch.nn.Linear(edim + dim, w),
            torch.nn.LayerNorm(w),
            torch.nn.GELU(),
            )
        
        self.cond = CondTimeEmbed(cdim+1,edim)
        
        self.blocks = torch.nn.ModuleList()       
        for i in range(layers):
            self.blocks.append(ResidualBlock(w + edim, w, w))
            
        self.last_layer = torch.nn.Linear(w + edim, out_dim)
            
    def forward(self, x, y, t):
        yt = self.cond(y, t)
        in_x = torch.cat([x, yt], -1)
        out = self.first_layer(in_x)
        
        for i in range(len(self.blocks)):
            out = self.blocks[i](out, yt)
        
        in_x = torch.cat([out, yt], -1)
        out = self.last_layer(in_x)
        
        return out
    
class torchdyn_wrapper(torch.nn.Module):
    def __init__(self, model, y):
        super().__init__()
        self.model = model
        self.y = y
        
    def forward(self, t, x, args=None):  
        return self.model(x, self.y, t.repeat(x.shape[0])[:, None])
    
class GradModel(torch.nn.Module):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def forward(self, x):
        x = x.requires_grad_(True)
        grad = torch.autograd.grad(torch.sum(self.action(x)), x, create_graph=True)[0]
        return grad[:, :-1] 