import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
import gpytorch
from torchdiffeq import odeint
from models.mogp import NGLMCGPModel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim,
                 w=512,
                 num_heads=8,
                 dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, w),
            nn.LayerNorm(w),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(w, out_dim),
            nn.Dropout(dropout),
        )
        
        self.attn_norm = nn.LayerNorm(w)
        self.attn = nn.MultiheadAttention(
            embed_dim=w,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
                
    def forward(self, x, yt):
        in_x = torch.cat([x, yt], -1)

        residual = x
        out = self.net(in_x)

        out = out * self.gate + residual * (1 - self.gate)
        out = out * self.residual_scale
        
        attn_out = self.attn(
            self.attn_norm(out),
            self.attn_norm(out),
            self.attn_norm(out),
            need_weights=False
        )[0]
            
        return out + attn_out

class TimeContextEmbedding(nn.Module):
    def __init__(self, cdim, edim,
                 w=512,
                 num_heads=8,
                 dropout=0.1):
        super(TimeContextEmbedding, self).__init__()
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, edim*4),
            nn.LayerNorm(edim*4),
            nn.SiLU(),
            nn.Linear(edim*4, edim),
        )
        
        self.context_embed = nn.Sequential(
            nn.Linear(cdim, edim*4),
            nn.LayerNorm(edim*4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edim*4, edim),
        )
        
        self.cross_attn = nn.MultiheadAttention(
            edim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.alpha = nn.Parameter(torch.tensor(1e-6))

    def forward(self, y, t):
        t_emb = self.time_embed(t)
        
        y_emb = self.context_embed(y)
        
        yt_emb = self.cross_attn(
            y_emb + self.alpha,
            t_emb,
            t_emb,
            need_weights=False
        )[0]
        
        return F.silu(yt_emb + y_emb + t_emb)

class MLP(nn.Module):
    def __init__(self, dim, cdim, edim,
                 layers=3,
                 w=512,
                 num_heads=8,
                 dropout=0.1):
        super(MLP, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(dim + edim, w),
            nn.LayerNorm(w),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        self.cond_yt = TimeContextEmbedding(cdim, edim, w, num_heads, dropout)
        
        self.blocks = nn.ModuleList([
            ResidualBlock(
                w + edim, w, w, num_heads, dropout
            ) for _ in range(layers)
        ])
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(w + edim),
            nn.SiLU(),
            nn.Linear(w + edim, w),
            nn.SiLU(),
            nn.Linear(w, dim),
        )

    def forward(self, x, y, t):
        yt = self.cond_yt(y, t)
        
        x = torch.cat([x, yt], -1)
        x = self.input_layer(x)
        
        for block in self.blocks:
            x = block(x, yt)
            
        return self.output_layer(torch.cat([x, yt], -1))
    
class FlowModel(pl.LightningModule):  
    def __init__(self, model, forward_model, hparams):
        super(FlowModel, self).__init__()
        self.save_hyperparameters(hparams)
        self.model = model
        self.forward_model = forward_model
        self.forward_model.requires_grad_(False)
        
        self.sigma = self.hparams['model']['sigma']
        self.ndim = self.hparams['model']['ndim']
        self.batch_size = self.hparams['train']['batch_size']
    
    def forward(self, x, y, t):
        out = self.model(x, y, t)
        return out
        
    def training_step(self, batch, batch_idx):
        
        t = torch.rand(self.batch_size, 1).to(self.device)
        
        # Sample from prior over process param space
        x1 = 2*torch.rand((self.batch_size, self.ndim)).to(self.device) - 1
        
        # Evaluate forward linkages
        y = self.forward_model(x1)
        
        mu_t = t*x1
        sigma_t = 1 - (1 - self.sigma)*t
        x = mu_t + sigma_t * torch.randn_like(x1).to(self.device)
        ut = (x1 - (1 - self.sigma)*x)/(1 - (1 - self.sigma)*t) 
        vt = self(x, y, t)    

        loss = torch.mean((vt - ut) ** 2)
        
        self.log("loss", loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 batch_size=self.batch_size,
                 sync_dist=True)
        
        return loss 
        
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
                        lr=self.hparams['train']['lr_init'],
                        weight_decay=self.hparams['train']['wdecay'])
        scheduler = CosineAnnealingLR(optimizer,
                                    self.hparams['train']['n_epoch'],
                                    self.hparams['train']['lr_end'])
                                            
        return [optimizer], [scheduler] 
    
    def on_after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.hparams['train']['clip'])
        
    def train_dataloader(self):
        dummy_data = TensorDataset(torch.zeros(self.hparams['train']['n_steps_epoch'] * self.hparams['train']['batch_size']),
                                   torch.zeros(self.hparams['train']['n_steps_epoch'] * self.hparams['train']['batch_size']))
        return DataLoader(dummy_data, batch_size=self.hparams['train']['batch_size'])

class ForwardModel(nn.Module):
    def __init__(self, fnames_model1, fnames_model2=None):
        super(ForwardModel, self).__init__()
        
        self.likelihood2 = None
        self.model2 = None
        
        likelihood_file = fnames_model1['likelihood']
        model_file = fnames_model1['model']
        likelihood1, model1 = self.load_mogp(likelihood_file,model_file,6,5,5,int(0.10*8000),3)
        self.likelihood1 = likelihood1
        self.model1 = model1

        if fnames_model2 is not None:
            likelihood_file = fnames_model2['likelihood']
            model_file = fnames_model2['model']
            likelihood2, model2 = self.load_mogp(likelihood_file,model_file,4,5,5,int(0.02*8000),5)
            self.likelihood2 = likelihood2
            self.model2 = model2
            
    def load_mogp(self,likelihood_file,model_file,num_mix,num_latents,num_tasks,num_inducing,input_dims):
        
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        model = NGLMCGPModel(num_latents,num_tasks,num_inducing,input_dims,num_mix=None)
        state_dict_model = torch.load(model_file)
        state_dict_likelihood = torch.load(likelihood_file)
        model.load_state_dict(state_dict_model)
        likelihood.load_state_dict(state_dict_likelihood)
        
        return likelihood, model
            
    def forward(self, x):
    
        y = self.likelihood1(self.model1(x)).rsample() 
        if self.model2 is not None:
            y = self.likelihood2(self.model2(y)).rsample()
            
        return y    

class torch_wrapper(nn.Module):
    def __init__(self, model, y):
        super().__init__()
        self.model = model
        self.y = y
        
    def forward(self, t, x, args=None):  
        return self.model(x, self.y, t.repeat(x.shape[0])[:, None])
    
def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ

def hutch_trace(x_out, x_in, noise=None, **kwargs):
    """Hutchinson's trace Jacobian estimator, O(1) call to autograd"""
    jvp = torch.autograd.grad(x_out, x_in, noise, create_graph=True)[0]
    trJ = torch.einsum('bi,bi->b', jvp, noise)

    return trJ    
    
def sample_model(model, y, cdim, ndim, n_sample=2048, n_traj=100):
    x0 = torch.randn((n_sample,ndim), device=model.device)
    t_span = torch.linspace(0, 1, n_traj, device=model.device)
    
    # node = NeuralODE(
    #     torch_wrapper(model, y.expand(n_sample,cdim)),
    #     solver="dopri5", sensitivity="adjoint", atol=1e-5, rtol=1e-5
    #     ).to(model.device)
    # traj = node.trajectory(
    #     x0,
    #     t_span=t_span,
    #     )       
    
    with torch.no_grad():
        traj = odeint(
            torch_wrapper(model, y.expand(n_sample,cdim)),
            x0, t_span, rtol=1e-4, atol=1e-4, method='dopri5'
        )
        
    return traj

def base_log_prob(z):
    return -0.5 * torch.sum(z**2, dim=1)

def compute_log_likelihood(x, y, flow_field, base_log_prob, T=1.0, n_steps=100, method='hutch'):
    # Wrap the model with conditional input y
    flow_wrapper = torch_wrapper(flow_field, y)
    
    # Initial condition is the data point(s) x
    z_0 = x
    # Time spans from T to 0 (reverse integration)
    t_span = torch.linspace(T, 0, n_steps).to(x.device)
    
    # Solve the ODE backward in time (from T to 0)
    z_t = odeint(
        flow_wrapper, 
        z_0, 
        t_span, 
        rtol=1e-4, 
        atol=1e-4, 
        method='dopri5'
    )
    
    # Final state at t=0 (base distribution)
    z_final = z_t[-1]
    log_p_z_final = base_log_prob(z_final)
    
    # Compute integral of trace of Jacobian
    trace_term = 0.0
    for i in range(n_steps - 1):
        t = t_span[i]
        z = z_t[i]
        x_out = flow_wrapper(t, z)
        
        # Choose trace estimation method
        if method == 'hutch':
            noise = torch.randn_like(z)
            trace_jacobian = hutch_trace(x_out, z, noise=noise)
        elif method == 'autograd':
            trace_jacobian = autograd_trace(x_out, z)
        else:
            raise ValueError(f"Unknown trace method: {method}. Trace must be one of ['hutch','autograd']")
        
        dt = t_span[i+1] - t_span[i]
        trace_term += trace_jacobian * dt

    log_p_x = log_p_z_final + trace_term
    return log_p_x