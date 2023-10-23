import math
import os
import time
import h5py
import argparse

import matplotlib.pyplot as plt
import corner
import numpy as np
#import ot as pot
import torch
import torchdyn
import gpytorch
from torchdyn.core import NeuralODE
from scipy.stats import binom
from models.mogp import MultitaskGPModel, NGDMultitaskGPModel
from models.dkl import FeatureExtractor, DKLModel

plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)

parser = argparse.ArgumentParser(description="Continuous Flow")
parser.add_argument("--train", action='store_false', help="train (True) cuda")
parser.add_argument("--load", action='store_true', help="load pretrained model")
parser.add_argument("--n_epoch", default=5000, type=int, help="number of epochs for training RealNVP")
parser.add_argument("--lr_init", default=1e-4, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=1e-10, type=float, help="end learning rate")
parser.add_argument("--batch_size", default=1024, type=int, help="minibatch size")
parser.add_argument("--sigma", default=1e-6, type=float, help="noise")
args = parser.parse_args()

class SinPosEmbed(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w=512):
        super(ResidualBlock, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, w),
            torch.nn.GELU(),
            torch.nn.LayerNorm(w),
            torch.nn.Linear(w, out_dim),
            )
        self.gelu = torch.nn.GELU()
        self.bn = torch.nn.LayerNorm(out_dim)
        
    def forward(self, x, yt):
        in_x = torch.cat([x, yt], -1)
        residual = x
        out = self.net(in_x)
        out += residual
        out = self.gelu(out)
        out = self.bn(out)
        return out
    
class CondTimeEmbed(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w=32):
        super(CondTimeEmbed, self).__init__()
        
        self.sinembed = SinPosEmbed(4)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim + 3, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, out_dim),
            )
        
    def forward(self, y, t):
        temb = self.sinembed(t).squeeze()
        in_x = torch.cat([y, temb], -1)
        out = self.net(in_x)
        return out   
    
class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, edim, out_dim=None, layers=3, w=512):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = dim
        self.first_layer = torch.nn.Sequential(
            torch.nn.Linear(edim + dim, w),
            torch.nn.GELU(),
            torch.nn.LayerNorm(w),
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
    
def plot_trajectories(traj,micro):   
    fig, axes = plt.subplots(3,3, figsize=(12, 12), sharex=False, sharey=False)
    for i in range(3):
        ax = axes[i, i]
        ax.hist(traj[-1,:,i], bins=20, rwidth=0.9, density=True, color="maroon",alpha=1)
        ax.set_yticks([])
    
    for yi in range(3):
        for xi in range(yi):
            ax = axes[yi,xi]
            ax.scatter(traj[0,:,xi], traj[0,:,yi], s=10, alpha=0.6, c="black")
            ax.scatter(traj[:,:,xi], traj[:,:,yi], s=0.2, alpha=0.2, c="slategray")
            ax.scatter(traj[-1,:,xi], traj[-1,:,yi], s=4, alpha=1, c="maroon")
            ax.set_ylabel(r'$\theta_{'+str(yi)+'}$')  
            ax.set_xlabel(r'$\theta_{'+str(xi)+'}$')
            
    for xi in range(3):
        for yi in range(xi):
            ax = axes[yi,xi]
            fig.delaxes(ax)
    plt.savefig('./images/traj_fm_selu' + str(micro) + '.png', bbox_inches='tight') 

def plot_trajectories_1d(traj, t_span,micro):
    fig, axes = plt.subplots(1,3, figsize=(30, 5), sharex=False)
    for i in range(3):
        ax = axes[i]
        ax.plot(t_span, traj[:,:,i], alpha=.05, color='slategray')
    axes[0].set_title("Dim. 0") ; axes[1].set_title("Dim. 1") ; axes[2].set_title("Dim. 3")
    axes[0].set_xlabel("t"); axes[1].set_xlabel("t"); axes[2].set_xlabel("t")
    plt.savefig('./images/traj1d_fm_selu' + str(micro) + '.png', bbox_inches='tight') 
    

def plot_corner_theta(traj,pr_max,pr_min,micro,lbl_theta):
    theta = 0.5*(traj[-1,:,:] + 1)*(pr_max - pr_min) + pr_min
    #theta = traj[-1,:,:]*pr_s + pr_m
    theta = theta.detach().cpu().numpy() 
    fig = corner.corner(theta, labels=lbl_theta, hist_bin_factor=2, smooth=True, truths=presults[micro,:])
    plt.savefig('./images/corner_theta_fm_selu' + str(micro) + '.png', bbox_inches='tight') 

def plot_corner_prop(traj,y_cond,out_m,out_s,micro,lbl_prop):
    y_calc = predict(traj[-1,:,:])
    #y_calc = 0.5*(y_calc + 1)*(out_max - out_min) + out_min
    y_calc = y_calc*out_s + out_m
    y_calc = y_calc.detach().cpu().numpy()
    y_calc[:,:3] = y_calc[:,:3]/1e3
    #y_cond = 0.5*(y_cond + 1)*(out_max - out_min) + out_min
    y_cond = (y_cond*out_s + out_m).squeeze()
    y_cond[:3] = y_cond[:3]/1e3
    fig = corner.corner(y_calc, labels=lbl_prop, hist_bin_factor=2, smooth=False, truths=y_cond.detach().cpu().numpy())
    plt.savefig('./images/corner_prop_fm_selu' + str(micro) + '.png', bbox_inches='tight') 
      
def SBC(L,N,ndim):
    ''' implements simulation-based calibration Gelman (2018) 
    https://arxiv.org/abs/1804.06788 '''
    r = np.zeros((N*L, ndim))
    for i in range(N*L):
        x = 2*torch.rand((1,ndim)).to(device) - 1
        y = predict(x)
        node = NeuralODE(
            torchdyn_wrapper(model, y.expand(L,cdim)), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
            ).to(device)
        with torch.no_grad():
            traj = node.trajectory(
                    torch.randn((L,ndim)).to(device),
                    t_span=torch.linspace(0, 1, 100),
                    )
        xp = traj[-1,...]
        for j in range(ndim):
            r[i,j] = np.sum(1*(xp[:,j] < x[:,j]).detach().cpu().numpy())
        
        if i % 10:
            print(f"Validation: {i+1}/{N*L}")
            
    mean = binom.stats(L*N, 1/L, moments='m')
    lb = binom.ppf(0.005, L*N, 1/L)              
    ub = binom.ppf(0.995, L*N, 1/L)
    
    fig, axes = plt.subplots(1,ndim, figsize=(20, 5), sharex=False)
    for i in range(ndim):
        ax = axes[i]
        ax.hist(r[:,i], color='slategray', rwidth=0.9, bins=L, edgecolor = "gray")
        ax.plot(np.arange(0,L+1,1), mean*np.ones((L+1)),color='maroon')
        ax.fill_between(np.arange(0,L+1,1), lb, ub, color='maroon', alpha=.15)
        ax.set_xlabel(f'Rank Statistic {i}')
    plt.savefig('./images/calibration_fm_selu.png', bbox_inches='tight')   
    
def plot_annotate(output):
    fig, ax = plt.subplots(figsize=(25,20))
    ax.scatter(output[:,0],output[:,1],s=20,color='tab:blue')
    for i, txt in enumerate (np.arange(0,output.shape[0],1)):
        ax.annotate(txt, (output[i,0],output[i,1]))
    plt.xlabel(r'$E_1$')
    plt.ylabel(r'$E_2$') 

    fig, ax = plt.subplots(figsize=(25,20))
    ax.scatter(output[:,0],output[:,1],s=20,color='tab:blue')
    for i, txt in enumerate (np.arange(0,output.shape[0],1)):
        ax.annotate(txt, (output[i,0],output[i,1]))
    plt.xlabel(r'$E_1$')
    plt.ylabel(r'$G_{12}$') 
        
def norm_scaling(x):
    x = torch.from_numpy(x).float().to(device)
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x, m, s

def unit_scaling(x):
    x = torch.from_numpy(x).float().to(device)
    x_min = x.min(0)[0]
    x_max = x.max(0)[0]
    x = 2 * (x - x_min)/(x_max - x_min) - 1
    
    return x, x_min, x_max    

def load_mogp(likelihood_file,model_file,num_mix,num_latents,num_tasks,num_inducing,input_dims):
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = NGDMultitaskGPModel(num_latents,num_tasks,num_inducing,input_dims,num_mix)
    state_dict_model = torch.load(likelihood_file)
    state_dict_likelihood = torch.load(model_file)
    model.load_state_dict(state_dict_model)
    likelihood.load_state_dict(state_dict_likelihood)
    
    return likelihood, model 

def load_data(file):
    with h5py.File(file, "r") as f:
        mresults = f['mech'][()]
        tresults = f['thermal'][()]
        presults = f['params'][()]
    f.close()

    mresults[:, [3, 2]] = mresults[:, [2, 3]]
    mresults = np.hstack((mresults[:,1:3], mresults[:,5][...,None]))                     
    output = np.hstack((mresults,tresults[:,1:]))
   
    return output, presults

def predict(x):
    ps_samples = likelihood_ps(model_ps(x)).rsample() 
    y = likelihood_sp(model_sp(ps_samples)).rsample()
    
    return y  
   
device = torch.device("cuda")    

output, presults = load_data("abq_results_memphis.h5")
pr_scaled, pr_min, pr_max = unit_scaling(presults)
#output, out_min, out_max = unit_scaling(output)
#pr_scaled, pr_m, pr_s = norm_scaling(presults)
output, out_m, out_s = norm_scaling(output)

numPCs = 5
pcs = np.load('microsPCs_memphis.npy')[:,:numPCs]
pcs, pcs_m, pcs_s = norm_scaling(pcs)
pcs_min = pcs.min(0)[0]
pcs_max = pcs.max(0)[0]

microindx_array = [7216,4027,9682,131]

# Load in linkages    
likelihood_file = 'mogp_model_state_psNGc.pth'#_unit.pth'
model_file = 'mogp_likelihood_state_psNGc.pth'#_unit.pth'
likelihood_ps, model_ps = load_mogp(likelihood_file,model_file,4,3,3,int(0.1*7992),3)
likelihood_ps = likelihood_ps.to(device)
model_ps = model_ps.to(device)

likelihood_file = 'mogp_model_state_spNGc.pth'#_unit.pth'
model_file = 'mogp_likelihood_state_spNGc.pth'#_unit.pth'
likelihood_sp, model_sp = load_mogp(likelihood_file,model_file,4,5,5,int(0.02*7992),3)
likelihood_sp = likelihood_sp.to(device)
model_sp = model_sp.to(device)

ndim = 3
cdim = 5
layers = 3
width = 512
model = MLP(dim=ndim, cdim=cdim, edim=32, layers=layers, w=width).to(device)

#cnf_fname = f'fm_gelu_{layers+2}_{width}_ema_sin_200k_norm.pth'
cnf_fname = f'fm_gelu_{layers+2}_{width}_ema_sin_norm.pth'
print('Total Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if args.load:
    state_dict = torch.load(cnf_fname, map_location=device)
    model.load_state_dict(state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_init)
ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
0.1 * averaged_model_parameter + 0.9 * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)

swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
                                            swa_lr=args.lr_end,
                                            anneal_epochs=args.n_epoch,
                                            anneal_strategy='cos',
                                            last_epoch=-1)

if args.train:
    lr = []
    for k in range(args.n_epoch):
        optimizer.zero_grad()
        t = torch.rand(args.batch_size, 1).to(device)
        x1 = 2*torch.rand((args.batch_size,ndim)).to(device) - 1
        y = predict(x1)
        
        # flow matching loss Lipman (2023)
        mu_t = t*x1
        sigma_t = 1 - (1 - args.sigma)*t
        x = mu_t + sigma_t * torch.randn(args.batch_size, ndim).to(device)
        ut = (x1 - (1 - args.sigma)*x)/(1 - (1 - args.sigma)*t) 
        vt = model(x, y, t)    

        loss = torch.mean((vt - ut) ** 2)
        loss.backward()
        optimizer.step()
        
        ema_model.update_parameters(model)
        swa_scheduler.step()

        if (k + 1) % 50 == 0:
            print(f"{k+1}: loss {loss.item():0.3f} - lr {swa_scheduler.get_last_lr()[0]:0.3e}")
        
        if (k + 1) % 5000 == 0:
            torch.save(model.state_dict(), cnf_fname)
        
# Compute trajectories and plot
n_sample = 2048
lbl_prop = [r'$E_1$ (GPa)',r'$E_2$ (GPa)', r'$G_{12}$ (GPa)', r'$k_1$ (W/mK)', r'$k_2$ (W/mK)']
lbl_theta = [r'$\theta_{0}$',r'$\theta_{1}$',r'$\theta_{2}$']
    
for micro in microindx_array:
    y_cond = output[micro,:]

    node = NeuralODE(
        torchdyn_wrapper(model, y_cond.expand(n_sample,cdim)), solver="dopri5", sensitivity="adjoint", atol=1e-5, rtol=1e-5
        ).to(device)

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn((n_sample,ndim)).to(device),
            t_span=torch.linspace(0, 1, 100),
            )
        plot_trajectories(traj.detach().cpu().numpy(),micro)
        #plot_trajectories_1d(traj.detach().cpu().numpy(), torch.linspace(0, 1, 100))
        plot_corner_theta(traj,pr_max,pr_min,micro,lbl_theta)
        plot_corner_prop(traj,y_cond,out_m,out_s,micro,lbl_prop)

SBC(11,100,ndim)

output, presults = load_data("abq_results_hi.h5")
output[:,:3] = output[:,:3]/1e3
fig = corner.corner(output, labels=lbl_prop, hist_bin_factor=2, smooth=False)
axes = np.array(fig.axes).reshape((output.shape[1], output.shape[1]))
for micro in microindx_array:
    for yi in range(cdim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.plot(output[micro,xi], output[micro,yi], "o",label=micro)

ax.legend(bbox_to_anchor=(0., 2.0, 2., .0), loc=4)
plt.savefig('./images/corner_cases.png', bbox_inches='tight') 

#plot_annotate(output)


