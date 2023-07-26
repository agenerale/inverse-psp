import math
import os
import time
import h5py
import argparse

import matplotlib.pyplot as plt
import corner
import numpy as np
import torch
import torchdyn
import gpytorch
from torchdyn.core import NeuralODE
from scipy.stats import binom
from models.mogp import MultitaskGPModel, NGDMultitaskGPModel
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)

parser = argparse.ArgumentParser(description="Continuous Flow")
parser.add_argument("--train", action='store_false', help="train (True) cuda")
parser.add_argument("--load", action='store_true', help="load pretrained model")
parser.add_argument("--n_epoch", default=200000, type=int, help="number of epochs for training RealNVP")
parser.add_argument("--lr_init", default=1e-4, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=1e-10, type=float, help="end learning rate")
parser.add_argument("--batch_size", default=1024, type=int, help="minibatch size")
parser.add_argument("--sigma", default=1e-6, type=float, help="noise")
parser.add_argument("--swa_start", default=40000, type=float, help="SWA")
args = parser.parse_args()

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w=512):
        super(ResidualBlock, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, w),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(w),
            torch.nn.Linear(w, out_dim),
            )
        self.gelu = torch.nn.GELU()
        self.bn = torch.nn.BatchNorm1d(out_dim)
        
    def forward(self, x, yt):
        in_x = torch.cat([x, yt], -1)
        residual = x
        out = self.net(in_x)
        out += residual
        out = self.gelu(out)
        out = self.bn(out)
        return out
    
class cEmbed(torch.nn.Module):
    def __init__(self, in_dim, out_dim, w=32):
        super(cEmbed, self).__init__()
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, w),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(w),
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(w),
            torch.nn.Linear(w, out_dim),
            torch.nn.GELU(),   
            torch.nn.BatchNorm1d(out_dim),
            )
        
    def forward(self, y, t):
        in_x = torch.cat([y, t], -1)
        out = self.net(in_x)
        return out  
    
class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, out_dim=None, layers=3, w=512):
        super(MLP, self).__init__()
        if out_dim is None:
            out_dim = dim
        self.first_layer = torch.nn.Sequential(
            torch.nn.Linear(dim + cdim + 1, w),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(w),
            )
        
        self.cond = cEmbed(cdim+1,cdim+1)
        
        self.blocks = torch.nn.ModuleList()       
        for i in range(layers):
            self.blocks.append(ResidualBlock(w + cdim + 1, w, w))
            
        self.last_layer = torch.nn.Linear(w + cdim + 1, out_dim)
            
    def forward(self, x, y, t):
        yt = self.cond(y, t)
        #yt = torch.cat([y, t], -1)
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
    theta = theta.detach().cpu().numpy() 
    fig = corner.corner(theta, labels=lbl_theta, hist_bin_factor=2, smooth=True, truths=presults[micro,:])
    plt.savefig('./images/corner_theta_fm_selu' + str(micro) + '.png', bbox_inches='tight') 

def plot_corner_prop(traj,y_cond,out_max,out_min,micro,lbl_prop):
    y_calc = predict(traj[-1,:,:])
    y_calc = 0.5*(y_calc + 1)*(out_max - out_min) + out_min
    y_calc = y_calc.detach().cpu().numpy()
    y_calc[:,:2] = y_calc[:,:2]/1e3
    y_cond = 0.5*(y_cond + 1)*(out_max - out_min) + out_min
    y_cond[:2] = y_cond[:2]/1e3
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
                t_span=torch.linspace(0, 1, 5),
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

def load_mogp(likelihood_file,model_file):
    num_latents = 3
    num_tasks = 3
    num_inducing = int(0.02*7992)
    input_dims = 3   
    
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = NGDMultitaskGPModel(num_latents,num_tasks,num_inducing,input_dims,1e-6)
    state_dict_model = torch.load(likelihood_file)
    state_dict_likelihood = torch.load(model_file)
    model.load_state_dict(state_dict_model)
    likelihood.load_state_dict(state_dict_likelihood)
    
    return likelihood, model

def load_data(file):
    with h5py.File(file, "r") as f:
        #print("Keys: %s" % f.keys())
        mresults = f['mech'][()]
        tresults = f['thermal'][()]
        presults = f['params'][()]
        f.close()

    mresults[:, [3, 2]] = mresults[:, [2, 3]]
    mresults_mean = np.array([np.mean(mresults[:,1:2],axis=1),
                              mresults[:,5]]).T
    tresults_mean = np.mean(tresults[:,1:],axis=1)[...,None]
    output = np.hstack((mresults_mean,tresults_mean))
    
    return output, presults

def predict(x):
    ps_samples = likelihood_ps(model_ps(x)).rsample() 
    ps_samples[ps_samples < -1] = -1
    ps_samples[ps_samples > 1] = 1
    y = likelihood_sp(model_sp(ps_samples)).rsample()
    
    return y  
   
device = torch.device("cuda")    
   
output, presults = load_data("abq_results.h5")
pr_scaled, pr_min, pr_max = unit_scaling(presults)
output, out_min, out_max = unit_scaling(output)

#microindx_array = [7838,2498,3882,4148,1005]
microindx_array = [7838,2498,3882,409,9473]#1005]

# Load in linkages    
likelihood_file = "mogp_model_state_psNG_unit.pth"
model_file = "mogp_likelihood_state_psNG_unit.pth"
likelihood_ps, model_ps = load_mogp(likelihood_file,model_file)
likelihood_ps = likelihood_ps.to(device)
model_ps = model_ps.to(device)

likelihood_file = 'mogp_model_state_spNG_unit.pth'
model_file = 'mogp_likelihood_state_spNG_unit.pth'
likelihood_sp, model_sp = load_mogp(likelihood_file,model_file)
likelihood_sp = likelihood_sp.to(device)
model_sp = model_sp.to(device)

ndim = 3
cdim = 3
layers = 3
width = 512
model = MLP(dim=ndim, cdim=cdim, layers=layers, w=width).to(device)

cnf_fname = f'fm_swish_{layers+2}_{width}_ema_res_bn.pth'
#cnf_fname = 'fm_selu.pth'
print('Total Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if args.load:
    state_dict = torch.load(cnf_fname, map_location=device)
    model.load_state_dict(state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr_init)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.n_epoch-args.swa_start,args.lr_end)

ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
0.1 * averaged_model_parameter + 0.9 * model_parameter
ema_model = torch.optim.swa_utils.AveragedModel(model, avg_fn=ema_avg)
swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, swa_lr=args.lr_end)

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
        
        if k > args.n_epoch-args.swa_start:
            ema_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        
        if (k + 1) % 50 == 0:
            #print(f"{k+1}: loss {loss.item():0.3f} - lr {optimizer.param_groups[0]['lr']:0.3e}")
            print(f"{k+1}: loss {loss.item():0.3f} - lr {scheduler.get_last_lr()[0]:0.3e}")
        
        if (k + 1) % 5000 == 0:
            torch.save(model.state_dict(), cnf_fname)
            
        #lr.append(optimizer.param_groups[0]['lr'])

#plt.figure(figsize=(7.5,6.5))
#plt.plot(np.arange(0,len(lr),1),lr)
#plt.savefig('./lr.png', bbox_inches='tight')
        
# Compute trajectories and plot
n_sample = 2048
lbl_prop = [r'$E$ (GPa)', r'$G$ (GPa)', r'$k$ (W/mK)']
lbl_theta = [r'$\theta_{0}$',r'$\theta_{1}$',r'$\theta_{2}$']
    
for micro in microindx_array:
    y_cond = output[micro,:]

    node = NeuralODE(
        torchdyn_wrapper(model, y_cond.expand(n_sample,cdim)), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        ).to(device)

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn((n_sample,ndim)).to(device),
            t_span=torch.linspace(0, 1, 100),
            )
        plot_trajectories(traj.detach().cpu().numpy(),micro)
        #plot_trajectories_1d(traj.detach().cpu().numpy(), torch.linspace(0, 1, 100))
        plot_corner_theta(traj,pr_max,pr_min,micro,lbl_theta)
        plot_corner_prop(traj,y_cond,out_max,out_min,micro,lbl_prop)

#SBC(11,100,ndim)