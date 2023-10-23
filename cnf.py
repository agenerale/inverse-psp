import argparse
import numpy as np
import torch
import h5py
from torchdyn.core import NeuralODE
from models.cnf import MLP,torchdyn_wrapper
from helpers.function_helpers import unit_scaling,norm_scaling,load_mogp,load_data
from helpers.plotting_helpers import plot_trajectories,plot_corner_theta,plot_corner_prop,SBC,plot_cases_prop,plot_cases_pcs

import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)

parser = argparse.ArgumentParser(description="Continuous Normalizing Flow")
parser.add_argument("--train", action='store_true', help="train")
parser.add_argument("--load", action='store_false', help="load pretrained model")
parser.add_argument("--n_epoch", default=1000, type=int, help="number of epochs for training")
parser.add_argument("--lr_init", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--lr_end", default=1e-10, type=float, help="end learning rate")
parser.add_argument("--batch_size", default=1024, type=int, help="minibatch size")
parser.add_argument("--sigma", default=1e-3, type=float, help="noise")
args = parser.parse_args()

device = torch.device("cuda")    

# Load in dataset and specify target process params/properties
output, presults = load_data("abq_results_memphis.h5")
pr_scaled, pr_min, pr_max = unit_scaling(presults,device)
output, out_m, out_s = norm_scaling(output,device)

microindx_array = [4027,131]

# Load in linkages    
likelihood_file = 'mogp_model_state_psNG.pth'
model_file = 'mogp_likelihood_state_psNG.pth'
likelihood_ps, model_ps = load_mogp(likelihood_file,model_file,6,5,5,int(0.10*8000),3)
likelihood_ps = likelihood_ps.to(device)
model_ps = model_ps.to(device)

likelihood_file = 'mogp_model_state_spNG.pth'
model_file = 'mogp_likelihood_state_spNG.pth'
likelihood_sp, model_sp = load_mogp(likelihood_file,model_file,4,5,5,int(0.02*8000),5)
likelihood_sp = likelihood_sp.to(device)
model_sp = model_sp.to(device)

def predict(x):
    ps_samples = likelihood_ps(model_ps(x)).rsample() 
    y = likelihood_sp(model_sp(ps_samples)).rsample()
    
    return y 

ndim = 3
cdim = 5
layers = 2
width = 256
model = MLP(dim=ndim, cdim=cdim, edim=16, layers=layers, w=width).to(device)

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
model.eval()

n_sample = 2048
n_traj = 100
lbl_prop = [r'$E_1$ (GPa)',r'$E_2$ (GPa)', r'$G_{12}$ (GPa)', r'$k_1$ (W/mK)', r'$k_2$ (W/mK)']
lbl_theta = [r'$\theta_{1}$',r'$\theta_{2}$',r'$\theta_{3}$']
    
traj_acc = torch.zeros((len(microindx_array),n_traj,n_sample,ndim)).to(device)
for micro in microindx_array:
    y_cond = output[micro,:]
    indx = microindx_array.index(micro)

    node = NeuralODE(
        torchdyn_wrapper(model, y_cond.expand(n_sample,cdim)), solver="dopri5", sensitivity="adjoint", atol=1e-5, rtol=1e-5
        ).to(device)

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn((n_sample,ndim)).to(device),
            t_span=torch.linspace(0, 1, n_traj),
            )      
        traj_acc[indx,...] = traj
        plot_trajectories(traj,pr_max,pr_min,micro,presults)
        plot_corner_theta(traj,pr_max,pr_min,micro,lbl_theta,presults)
        plot_corner_prop(traj,predict,y_cond,out_m,out_s,micro,lbl_prop)
        
SBC(model,predict,ndim,cdim,device,N=2000,L=100,
        n_bins=None,
        bin_interval=0.95,
        stacked=False,
        fig_size=None,
        param_names=None,
        difference=False)
