import os
import scipy.io as sio
import numpy as np
import h5py
import torch
import gpytorch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from scipy import interpolate
from matplotlib import pyplot as plt
from models.mogp import MultitaskGPModel, NGDMultitaskGPModel, NGDMultitaskGPModelNorm
import argparse

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 22}
plt.rc('font', **font)

parser = argparse.ArgumentParser(description="Deep Probabilistic Inverse Microstructure Training beta-VAE")
parser.add_argument("--train", action='store_false', help="train (True) cuda")
parser.add_argument("--load", action='store_true', help="load pretrained model")
parser.add_argument("--batch", default=1024, type=int, help="minibatch training size")
parser.add_argument("--num_latent", default=5, type=int, help="# latent GPs")
parser.add_argument("--num_inducing", default=0.02, type=float, help="% inducing points of training set size")
parser.add_argument("--num_epochs", default=4000, type=int, help="# training epochs")
parser.add_argument("--num_mix", default=4, type=int, help="# spectral mixture")
parser.add_argument("--lr_init", default=1e-2, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=0, type=float, help="end learning rate")
args = parser.parse_args()

#device = torch.device("cuda" if (torch.cuda.is_available() and args.train) else "cpu")
device = torch.device("cuda")

###############################################################################
numPCs = 5
#pcs = np.load('microsPCs_autocorr.npy')[:,:numPCs]
pcs = np.load('microsPCs_memphis.npy')[:,:numPCs]

def normScaling(x):
    x = torch.from_numpy(x).float().to(device)
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    print(x.max(0)[0])
    print(x.min(0)[0])
    x = x.detach().cpu().numpy()
    
    return x, m, s
    
def unitScaling(x):
    x = torch.from_numpy(x).float().to(device)
    x_min = x.min(0)[0]
    x_max = x.max(0)[0]
    x = 2 * (x - x_min)/(x_max - x_min) - 1
    x = x.detach().cpu().numpy()
    
    return x, x_min, x_max 

# Standard scaling
pcs, m, s = normScaling(pcs)

with h5py.File("abq_results_memphis.h5", "r") as f:
    print("Keys: %s" % f.keys())
    mresults = f['mech'][()]
    tresults = f['thermal'][()]
    presults = f['params'][()]
    print(mresults.shape)
    print(tresults.shape)
    print(presults.shape)
    f.close()

mresults[:, [3, 2]] = mresults[:, [2, 3]]
mresults = np.hstack((mresults[:,1:3], mresults[:,5][...,None]))                     
output = np.hstack((mresults,tresults[:,1:]))
ndim = output.shape[1]

print(output.shape)
output, mo, so = normScaling(output)
#output, output_min, output_max = unitScaling(output)

print(pcs.min())
print(pcs.max())
###############################################################################
xtr, xte, ytr, yte = train_test_split(pcs, output, test_size=0.2, random_state=2)#17

train_x = torch.from_numpy(xtr).float().to(device)
test_x = torch.from_numpy(xte).float().to(device)
train_y = torch.from_numpy(ytr).float().to(device)
test_y = torch.from_numpy(yte).float().to(device)

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch, drop_last=False, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch, drop_last=False, shuffle=True)
###############################################################################
#   Train MOGP model for curve points
###############################################################################
num_latents = args.num_latent
num_tasks = train_y.shape[1]
num_inducing = int(args.num_inducing*train_x.shape[0])
input_dims = train_x.shape[1]
choel_mean_init = torch.std(train_y,0).mean()
num_mix = args.num_mix

#print(f'num_latents: {num_latents}')
#print(f'num_tasks: {num_tasks}')
#print(f'num_inducing: {num_inducing}')
#print(f'num_mix: {num_mix}')

model = NGDMultitaskGPModel(num_latents,num_tasks,num_inducing,input_dims,num_mix).to(device)
#model.covar_module.initialize_from_data(train_x, train_y) 
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)

print('Likelihood Parameters: ' + str(sum(p.numel() for p in likelihood.parameters() if p.requires_grad)))
print('Model Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

model_fname = 'mogp_model_state_spNGsm.pth'
lik_fname = 'mogp_likelihood_state_spNGsm.pth'

if args.load:
    state_dict_model = torch.load(model_fname, map_location=device)
    state_dict_likelihood = torch.load(lik_fname, map_location=device)
    model.load_state_dict(state_dict_model)
    likelihood.load_state_dict(state_dict_likelihood)
    
if args.train:
    model.train()
    likelihood.train()
   
    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=args.lr_init)

    hyperparameter_optimizer = torch.optim.Adam([
            {'params': model.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=args.lr_init)
    

    num_epochs = args.num_epochs
    hyperparameter_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hyperparameter_optimizer,num_epochs,0)
    variational_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(variational_ngd_optimizer,num_epochs,0)
    #hyperparameter_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(hyperparameter_optimizer, int(2500), 1, args.lr_end)
    #variational_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(variational_ngd_optimizer, int(2500), 1, args.lr_end)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))    # loss object ELBO
    loss_list = []
    for i in range(num_epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()
            batch_losses.append(loss.cpu().detach())

        hyperparameter_scheduler.step()
        variational_scheduler.step()
        loss_mean = np.mean(batch_losses)
    
        if (i + 1) % 5 == 0:
            print(f"epoch: {(i+1):}, loss: {loss_mean:.5f}, lr: {hyperparameter_scheduler.get_last_lr()[0]:.5f}")
            loss_list.append(loss_mean)
            
        if (i + 1) % 500 == 0:
            # Save model
            torch.save(model.state_dict(), model_fname)
            torch.save(likelihood.state_dict(), lik_fname)

# Make predictions
m = mo.detach().cpu().numpy()
s = so.detach().cpu().numpy()

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    train_mean = torch.zeros((train_y.shape))
    trainlower = torch.zeros((train_y.shape))
    trainupper = torch.zeros((train_y.shape))
    for i in range(int(train_y.shape[0])):
        train_predictions = likelihood(model(train_x[i:(i+1),:]))
        train_mean[i:(i+1),:] = train_predictions.mean.cpu()*s + m
        trainlower[i:(i+1),:], trainupper[i:(i+1),:] = train_predictions.confidence_region()
        trainupper[i:(i+1),:] = trainupper[i:(i+1),:].cpu()*s + m
        trainlower[i:(i+1),:] = trainlower[i:(i+1),:].cpu()*s + m
        
    test_mean = torch.zeros((test_y.shape))
    testlower = torch.zeros((test_y.shape))
    testupper = torch.zeros((test_y.shape))
    for i in range(int(test_y.shape[0])):
        test_predictions = likelihood(model(test_x[i:(i+1),:]))
        test_mean[i:(i+1),:] = test_predictions.mean.cpu()*s + m
        testlower[i:(i+1),:], testupper[i:(i+1),:] = test_predictions.confidence_region()
        testupper[i:(i+1),:] = testupper[i:(i+1),:].cpu()*s + m
        testlower[i:(i+1),:] = testlower[i:(i+1),:].cpu()*s + m  
    
#train_mean = scaler_out.inverse_transform(train_mean)
#test_mean = scaler_out.inverse_transform(test_mean)
#train_y = scaler_out.inverse_transform(train_y.detach().cpu().numpy())
#test_y = scaler_out.inverse_transform(test_y.detach().cpu().numpy())
 
train_mean = train_mean.detach().cpu().numpy()
test_mean = test_mean.detach().cpu().numpy()
train_y = train_y.detach().cpu().numpy()*s + m
test_y = test_y.detach().cpu().numpy()*s + m

for i in [0,1,2]:
    train_mean[:,i] = train_mean[:,i]/1e3
    test_mean[:,i] = test_mean[:,i]/1e3
    trainlower[:,i] = trainlower[:,i]/1e3
    trainupper[:,i] = trainupper[:,i]/1e3
    testlower[:,i] = testlower[:,i]/1e3
    testupper[:,i] = testupper[:,i]/1e3
    train_y[:,i] = train_y[:,i]/1e3
    test_y[:,i] = test_y[:,i]/1e3
    
# Parity plots every 10 points
#albl = [r'Actual $E_1$ (GPa)',r'Actual $E_2$ (GPa)', r'Actual $G_{12}$ (GPa)', r'Actual $k_1$ (W/mK)',r'Actual $k_2$ (W/mK)']
#plbl = [r'Predicted $E_1$ (GPa)',r'Predicted $E_2$ (GPa)', r'Predicted $G_{12}$ (GPa)', r'Predicted $k_1$ (W/mK)',r'Predicted $k_2$ (W/mK)']
title = [r'$E_1$ (GPa)',r'$E_2$ (GPa)', r'$G_{12}$ (GPa)', r'$k_1$ (W/mK)',r'$k_2$ (W/mK)']
fig, axes = plt.subplots(1,ndim, figsize=(25, 5), sharex=False)
for i in range(ndim):
    ax = axes[i]
    ax.errorbar(train_y[:,i],train_mean[:,i],yerr=(trainupper[:,i].numpy()-trainlower[:,i].numpy()),fmt='o',capsize=2,color='tab:blue',label=r'$\alpha_{train}$',alpha=0.2)
    ax.errorbar(test_y[:,i],test_mean[:,i],yerr=(testupper[:,i].numpy()-testlower[:,i].numpy()),fmt='o',capsize=2,color='tab:red',label=r'$\alpha_{test}$',alpha=0.2)
    ax.plot([np.min(train_y[:,i]),np.max(train_y[:,i])],
                 [np.min(train_y[:,i]),np.max(train_y[:,i])],'k--',linewidth=3) 
    ax.set_title(title[i])
    ax.set_xlabel('Actual')
    if i == 0: ax.set_ylabel('Predicted')
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
axes[0].legend()
plt.tight_layout()
plt.savefig('gpyPoints_sp.png', bbox_inches='tight')   

print('*******Mean*******')
nmae_train = np.mean(abs(train_mean - train_y)/np.mean(abs(train_y),axis=0),axis=0)*100
nmae_test = np.mean(abs(test_mean - test_y)/np.mean(abs(train_y),axis=0),axis=0)*100
print(nmae_train)
print(nmae_test)
print('**************')
mae_train = np.mean(abs(train_mean - train_y),axis=0)
mae_test = np.mean(abs(test_mean - test_y),axis=0)
print(mae_train)
print(mae_test)
print('**************')
mse_train = np.mean((train_mean - train_y)**2,axis=0)
mse_test = np.mean((test_mean - test_y)**2,axis=0)
print(mse_train)
print(mse_test)
print('*******Std*******')
nmae_std_train = np.std(abs(train_mean - train_y)/np.mean(abs(train_y),axis=0),axis=0)*100
nmae_std_test = np.std(abs(test_mean - test_y)/np.mean(abs(train_y),axis=0),axis=0)*100
print(nmae_std_train)
print(nmae_std_test)
print('**************')
mae_std_train = np.std(abs(train_mean - train_y),axis=0)
mae_std_test = np.std(abs(test_mean - test_y),axis=0)
print(mae_std_train)
print(mae_std_test)

fig = plt.figure(figsize=(15, 12.5))
plt.plot(np.arange(0,ndim,1),nmae_train,linestyle='-', marker='o',)
plt.plot(np.arange(0,ndim,1),nmae_test,linestyle='-', marker='o',)
plt.ylabel(r'$NMAE$') 
plt.legend(['Train','Test'])
plt.tight_layout()
plt.savefig('gpyNMAE_sp.png', bbox_inches='tight')