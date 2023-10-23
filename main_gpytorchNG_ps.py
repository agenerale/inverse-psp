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
from models.mogp import NGDMultitaskGPModel, NGDMultitaskGPModelNorm
import argparse

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
font = {'family' : 'serif','weight' : 'normal','size'   : 22}
plt.rc('font', **font)

parser = argparse.ArgumentParser(description="Deep Probabilistic Inverse Microstructure Training beta-VAE")
parser.add_argument("--train", action='store_false', help="train (True) cuda")
parser.add_argument("--load", action='store_true', help="load pretrained model")
parser.add_argument("--batch", default=2048, type=int, help="minibatch training size")
parser.add_argument("--num_latent", default=5, type=int, help="# latent GPs")
parser.add_argument("--num_inducing", default=0.1, type=float, help="% inducing points of training set size")
parser.add_argument("--num_epochs", default=10000, type=int, help="# training epochs")
parser.add_argument("--lr_init", default=1e-2, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=0, type=float, help="end learning rate")
args = parser.parse_args()

device = torch.device("cuda")# if (torch.cuda.is_available() and args.train) else "cpu")

###############################################################################
numPCs = 5
pcs = np.load('./data/microsPCs_memphis.npy')[:,:numPCs]

with h5py.File("./data/abq_results_memphis.h5", "r") as f:
    print("Keys: %s" % f.keys())
    mresults = f['mech'][()]
    tresults = f['thermal'][()]
    presults = f['params'][()]
    print(mresults.shape)
    print(tresults.shape)
    print(presults.shape)
    f.close()
    
presults[:,:2] = np.log(presults[:,:2]) # scale to uniform distributions

# Manual scaling -1 to 1
presults = torch.from_numpy(presults).float().to(device)
presults_orig = presults
presults_min = presults.min(0)[0]
presults_max = presults.max(0)[0]
presults = 2 * (presults - presults_min)/(presults_max - presults_min) - 1
presults = presults.detach().cpu().numpy()

#pcs = torch.from_numpy(pcs).float().to(device)
#pcs_orig = pcs
#pcs_min = pcs.min(0)[0]
#pcs_max = pcs.max(0)[0]
#pcs = 2 * (pcs - pcs_min)/(pcs_max - pcs_min) - 1
#pcs = pcs.detach().cpu().numpy()

# Standard scaling
pcs = torch.from_numpy(pcs).float().to(device)
m = pcs.mean(0, keepdim=True)
s = pcs.std(0, unbiased=False, keepdim=True)
pcs -= m
pcs /= s 
pcs = pcs.detach().cpu().numpy()

output = pcs

# Standard scaling
#from sklearn.preprocessing import MinMaxScaler
#from pickle import dump
#scaler = StandardScaler()
#scaler.fit(z)
#z = scaler_out.transform(z)
#z = torch.from_numpy(z).float().to(device)
#dump(scaler_in, open('scaler_in.pkl', 'wb'))

print(presults_min)
print(presults_max)
print(presults.min())
print(presults.max())
###############################################################################
xtr, xte, ytr, yte = train_test_split(presults, output, test_size=0.2, random_state=10)#17

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
#choel_mean_init = torch.std(train_y,0).mean()
num_mix = 10#6

model = NGDMultitaskGPModelNorm(num_latents,num_tasks,num_inducing,input_dims,num_mix).to(device)
#model.covar_module.initialize_from_data_empspect(train_x, train_y) 
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks).to(device)

print('Likelihood Parameters: ' + str(sum(p.numel() for p in likelihood.parameters() if p.requires_grad)))
print('Model Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

model_fname = 'mogp_model_state_psNGsm_10m.pth'
lik_fname = 'mogp_likelihood_state_psNGsm_10m.pth'

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
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(hyperparameter_optimizer, int(5000), 1, args.lr_end)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))    # loss object ELBO
    loss_list = []
    loss_list_test = []
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
        
        # Test
        batch_losses_test = []
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            loss = -mll(output, y_batch)
            batch_losses_test.append(loss.cpu().detach())   
            
        loss_mean_test = np.mean(batch_losses_test)
        
        if (i + 1) % 5 == 0:
            print(f"epoch: {(i+1):}, loss: {loss_mean:.5f}, lr: {hyperparameter_scheduler.get_last_lr()[0]:.5f}")
            loss_list.append(loss_mean)
            loss_list_test.append(loss_mean_test)
            
        if (i + 1) % 500 == 0:
            # Save model
            torch.save(model.state_dict(), model_fname)
            torch.save(likelihood.state_dict(), lik_fname)

# Make predictions
m = m.detach().cpu().numpy()
s = s.detach().cpu().numpy()

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

# Parity plots every 10 points
fig, axes = plt.subplots(1,numPCs, figsize=(25, 5), sharex=False)
for i in range(numPCs):
    ax = axes[i]
    ax.errorbar(train_y[:,i],train_mean[:,i],yerr=(trainupper[:,i].numpy()-trainlower[:,i].numpy()),fmt='o',capsize=2,color='tab:blue',label=r'$\theta_{train}$',alpha=0.2)
    ax.errorbar(test_y[:,i],test_mean[:,i],yerr=(testupper[:,i].numpy()-testlower[:,i].numpy()),fmt='o',capsize=2,color='tab:red',label=r'$\theta_{test}$',alpha=0.2)
    ax.plot([np.min(train_y[:,i]),np.max(train_y[:,i])],
                 [np.min(train_y[:,i]),np.max(train_y[:,i])],'k--',linewidth=3) 
    #ax.set_xlabel(r'Actual $\alpha_{'+str(int(i+1))+'}$')
    #ax.set_ylabel(r'Predicted $\alpha_{'+str(int(i+1))+'}$')
    #ax.set_title(r'$k_{'+str(int(i+1))+str(int(i+1))+'}$')
    ax.set_title(r'$\alpha_{'+str(int(i+1))+'}$')
    ax.set_xlabel('Actual')
    if i == 0: ax.set_ylabel('Predicted')
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
axes[0].legend()
plt.tight_layout()
plt.savefig('gpyPoints_ps.png', bbox_inches='tight')   

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
plt.plot(np.arange(0,numPCs,1),nmae_train,linestyle='-', marker='o',)
plt.plot(np.arange(0,numPCs,1),nmae_test,linestyle='-', marker='o',)
plt.ylabel(r'$NMAE$') 
plt.legend(['Train','Test'])
plt.tight_layout()
plt.savefig('gpyNMAE_ps.png', bbox_inches='tight')




