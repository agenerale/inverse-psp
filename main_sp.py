import numpy as np
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from models.mogp import NGLMCGPModel
from helpers.function_helpers import load_data, norm_scaling, eval_dataloaders
from helpers.plotting_helpers import plot_parity
import argparse

parser = argparse.ArgumentParser(description="SVMOGP Structure-Property Linkage")
parser.add_argument("--train", action='store_true', help="train (True) cuda")
parser.add_argument("--load", action='store_false', help="load pretrained model")
parser.add_argument("--spectral", action='store_false', help="spectral mixture kernel")
parser.add_argument("--batch", default=512, type=int, help="minibatch training size")
parser.add_argument("--n_latent", default=5, type=int, help="# latent GPs")
parser.add_argument("--n_inducing", default=0.02, type=float, help="% inducing points of training set size")
parser.add_argument("--n_epochs", default=500, type=int, help="# training epochs")
parser.add_argument("--n_smix", default=6, type=int, help="number of spectral mixtures in kernel")
parser.add_argument("--n_pcs", default=5, type=int, help="number of pcs for microstructure 2pt. statistics")
parser.add_argument("--lr_init", default=1e-2, type=float, help="init. learning rate")
parser.add_argument("--lr_end", default=1e-6, type=float, help="end learning rate")
args = parser.parse_args()

device = torch.device("cuda" if (torch.cuda.is_available() and args.train) else "cpu")

fname_model = 'mogp_model_state_spNGsm_test.pth'
fname_lik = 'mogp_likelihood_state_spNGsm_test.pth'

fname = './data/data.h5'
data = load_data(fname, test_size=0.2, random_state=10, device='cpu')

train_x = data['mid_train'][:,:args.n_pcs].float()
test_x = data['mid_test'][:,:args.n_pcs].float()
train_y = data['output_train'].float()
test_y = data['output_test'].float()
mid_scaler = data['mid_scaler']
output_scaler = data['output_scaler']

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=args.batch,
                          drop_last=False, shuffle=False)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=args.batch, 
                         drop_last=False, shuffle=False)

# Model
n_inducing = int(args.n_inducing*train_x.shape[0])
model = NGLMCGPModel(args.n_latent,
                    train_y.shape[1],
                    n_inducing,
                    train_x.shape[1],
                    spectral=args.spectral,
                    num_mix=args.n_smix).to(device)
if args.spectral: model.covar_module.initialize_from_data_empspect(train_x, train_y) 
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=train_y.shape[1]).to(device)

print('Likelihood Parameters: ' + str(sum(p.numel() for p in likelihood.parameters() if p.requires_grad)))
print('Model Parameters: ' + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if args.load:
    state_dict_model = torch.load(fname_model, map_location=device)
    state_dict_likelihood = torch.load(fname_lik, map_location=device)
    model.load_state_dict(state_dict_model)
    likelihood.load_state_dict(state_dict_likelihood)
    
if args.train:
    model.train()
    likelihood.train()
   
    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.shape[0], lr=args.lr_init)

    hyperparameter_optimizer = torch.optim.Adam([
            {'params': model.hyperparameters()},
            {'params': likelihood.parameters()},
        ], lr=args.lr_init)
    
    hyperparameter_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hyperparameter_optimizer,args.n_epochs, args.lr_end)
    variational_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(variational_ngd_optimizer,args.n_epochs, args.lr_end)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.shape[0])
    loss_list = []
    for i in range(args.n_epochs):
        batch_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
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
            torch.save(model.state_dict(), fname_model)
            torch.save(likelihood.state_dict(), fname_lik)

# %%
# predictions
out = eval_dataloaders(model,
                    likelihood,
                    train_loader,
                    test_loader,
                    output_scaler,
                    device=device)
train_mean = out['train']['mean']
trainlower = out['train']['low']
trainupper = out['train']['up']
test_mean = out['test']['mean']
testlower = out['test']['low']
testupper = out['test']['up']

train_y = output_scaler.inverse_transform(train_y.numpy())
test_y = output_scaler.inverse_transform(test_y.numpy())

plot_parity(train_y, test_y,
                train_mean, test_mean,
                trainlower, trainupper,
                testlower, testupper)

print('******* NMAE *******')
nmae_train = np.mean(abs(train_mean - train_y)/np.mean(abs(train_y),axis=0),axis=0)*100
nmae_test = np.mean(abs(test_mean - test_y)/np.mean(abs(train_y),axis=0),axis=0)*100
print(nmae_train)
print(nmae_test)
print('******* MAE *******')
mae_train = np.mean(abs(train_mean - train_y),axis=0)
mae_test = np.mean(abs(test_mean - test_y),axis=0)
print(mae_train)
print(mae_test)
