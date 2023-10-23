import h5py
import numpy as np
import torch
import gpytorch
from models.mogp import MultitaskGPModel, NGDMultitaskGPModel
from models.dkl import FeatureExtractor, DKLModel

def norm_scaling(x,device):
    x = torch.from_numpy(x).float().to(device)
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x, m, s

def unit_scaling(x,device):
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
    
def load_dkl(likelihood_file,model_file,feature_file):
    num_latents = 3
    num_tasks = 3
    num_inducing = int(0.02*8000)
    input_dims = 3   
    
    out_dim = input_dims  
    layers = 3
    width = 64   
    num_mix = 4
    
    feature_extractor = FeatureExtractor(input_dims,out_dim,layers=layers,w=width)
    model = DKLModel(feature_extractor, out_dim, num_tasks, num_inducing, num_mix)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    
    state_dict_model = torch.load(likelihood_file)
    state_dict_likelihood = torch.load(model_file)
    state_dict_feature = torch.load(feature_file)
    model.load_state_dict(state_dict_model)
    likelihood.load_state_dict(state_dict_likelihood)
    feature_extractor.load_state_disct(state_dict_feature)
    
    return likelihood, model    

def load_data(file):
    with h5py.File(file, "r") as f:
        mresults = f['mech'][()]
        tresults = f['thermal'][()]
        presults = f['params'][()]
    f.close()
    
    presults[:,:2] = np.log(presults[:,:2]) # scale to uniform distributions

    mresults[:, [3, 2]] = mresults[:, [2, 3]]
    mresults = np.hstack((mresults[:,1:3], mresults[:,5][...,None]))                     
    output = np.hstack((mresults,tresults[:,1:]))
   
    return output, presults 