import h5py
import numpy as np
import torch
import gpytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def norm_scaling(x, device='cpu'):
    x = torch.from_numpy(x).float().to(device)
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x -= m
    x /= s
    return x, m, s

def unit_scaling(x, device='cpu'):
    x = torch.from_numpy(x).float().to(device)
    x_min = x.min(0)[0]
    x_max = x.max(0)[0]
    x = 2 * (x - x_min)/(x_max - x_min) - 1
    
    return x, x_min, x_max    

def load_data(file, cutoff=5, test_size=0.2, random_state=10, device='cpu'):
    with h5py.File(file, "r") as f:
        mresults = f['mech'][()]
        tresults = f['thermal'][()]
        presults = f['params'][()]
        pcs = f['pcs'][()]
    f.close() 
    
    presults[:,:2] = np.log(presults[:,:2]) # scale to uniform distributions

    mresults[:, [3, 2]] = mresults[:, [2, 3]]
    mresults = np.hstack((mresults[:,1:3], mresults[:,5][...,None]))                     
    output = np.hstack((mresults,tresults[:,1:]))
   
    data = {'input': torch.tensor(presults, device=device),
            'output': torch.tensor(output, device=device)}
    
    indices = np.arange(len(presults))
    xtr, xte, tr_indx, te_indx = train_test_split(presults,
                                                  indices,
                                                  test_size=test_size, 
                                                  random_state=random_state)    
   
    ytr_mid = pcs[tr_indx, :cutoff]
    yte_mid = pcs[te_indx, :cutoff]
    ytr = output[tr_indx]
    yte = output[te_indx]
    
    minmax = MinMaxScaler(feature_range=(-1,1))
    standard = StandardScaler()
    standard_mid = StandardScaler()
    
    xtr = minmax.fit_transform(xtr)
    xte = minmax.transform(xte)
    ytr_mid = standard_mid.fit_transform(ytr_mid)
    yte_mid = standard_mid.transform(yte_mid)
    ytr = standard.fit_transform(ytr)
    yte = standard.transform(yte)
    
    data['input_train'] = torch.tensor(xtr, device=device)
    data['input_test'] = torch.tensor(xte, device=device)
    data['mid_train'] = torch.tensor(ytr_mid, device=device)
    data['mid_test'] = torch.tensor(yte_mid, device=device)
    data['output_train'] = torch.tensor(ytr, device=device)
    data['output_test'] = torch.tensor(yte, device=device)
    data['input_scaler'] = minmax
    data['output_scaler'] = standard
    data['mid_scaler'] = standard_mid
            
    return data

def eval_mae(traj, forward_model, y_target, scaler):
    in_x = traj[-1,...]
    y_pred = forward_model(in_x) 
    y_pred = y_pred.detach().cpu().numpy()
    y_pred = scaler.inverse_transform(y_pred)
    y_pred = np.mean(y_pred, axis=0)
    
    y_target = y_target.unsqueeze(0).detach().cpu().numpy()
    y_target = scaler.inverse_transform(y_target)

    err = (y_target - y_pred)
    mae = np.abs(err)
    nmae = np.abs(err / y_target)
    
    return mae, nmae

def eval_dataloaders(model,
                     likelihood,
                     train_loader,
                     test_loader,
                     scaler,
                     device='cpu'):
    model.eval()
    likelihood.eval()        

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        
        train_mean, trainlower, trainupper = [], [], []
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            train_predictions = likelihood(model(x_batch))
            lower_batch, upper_batch = train_predictions.confidence_region()
            
            mean = scaler.inverse_transform(train_predictions.mean.detach().cpu().numpy())
            lower = scaler.inverse_transform(lower_batch.detach().cpu().numpy())
            upper = scaler.inverse_transform(upper_batch.detach().cpu().numpy())
            
            train_mean.append(mean)
            trainlower.append(lower)
            trainupper.append(upper)
        train_mean = np.vstack(train_mean)
        trainlower = np.vstack(trainlower)
        trainupper = np.vstack(trainupper)
        
        test_mean, testlower, testupper = [], [], []
        for x_batch, _ in test_loader:
            x_batch = x_batch.to(device)
            test_predictions = likelihood(model(x_batch))
            lower_batch, upper_batch = test_predictions.confidence_region()
            
            mean = scaler.inverse_transform(test_predictions.mean.detach().cpu().numpy())
            lower = scaler.inverse_transform(lower_batch.detach().cpu().numpy())
            upper = scaler.inverse_transform(upper_batch.detach().cpu().numpy())            
            
            test_mean.append(mean)
            testlower.append(lower)
            testupper.append(upper)
        test_mean = np.vstack(test_mean)
        testlower = np.vstack(testlower)
        testupper = np.vstack(testupper)
        
    out = {'train':{'mean': train_mean,
                    'low': trainlower,
                    'up': trainupper},
           'test':{'mean': test_mean,
                   'low': testlower,
                   'up': testupper}}
                
    return out