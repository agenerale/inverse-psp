import corner
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)
    
#(traj, data['input_scaler'], indx, x_target, save_path=imgs_subdir)
def plot_trajectories(traj, scaler, indx, x_target, save_path=None):  
    T, B, D = traj.shape
    x_target = x_target.detach().cpu().numpy()
    traj = traj.detach().cpu().numpy()
    traj = traj.reshape(-1, 3) 
    traj = scaler.inverse_transform(traj)
    traj = traj.reshape(T, B, D)
    
    fig, axes = plt.subplots(traj.shape[-1],traj.shape[-1], figsize=(12, 12),
                             sharex=False, sharey=False, constrained_layout=True)
    for i in range(traj.shape[-1]):
        ax = axes[i, i]
        ax.hist(traj[-1,:,i], bins=20, rwidth=0.9, density=True, color="maroon",alpha=1)
        ax.set_yticks([])
        ax.axvline(x_target[i], color="black",linestyle='--', alpha=1.0,linewidth=4)
        
        if i == 0:
            ax.set_ylabel(r'$\theta_{'+str(i+1)+'}$')
        if i == traj.shape[-1]-1:
            ax.set_xlabel(r'$\theta_{'+str(i+1)+'}$')
    
    for yi in range(traj.shape[-1]):
        for xi in range(yi):
            ax = axes[yi,xi]
            ax.scatter(traj[0,:,xi], traj[0,:,yi], s=4, alpha=0.4, c="black")
            ax.scatter(traj[:,:,xi], traj[:,:,yi], s=0.2, alpha=0.1, c="slategray")
            ax.scatter(traj[-1,:,xi], traj[-1,:,yi], s=4, alpha=1, c="maroon")
            
            ax.axvline(x_target[xi], color="black",linestyle='--', alpha=1.0,linewidth=2)
            ax.axhline(x_target[yi], color="black",linestyle='--', alpha=1.0,linewidth=2)
            ax.scatter(x_target[xi],x_target[yi],color="black",s=100)
            
            if yi == traj.shape[-1]-1:
                ax.set_xlabel(r'$\theta_{'+str(xi)+'}$')
            if xi == 0:
                ax.set_ylabel(r'$\theta_{'+str(yi)+'}$')  
                
    for xi in range(traj.shape[-1]):
        for yi in range(xi):
            ax = axes[yi,xi]
            fig.delaxes(ax)
    
    if save_path is not None:
        plt.savefig(save_path / f'./traj_corner_{indx}.png') 
        plt.close()
        
def plot_corner_theta(traj, scaler, indx, x_target, save_path=None):
    T, B, D = traj.shape
    x_target = x_target.detach().cpu().numpy()
    traj = traj.detach().cpu().numpy()
    traj = traj.reshape(-1, 3) 
    traj = scaler.inverse_transform(traj)
    traj = traj.reshape(T, B, D)
    theta = traj[-1,...]
    
    lbl = [fr'$\theta_{i}$' for i in range(theta.shape[-1])]
    corner.corner(theta, labels=lbl, hist_bin_factor=2, smooth=True, truths=x_target)
    if save_path is not None:
        plt.savefig(save_path / f'./corner_theta_fm_{indx}.png') 
        plt.close()

def plot_trajectories_1d(traj, t_span, micro):
    fig, axes = plt.subplots(1,3, figsize=(30, 5), sharex=False, constrained_layout=True)
    for i in range(3):
        ax = axes[i]
        ax.plot(t_span, traj[:,:,i], alpha=.05, color='slategray')
    axes[0].set_title("Dim. 0") ; axes[1].set_title("Dim. 1") ; axes[2].set_title("Dim. 3")
    axes[0].set_xlabel("t"); axes[1].set_xlabel("t"); axes[2].set_xlabel("t")
    plt.savefig('./images/traj1d_fm_' + str(micro) + '.png', bbox_inches='tight') 

def plot_corner_prop(traj, forward_model, y_target, scaler, indx, save_path=None):
    lbl_prop = [r'$E_1$ (GPa)',r'$E_2$ (GPa)', r'$G_{12}$ (GPa)', r'$k_1$ (W/mK)', r'$k_2$ (W/mK)']
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_calc = forward_model(traj[-1,:,:])
    
    y_calc = y_calc.detach().cpu().numpy()
    y_calc = scaler.inverse_transform(y_calc)
    y_calc[:,:3] = y_calc[:,:3]/1e3
    
    y_target = y_target.unsqueeze(0).detach().cpu().numpy()
    y_target = scaler.inverse_transform(y_target)
    y_target[:,:3] = y_target[:,:3]/1e3
    
    corner.corner(y_calc, labels=lbl_prop, hist_bin_factor=2, smooth=False, truths=y_target.squeeze())
    if save_path is not None:
        plt.savefig(save_path / f'./corner_prop_fm_{indx}.png') 
        plt.close()
        
def plot_annotate(output, pcs):
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
    
    fig, ax = plt.subplots(figsize=(25,20))
    ax.scatter(pcs[:,1],pcs[:,2],s=20,color='tab:blue')
    for i, txt in enumerate (np.arange(0,pcs.shape[0],1)):
        ax.annotate(txt, (pcs[i,1],pcs[i,2]))
    plt.xlabel(r'$\alpha_2$')
    plt.ylabel(r'$\alpha_3$')     
    
def plot_cases_prop(output,lbl_prop,microindx_array,cdim):
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
    
    clr_dict = {
        7216 : 'red',
        4027 : 'aqua',
        #9682 : 'lime',  
        440 : 'lime',    
        131  : 'magenta',
        }  

    marker = ['o','^','s','d']
    
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    font = {'family' : 'serif','weight' : 'normal','size'   : 20}
    plt.rc('font', **font)
    plt.rc('font', family='serif')
    fig, axes = plt.subplots(1,output.shape[-1]-1, figsize=(25, 6), sharex=False)
    for i in range(output.shape[-1]-1):
        axes[i].scatter(output[:,i],output[:,i+1],s=10,color='black')

        for j in range(len(microindx_array)):
            axes[i].scatter(output[microindx_array[j],i],output[microindx_array[j],i+1],s=200,
                            facecolor=clr_dict[microindx_array[j]],edgecolor='black',label=r'$p_{'+str(j+1)+'}$',marker=marker[j])

        axes[i].set_xlabel(lbl_prop[i])
        axes[i].set_ylabel(lbl_prop[i+1])

    axes[0].legend()
    plt.tight_layout()
    plt.savefig('./images/prop_ensemble_paper.png', bbox_inches='tight')
    
def plot_cases_pcs(pcs,microindx_array):
    clr_dict = {
        7216 : 'red',
        4027 : 'aqua',
        #9682 : 'lime',  
        440 : 'lime',    
        131  : 'magenta',
        }  

    marker = ['o','^','s','d']

    #plt.rc('xtick',labelsize=14)
    #plt.rc('ytick',labelsize=14)
    #font = {'family' : 'serif','weight' : 'normal','size'   : 20}
    #plt.rc('font', **font)
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.plot(pcs[:,0],pcs[:,1],pcs[:,2],'.',markersize=5,color='black',alpha=0.5)
    for j in range(len(microindx_array)):
        ax.plot(pcs[microindx_array[j],0],pcs[microindx_array[j],1],pcs[microindx_array[j],2],markersize=15,markeredgecolor='black',color=clr_dict[microindx_array[j]],label=r'$m_{'+str(j+1)+'}$',marker=marker[j])
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.set_xlabel(r'$\alpha_{1}$')
    ax.set_ylabel(r'$\alpha_{2}$')
    ax.set_zlabel(r'$\alpha_{3}$')
    ax.legend()
    plt.show()
    plt.savefig('./images/pc_ensemble_3d.png', bbox_inches='tight')
    
    
    fig, axes = plt.subplots(1,3, figsize=(25, 6), sharex=False)
    for i in range(3):
        ax = axes[i]
        ax.scatter(pcs[:,i],pcs[:,i+1],s=10,color='black')
        ax.set_xlabel(r'$\alpha_{'+str(int(i+1))+'}$')
        ax.set_ylabel(r'$\alpha_{'+str(int(i+2))+'}$')
        for j in range(len(microindx_array)):
            ax.scatter(pcs[microindx_array[j],i],pcs[microindx_array[j],i+1],s=300,
                       facecolor=clr_dict[microindx_array[j]],edgecolor='black',label=r'$p_{'+str(j+1)+'}$',marker=marker[j])
    axes[0].legend()
    plt.savefig('./images/pc_ensemble.png', bbox_inches='tight')
    
def plot_parity(train_y, test_y,
                train_mean, test_mean,
                trainlower, trainupper,
                testlower, testupper,
                save_path=None):
    
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    font = {'family' : 'serif','weight' : 'normal','size'   : 22}
    plt.rc('font', **font)

    fig, axes = plt.subplots(1, train_y.shape[1], figsize=(25, 5), sharex=False, constrained_layout=True)
    for i in range(train_y.shape[1]):
        ax = axes[i]
        ax.errorbar(train_y[:,i],train_mean[:,i],
                    yerr=(trainupper[:,i]-trainlower[:,i]),
                    fmt='o',
                    capsize=2,
                    color='tab:blue',
                    label='train',
                    alpha=0.2)
        
        ax.errorbar(test_y[:,i],test_mean[:,i],
                    yerr=(testupper[:,i]-testlower[:,i]),
                    fmt='o',
                    capsize=2,
                    color='tab:red',
                    label='test',
                    alpha=0.2)
        
        ax.plot([np.min(train_y[:,i]),np.max(train_y[:,i])],
                     [np.min(train_y[:,i]),np.max(train_y[:,i])],'k--',linewidth=3) 
        ax.set_title(r'$\alpha_{'+str(int(i+1))+'}$')
        ax.set_xlabel('Actual')
        if i == 0: ax.set_ylabel('Predicted')
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    axes[0].legend()
        

