import corner
import numpy as np
import torch
import torchdyn
import gpytorch
from torchdyn.core import NeuralODE
from scipy.stats import binom
from models.cnf import MLP,torchdyn_wrapper
#from models.mogp import MultitaskGPModel, NGDMultitaskGPModel
#from models.dkl import FeatureExtractor, DKLModel
import matplotlib.pyplot as plt
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
font = {'family' : 'serif','weight' : 'normal','size'   : 20}
plt.rc('font', **font)

    
def plot_trajectories(traj,pr_max,pr_min,micro,presults):  
    traj = 0.5*(traj + 1)*(pr_max - pr_min) + pr_min
    traj = traj.detach().cpu().numpy()
    
    target = presults[micro,:]
    
    fig, axes = plt.subplots(traj.shape[-1],traj.shape[-1], figsize=(12, 12), sharex=False, sharey=False)
    for i in range(traj.shape[-1]):
        ax = axes[i, i]
        ax.hist(traj[-1,:,i], bins=20, rwidth=0.9, density=True, color="maroon",alpha=1)
        ax.set_yticks([])
        ax.axvline(target[i], color="black",linestyle='--', alpha=1.0,linewidth=4)
        
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
            
            ax.axvline(target[xi], color="black",linestyle='--', alpha=1.0,linewidth=2)
            ax.axhline(target[yi], color="black",linestyle='--', alpha=1.0,linewidth=2)
            ax.scatter(target[xi],target[yi],color="black",s=100)
            
            if yi == traj.shape[-1]-1:
                ax.set_xlabel(r'$\theta_{'+str(xi+1)+'}$')
            if xi == 0:
                ax.set_ylabel(r'$\theta_{'+str(yi+1)+'}$')  
                
    for xi in range(traj.shape[-1]):
        for yi in range(xi):
            ax = axes[yi,xi]
            fig.delaxes(ax)
    plt.savefig('./images/traj_corner' + str(micro) + '.png', bbox_inches='tight') 

def plot_trajectories_1d(traj, t_span,micro):
    fig, axes = plt.subplots(1,3, figsize=(30, 5), sharex=False)
    for i in range(3):
        ax = axes[i]
        ax.plot(t_span, traj[:,:,i], alpha=.05, color='slategray')
    axes[0].set_title("Dim. 0") ; axes[1].set_title("Dim. 1") ; axes[2].set_title("Dim. 3")
    axes[0].set_xlabel("t"); axes[1].set_xlabel("t"); axes[2].set_xlabel("t")
    plt.savefig('./images/traj1d_fm_' + str(micro) + '.png', bbox_inches='tight') 
    
def plot_corner_theta(traj,pr_max,pr_min,micro,lbl_theta,presults):
    theta = 0.5*(traj[-1,:,:] + 1)*(pr_max - pr_min) + pr_min
    theta = theta.detach().cpu().numpy()
    #theta[:,:2] = np.exp(theta[:,:2])
    target = presults[micro,:]
    #target[:2] = np.exp(target[:2])
    fig = corner.corner(theta, labels=lbl_theta, hist_bin_factor=2, smooth=True, truths=target)
    plt.savefig('./images/corner_theta_fm_' + str(micro) + '.png', bbox_inches='tight') 

def plot_corner_prop(traj,predict,y_cond,out_m,out_s,micro,lbl_prop):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_calc = predict(traj[-1,:,:])
    #y_calc = 0.5*(y_calc + 1)*(out_max - out_min) + out_min
    y_calc = y_calc*out_s + out_m
    y_calc = y_calc.detach().cpu().numpy()
    y_calc[:,:3] = y_calc[:,:3]/1e3
    #y_cond = 0.5*(y_cond + 1)*(out_max - out_min) + out_min
    y_cond = (y_cond*out_s + out_m).squeeze()
    y_cond[:3] = y_cond[:3]/1e3
    fig = corner.corner(y_calc, labels=lbl_prop, hist_bin_factor=2, smooth=False, truths=y_cond.detach().cpu().numpy())
    plt.savefig('./images/corner_prop_fm_' + str(micro) + '.png', bbox_inches='tight') 
        

def get_coverage_probs(z, u):
    """Vectorized function to compute the minimal coverage probability for uniform
    ECDFs given evaluation points z and a sample of samples u.

    Parameters
    ----------
    z  : np.ndarray of shape (num_points, )
        The vector of evaluation points.
    u  : np.ndarray of shape (num_simulations, num_samples)
        The matrix of simulated draws (samples) from U(0, 1)
    """

    N = u.shape[1]
    F_m = np.sum((z[:, np.newaxis] >= u[:, np.newaxis, :]), axis=-1) / u.shape[1]
    bin1 = binom(N, z).cdf(N * F_m)
    bin2 = binom(N, z).cdf(N * F_m - 1)
    gamma = 2 * np.min(np.min(np.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma

def simultaneous_ecdf_bands(
    num_samples, num_points=None, num_simulations=1000, confidence=0.95, eps=1e-5, max_num_points=1000
):
    """Computes the simultaneous ECDF bands through simulation according to
    the algorithm described in Section 2.2:

    https://link.springer.com/content/pdf/10.1007/s11222-022-10090-6.pdf

    Depends on the vectorized utility function `get_coverage_probs(z, u)`.

    Parameters
    ----------
    num_samples     : int
        The sample size used for computing the ECDF. Will equal to the number of posterior
        samples when used for calibrarion. Corresponds to `N` in the paper above.
    num_points      : int, optional, default: None
        The number of evaluation points on the interval (0, 1). Defaults to `num_points = num_samples` if
        not explicitly specified. Correspond to `K` in the paper above.
    num_simulations : int, optional, default: 1000
        The number of samples of size `n_samples` to simulate for determining the simultaneous CIs.
    confidence      : float in (0, 1), optional, default: 0.95
        The confidence level, `confidence = 1 - alpha` specifies the width of the confidence interval.
    eps             : float, optional, default: 1e-5
        Small number to add to the lower and subtract from the upper bound of the interval [0, 1]
        to avoid edge artefacts. No need to touch this.
    max_num_points  : int, optional, default: 1000
        Upper bound on `num_points`. Saves computation time when `num_samples` is large.

    Returns
    -------
    (alpha, z, L, U) - tuple of scalar and three arrays of size (num_samples,) containing the confidence level as well as
                       the evaluation points, the lower, and the upper confidence bands, respectively.
    """

    # Use shorter var names throughout
    N = num_samples
    if num_points is None:
        K = min(N, max_num_points)
    else:
        K = min(num_points, max_num_points)
    M = num_simulations

    # Specify evaluation points
    z = np.linspace(0 + eps, 1 - eps, K)

    # Simulate M samples of size N
    u = np.random.uniform(size=(M, N))

    # Get alpha
    alpha = 1 - confidence

    # Compute minimal coverage probabilities
    gammas = get_coverage_probs(z, u)

    # Use insights from paper to compute lower and upper confidence interval
    gamma = np.percentile(gammas, 100 * alpha)
    L = binom(N, z).ppf(gamma / 2) / N
    U = binom(N, z).ppf(1 - gamma / 2) / N
    return alpha, z, L, U

def SBC_samples(model,predict,N,L,ndim,cdim,device):
    prior_samples = np.asarray([np.random.uniform(-1,1,N) for i in range(ndim)]).T
    prior_samples = torch.from_numpy(prior_samples).float()
    post_samples = torch.zeros((N,L,ndim))
    
    for i in range(prior_samples.shape[0]):
        x = prior_samples[i,:][None,...].to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y = predict(x)
        node = NeuralODE(
            torchdyn_wrapper(model, y.expand(L,cdim)), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
            ).to(device)
        #node = NeuralODE(
        #    torchdyn_wrapper(model, y.expand(L,cdim)), solver="rk4", sensitivity="adjoint", solver_adjoint='dopri5', atol=1e-4, rtol=1e-4
        #    ).to(device)
        with torch.no_grad():
            traj = node.trajectory(
                    torch.randn((L,ndim)).to(device),
                    t_span=torch.linspace(0, 1, 100),
                    )
        post_samples[i,...] = traj[-1,...].detach().cpu()
        if (i + 1) % 50 == 0:
            print(f"Validation: {i+1}/{N}")
        
    post_samples = post_samples.detach().cpu().numpy() 
    prior_samples = prior_samples.detach().cpu().numpy() 
    
    return post_samples, prior_samples
      
def SBC(model,predict,ndim,cdim,device,N=2000,L=100,
        n_bins=None,
        bin_interval=0.95,
        stacked=False,
        fig_size=None,
        param_names=None,
        difference=False
        ):
    ''' implements simulation-based calibration Gelman (2018) 
    https://arxiv.org/abs/1804.06788 '''
    
    post_samples, prior_samples = SBC_samples(model,predict,N,L,ndim,cdim,device)
    
    ratio = int(N/L)
    print(f"Ratio: {ratio}")
    if n_bins is None:
        n_bins = int(ratio / 2)
        print(f"Bins: {n_bins}")
    
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)
            
    # Compute confidence interval and mean
    # uniform distribution expected -> for all bins: equal probability
    # p = 1 / num_bins that a rank lands in that bin
    endpoints = binom.interval(bin_interval, N, 1 / n_bins)
    mean = N / n_bins  # corresponds to binom.mean(N, 1 / num_bins)
    
    fig, axes = plt.subplots(1,ndim, figsize=(20, 5), sharex=False)
    for i in range(ndim):
        ax = axes[i]
        ax.hist(ranks[:,i], color='maroon', rwidth=0.9, bins=n_bins, edgecolor = "gray")
        ax.axhspan(endpoints[0], endpoints[1], facecolor="slategray", alpha=0.15)
        ax.axhline(mean, color="slategray", alpha=0.9)
        ax.set_xlabel(f'Rank Statistic {i}')
        ax.set_title(r'$\theta_{'+str(int(i+1))+'}$')
    plt.savefig('./images/sbc_gelu.png', bbox_inches='tight')
    
    '''Creates the empirical CDFs for each marginal rank distribution and plots it against
    a uniform ECDF. ECDF simultaneous bands are drawn using simulations from the uniform,
    as proposed by [1].

    For models with many parameters, use `stacked=True` to obtain an idea of the overall calibration
    of a posterior approximator.

    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing,
    32(2), 1-21. https://arxiv.org/abs/2103.10522'''  
    
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]
    
    # Prepare figure
    if stacked:
        n_row, n_col = 1, 1
        f, ax = plt.subplots(1, 1, figsize=fig_size)
    else:
        # Determine n_subplots dynamically
        n_row = int(np.ceil(ndim / 6))
        n_col = int(np.ceil(ndim / n_row))

        # Determine fig_size dynamically, if None
        if fig_size is None:
            fig_size = (int(5 * n_col), int(5 * n_row))

        # Initialize figure
        f, ax = plt.subplots(n_row, n_col, figsize=fig_size)

    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = np.sort(ranks[:, j])
        xx = ecdf_single
        yy = np.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                ax.plot(xx, yy, color='maroon', alpha=0.95, label="Rank ECDFs")
            else:
                ax.plot(xx, yy, color='maroon', alpha=0.95)
        else:
            ax.flat[j].plot(xx, yy, color='maroon', alpha=0.95, label="Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0])

    # Difference, if specified
    if difference:
        L -= z
        H -= z
        ylab = "ECDF difference"
    else:
        ylab = "ECDF"

    # Add simultaneous bounds
    if stacked:
        titles = [None]
        axes = [ax]
    else:
        axes = ax.flat
        if param_names is None:
            titles = [f"$\\theta_{{{i}}}$" for i in range(1, ndim + 1)]
        else:
            titles = param_names

    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color='slategray', alpha=0.2, label=rf"{int((1-alpha) * 100)}$\%$ Confidence Bands")

        # Prettify plot
        #sns.despine(ax=_ax)
        #_ax.grid(alpha=0.35)
        #_ax.legend(fontsize=legend_fontsize)
        _ax.set_title(title)
        #_ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        #_ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    if stacked:
        bottom_row = [ax]
    else:
        bottom_row = ax if n_row == 1 else ax[-1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Fractional rank statistic")

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axes[0].set_ylabel(ylab)
    else:  # if there is more than one row, the ax array is 2D
        for _ax in ax[:, 0]:
            _ax.set_ylabel(ylab)

    # Remove unused axes entirely
    for _ax in axes[ndim:]:
        _ax.remove()

    f.tight_layout()    
    plt.savefig('./images/ecdf_gelu.png', bbox_inches='tight')
    
def plot_annotate(output,pcs):
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
        

