# Bayesian Approach to Designing Microstructures and Processing Pathways for Tailored Material Properties
This repository contains research code associated with a forthcoming article entitled
*"Bayesian Approach to Designing Microstructures and Processing Pathways for Tailored Material Properties"*. The proposed framework utilizes a conditional continuous normalizing flow trained with the flow matching objective proposed by [https://openreview.net/forum?id=PqvMRDCJT9t](Lipman *et al.* (2023)), and a composite process--structure and structure-property linkage as the forward model for implicitly learning the joint distribution between processing parameters and an arbitrary property set.

If you find this code useful, interesting, and are open to collaboration, please reach out! 
Alternatively, if you have any questions regarding the contents of this repository, feel free
to as well at: [agenerale3@gatech.edu](agenerale3@gatech.edu).

## Contents
This section provides a brief description of the contents of this repository.

1. *Models*: Contains code for instantiating the sparse variational multi-output
 Gaussian process (SV-MOGP) used in this work.
 
2. *abq_results_memphis.h5*: Contains processing parameters and resulting property set (elastic + thermal properties) for initial dataset.

3. *microsPCs_memphis*: Contains associated PC scores of 2-point spatial correlations for micorstructures in the initial dataset
 
3. *mogp_likelihood_state.pth, mogp_model_state.pth*: Model state dictionaries for the SV-MOGP forward model.

4. *normflow_k_vae_{micro}_1.0_48.pth*: Model state dictionaries for final solutions of conditional distributions
 of microstructure PC scores given target orthotropic thermal conductivity.

5. *vae_1024d_64_beta100.pth*: Model state dictionary for the trained VAE.

6. *main.py*: Main executable for training and post-processing results from the flow-based generative model.

## Execute
Inference of the conditional microstructure distributions provided above can be replicated as
```
python main.py --micro 0
```
where the *micro* flag can be swept from 0-2 for the three current test cases.
