# Bayesian Approach to Designing Microstructures and Processing Pathways for Tailored Material Properties
This repository contains research code associated with a forthcoming article entitled
*"Bayesian Approach to Designing Microstructures and Processing Pathways for Tailored Material Properties"*. The proposed framework utilizes a conditional continuous normalizing flow trained with the flow matching objective proposed by [Lipman *et al.* (2023)](https://arxiv.org/abs/2210.02747), and a composite process--structure and structure-property linkage as the forward model for implicitly learning the joint distribution between processing parameters and an arbitrary property set.

If you find this code useful, interesting, and are open to collaboration, please reach out! 
Alternatively, if you have any questions regarding the contents of this repository, feel free
to as well at: [agenerale3@gatech.edu](agenerale3@gatech.edu).

## Contents
This section provides a brief description of the contents of this repository.

1. *Data*: Contains processing parameters, resulting property set (elastic + thermal properties), and PC scores of 2-point spatial correlations of microstructures in the initial dataset.

2. *Helpers*: Code for helper functions utilized in main script *cnf.py*.

3. *Models*: Contains code for instantiating the sparse variational multi-output
 Gaussian process (SV-MOGP) used in this work, along with the actual trained models utilized.
 
4. *cnf.py*: Main executable for training and post-processing results from the conditional continuous normalizing flow.

5. *main_gpytorchNG_ps.py*: Main executable for training of the SV-MOGP process-structure linkage.
 
6. *main_gpytorchNG_sp.py*: Main executable for training of the SV-MOGP structure-property linkage.

## Execute
Inference of the conditional microstructure distributions provided above can be replicated as
```
python cnf.py
```
where the *train* and *load* flags can be turned on to prevent trainig and load an existing model as
```
python cnf.py --train --load
```
allowing for postprocessing of the results.
