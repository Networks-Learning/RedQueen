# RedQueen

This is a repository containing code for the paper:

> RedQueen: An Online Algorithm for Smart Broadcasting in Social Networks.
> Ali Zarezade, MPI-SWS
> Utkarsh Upadhyay, MPI-SWS
> Hamid Rabiee, Sharif University of Technology
> Manuel Gomez Rodriguez, MPI for Software Systems

## Pre-requisites

This code depends on the following packages:

 1. `decorated_options`: Installation instructions are at [musically-ut/decorated_options](https://github.com/musically-ut/decorated_options)
 2. `broadcast_ref` package (i.e. Karimi et.al.'s method, which is the current benchmark). This repository is currently private, but will be released before this code is made public.


## Code structure

 - `opt_models.py` contains models for various broadcasters and baselines:
   - `Poisson` (random posting)
   - `Hawkes` (bursty posting)
   - `PiecewiseConst` (different rates at different times)
   - `RealData` (emulates behavior of a real user in our dataset)
   - `RedQueen` (our proposed algorithm)

 - `utils.py` contains common utility functions for metric calculation and plotting.
 - `opt_runs.py` contains functions to execute the simulations.

 - `real_data_gen.py` and `read_real_data.py` files deal with conversion of real Twitter data to and from formats helpful for our models.


## Execution

The code execution is detailed in the included IPython notebooks.

