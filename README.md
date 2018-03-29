# RedQueen

This is a repository containing code for the paper:

> A. Zarezade, U. Upadhyay, H. R. Raibee, M. Gomez-Rodriguez. RedQueen: An Online Algorithm for Smart Broadcasting in Social Networks. In Proceedings of the 10th ACM International Conference on Web Search and Data Mining (WSDM), 2017.

## Pre-requisites

This code depends on the following packages:

 1. `decorated_options`: Installation instructions are at [musically-ut/decorated_options](https://github.com/musically-ut/decorated_options) or `pip install decorated_options`.
 2. `broadcast_ref` package (i.e. Karimi et.al.'s method, which is the used as a baseline). Follow the instructions at [Networks-Learning/broadcast_ref](https://github.com/Networks-Learning/broadcast_ref).


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

As an example, say if we have the following structure in our network of
broadcasters (i.e. sources) and followers (i.e. sinks), with `Source 1` being
the broadcaster we control:

```
   Source 1**      Source 2      Source 3
    +   +             +                 +
    |   |             |                 |
    |   +----------------------------+  |
    |                 |              |  |
    |    +------------+-----------+  |  |
    |    |            |           |  |  |
   +v----v-+      +---v---+      +v--v--v+
   |       |      |       |      |       |
   |       |      |       |      |       |
   |       |      |       |      |       |
   |       |      |       |      |       |
   |       |      |       |      |       |
   |       |      |       |      |       |
   +-------+      +-------+      +-------+
    Sink 1          Sink 2         Sink 3
```

This will be represented in the following way:

```python
simOpts = SimOpts(
   src_id = 1,
   end_time = 100, # When the simulations stop
   q_vec = { 1: 1.0, 3: 1.0 }, # Weights of followers of source 1
   s=1.0, # Control parameter for RedQueen
   sink_ids = [1, 2, 3],
   other_sources = [
      ('Poisson2', { 'src_id': 2,
                    'seed': 42,
                    'rate': 10
                  }),
      ('Hawkes',  { 'src_id': 3,
                    'seed': 43,
                    'l_0': 10,
                    'alpha': 1.0,
                    'beta': 10.0
                  })
   ],
   edge_list=[(1, 1), (1, 3), 
              (2, 1), (2, 2), (2, 3),
              (3, 3)]
);
```

These `SimOpts` objects are immutable and can be used to create multiple simulations.

Then to run the simulation, we need to create a simulation `Manager` by instantiating `Source 1`
to be a kind of broadcaster, or by removing it altogether.

```python
manager = simOpts.create_manager_with_opt(seed=101)
# or 
manager = simOpts.create_manager_for_wall()
```

Finally, run the simulation by calling `.run_dynamic`:

```python
manager.run_dynamic()
```

Finally, the list of events can be extracted for further analysis:

```python
df = manager.state.get_dataframe()
```


The file `utils.py` contains some functions which can assist in calculation of
certain metrics:

```python
import redqueen.utils as U
perf_1 = U.time_in_top_k(df=df, K=1, sim_opts=simOpts)
perf_2 = U.average_rank(df=df, sim_opts=simOpts)
```
