# Helper functions

import matplotlib
from matplotlib import pyplot as py
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import datetime as D
from options import Options, optioned

## Utilities

def mb(val, default):
    return val if val is not None else default


def logTime(chkpoint):
    print('*** \x1b[31m{}\x1b[0m Checkpoint: {}'.format(D.datetime.now(), chkpoint))
    sys.stdout.flush()


def def_q_vec(num_followers):
    """Returns the default q_vec for the given number of followers."""
    return np.ones(num_followers, dtype=float) / (num_followers ** 2)


def is_sorted(x, ascending=True):
    """Determines if a given numpy.array-like is sorted in ascending order."""
    return np.all((np.diff(x) * (1.0 if ascending else -1.0) >= 0))


## Metrics

def rank_of_src_in_df(df, src_id, fill=True, with_time=True):
    """Calculates the rank of the src_id at each time instant in the list of events."""

    assert is_sorted(df.t.values), "Array not sorted by time."

    def steps_to(x):
        return (np.arange(1, len(x) + 1) -
                np.maximum.accumulate(
                    np.where(x == src_id, range(1, len(x) + 1), 0)))

    df2 = df.copy()
    df2['rank'] = (df.groupby('sink_id')
                     .src_id
                     .transform(steps_to)
                     .astype(float))

    pivot_ranks = df2.pivot_table(index='t' if with_time else 'event_id',
                                  columns='sink_id', values='rank')
    return pivot_ranks.fillna(method='ffill') if fill else pivot_ranks


def u_int_opt(df, src_id=None, end_time=None, q_vec=None, s=None,
              follower_ids=None, sim_opts=None):
    """Calculate the ∫u(t)dt for the given src_id assuming that the broadcaster
    was following the optimal strategy."""

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)
        q_vec        = mb(q_vec, sim_opts.q_vec)
        s            = mb(s, sim_opts.s)
        follower_ids = mb(follower_ids, sim_opts.sink_ids)

    if follower_ids is None:
        follower_ids = sorted(df.sink_id[df.src_id == src_id].unique())
    else:
        assert is_sorted(follower_ids)

    r_t      = rank_of_src_in_df(df, src_id)
    u_values = r_t[follower_ids].values.dot(np.sqrt(q_vec / s))
    u_dt     = np.diff(np.concatenate([r_t.index.values, [end_time]]))

    return np.sum(u_values * u_dt)


def time_in_top_k(df, K, src_id=None, end_time=None, sim_opts=None):
    """Calculate ∫I(r(t) <= k)dt for the given src_id."""

    # if follower_ids is None:
    #     follower_ids = sorted(df[df.src_id == src_id].sink_id.unique())

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)

    r_t      = rank_of_src_in_df(df, src_id)
    I_values = np.where(r_t.mean(1) <= K - 1, 1.0, 0.0)
    I_dt     = np.diff(np.concatenate([r_t.index.values, [end_time]]))

    return np.sum(I_values * I_dt)


def average_rank(df, src_id=None, end_time=None, sim_opts=None, **kwargs):
    """Calculate ∫r(t)dt for the given src_id."""

    # if follower_ids is None:
    #     follower_ids = sorted(df[df.src_id == src_id].sink_id.unique())

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)

    r_t  = rank_of_src_in_df(df, src_id).mean(1)
    r_dt = np.diff(np.concatenate([r_t.index.values, [end_time]]))

    return np.sum(r_t * r_dt)

def int_r_2(df, sim_opts):
    """Returns ∫r²(t)dt for the given source id."""
    r_t = rank_of_src_in_df(df, sim_opts.src_id).mean(1)
    r_dt = np.diff(np.concatenate([r_t.index.values, [sim_opts.end_time]]))
    return np.sum(r_t ** 2 * r_dt)


def calc_loss_poisson(df, u_const, src_id=None, end_time=None,
                      q_vec=None, s=None, follower_ids=None, sim_opts=None):
    """Calculate the loss for the given source assuming that it was Poisson
    with rate u_const."""

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)
        q_vec        = mb(q_vec, sim_opts.q_vec)
        s            = mb(s, sim_opts.s)
        follower_ids = mb(follower_ids, sim_opts.sink_ids)

    assert is_sorted(follower_ids)

    if q_vec is None:
        q_vec = def_q_vec(len(follower_ids))

    r_t = rank_of_src_in_df(df, src_id)
    q_t = 0.5 * np.square(r_t[follower_ids].values).dot(q_vec)
    s_t = 0.5 * s * np.ones(r_t.shape[0], dtype=float) * (u_const ** 2)

    return pd.Series(data=q_t + s_t, index=r_t.index)


def calc_loss_opt(df, sim_opts):
    """Calculate the loss for the given source assuming that it was the
    optimal broadcaster."""

    follower_ids = sim_opts.sink_ids
    q_vec        = sim_opts.q_vec
    src_id       = sim_opts.src_id

    r_t = rank_of_src_in_df(df, src_id)
    q_t = 0.5 * np.square(r_t[follower_ids].values).dot(q_vec)
    s_t = q_t # For the optimal solution, the q_t is the same is s_t

    return pd.Series(data=q_t + s_t, index=r_t.index)


## Oracle

def oracle_ranking(df, sim_opts, omit_src_ids=None, follower_ids=None):
    """Returns the best places the oracle would have put events. Optionally, it
    can remove sources, use a custom weight vector and have a custom list of
    followers."""

    if omit_src_ids is not None:
        df = df[~df.src_id.isin(omit_src_ids)]

    if follower_ids is not None:
        df = sorted(df[df.sink_id.isin(follower_ids)])
    else:
        follower_ids = sorted(df.sink_id.unique())

    assert len(follower_ids) == 1, "Oracle has been implemented only for 1 follower."

    # TODO: Will need to update q_vec manually if we want to run the method
    # for a subset of the user's followers.
    q_vec = sim_opts.q_vec
    s = sim_opts.s

    assert is_sorted(df.t.values), "Dataframe is not sorted by time."
    event_times = df.groupby('event_id').t.mean()

    n = event_times.shape[0]
    if n > 1e6:
        print('Not running for n > 1e6 events')
        return []

    w = np.diff(np.concatenate([[0.0], [0.0], event_times.values, [sim_opts.end_time]]))

    # TODO: Check index/off by one.
    # Initialization sets the final penalty.
    J = np.zeros((n + 1, n + 2), dtype=float)

    J[:, n + 1] = (np.arange(n + 1) ** 2) / 2

    for k in range(n, -1, -1):
        # This can be made parallel and vectorized
        # Also, not the whole matrix needs to be filled in (reduce run-time by 50%)
        for r in range(min(k + 1, n)):
            J[r, k] = min(0.5 * s + J[0, k + 1],
                          0.5 * q_vec * w[k + 1] * ((r + 1) ** 2) + J[r + 1, k + 1])

    # We are implicitly assuming that the oracle starts with rank 0
    oracle_ranks = np.zeros(n + 1, dtype=int)
    u_star = np.zeros(n + 1, dtype=int)
    for k in range(n):
        lhs = 0.5 * s + J[0, k + 1]
        rhs = 0.5 * q_vec * w[k + 1] * ((oracle_ranks[k] + 1) ** 2) + J[oracle_ranks[k] + 1, k + 1]
        if lhs < rhs:
            u_star[k] = 1
            oracle_ranks[k + 1] = 0
        else:
            u_star[k] = 0
            oracle_ranks[k + 1] = oracle_ranks[k] + 1

    oracle_df = pd.DataFrame.from_dict({
        'ranks'   : oracle_ranks,
        'events'  : u_star,
        'at'      : np.concatenate([[0.0], event_times.values]),
        't'       : np.concatenate([[0.0], event_times.values]),
        't_delta' : w[1:]
    })

    return oracle_df, J[0, 0]


def get_oracle_df(sim_opts, with_cost=False):
    wall_mgr = sim_opts.create_manager_for_wall()
    wall_mgr.run()
    oracle_df, cost = oracle_ranking(df=wall_mgr.state.get_dataframe(),
                                     sim_opts=sim_opts)

    if with_cost:
        return oracle_df, cost
    else:
        return oracle_df


def find_opt_oracle(target_events, sim_opts, tol=1e-2, verbose=False):
    """Sweep the 's' parameter and get the best run of the oracle."""
    s_hi, s_init, s_lo = 1.0 * 2, 1.0, 1.0 / 2

    def oracle_num_events(s):
        oracle_df = get_oracle_df(sim_opts.update({ 's': s }))
        return oracle_df.events.sum()

    num_events = oracle_num_events(s_init)

    if num_events > target_events:
        while True:
            s_lo = s_init
            s_init *= 2
            s_hi = s_init
            num_events = oracle_num_events(s_init)
            if verbose:
                logTime('s_lo = {}, s_hi = {}, num_events = {} '
                        .format(s_lo, s_hi, num_events))
            if num_events < target_events:
                break
    elif num_events < target_events:
        while True:
            s_hi = s_init
            s_init /= 2
            s_lo = s_init
            num_events = oracle_num_events(s_init)
            if verbose:
                logTime('s_lo = {}, s_hi = {}, num_events = {} '
                        .format(s_lo, s_hi, num_events))
            if num_events > target_events:
                break

    if verbose:
        logTime('s_lo = {}, s_hi = {}'.format(s_lo, s_hi))

    while True:
        s_try = (s_lo + s_hi) / 2.0
        oracle_df, cost = get_oracle_df(sim_opts.update({ 's': s_try }),
                                        with_cost=True)
        opt_events = oracle_df.events.sum()

        if verbose:
            logTime('s_try = {}, events = {}, cost = {}'.format(s_try, opt_events, cost))

        if np.abs(opt_events - target_events) / (target_events * 1.0) < tol or \
            (opt_events == np.ceil(target_events)) or \
            (opt_events == np.floor(target_events)):
            return {
                's': s_try,
                'cost': cost,
                'df': oracle_df
            }
        elif opt_events < target_events:
            s_hi = s_try
        else:
            s_lo = s_try


def find_opt_oracle_s(target_events, sim_opts, tol=1e-1, verbose=False):
    res = find_opt_oracle(target_events, sim_opts, tol, verbose)
    return res['s']


def find_opt_oracle_time_top_k(target_events, K, sim_opts, tol=1e-1, verbose=False):
    print('This method is incorrect.', file=sys.stderr)
    res = find_opt_oracle(target_events, sim_opts, tol, verbose)
    df = res['df']
    return np.sum(df.t_delta[df.ranks <= K - 1])


## LaTeX

SPINE_COLOR = 'grey'
def latexify(fig_width=None, fig_height=None, columns=1, largeFonts=False):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9 # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0    # Aesthetic ratio
        fig_height = fig_width*golden_mean # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\\usepackage{gensymb}'],
              'axes.labelsize': 10 if largeFonts else 7, # fontsize for x and y labels (was 10)
              'axes.titlesize': 10 if largeFonts else 7,
              'font.size': 10 if largeFonts else 7, # was 10
              'legend.fontsize': 10 if largeFonts else 7, # was 10
              'xtick.labelsize': 10 if largeFonts else 7,
              'ytick.labelsize': 10 if largeFonts else 7,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif',
              'xtick.minor.size': 0.5,
              'xtick.major.pad': 1.5,
              'xtick.major.size': 1,
              'ytick.minor.size': 0.5,
              'ytick.major.pad': 1.5,
              'ytick.major.size': 1
    }

    matplotlib.rcParams.update(params)
    py.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


## Sweeping s

def q_int_worker(params):
    sim_opts, seed = params
    m = sim_opts.create_manager_with_opt(seed)
    m.run()
    return u_int_opt(m.state.get_dataframe(), sim_opts=sim_opts)


def calc_q_capacity_iter(sim_opts_gen, s, seeds=None, parallel=True):
    if seeds is None:
        seeds = range(10)

    def _get_sim_opts():
        return sim_opts_gen().update({ 's': s })

    capacities = np.zeros(len(seeds), dtype=float)
    if not parallel:
        for idx, seed in enumerate(seeds):
            m = _get_sim_opts().create_manager(seed)
            m.run()
            capacities[idx] = u_int_opt(m.state.get_dataframe(),
                                        sim_opts=_get_sim_opts())
    else:
        num_workers = min(len(seeds), mp.cpu_count())
        with mp.Pool(num_workers) as pool:
            for (idx, capacity) in \
                enumerate(pool.imap(q_int_worker, [(_get_sim_opts(), x)
                                                   for x in seeds])):
                capacities[idx] = capacity

    return capacities


def sweep_s(sim_opts_gen, capacity_cap, tol=1e-2, verbose=False, s_init=1.0):
    # We know that on average, the ∫u(t)dt decreases with increasing 's'

    # Step 1: Find the upper/lower bound by exponential increase/decrease
    init_cap = calc_q_capacity_iter(sim_opts_gen, s_init).mean()

    if verbose:
        logTime('Initial capacity = {}'.format(init_cap))

    s = s_init
    if init_cap < capacity_cap:
        while True:
            s_hi = s
            s /= 2.0
            s_lo = s
            if calc_q_capacity_iter(sim_opts_gen, s).mean() > capacity_cap:
                break
    else:
        while True:
            s_lo = s
            s *= 2.0
            s_hi = s
            if calc_q_capacity_iter(sim_opts_gen, s).mean() < capacity_cap:
                break

    if verbose:
        logTime('s_hi = {}, s_lo = {}'.format(s_hi, s_lo))

    # Step 2: Keep bisecting on 's' until we arrive at a close enough solution.
    while True:
        s = (s_hi + s_lo) / 2.0
        new_capacity = calc_q_capacity_iter(sim_opts_gen, s).mean()

        if verbose:
            logTime('new_capacity = {}, s = {}'.format(new_capacity, s))

        if abs(new_capacity - capacity_cap) / capacity_cap < tol:
            # Have converged
            break
        elif new_capacity > capacity_cap:
            s_lo = s
        else:
            s_hi = s

    # Step 3: Return
    return s



## Workers for metrics

import broadcast.opt.optimizer as Bopt

# This is how much after the event that the Oracle tweets.
oracle_eps = 1e-10

Ks = [1, 3, 5, 10, 20]
performance_fields = ['seed', 's', 'type'] + ['top_' + str(k) for k in Ks] + ['avg_rank', 'r_2']

def add_perf(op, df, sim_opts):
    for k in Ks:
        op['top_' + str(k)] = time_in_top_k(df=df, K=k, sim_opts=sim_opts)

    op['avg_rank'] = average_rank(df, sim_opts=sim_opts)
    op['r_2'] =  r_2(df, sim_opts=sim_opts)


def worker_opt(params):
    seed, sim_opts, queue = params
    sim_mgr = sim_opts.create_manager_with_opt(seed=seed)
    sim_mgr.run()
    df = sim_mgr.state.get_dataframe()
    capacity = u_int_opt(df=df, sim_opts=sim_opts)
    op = {
        'type': 'Opt',
        'seed': seed,
        'capacity': capacity,
        'sim_opts': sim_opts,
        's': sim_opts.s,
        'num_events': np.sum(df.src_id == sim_opts.src_id)
    }

    add_perf(op, df, sim_opts)

    if queue is not None:
        queue.put(op)

    return op


def worker_poisson(params):
    seed, capacity, sim_opts, queue = params
    sim_mgr = sim_opts.create_manager_with_poisson(seed=seed, capacity=capacity)
    sim_mgr.run()
    op = {
        'type': 'Poisson',
        'seed': seed,
        'sim_opts': sim_opts,
        's': sim_opts.s
    }

    df = sim_mgr.state.get_dataframe()

    add_perf(op, df, sim_opts)

    if queue is not None:
        queue.put(op)

    return op


def worker_oracle(params):
    seed, capacity, sim_opts, queue = params
    opt_oracle = find_opt_oracle(capacity, sim_opts)
    oracle_df = opt_oracle['df']
    opt_oracle_mgr = sim_opts.create_manager_with_times(oracle_df.t[oracle_df.events == 1] + oracle_eps)
    opt_oracle_mgr.run()
    df = opt_oracle_mgr.state.get_dataframe()

    op = {
        'type': 'Oracle',
        'seed': seed,
        'sim_opts': sim_opts,
        's': sim_opts.s,
        'r0_num_events': np.sum(oracle_df.events == 1),
        'num_events': np.sum(df.src_id == sim_opts.src_id)
    }


    add_perf(op, df, sim_opts)

    if queue is not None:
        queue.put(op)

    return op


def worker_kdd(params):
    seed, capacity, num_segments, sim_opts, world_changing_rates, queue = params

    T = sim_opts.end_time
    seg_len = T / num_segments

    if world_changing_rates is None:
        wall_mgr = sim_opts.create_manager_for_wall()
        wall_mgr.run()
        wall_df = wall_mgr.state.get_dataframe()
        seg_idx = (wall_df.t.values / T * num_segments).astype(int)
        wall_intensities = wall_df.groupby(seg_idx).size() / (T / num_segments)
    else:
        wall_intensities = world_changing_rates

    follower_wall_intensities = np.array([wall_intensities])
    follower_conn_prob = np.asarray([[1.0] * num_segments])
    follower_weights = [1.0]

    upper_bounds = np.array([1e6] * num_segments)
    threshold = 0.005

    op = {
        'type'   : 'kdd',
        'seed'   : seed,
        'sim_opts': sim_opts,
        's'      : sim_opts.s
    }

    best_avg_rank, best_avg_k = np.inf, -1
    best_r_2, best_r_2_k = np.inf, -1

    for k in Ks:
        def _util(x):
            return Bopt.utils.weighted_top_k(x,
                                             follower_wall_intensities,
                                             follower_conn_prob,
                                             follower_weights,
                                             k)
        def _util_grad(x):
            return Bopt.utils.weighted_top_k_grad(x,
                                                  follower_wall_intensities,
                                                  follower_conn_prob,
                                                  follower_weights,
                                                  k)

        x0 = np.ones(num_segments)

        kdd_opt = Bopt.optimize(util=_util,
                                util_grad=_util_grad,
                                budget=capacity,
                                upper_bounds=upper_bounds,
                                threshold=threshold,
                                x0=x0)

        piecewise_const_mgr = sim_opts.create_manager_with_piecewise_const(
            seed=seed,
            change_times=np.arange(num_segments) * seg_len,
            rates=kdd_opt / seg_len
        )
        piecewise_const_mgr.run()
        df = piecewise_const_mgr.state.get_dataframe()
        perf = time_in_top_k(df=df, K=K, sim_opts=sim_opts)
        op['top_' + str(k)] = perf

        avg_rank = average_rank(df, sim_opts=sim_opts)
        r_2 = int_r_2(df, sim_opts=sim_opts)

        op['avg_rank_' + str(k)] = avg_rank
        op['r_2_' + str(k)] = r_2


        if avg_rank < best_avg_rank:
            best_avg_rank = avg_rank
            best_avg_k = k

        if r_2 < best_r_2:
            best_r_2 = r_2
            best_r_2_k = k

    op['avg_rank']   = best_avg_rank
    op['avg_rank_k'] = best_avg_k
    op['r_2']        = best_r_2
    op['r_2_k']      = best_r_2_k

    if queue is not None:
        queue.put(op)

    return op



# This is an approach using the multiprocessing module without the Pool and using a queue to accumulate the results.
# This will lead to a better utilization of the CPU resources (hopefully) because the previous method only allowed
# parallization of the number of seeds.


@optioned(option_arg='opts')
def piecewise_sim_opt_factory(N, T, num_segments, world_rate):
    random_state = np.random.RandomState(42)
    world_changing_rates = random_state.uniform(low=world_rate / 2.0, high=world_rate, size=num_segments)
    world_change_times = np.arange(num_segments) * T / num_segments

    def sim_opts_gen(seed):
        return SimOpts.std_piecewise_const(world_rates=world_changing_rates,
                                           world_change_times=world_change_times,
                                           world_seed=seed + 42).update({'end_time': T })

    return Options(N=N, T=T, num_segments=num_segments, sim_opts_gen=sim_opts_gen)

simulation_opts = Options(world_rate=1000.0, world_alphs=1.0, world_beta=2.0,
                          N=10, T=1.0, num_segments=10)

poisson_inf_opts = simulation_opts.set_new(
    sim_opts_gen=lambda seed: SimOpts.std_poisson(world_rate=simulation_opts.world_rate,
                                                  world_seed=seed))
piecewise_inf_opts = piecewise_sim_opt_factory(opts=simulation_opts)
hawkes_inf_opts = simulation_opts.set_new(
    sim_opts_gen=lambda seed: SimOpts.std_hawkes(world_seed=seed,
                                                 world_lambda_0=simulation_opts.world_rate,
                                                 world_alpha=simulation_opts.world_alpha,
                                                 world_beta=simulation_opts.world_beta))

@optioned(option_arg='opts')
def run_inference(N, T, num_segments, sim_opts_gen):

    def extract_perf_fields(return_obj):
        """Extracts the relevant fields from the return object and returns them in a new dict."""
        result_dict = {}
        for field in performance_fields:
            result_dict[field] = return_obj[field]

        return result_dict

    processes = []
    queue = mp.Queue()
    results = []
    capacities = {}
    raw_results = []

    try:
        active_processes = 0
        for s in np.logspace(-8, 1, num=10):
            capacities[s] = []
            for seed in range(N):
                active_processes += 1
                sim_opts = sim_opts_gen(seed).update({ 's' : s })
                p = mp.Process(target=worker_opt,
                               args=((seed, sim_opts, queue),))
                processes.append(p)
                p.daemon = True
                p.start()

        logTime('Started all processes: {}'.format(active_processes))

        while active_processes > 0:
            # logTime('active_processes = {}'.format(active_processes))
            r = queue.get()
            raw_results.append(r)
            results.append(extract_perf_fields(r))

            if r['type'] == 'Opt':
                active_processes -= 1

                seed = r['seed']
                capacity = r['capacity']
                s = r['sim_opts'].s
                sim_opts = r['sim_opts']
                capacities[s].append(capacity)

                # Poisson

                p = mp.Process(target=worker_poisson, args=((seed, capacity, sim_opts, queue),))
                processes.append(p)
                p.daemon = True
                p.start()
                active_processes += 1

                # Oracle

                oracle_args = (seed, capacity, sim_opts, queue)
                p = mp.Process(target=worker_oracle, args=(oracle_args,))
                processes.append(p)
                p.daemon = True
                p.start()
                active_processes += 1

                # KDD solution

                # kdd_args = (seed, capacity, num_segments, sim_opts, world_changing_rates, queue)
                kdd_args = (seed, capacity, num_segments, sim_opts, None, queue)
                p = mp.Process(target=worker_kdd, args=(kdd_args,))
                processes.append(p)
                p.daemon = True
                p.start()
                active_processes += 1

            elif r['type'] == 'Poisson':
                active_processes -= 1
                if active_processes % 10 == 0:
                    logTime('Active processes = {}'.format(active_processes))

            elif r['type'] == 'Oracle':
                active_processes -= 1
                if active_processes % 10 == 0:
                    logTime('Active processes = {}'.format(active_processes))

            elif r['type'] == 'kdd':
                active_processes -= 1
                if active_processes % 10 == 0:
                    logTime('Active processes = {}'.format(active_processes))

            else:
                raise ValueError('Unknown type: {}'.format(r['type']))
    finally:
        # Attempt at cleanup
        print("Cleaning up {} processes".format(len(processes)))
        for p in processes:
            p.terminate()
            p.join()

    return Options(df=pd.DataFrame.from_records(results),
                   raw_results=raw_results,
                   capacities=capacities)
