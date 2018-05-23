# Helper functions

import matplotlib
import logging
from matplotlib import pyplot as py
import multiprocessing as mp
import numpy as np
import pandas as pd
import sys
import datetime as D

from decorated_options import optioned

# Utilities


def mb(val, default):
    return val if val is not None else default


def logTime(chkpoint):
    print('*** \x1b[31m{}\x1b[0m Checkpoint: {}'.format(D.datetime.now(), chkpoint))
    sys.stdout.flush()


def def_s_vec(num_followers):
    """Returns the default s for the given number of followers."""
    return np.ones(num_followers, dtype=float) / (num_followers ** 2)


def is_sorted(x, ascending=True):
    """Determines if a given numpy.array-like is sorted in ascending order."""
    return np.all((np.diff(x) * (1.0 if ascending else -1.0) >= 0))


# Metrics

def rank_of_src_in_df(df, src_id, fill=True, with_time=True):
    """Calculates the rank of the src_id at each time instant in the list of events."""

    # assert is_sorted(df.t.values), "Array not sorted by time."

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


def u_int_opt(df, src_id=None, end_time=None, s=None, q=None,
              follower_ids=None, sim_opts=None):
    """Calculate the ∫u(t)dt for the given src_id assuming that the broadcaster
    was following the optimal strategy."""

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)
        s            = mb(s, sim_opts.s)
        q            = mb(q, sim_opts.q)
        follower_ids = mb(follower_ids, sim_opts.sink_ids)

    if follower_ids is None:
        follower_ids = sorted(df.sink_id[df.src_id == src_id].unique())
    else:
        pass
        # assert is_sorted(follower_ids)

    r_t      = rank_of_src_in_df(df, src_id)
    u_values = r_t[follower_ids].values.dot(np.sqrt(s / q))
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
    I_values = np.where(r_t <= K - 1, 1.0, 0.0).mean(1)
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
    """Returns ∫avg-rank²(t)dt for the given source id."""
    r_t = rank_of_src_in_df(df, sim_opts.src_id).mean(1)
    r_dt = np.diff(np.concatenate([r_t.index.values, [sim_opts.end_time]]))
    return np.sum(r_t ** 2 * r_dt)


def int_r_2_true(df, sim_opts):
    """Returns ∫avg(rank²(t))dt for the given source id."""
    r_t = rank_of_src_in_df(df, sim_opts.src_id)
    r_dt = np.diff(np.concatenate([r_t.index.values, [sim_opts.end_time]]))
    return np.sum((r_t ** 2).mean(1) * r_dt)


def calc_loss_poisson(df, u_const, src_id=None, end_time=None,
                      q=None, s=None, follower_ids=None, sim_opts=None):
    """Calculate the loss for the given source assuming that it was Poisson
    with rate u_const."""

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)
        q            = mb(q, sim_opts.q)
        s            = mb(s, sim_opts.s)
        follower_ids = mb(follower_ids, sim_opts.sink_ids)

    assert is_sorted(follower_ids)

    if s is None:
        s = def_s_vec(len(follower_ids))

    r_t = rank_of_src_in_df(df, src_id)
    s_t = 0.5 * np.square(r_t[follower_ids].values).dot(s)
    q_t = 0.5 * q * np.ones(r_t.shape[0], dtype=float) * (u_const ** 2)

    return pd.Series(data=q_t + s_t, index=r_t.index)


def calc_loss_opt(df, sim_opts):
    """Calculate the loss for the given source assuming that it was the
    optimal broadcaster."""

    follower_ids = sim_opts.sink_ids
    s            = sim_opts.s
    src_id       = sim_opts.src_id

    r_t = rank_of_src_in_df(df, src_id)
    s_t = 0.5 * np.square(r_t[follower_ids].values).dot(s)
    q_t = s_t  # For the optimal solution, the q_t is the same is s_t

    return pd.Series(data=q_t + s_t, index=r_t.index)


def num_tweets_of(df, broadcaster_id=None, sim_opts=None):
    """Returns number of tweets made by the given broadcaster in the data-frame."""
    if sim_opts is not None:
        broadcaster_id = mb(broadcaster_id, sim_opts.src_id)

    assert broadcaster_id is not None, "Must either provide either broadcaster_id or sim_opts."
    return 1.0 * df.event_id[df.src_id == broadcaster_id].nunique()


# Oracle

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
    q = sim_opts.q
    s = sim_opts.s

    # assert is_sorted(df.t.values), "Dataframe is not sorted by time."
    event_times = df.groupby('event_id').t.mean()

    n = event_times.shape[0]
    if n > 1e6:
        logging.error('Not running for n > 1e6 events')
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
            J[r, k] = min(0.5 * q + J[0, k + 1],
                          0.5 * s * w[k + 1] * ((r + 1) ** 2) + J[r + 1, k + 1])

    # We are implicitly assuming that the oracle starts with rank 0
    oracle_ranks = np.zeros(n + 1, dtype=int)
    u_star = np.zeros(n + 1, dtype=int)
    for k in range(n):
        lhs = 0.5 * q + J[0, k + 1]
        rhs = 0.5 * s * w[k + 1] * ((oracle_ranks[k] + 1) ** 2) + J[oracle_ranks[k] + 1, k + 1]
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
    wall_mgr.run_dynamic()
    oracle_df, cost = oracle_ranking(df=wall_mgr.state.get_dataframe(),
                                     sim_opts=sim_opts)

    if with_cost:
        return oracle_df, cost
    else:
        return oracle_df


def find_opt_oracle(target_events, sim_opts, max_events=None, tol=1e-2, verbose=False):
    """Sweep the 's' parameter and get the best run of the oracle."""
    q_hi, q_init, q_lo = 1.0 * 2, 1.0, 1.0 / 2

    def terminate_cond(opt_events):
        return np.abs(opt_events - target_events) / (target_events * 1.0) < tol or \
            (opt_events == np.ceil(target_events)) or \
            (opt_events == np.floor(target_events))

    oracle_df, cost = get_oracle_df(sim_opts.update({'q': q_init}),
                                    with_cost=True)
    num_events = oracle_df.events.sum()

    if terminate_cond(num_events):
        return {
            'q': q_init,
            'cost': cost,
            'oracle_df': oracle_df
        }

    if num_events > target_events:
        while True:
            q_lo = q_init
            q_init *= 2
            q_hi = q_init
            oracle_df, cost = get_oracle_df(sim_opts.update({'q': q_init}),
                                            with_cost=True)
            num_events = oracle_df.events.sum()
            if verbose:
                logTime('q_lo = {}, q_hi = {}, num_events = {} '
                        .format(q_lo, q_hi, num_events))
            if terminate_cond(num_events):
                return {
                    'q': q_init,
                    'cost': cost,
                    'df': oracle_df
                }
            if num_events <= target_events:
                break
    elif num_events < target_events:
        while True:
            q_hi = q_init
            q_init /= 2
            q_lo = q_init
            oracle_df, cost = get_oracle_df(sim_opts.update({'q': q_init}),
                                            with_cost=True)
            num_events = oracle_df.events.sum()
            if verbose:
                logTime('q_lo = {}, q_hi = {}, num_events = {} '
                        .format(q_lo, q_hi, num_events))
            if terminate_cond(num_events):
                return {
                    'q': q_init,
                    'cost': cost,
                    'df': oracle_df
                }
            if num_events >= target_events or num_events == max_events:
                break

    if verbose:
        logTime('q_lo = {}, q_hi = {}'.format(q_lo, q_hi))

    while True:
        q_try = (q_lo + q_hi) / 2.0
        oracle_df, cost = get_oracle_df(sim_opts.update({'q': q_try}),
                                        with_cost=True)
        opt_events = oracle_df.events.sum()

        if verbose:
            logTime('q_try = {}, events = {}, cost = {}'.format(q_try, opt_events, cost))

        if terminate_cond(opt_events):
            return {
                'q': q_try,
                'cost': cost,
                'df': oracle_df
            }
        elif opt_events < target_events:
            q_hi = q_try
        else:
            q_lo = q_try


def find_opt_oracle_q(target_events, sim_opts, tol=1e-1, verbose=False):
    res = find_opt_oracle(target_events, sim_opts, tol, verbose)
    return res['q']


def find_opt_oracle_time_top_k(target_events, K, sim_opts, tol=1e-1, verbose=False):
    logTime('This method is incorrect.')
    res = find_opt_oracle(target_events, sim_opts, tol, verbose)
    df = res['df']
    return np.sum(df.t_delta[df.ranks <= K - 1])


# LaTeX

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

    assert(columns in [1, 2])

    if fig_width is None:
        fig_width = 3.39 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (np.sqrt(5) - 1.0) / 2.0    # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        logging.warning("WARNING: fig_height too large:" + fig_height +
                        "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
        'backend': 'ps',
        'text.latex.preamble': ['\\usepackage{gensymb}'],
        'axes.labelsize': 10 if largeFonts else 7,  # fontsize for x and y labels (was 10)
        'axes.titlesize': 10 if largeFonts else 7,
        'font.size': 10 if largeFonts else 7,  # was 10
        'legend.fontsize': 10 if largeFonts else 7,  # was 10
        'xtick.labelsize': 10 if largeFonts else 7,
        'ytick.labelsize': 10 if largeFonts else 7,
        'text.usetex': True,
        'figure.figsize': [fig_width, fig_height],
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


# Sweeping q

def q_int_worker(params):
    sim_opts, seed, dynamic, max_events = params
    m = sim_opts.create_manager_with_opt(seed)
    if dynamic:
        m.run_dynamic(max_events=max_events)
    else:
        m.run()
    # return u_int_opt(m.state.get_dataframe(), sim_opts=sim_opts)
    return num_tweets_of(m.state.get_dataframe(), sim_opts=sim_opts)



def calc_q_capacity_iter(sim_opts, q, seeds=None, parallel=True, dynamic=True, max_events=None):
    if seeds is None:
        seeds = range(100, 120)

    sim_opts = sim_opts.update({'q': q})

    capacities = np.zeros(len(seeds), dtype=float)
    if not parallel:
        for idx, seed in enumerate(seeds):
            m = sim_opts.create_manager_with_opt(seed)
            if dynamic:
                m.run_dynamic(max_events=max_events)
            else:
                m.run()
            capacities[idx] = num_tweets_of(m.state.get_dataframe(), sim_opts=sim_opts)
    else:
        num_workers = min(len(seeds), mp.cpu_count())
        with mp.Pool(num_workers) as pool:
            for (idx, capacity) in \
                enumerate(pool.imap(q_int_worker, [(sim_opts, x, dynamic, max_events)
                                                   for x in seeds])):
                capacities[idx] = capacity

    return capacities


@optioned('opts')
def convert_to_bins(ts, start_time, num_segments, segment_length=None, time_period=None):
        """Changes post-times to segment indexes."""
        if time_period is None:
            time_period = num_segments * segment_length

        return (((np.asarray(ts) - start_time) % time_period / time_period) * num_segments).astype(int)


def significance_q_int_worker(params):
    sim_opts, seed, time_period = params
    m = sim_opts.create_manager_with_significance(seed,
                                                  significance=sim_opts.s,
                                                  time_period=time_period)
    m.run_dynamic()
    # return u_int_opt(m.state.get_dataframe(), sim_opts=sim_opts)
    # Opting for the simpler measurement of \int u(t) dt.
    return num_tweets_of(m.state.get_dataframe(), sim_opts=sim_opts)


@optioned('opts')
def calc_significance_capacity_iter(sim_opts, q, time_period, seeds=None, parallel=True, max_events=None):
    if seeds is None:
        seeds = range(1000, 1025)

    sim_opts = sim_opts.update({'q': q})
    capacities = np.zeros(len(seeds), dtype=float)
    if not parallel:
        for idx, seed in enumerate(seeds):
            m = sim_opts.create_manager_with_significance(seed=seed,
                                                          significance=sim_opts.s,
                                                          time_period=time_period)
            m.run_dynamic()
            capacities[idx] = u_int_opt(m.state.get_dataframe(),
                                        sim_opts=sim_opts)
    else:
        num_workers = min(len(seeds), mp.cpu_count())
        with mp.Pool(num_workers) as pool:
            for (idx, capacity) in \
                enumerate(pool.imap(significance_q_int_worker, [(sim_opts, x, time_period)
                                                                 for x in seeds])):
                capacities[idx] = capacity

    return capacities


# There are so many ways this can go south. Particularly, if the user capacity
# is much higher than the average of the wall of other followees.
def sweep_q(sim_opts, capacity_cap, tol=1e-2, verbose=False, q_init=None, parallel=True,
            dynamic=True, max_events=None, max_iters=float('inf'), only_tol=False):
    # We know that on average, the ∫u(t)dt decreases with increasing 'q'

    def terminate_cond(new_capacity):
        return abs(new_capacity - capacity_cap) / capacity_cap < tol or \
            (not only_tol and np.ceil(capacity_cap - 1) <= new_capacity <= np.ceil(capacity_cap + 1))
            # np.ceil(capacity_cap - 1) <= new_capacity <= np.ceil(capacity_cap + 1)

    if q_init is None:
        wall_mgr = sim_opts.create_manager_for_wall()
        wall_mgr.run_dynamic()
        r_t = rank_of_src_in_df(wall_mgr.state.get_dataframe(), -1)
        q_init = (4 * (r_t.iloc[-1].mean() ** 2) * (sim_opts.end_time) ** 2) / (np.pi * np.pi * (capacity_cap + 1) ** 4)
        if verbose:
            logTime('q_init = {}'.format(q_init))

    # Step 1: Find the upper/lower bound by exponential increase/decrease
    init_cap = calc_q_capacity_iter(sim_opts, q_init, dynamic=dynamic, parallel=parallel, max_events=max_events).mean()

    if terminate_cond(init_cap):
        return q_init

    if verbose:
        logTime('Initial capacity = {}, target capacity = {}, q_init = {}'
                .format(init_cap, capacity_cap, q_init))

    q = q_init
    if init_cap < capacity_cap:
        iters = 0
        while True:
            iters += 1
            q_hi = q
            q /= 2.0
            q_lo = q
            capacity = calc_q_capacity_iter(sim_opts, q, dynamic=dynamic, parallel=parallel, max_events=max_events).mean()
            if verbose:
                logTime('q = {}, capacity = {}'.format(q, capacity))
            if terminate_cond(capacity):
                return q
            if capacity >= capacity_cap:
                break
            if iters > max_iters:
                if verbose:
                    logTime('Breaking because of max-iters: {}.'.format(max_iters))
                return q
    else:
        iters = 0
        while True:
            iters += 1
            q_lo = q
            q *= 2.0
            q_hi = q
            capacity = calc_q_capacity_iter(sim_opts, q, dynamic=dynamic, parallel=parallel, max_events=max_events).mean()
            if verbose:
                logTime('q = {}, capacity = {}'.format(q, capacity))
            # TODO: will break if capacity_cap is too low ~ 1 event.
            if terminate_cond(capacity):
                return q
            if capacity <= capacity_cap:
                break
            if iters > max_iters:
                if verbose:
                    logTime('Breaking because of max-iters: {}.'.format(max_iters))
                return q

    if verbose:
        logTime('q_hi = {}, q_lo = {}'.format(q_hi, q_lo))

    # Step 2: Keep bisecting on 's' until we arrive at a close enough solution.
    while True:
        q = (q_hi + q_lo) / 2.0
        new_capacity = calc_q_capacity_iter(sim_opts, q, dynamic=dynamic, parallel=parallel, max_events=max_events).mean()

        if verbose:
            logTime('new_capacity = {}, q = {}'.format(new_capacity, q))

        if terminate_cond(new_capacity):
            # Have converged
            break
        elif new_capacity > capacity_cap:
            q_lo = q
        else:
            q_hi = q

    # Step 3: Return
    return q


# There are so many ways this can go south. Particularly, if the user capacity
# is much higher than the average of the wall of other followees.
def sweep_q_with_significance(sim_opts,
                              capacity_cap,
                              time_period,
                              tol=1e-2,
                              parallel=True,
                              verbose=False,
                              q_init=None):
    # We know that on average, the ∫u(t)dt decreases with increasing 'q'

    def terminate_cond(new_capacity):
        return abs(new_capacity - capacity_cap) / capacity_cap < tol or \
            np.ceil(capacity_cap - 1) <= new_capacity <= np.ceil(capacity_cap + 1)

    if q_init is None:
        wall_mgr = sim_opts.create_manager_for_wall()
        wall_mgr.run_dynamic()
        r_t = rank_of_src_in_df(wall_mgr.state.get_dataframe(), -1)
        q_init = (4 * (r_t.iloc[-1].mean() ** 2) * (sim_opts.end_time) ** 2) / (np.pi * np.pi * (capacity_cap + 1) ** 4)
        if verbose:
            logTime('q_init = {}'.format(q_init))

    # Step 1: Find the upper/lower bound by exponential increase/decrease
    init_cap = calc_significance_capacity_iter(
        sim_opts=sim_opts,
        q=q_init,
        time_period=time_period,
        parallel=parallel
    ).mean()

    if terminate_cond(init_cap):
        logTime('q_init meets the conditions.')
        return q_init

    if verbose:
        logTime('Initial capacity = {}, target capacity = {}, q_init = {}'
                .format(init_cap, capacity_cap, q_init))

    q = q_init
    if init_cap < capacity_cap:
        while True:
            q_hi = q
            q /= 2.0
            q_lo = q
            capacity = calc_significance_capacity_iter(
                sim_opts=sim_opts,
                q=q,
                time_period=time_period,
                parallel=parallel
            ).mean()
            if verbose:
                logTime('q = {}, capacity = {}'.format(q, capacity))
            if terminate_cond(capacity):
                return q
            if capacity >= capacity_cap:
                break
    else:
        while True:
            q_lo = q
            q *= 2.0
            q_hi = q
            capacity = calc_significance_capacity_iter(
                sim_opts=sim_opts,
                q=q,
                time_period=time_period,
                parallel=parallel
            ).mean()
            if verbose:
                logTime('q = {}, capacity = {}'.format(q, capacity))
            # TODO: will break if capacity_cap is too low ~ 1 event.
            if terminate_cond(capacity):
                return q
            if capacity <= capacity_cap:
                break

    if verbose:
        logTime('q_hi = {}, q_lo = {}'.format(q_hi, q_lo))

    # Step 2: Keep bisecting on 's' until we arrive at a close enough solution.
    while True:
        q = (q_hi + q_lo) / 2.0
        new_capacity = calc_significance_capacity_iter(
            sim_opts=sim_opts,
            q=q,
            time_period=time_period,
            parallel=parallel
        ).mean()

        if verbose:
            logTime('new_capacity = {}, q = {}'.format(new_capacity, q))

        if terminate_cond(new_capacity):
            # Have converged
            break
        elif new_capacity > capacity_cap:
            q_lo = q
        else:
            q_hi = q

    # Step 3: Return
    return q
