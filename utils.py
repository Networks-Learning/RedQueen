# Helper functions

import matplotlib
from matplotlib import pyplot as py
import numpy as np
import pandas as pd
import sys
import datetime as D


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
    return np.all((np.diff(x) * 1.0 if ascending else -1.0) >= 0)


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


def u_int(df, src_id=None, end_time=None, q_vec=None, s=None,
          follower_ids=None, sim_opts=None):
    """Calculate the ∫u(t)dt for the given src_id."""

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)
        q_vec        = mb(q_vec, sim_opts.q_vec)
        s            = mb(s, sim_opts.s)
        follower_ids = mb(follower_ids, sim_opts.sink_ids)

    assert is_sorted(follower_ids)

    r_t      = rank_of_src_in_df(df, src_id)
    u_values = r_t[follower_ids].values.dot(np.sqrt(q_vec / s))
    u_dt     = np.diff(np.concatenate([r_t.index.values, [end_time]]))

    return np.sum(u_values * u_dt)


def time_in_top_k(df, src_id=None, K=None, end_time=None, q_vec=None, s=None,
                  follower_ids=None, sim_opts=None):
    """Calculate ∫I(r(t) <= k)dt for the given src_id."""

    # if follower_ids is None:
    #     follower_ids = sorted(df[df.src_id == src_id].sink_id.unique())

    if sim_opts is not None:
        src_id       = mb(src_id, sim_opts.src_id)
        end_time     = mb(end_time, sim_opts.end_time)
        q_vec        = mb(q_vec, sim_opts.q_vec)
        s            = mb(s, sim_opts.s)
        follower_ids = mb(follower_ids, sim_opts.sink_ids)

    assert is_sorted(follower_ids)

    r_t      = rank_of_src_in_df(df, src_id)
    u_values = r_t[follower_ids].values.dot(np.sqrt(q_vec / s))
    I_values = np.where(u_values <= K - 1, 1.0, 0.0)
    I_dt     = np.diff(np.concatenate([r_t.index.values, [end_time]]))

    return np.sum(I_values * I_dt)


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


def calc_loss_opt(df, src_id, end_time,
                  q_vec=None, s=1.0, follower_ids=None):
    """Calculate the loss for the given source assuming that it was the
    optimal broadcaster."""

    if follower_ids is None:
        follower_ids = sorted(df[df.src_id == src_id].sink_id.unique())

    if q_vec is None:
        q_vec = def_q_vec(len(follower_ids))

    r_t = rank_of_src_in_df(df, src_id)
    q_t = 0.5 * np.square(r_t[follower_ids].values).dot(q_vec)
    s_t = q_t # For the optimal solution, the q_t is the same is s_t

    return pd.Series(data=q_t + s_t, index=r_t.index)


# def oracle_ranking(df, K,
#                    omit_src_ids=None,
#                    follower_ids=None,
#                    q_vec=None,
#                    s=1.0):
#     """Returns the best places the oracle would have put events. Optionally, it
#     can remove sources, use a custom weight vector and have a custom list of
#     followers.."""
#
#     if omit_src_ids is not None:
#         df = df[~df.src_id.isin(omit_src_ids)]
#
#     if follower_ids is not None:
#         df = sorted(df[df.sink_id.isin(follower_ids)])
#     else:
#         follower_ids = sorted(df.sink_id.unique())
#
#     if q_vec is None:
#         q_vec = def_q_vec(len(follower_ids))
#
#     assert is_sorted(df.t.values), "Dataframe is not sorted by time."
#
#     # Find ∫r(t)dt if the Oracle had tweeted once in the beginning and then
#     # had not tweeted ever afterwards.
#
#     # There will be some NaN till the point all the sources had tweeted.
#     # Also, the ranks will start from 0. Fixing that:
#     oracle_rank = rank_of_src_in_df(df, src_id=None, with_time=False).fillna(0)
#
#     # Though each event should have a unique time, we take the 'mean' to catch
#     # bugs which may otherwise go unnoticed if we used .min or .first
#     event_times = df.groupby('event_id').t.mean()
#     event_ids = oracle_rank.index.values
#
#     r_values = oracle_rank[follower_ids].values.dot(np.sqrt(q_vec / s))
#     r = pd.Series(data=r_values, index=event_ids)
#
#     T = np.inf
#     oracle_event_times = [None] * K
#     K_orig = K
#
#     while K > 0:
#         all_moves = [(r[e_id] * (T - event_times[e_id]), event_times[e_id])
#                      for e_id in event_times.index[event_times < T]]
#         if len(all_moves) == 0:
#             print('Ran out of moves after {} steps'.format(K_orig - K))
#             break
#
#         best_move = sorted(all_moves, reverse=True)[0]
#         K -= 1
#         oracle_event_times[K] = best_move[1]
#         T = best_move[1]
#
#     return oracle_event_times


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


## Determining the q for a user

def q_int_worker(params):
    sim_opts, seed = params
    m = sim_opts.create_manager(seed)
    m.run_till(sim_opts.end_time)
    return u_int(m.state.get_dataframe(), sim_opts=sim_opts)


def calc_q_capacity_iter(sim_opts, seeds=None, parallel=True):
    if seeds is None:
        seeds = range(10)

    capacities = np.zeros(len(seeds), dtype=float)
    if not parallel:
        for idx, seed in enumerate(seeds):
            m = sim_opts.create_manager(seed)
            m.run_till(sim_opts.end_time)
            capacities[idx] = u_int(m.state.get_dataframe(), sim_opts=sim_opts)
    else:
        num_workers = min(len(seeds), mp.cpu_count())
        with mp.Pool(num_workers) as pool:
            for (idx, capacity) in \
                enumerate(pool.imap_unordered(q_int_worker, [(sim_opts, x)
                                                             for x in seeds])):
                capacities[idx] = capacity

    return capacities


def sweep_s(sim_opts, capacity_cap, rel_tol=1e-2, verbose=False, s_init=1.0):
    # We know that on average, the ∫u(t)dt decreases with increasing 's'

    # Step 1: Find the upper/lower bound by exponential increase/decrease
    init_cap = calc_q_capacity_iter(sim_opts.update({ 's': s_init })).mean()

    if verbose:
        logTime('Initial capacity = {}'.format(init_cap))

    s = s_init
    if init_cap < capacity_cap:
        while calc_q_capacity_iter(sim_opts.update({ 's': s })).mean() < capacity_cap:
            s_hi = s
            s /= 2.0
            s_lo = s
    else:
        while calc_q_capacity_iter(sim_opts.update({ 's': s })).mean() > capacity_cap:
            s_lo = s
            s *= 2.0
            s_hi = s

    if verbose:
        logTime('s_hi = {}, s_lo = {}'.format(s_hi, s_lo))

    # Step 3: Keep bisecting on 's' until we arrive at a close enough solution.
    old_capacity = np.inf

    while True:
        s = (s_hi + s_lo) / 2.0
        new_capacity = calc_q_capacity_iter(sim_opts.update({ 's': s })).mean()

        if verbose:
            logTime('new_capacity = {}, s = {}'.format(new_capacity, s))

        if abs(new_capacity - old_capacity) / old_capacity < rel_tol:
            # Have converged
            break
        else:
            old_capacity = new_capacity

        if new_capacity > capacity_cap:
            s_lo = s
        else:
            s_hi = s

    # Step 4: Return
    return s

