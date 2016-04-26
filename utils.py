# Helper functions

import numpy as np
import pandas as pd


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


def u_int(df, src_id, end_time, q_vec=None, s=1.0, follower_ids=None):
    """Calculate the ∫u(t)dt for the given src_id."""

    if follower_ids is None:
        follower_ids = sorted(df[df.src_id == src_id].sink_id.unique())

    if q_vec is None:
        q_vec = np.ones_like(follower_ids, dtype=float)

    r_t = rank_of_src_in_df(df, src_id)
    u_values = r_t[follower_ids].values.dot(q_vec)
    u_dt = np.diff(np.concatenate([r_t.index.values, [end_time]]))

    return np.sum(u_values * u_dt)


def oracle_ranking(df, K,
                   omit_src_ids=None,
                   follower_ids=None,
                   q_vec=None,
                   s=1.0):
    """Returns the best places the oracle would have put events. Optionally, it
    can remove sources, use a custom weight vector and have a custom list of
    followers.."""

    if omit_src_ids is not None:
        df = df[~df.src_id.isin(omit_src_ids)]

    if follower_ids is not None:
        df = sorted(df[df.sink_id.isin(follower_ids)])
    else:
        follower_ids = df.sink_id.unique()

    if q_vec is None:
        q_vec = np.ones(len(follower_ids), dtype=float)

    assert is_sorted(df.t.values), "Dataframe is not sorted by time."

    # Find ∫r(t)dt if the Oracle had tweeted once in the beginning and then
    # had not tweeted ever afterwards.

    # There will be some NaN till the point all the sources had tweeted.
    # Also, the ranks will start from 0. Fixing that:
    oracle_rank = rank_of_src_in_df(df, src_id=None, with_time=False).fillna(0)

    # Though each event should have a unique time, we take the 'mean' to catch
    # bugs which may otherwise go unnoticed if we used .min or .first
    event_times = df.groupby('event_id').t.mean()
    event_ids = oracle_rank.index.values

    r_values = oracle_rank[follower_ids].values.dot(np.sqrt(q_vec) / s)
    r = pd.Series(data=r_values, index=event_ids)

    T = np.inf
    oracle_event_times = [None] * K
    K_orig = K

    while K > 0:
        all_moves = [(r[e_id] * (T - event_times[e_id]), event_times[e_id])
                     for e_id in event_times.index[event_times < T]]
        if len(all_moves) == 0:
            print('Ran out of moves after {} steps'.format(K_orig - K))
            break

        best_move = sorted(all_moves, reverse=True)[0]
        K -= 1
        oracle_event_times[K] = best_move[1]
        T = best_move[1]

    return oracle_event_times
