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


