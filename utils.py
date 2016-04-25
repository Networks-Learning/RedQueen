# Helper functions

import numpy as np

def is_sorted(x, ascending=True):
    """Determines if a given numpy.array-like is sorted in ascending order."""
    return np.all((np.diff(x) * 1.0 if ascending else -1.0) >= 0)

def rank_of_src_in_df(df, src_id):
    """Calculates the rank of the src_id at each time instant in the list of events."""
    # Assumption df is sorted by the time.

    assert is_sorted(df.t.values), "Array not sorted by time."

    df2 = df.copy()
    df2['rank'] = (df.groupby('sink_id')
                     .src_id
                     .transform(lambda x: range(0, len(x)) -
                                          np.maximum.accumulate(np.where(x == src_id, range(0, len(x)), 0)))
                     .astype(float))
    return df2.pivot(index='t', columns='sink_id', values='rank').fillna(method='ffill')
