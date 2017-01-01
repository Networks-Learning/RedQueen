### Reading real data

from opt_models import SimOpts

import os
import re

base = '/NL/ghtorrent/work/opt-broadcast'
user_data_regex = re.compile(r'user-([0-9]*)\.pickle')
sizes_files = sorted([(os.path.getsize(os.path.join(base, x)), x) for x in os.listdir(base) if user_data_regex.match(x)])
file_user_id = [(os.path.join(base, x), int(user_data_regex.match(x).group(1))) for _, x in sizes_files]


## Saving raw_data

def change_sim_opts(d):
    d2 = d.copy()
    del d2['sim_opts']
    d2['sim_opts_dict'] = d['sim_opts'].get_dict()
    return d2

def revert_to_sim_opts(d):
    d2 = d.copy()
    del d2['sim_opts_dict']
    d2['sim_opts'] = SimOpts(**d['sim_opts_dict'])
    return d2


def get_savable_raw_results(r):
    return [change_sim_opts(d) for d in r]


