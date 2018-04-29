# Reading real data

import warnings

try:
    from .opt_model import SimOpts
except:
    # May be in the current scope via %run -i
    warnings.warn('SimOpts not loaded via import.')

import os
import re

base = '/NL/redqueen/work/opt-broadcast'
user_data_regex = re.compile(r'user-([0-9]*)\.pickle')
sizes_files = sorted([(os.path.getsize(os.path.join(base, x)), x) for x in os.listdir(base) if user_data_regex.match(x)])
file_user_id = [(os.path.join(base, x), int(user_data_regex.match(x).group(1))) for _, x in sizes_files]


# Saving raw_data
def change_sim_opts(d):
    d2 = d.copy()
    del d2['sim_opts']
    d2['sim_opts_dict'] = d['sim_opts'].get_dict()
    return d2


def revert_to_sim_opts(d):
    d2 = d.copy()
    del d2['sim_opts_dict']
    sim_opts_dict = d['sim_opts_dict'].copy()
    sim_opts_dict['s'] = d['sim_opts_dict']['q_vec']
    sim_opts_dict['q'] = d['sim_opts_dict']['s']
    del sim_opts_dict['q_vec']
    d2['sim_opts'] = SimOpts(**sim_opts_dict)
    return d2


def get_savable_raw_results(r):
    return [change_sim_opts(d) for d in r]
