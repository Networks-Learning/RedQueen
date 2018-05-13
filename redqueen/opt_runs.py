import warnings
import pickle

import sys
from decorated_options import optioned, Options
import numpy as np
from collections import defaultdict
import logging
import multiprocessing as mp
import pandas as pd

try:
    from .utils import time_in_top_k, average_rank, int_r_2, logTime, find_opt_oracle, sweep_q
    from .opt_model import SimOpts
except:
    # Ignore imports because they may have been imported directly using
    # %run -i
    pass


try:
    import broadcast.opt.optimizer as Bopt
except:
    warnings.warn('broadcast.opt.optimizer was NOT imported. '
                  'Comparison against method of Karimi et. al. method will '
                  'not be possible.')


# Workers for metrics

# Ks = [1, 5, 10]
# Ks = [1, 5]
Ks = [1]
perf_opts = Options(oracle_eps=1e-10,  # This is how much after the event that the Oracle tweets.
                    Ks=Ks,
                    performance_fields=['seed', 'q', 'type'] +
                                       ['top_' + str(k) for k in Ks] +
                                       ['avg_rank', 'r_2', 'num_events', 'world_events'])


def add_perf(op, df, sim_opts):
    for k in perf_opts.Ks:
        op['top_' + str(k)] = time_in_top_k(df=df, K=k, sim_opts=sim_opts)

    op['avg_rank'] = average_rank(df, sim_opts=sim_opts)
    op['r_2'] = int_r_2(df, sim_opts=sim_opts)
    op['world_events'] = len(df.event_id[df.src_id != sim_opts.src_id].unique())
    op['num_events'] = len(df.event_id[df.src_id == sim_opts.src_id].unique())


def worker_opt(params):
    try:
        seed, sim_opts, num_segments, queue = params
    except ValueError:
        logging.warning('Setting num_segments=10 for world-rate in worker_opt.')
        seed, sim_opts, queue = params
        num_segments = 10

    sim_mgr = sim_opts.create_manager_with_opt(seed=seed)
    sim_mgr.run_dynamic()
    df = sim_mgr.state.get_dataframe()
    # If Capacity if calculated this way, then the Optimal Broadcaster
    # May end up with number of tweets higher than the number of tweets
    # produced by the rest of the world.
    # capacity = u_int_opt(df=df, sim_opts=sim_opts)

    # All tweets except by the optimal worker are wall tweets.
    # this is valid only if other broadcasters do not react to the optimal
    # broadcaster or do not deviate from their strategy (modulo variation due
    # to different seeds of the random number generator).
    wall_df = df[df.src_id != sim_opts.src_id]
    T = sim_opts.end_time
    seg_idx = (wall_df.t.values / T * num_segments).astype(int)
    intensity_df = (
        (wall_df.groupby(['sink_id', pd.Series(seg_idx, name='segment')]).size() / (T / num_segments))
        .sort_index()
        .reset_index(name='intensity')
    )

    # Order of walls is ambiguous here.
    wall_intensities = (
        intensity_df.pivot_table(values='intensity', index='sink_id', columns='segment')
        .ix[sim_opts.sink_ids]  # Sort the data according to the sink_ids in sim_opts.
        .values
    )

    # Note: this works only if the optimal follower has exactly one follower. It is better to count the number
    # of distinct times that the optimal broadcaster tweeted.
    # capacity = (df.src_id == sim_opts.src_id).sum() * 1.0
    num_events = len(df.event_id[df.src_id == sim_opts.src_id].unique())
    capacity = num_events * 1.0
    op = {
        'type'             : 'Opt',
        'seed'             : seed,
        'capacity'         : capacity,
        'sim_opts'         : sim_opts,
        'q'                : sim_opts.q,
        'wall_intensities' : wall_intensities
    }

    add_perf(op, df, sim_opts)

    if queue is not None:
        queue.put(op)

    return op


def worker_poisson(params):
    seed, capacity, sim_opts, queue = params
    sim_mgr = sim_opts.create_manager_with_poisson(seed=seed, capacity=capacity)
    sim_mgr.run_dynamic()
    df = sim_mgr.state.get_dataframe()
    op = {
        'type': 'Poisson',
        'seed': seed,
        'sim_opts': sim_opts,
        'q': sim_opts.q
    }

    add_perf(op, df, sim_opts)

    if queue is not None:
        queue.put(op)

    return op


def worker_oracle(params):
    seed, capacity, max_events, sim_opts, queue = params
    opt_oracle = find_opt_oracle(capacity, sim_opts, max_events=max_events)
    oracle_df = opt_oracle['df']

    # TODO: This method of extracting times works before oracle is always run only
    # for one follower.
    opt_oracle_mgr = sim_opts.create_manager_with_times(oracle_df.t[oracle_df.events == 1] +
                                                        perf_opts.oracle_eps)
    opt_oracle_mgr.run_dynamic()
    df = opt_oracle_mgr.state.get_dataframe()

    op = {
        'type'          : 'Oracle',
        'seed'          : seed,
        'sim_opts'      : sim_opts,
        'q'             : sim_opts.q,
        'r0_num_events' : np.sum(oracle_df.events == 1),
        'num_events'    : np.sum(df.src_id == sim_opts.src_id)
    }

    add_perf(op, df, sim_opts)

    if queue is not None:
        queue.put(op)

    return op


def worker_kdd(params, window_start=0, verbose=False, Ks=None):
    seed, capacity, num_segments, sim_opts, world_changing_rates, queue = params

    T = sim_opts.end_time - window_start
    seg_len = T / num_segments

    if world_changing_rates is None:
        wall_mgr = sim_opts.create_manager_for_wall()
        wall_mgr.run_dynamic()
        wall_df = wall_mgr.state.get_dataframe()
        seg_idx = (wall_df.t.values / T * num_segments).astype(int)
        intensity_df = (wall_df.groupby(['sink_id', pd.Series(seg_idx, name='segment')]).size() / (T / num_segments)).reset_index(name='intensity')
        wall_intensities = intensity_df.pivot_table(values='intensity', index='sink_id', columns='segment').values
    else:
        wall_intensities = np.asarray(world_changing_rates)

    follower_wall_intensities = wall_intensities
    follower_conn_prob = np.asarray([[1.0] * num_segments] * len(sim_opts.sink_ids))
    follower_weights = [1.0] * len(sim_opts.sink_ids)

    upper_bounds = np.array([1e11] * num_segments)
    threshold = 0.02

    op = {
        'type'     : 'kdd',
        'seed'     : seed,
        'sim_opts' : sim_opts,
        'q'        : sim_opts.q
    }

    best_avg_rank, best_avg_k = np.inf, -1
    best_r_2, best_r_2_k = np.inf, -1

    if Ks is None:
        Ks = perf_opts.Ks

    for k in Ks:
        if k != 1:
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
        else:

            # For k = 1, special case of gradient calculation
            def _util(x):
                return Bopt.utils.weighted_top_one(x,
                                                   follower_wall_intensities,
                                                   follower_conn_prob,
                                                   follower_weights)

            def _util_grad(x):
                return Bopt.utils.weighted_top_one_grad(x,
                                                        follower_wall_intensities,
                                                        follower_conn_prob,
                                                        follower_weights)

        # Initial guess is close to Poisson solution
        x0 = np.ones(num_segments) * capacity / num_segments

        kdd_opt, iters = Bopt.optimize(
            util         = _util,
            util_grad    = _util_grad,
            budget       = capacity,
            upper_bounds = upper_bounds,
            threshold    = threshold,
            x0           = x0,
            verbose      = verbose,
            with_iter    = True
        )

        op['kdd_opt_' + str(k)] = kdd_opt
        op['kdd_opt_iters_' + str(k)] = iters

        if iters > 49900:
            logging.warning('Setting {} took {} iters to converge.'.format(op, iters),
                            file=sys.stderr)

        piecewise_const_mgr = sim_opts.create_manager_with_piecewise_const(
            seed=seed,
            change_times=window_start + np.arange(num_segments) * seg_len,
            rates=kdd_opt / seg_len
        )
        piecewise_const_mgr.state.time = window_start
        piecewise_const_mgr.run_dynamic()
        df = piecewise_const_mgr.state.get_dataframe()
        perf = time_in_top_k(df=df, K=k, sim_opts=sim_opts)
        op['top_' + str(k)] = perf
        op['top_' + str(k) + '_num_events'] = len(df.event_id[df.src_id == sim_opts.src_id].unique())

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
    op['world_events'] = len(df.event_id[df.src_id != sim_opts.src_id].unique())
    op['num_events'] = len(df.event_id[df.src_id == sim_opts.src_id].unique())

    if queue is not None:
        queue.put(op)

    return op


# This is an approach using the multiprocessing module without the Pool and using a queue to accumulate the results.
# This will lead to a better utilization of the CPU resources (hopefully) because the previous method only allowed
# parallization of the number of seeds.

dilation = 100.0
simulation_opts = Options(world_rate=1000.0 / dilation, world_alpha=1.0, world_beta=10.0,
                          N=10, T=1.0 * dilation, num_segments=10,
                          log_q_low=-6 + np.log10(dilation), log_q_high=5 + np.log10(dilation))


@optioned(option_arg='opts')
def piecewise_sim_opt_factory(N, T, num_segments, world_rate, opts):
    random_state = np.random.RandomState(42)
    world_changing_rates = random_state.uniform(low=world_rate / 2.0, high=world_rate, size=num_segments)
    world_change_times = np.arange(num_segments) * T / num_segments

    def sim_opts_gen(seed):
        return SimOpts.std_piecewise_const(world_rates=world_changing_rates,
                                           world_change_times=world_change_times,
                                           world_seed=seed + 42).update({'end_time': T})

    return opts.set_new(N=N, T=T, num_segments=num_segments, sim_opts_gen=sim_opts_gen)


poisson_inf_opts = simulation_opts.set_new(
    sim_opts_gen=lambda seed: SimOpts.std_poisson(world_rate=simulation_opts.world_rate,
                                                  world_seed=seed + 42)
                                     .update({'end_time': simulation_opts.T}))

piecewise_inf_opts = piecewise_sim_opt_factory(opts=simulation_opts)
hawkes_inf_opts = simulation_opts.set_new(
    sim_opts_gen=lambda seed: SimOpts.std_hawkes(world_seed=seed,
                                                 world_lambda_0=simulation_opts.world_rate,
                                                 world_alpha=simulation_opts.world_alpha,
                                                 world_beta=simulation_opts.world_beta)
                                     .update({'end_time': simulation_opts.T}))


def extract_perf_fields(return_obj, exclude_fields=None, include_fields=None):
    """Extracts the relevant fields from the return object and returns them in a new dict."""
    result_dict = {}
    include_fields = include_fields if include_fields is not None else set()
    exclude_fields = exclude_fields if exclude_fields is not None else set()
    fields = set(perf_opts.performance_fields).union(include_fields) - exclude_fields
    for field in fields:
        result_dict[field] = return_obj[field]

    return result_dict


real_performance_fields = [x for x in perf_opts.performance_fields if x != 'q'] + ['user_id']


def extract_real_perf_fields(return_obj, exclude_fields=None, include_fields=None):
    """Extracts the relevant fields from the return object and returns them in a new dict."""
    result_dict = {}
    include_fields = include_fields if include_fields is not None else set()
    exclude_fields = exclude_fields if exclude_fields is not None else set()
    fields = set(real_performance_fields).union(include_fields) - exclude_fields
    for field in fields:
        result_dict[field] = return_obj[field]

    return result_dict


@optioned(option_arg='opts')
def run_inference(N, T, num_segments, sim_opts_gen, log_q_high, log_q_low):
    """Run inference for the given sim_opts_gen by sweeping over 'q' and
    running the simulation for different seeds."""
    processes = []
    queue = mp.Queue()
    results = []
    capacities = {}
    raw_results = []

    try:
        active_processes = 0
        for q in np.logspace(log_q_low, log_q_high, num=10):
            capacities[q] = []
            for seed in range(N):
                active_processes += 1
                sim_opts = sim_opts_gen(seed).update({'q' : q})
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
            active_processes -= 1

            if r['type'] == 'Opt':
                seed = r['seed']
                capacity = r['capacity']
                s = r['sim_opts'].s
                sim_opts = r['sim_opts']
                world_events = r['world_events']
                capacities[s].append((seed, capacity))

                # Poisson

                p = mp.Process(target=worker_poisson, args=((seed, capacity, sim_opts, queue),))
                processes.append(p)
                p.daemon = True
                p.start()
                active_processes += 1

                # Oracle

                oracle_args = (seed, capacity, world_events, sim_opts, queue)
                p = mp.Process(target=worker_oracle, args=(oracle_args,))
                processes.append(p)
                p.daemon = True
                p.start()
                active_processes += 1

                # KDD solution

                # kdd_args = (seed, capacity, num_segments, sim_opts, world_changing_rates, queue)
                # kdd_args = (seed, capacity, num_segments, sim_opts, None, queue)
                # p = mp.Process(target=worker_kdd, args=(kdd_args,))
                # processes.append(p)
                # p.daemon = True
                # p.start()
                # active_processes += 1

            elif r['type'] == 'Poisson':
                if active_processes % 10 == 0:
                    logTime('Active processes = {}'.format(active_processes))

            elif r['type'] == 'Oracle':
                if active_processes % 10 == 0:
                    logTime('Active processes = {}'.format(active_processes))

            elif r['type'] == 'kdd':
                if active_processes % 10 == 0:
                    logTime('Active processes = {}'.format(active_processes))

            else:
                raise ValueError('Unknown type: {}'.format(r['type']))
    finally:
        # Attempt at cleanup
        logging.info("Cleaning up {} processes".format(len(processes)))
        for p in processes:
            p.terminate()
            p.join()

    return Options(df=pd.DataFrame.from_records(results),
                   raw_results=raw_results,
                   capacities=capacities)


def worker_combined(input_queue, output_queue):
    while True:
        broadcaster_type, broadcaster_args = input_queue.get()

        if broadcaster_type == 'Stop':
            break

        try:
            all_args = broadcaster_args + (output_queue,)
            if broadcaster_type == 'Opt':
                worker_opt(all_args)
            elif broadcaster_type == 'Poisson':
                worker_poisson(all_args)
            elif broadcaster_type == 'Oracle':
                worker_oracle(all_args)
            elif broadcaster_type == 'kdd':
                worker_kdd(all_args)
            else:
                raise RuntimeError('Unknown broadcaster type: {}'.format(broadcaster_type))
        except Exception as e:
            output_queue.put({
                'type'             : 'Exception',
                'error'            : e,
                'broadcaster_type' : broadcaster_type,
                'broadcaster_args' : broadcaster_args
            })
            raise


@optioned(option_arg='opts')
def run_inference_queue_kdd(N, T, num_segments, sim_opts_gen, log_q_high, log_q_low, num_procs=None):
    """Run inference for the given sim_opts_gen by sweeping over 'q' and
    running the simulation for different seeds."""

    if num_procs is None:
        num_procs = mp.cpu_count() - 1

    in_queue = mp.Queue()
    out_queue = mp.Queue()
    results = []
    raw_results = []
    capacities = {}

    # Start consumers
    processes = [mp.Process(target=worker_combined, args=(in_queue, out_queue))
                 for _ in range(num_procs)]

    for p in processes:
        p.daemon = True  # Terminate if the parent dies.
        p.start()

    active_procs = 0
    type_procs = defaultdict(lambda: 0)

    def add_task(task_type, args):
        in_queue.put((task_type, args))
        type_procs[task_type] += 1

    try:
        for q in np.logspace(log_q_low, log_q_high, num=10):
            capacities[q] = []
            for seed in range(N):
                in_queue.put(('Opt', (seed, sim_opts_gen(seed).update({'q': q}))))
                active_procs += 1

        type_procs['Opt'] = active_procs
        while active_procs > 0:
            r = out_queue.get()
            active_procs -= 1
            type_procs[r['type']] -= 1

            if active_procs % 10 == 0:
                logTime('active_procs = {}, procs = {}'
                        .format(active_procs, list(type_procs.items())))

            if r['type'] == 'Exception':
                logging.error('Exception while handling: ', r)
            else:
                raw_results.append(r)
                results.append(extract_perf_fields(r))

                if r['type'] == 'Opt':
                    seed = r['seed']
                    capacity = r['capacity']
                    q = r['sim_opts'].q
                    sim_opts = r['sim_opts']
                    world_events = r['world_events']
                    capacities[q].append((seed, capacity))

                    # add_task('Poisson', (seed, capacity, sim_opts))
                    # active_procs += 1

                    # add_task('Oracle', (seed, capacity, world_events, sim_opts))
                    # active_procs += 1

                    add_task('kdd', (seed, capacity, num_segments, sim_opts, None))
                    active_procs += 1

        for p in range(num_procs):
            in_queue.put(('Stop', None))

    except:
        # In case of exceptions, do not block the parent thread and just
        # discard all data on the queues.
        in_queue.cancel_join_thread()
        out_queue.cancel_join_thread()
        raise
    finally:
        logging.info('Cleaning up {} processes'.format(len(processes)))
        for p in processes:
            p.terminate()
            p.join()

    return Options(df=pd.DataFrame.from_records(results),
                   raw_results=raw_results,
                   capacities=capacities)


@optioned(option_arg='opts')
def run_inference_queue(N, T, num_segments, sim_opts_gen, log_q_high, log_q_low, num_procs=None):
    """Run inference for the given sim_opts_gen by sweeping over 'q' and
    running the simulation for different seeds."""

    if num_procs is None:
        num_procs = mp.cpu_count() - 1

    in_queue = mp.Queue()
    out_queue = mp.Queue()
    results = []
    raw_results = []
    capacities = {}

    # Start consumers
    processes = [mp.Process(target=worker_combined, args=(in_queue, out_queue))
                 for _ in range(num_procs)]

    for p in processes:
        p.daemon = True  # Terminate if the parent dies.
        p.start()

    active_procs = 0
    type_procs = defaultdict(lambda: 0)

    def add_task(task_type, args):
        in_queue.put((task_type, args))
        type_procs[task_type] += 1

    try:
        for q in np.logspace(log_q_low, log_q_high, num=10):
            capacities[q] = []
            for seed in range(N):
                in_queue.put(('Opt', (seed, sim_opts_gen(seed).update({'q': q}), num_segments)))
                active_procs += 1

        type_procs['Opt'] = active_procs
        while active_procs > 0:
            r = out_queue.get()
            active_procs -= 1
            type_procs[r['type']] -= 1

            if active_procs % 10 == 0:
                logTime('active_procs = {}, procs = {}'
                        .format(active_procs, list(type_procs.items())))

            if r['type'] == 'Exception':
                logging.error('Exception while handling: ', r)
            else:
                raw_results.append(r)
                results.append(extract_perf_fields(r))

                if r['type'] == 'Opt':
                    seed = r['seed']
                    capacity = r['capacity']
                    q = r['sim_opts'].q
                    sim_opts = r['sim_opts']
                    world_events = r['world_events']
                    capacities[q].append((seed, capacity))

                    add_task('Poisson', (seed, capacity, sim_opts))
                    active_procs += 1

                    add_task('Oracle', (seed, capacity, world_events, sim_opts))
                    active_procs += 1

                    add_task('kdd', (seed, capacity, num_segments, sim_opts, r['wall_intensities']))
                    active_procs += 1

        for p in range(num_procs):
            in_queue.put(('Stop', None))

    except:
        # In case of exceptions, do not block the parent thread and just
        # discard all data on the queues.
        in_queue.cancel_join_thread()
        out_queue.cancel_join_thread()
        raise
    finally:
        logging.info('Cleaning up {} processes'.format(len(processes)))
        for p in processes:
            p.terminate()
            p.join()

    return Options(df=pd.DataFrame.from_records(results),
                   raw_results=raw_results,
                   capacities=capacities)


## Experiment with multiple followers

def make_piecewise_const(num_segments):
    """Makes a piecewise constant semi-sinusoid curve with num_segments segments."""
    true_values = np.sin(np.arange(0, np.pi, step=0.001))
    seg_idx = np.arange(true_values.shape[0]) // (true_values.shape[0] / num_segments)
    return pd.Series(true_values).groupby(seg_idx).mean().tolist()


mk_edge_list_opts = Options(num_followers=100, num_broadcasters=100, degree=5, seed=42,
                                follower_id_offset=1000, broadcaster_id_offset=5000)


@optioned('opts')
def make_edge_list(num_followers, num_broadcasters, degree,
                   seed, follower_id_offset=0, broadcaster_id_offset=0,
                   preferential_attachment=False):
    """Creates a social network between the followers and the broadcasters based on the given seed."""

    RS = np.random.RandomState(seed)

    edge_list = []
    broadcaster_followers = np.ones(num_broadcasters)
    prob = broadcaster_followers / broadcaster_followers.sum()

    for sink_id in range(follower_id_offset, num_followers + follower_id_offset):

        if preferential_attachment:
            prob = broadcaster_followers / broadcaster_followers.sum()

        for src_id in RS.choice(num_broadcasters, degree, replace=False, p=prob):
            if preferential_attachment:
                broadcaster_followers[src_id] += 1

            edge_list.append((src_id + broadcaster_id_offset, sink_id))

    return edge_list


def create_phased_pwconst_broadcaster(src_id, seed, rel_rates, avg_rate, end_time, phase_shift):
    """Create a source which is PieceWise constant and with a given 'phase_shift'."""
    num_segments = len(rel_rates)

    assert int(phase_shift) == phase_shift, "The phase shift cannot be fractional."

    phase_shift %= num_segments

    change_times = np.arange(num_segments) * (end_time / num_segments)

    # The area under the sin x curve from 0 till 2*pi is == 2.
    # Scaling the area such that the total area is equal to the avg_rate till the end_time
    shifted_rates = np.asarray(rel_rates[phase_shift:] + rel_rates[:phase_shift])
    actual_rates = shifted_rates * (avg_rate * num_segments) / np.sum(shifted_rates)
    return ('PiecewiseConst', {'src_id': src_id, 'seed': seed, 'change_times': change_times, 'rates': actual_rates})


def trim_sim_opts(sim_opts):
    """Creates a new sim_opts after removing all the broadcasters/sinks which do not follow the src_id."""
    intended_followees = set([t for s, t in sim_opts.edge_list if s == sim_opts.src_id])
    num_followers = len(intended_followees)
    new_edge_list = [(s, t) for s, t in sim_opts.edge_list if t in intended_followees]
    reachable_broadcaster_ids = set([s for s, _ in new_edge_list])
    new_broadcasters = [src for src in sim_opts.other_sources if src[1]['src_id'] in reachable_broadcaster_ids]

    return sim_opts.update({
        'sink_ids': sorted(intended_followees),
        'edge_list': new_edge_list,
        'other_sources': new_broadcasters,
        'q': 1.0 * (num_followers ** 2)
    })


multiple_follower_opts = Options(seed=42, world_alpha=1.0, world_beta=10.0, world_rate=100.0,
                                 kind='PiecewiseConst', num_other_broadcasters=1000,
                                 max_num_followers=500, follower_other_degree=1)


@optioned('opts')
def prepare_multiple_followers_sim_opts(num_followers, num_other_broadcasters, max_num_followers,
                                        seed, world_rate, world_alpha, world_beta,
                                        follower_other_degree, kind):

    assert num_other_broadcasters >= follower_other_degree, ("There should be more other broadcasters" +
                                                             " than followers per node.")

    end_time = 100.0

    randomState = np.random.RandomState(seed)

    follower_id_offset = 1000
    follower_ids = follower_id_offset + np.arange(max_num_followers)

    broadcaster_id_offset = 5000
    other_broadcasters_ids = broadcaster_id_offset + np.arange(num_other_broadcasters)

    fixed_social_network =  make_edge_list(
        num_followers=max_num_followers,
        num_broadcasters=num_other_broadcasters,
        degree=follower_other_degree,
        seed=1024, # This seed keeps the network constant as if num_followers changes.
        follower_id_offset=follower_id_offset,
        broadcaster_id_offset=broadcaster_id_offset,
        opts=mk_edge_list_opts
    )

    optimal_broadcaster_id = 1

    if kind == 'PiecewiseConst':
        pcw_rates = make_piecewise_const(24)
        other_broadcasters = [create_phased_pwconst_broadcaster(src_id=x,
                                                                seed=seed + x,
                                                                rel_rates=pcw_rates,
                                                                avg_rate=world_rate,
                                                                end_time=end_time,
                                                                phase_shift=x
                                                                ) for x in other_broadcasters_ids]
    elif kind == 'Hawkes':
        other_broadcasters = [('Hawkes', {'src_id': x,
                                          'seed': seed + x,
                                          'l_0': world_rate,
                                          'alpha': world_alpha,
                                          'beta': world_beta})
                              for x in other_broadcasters_ids]
    elif kind == 'Poisson2':
        logging.warn('The rates are being randomised for Poisson2')
        other_broadcasters = [('Poisson2', {'src_id': x,
                                            'seed': seed + x,
                                            'rate': world_rate * np.abs(randomState.randn() + 1.0)})
                              for x in other_broadcasters_ids]
    else:
        raise ValueError('Cannot create broadcasters of kind "{}"'.format(kind))


    fixed_social_network.extend([(optimal_broadcaster_id, x)
                                 for x in randomState.choice(follower_ids, num_followers, replace=False)])

    sim_opts = SimOpts(
        src_id=optimal_broadcaster_id,
        end_time=end_time,
        s=np.asarray([1.0] * num_followers),
        sink_ids=follower_ids,
        other_sources=other_broadcasters,
        edge_list=fixed_social_network,
        q=1.0 * (num_followers ** 2)
    )

    return trim_sim_opts(sim_opts)

@optioned(option_arg='opts')
def run_multiple_followers(num_followers_list, num_segments, setup_opts, repetitions, num_procs=None):
    """Run experiment with multiple followers."""

    if num_procs is None:
        num_procs = mp.cpu_count()

    in_queue = mp.Queue()
    out_queue = mp.Queue()
    results = []
    raw_results = []

    # Start consumers
    processes = [mp.Process(target=worker_combined, args=(in_queue, out_queue))
                 for _ in range(num_procs)]

    for p in processes:
        p.daemon = True # Terminate if the parent dies.
        p.start()

    active_procs = 0
    type_procs = defaultdict(lambda: 0)

    def add_task(task_type, args):
        in_queue.put((task_type, args))
        type_procs[task_type] += 1

    total_procs = 0

    try:
        for num_followers in sorted(num_followers_list, reverse=True):
            sim_opts = prepare_multiple_followers_sim_opts(num_followers=num_followers,
                                                           opts=setup_opts)
            for n in range(repetitions):
                in_queue.put(('Opt', (setup_opts.seed * (n + 1) + 13, sim_opts, num_segments)))
                active_procs += 1

        output_period = 1
        while output_period * 10 < active_procs:
            output_period *= 10

        logging.info('Reporting will be done every {} runs.'.format(output_period))

        type_procs['Opt'] = active_procs
        while active_procs > 0:
            r = out_queue.get()
            active_procs -= 1
            type_procs[r['type']] -= 1
            total_procs += 1

            if total_procs % output_period == 0:
                logTime('active/total = {}/{}, procs = {}'
                        .format(active_procs, total_procs, list(type_procs.items())))

            if r['type'] == 'Exception':
                logging.error('Exception while handling: ', r)
            else:
                raw_results.append(r)
                perf = extract_perf_fields(r)
                perf['num_followers'] = len(r['sim_opts'].sink_ids)
                results.append(perf)

                if r['type'] == 'Opt':
                    seed = r['seed']
                    capacity = r['capacity']
                    sim_opts = r['sim_opts']
                    # world_events = r['world_events']

                    add_task('Poisson', (seed, capacity, sim_opts))
                    active_procs += 1

                    # add_task('Oracle', (seed, capacity, world_events, sim_opts))
                    # active_procs += 1

                    try:
                        wall_intensities = r['wall_intensities']
                    except KeyError:
                        wall_intensities = None

                    add_task('kdd', (seed, capacity, num_segments, sim_opts, wall_intensities))
                    active_procs += 1

        for p in range(num_procs):
            in_queue.put(('Stop', None))

    except:
        # In case of exceptions, do not block the parent thread and just
        # discard all data on the queues.
        in_queue.cancel_join_thread()
        out_queue.cancel_join_thread()
        raise
    finally:
        logging.info('Cleaning up {} processes'.format(len(processes)))
        for p in processes:
            p.terminate()
            p.join()

    return Options(df=pd.DataFrame.from_records(results),
                   raw_results=raw_results)





overlap_opts = Options(seed=451, world_rate=10.0,
                       broadcasters_per_follower=50,
                       world_alpha=1.0, world_beta=10.0, kind='PiecewiseConst')

@optioned('opts')
def prepare_overlapping_followees_sim_opts(num_overlap, broadcasters_per_follower,
                                           seed, world_rate, world_alpha, world_beta, kind):

    assert num_overlap <= broadcasters_per_follower, "Maximum overlap can be 100%"

    follower_ids = [1000, 1001]

    end_time = 100.0
    broadcaster_id_offset = 5000
    optimal_broadcaster_id = 1

    num_other_broadcasters = broadcasters_per_follower * 2 - num_overlap
    other_broadcasters_ids = broadcaster_id_offset + np.arange(num_other_broadcasters)

    if kind == 'Hawkes':
        other_broadcasters = [('Hawkes', {'src_id': x,
                                          'seed': seed + x,
                                          'l_0': world_rate,
                                          'alpha': world_alpha,
                                          'beta': world_beta}) for x in other_broadcasters_ids]
    elif kind == 'Poisson2':
        other_broadcasters = [('Poisson2', {'src_id': x,
                                            'seed': seed + x,
                                            'rate': world_rate}) for x in other_broadcasters_ids]
    elif kind == 'PiecewiseConst':
        pcw_rates = make_piecewise_const(24)
        other_broadcasters = [create_phased_pwconst_broadcaster(src_id=x,
                                                                seed=((seed + x) * 12),
                                                                rel_rates=pcw_rates,
                                                                avg_rate=world_rate,
                                                                end_time=end_time,
                                                                phase_shift=x * seed
                                                                ) for x in other_broadcasters_ids]
        rs = np.random.RandomState(seed)
        rs.shuffle(other_broadcasters)

    edge_list = ([(x[1]['src_id'], follower_ids[0]) for x in other_broadcasters[:num_overlap]] +
                 [(x[1]['src_id'], follower_ids[1]) for x in other_broadcasters[:num_overlap]] +
                 [(x[1]['src_id'], follower_ids[0]) for x in other_broadcasters[num_overlap:broadcasters_per_follower]] +
                 [(x[1]['src_id'], follower_ids[1]) for x in other_broadcasters[broadcasters_per_follower:]] +
                 [(1, follower_ids[0]), (1, follower_ids[1])])

    sim_opts = SimOpts(
        src_id=optimal_broadcaster_id,
        end_time=100.0,
        s=np.asarray([1.0] * len(follower_ids)),
        sink_ids=follower_ids,
        other_sources=other_broadcasters,
        edge_list=edge_list,
        q=1.0
    )

    return sim_opts


@optioned(option_arg='opts')
def run_overlapping_followees(overlap_list, num_segments, setup_opts, repetitions, num_procs=None):
    """Run experiment with multiple followers."""

    if num_procs is None:
        num_procs = mp.cpu_count()

    in_queue = mp.Queue()
    out_queue = mp.Queue()
    results = []
    raw_results = []

    # Start consumers
    processes = [mp.Process(target=worker_combined, args=(in_queue, out_queue))
                 for _ in range(num_procs)]

    for p in processes:
        p.daemon = True # Terminate if the parent dies.
        p.start()

    active_procs = 0
    type_procs = defaultdict(lambda: 0)

    def add_task(task_type, args):
        in_queue.put((task_type, args))
        type_procs[task_type] += 1

    total_procs = 0

    try:
        for overlap in overlap_list:
            sim_opts = prepare_overlapping_followees_sim_opts(
                            num_overlap=overlap,
                            seed=setup_opts.seed + overlap + 1337,
                            opts=setup_opts)

            for n in range(repetitions):
                in_queue.put(('Opt', (setup_opts.seed + n, sim_opts, num_segments)))
                active_procs += 1

        output_period = 1
        while output_period * 10 < active_procs:
            output_period *= 10

        logging.info('Reporting will be done every {} runs.'.format(output_period))

        type_procs['Opt'] = active_procs
        while active_procs > 0:
            r = out_queue.get()
            active_procs -= 1
            type_procs[r['type']] -= 1
            total_procs += 1

            if total_procs % output_period == 0:
                logTime('active/total = {}/{}, procs = {}'
                        .format(active_procs, total_procs, list(type_procs.items())))

            if r['type'] == 'Exception':
                logging.error('Exception while handling: ', r)
            else:
                raw_results.append(r)
                perf = extract_perf_fields(r)
                perf['num_followers'] = len(r['sim_opts'].sink_ids)
                results.append(perf)

                if r['type'] == 'Opt':
                    seed = r['seed']
                    capacity = r['capacity']
                    sim_opts = r['sim_opts']
                    # world_events = r['world_events']

                    add_task('Poisson', (seed, capacity, sim_opts))
                    active_procs += 1

                    # add_task('Oracle', (seed, capacity, world_events, sim_opts))
                    # active_procs += 1

                    wall_intensities = r['wall_intensities']

                    add_task('kdd', (seed, capacity, num_segments, sim_opts, wall_intensities))
                    active_procs += 1

        for p in range(num_procs):
            in_queue.put(('Stop', None))

    except:
        # In case of exceptions, do not block the parent thread and just
        # discard all data on the queues.
        in_queue.cancel_join_thread()
        out_queue.cancel_join_thread()
        raise
    finally:
        logging.info('Cleaning up {} processes'.format(len(processes)))
        for p in processes:
            p.terminate()
            p.join()

    return Options(df=pd.DataFrame.from_records(results),
                   raw_results=raw_results)


## Workers for real data

def real_worker_base(params):
    user_id, user_event_times, sim_opts, queue = params
    user_mgr = sim_opts.create_manager_with_times(user_event_times)
    user_mgr.run_dynamic()

    op = {
        'user_id'  : user_id,
        'type'     : 'RealData',
        'seed'     : 0,
        'sim_opts' : sim_opts,
        'capacity' : len(user_event_times) * 1.0
    }

    add_perf(op, user_mgr.state.get_dataframe(), sim_opts)

    if queue is not None:
        queue.put(op)

    return op


def real_worker_poisson(params):
    user_id, seeds, user_budget, sim_opts, queue = params

    ops = []
    for seed in seeds:
        poisson_mgr = sim_opts.create_manager_with_poisson(seed, capacity=user_budget)
        poisson_mgr.run_dynamic()

        op = {
            'user_id'  : user_id,
            'sim_opts' : sim_opts,
            'seed'     : seed,
            'type'     : 'Poisson'
        }

        add_perf(op, poisson_mgr.state.get_dataframe(), sim_opts)

        if queue is not None:
            queue.put(op)

        ops.append(op)

    return ops


def real_worker_opt(params, verbose=False):
    user_id, seeds, num_user_events, sim_opts, queue = params
    q_opt = sweep_q(sim_opts, capacity_cap=num_user_events, tol=0.05, dynamic=True, verbose=verbose)
    opt_sim_opts = sim_opts.update({'q' : q_opt})

    ops = []
    for seed in seeds:
        opt_mgr = opt_sim_opts.create_manager_with_opt(seed)
        opt_mgr.run_dynamic()
        df = opt_mgr.state.get_dataframe()

        num_events = len(df.event_id[df.src_id == sim_opts.src_id].unique())
        capacity = num_events * 1.0

        op = {
            'user_id'    : user_id,
            'seed'       : seed,
            'q_opt'      : q_opt,
            'capacity'   : capacity,
            'sim_opts'   : sim_opts,
            'num_events' : num_events,
            'type'       : 'Opt'
        }

        add_perf(op, df, sim_opts)

        if queue is not None:
            queue.put(op)

        ops.append(op)

    return ops


def _follower_intensity_factory(T, num_segments):
    """Returns a function which can calculate the intensities for one user."""
    # Assumption: start time is always zero
    seg_len = T / num_segments

    def _follower_intensity_calc(x):
        seg_idx = (x.t.values / T * num_segments).astype(int)
        assert is_sorted(x.t.values)
        ret_val = x.groupby(seg_idx).size() / seg_len
        ret_val.index.rename('segment', inplace=True)
        return ret_val

    return _follower_intensity_calc


def real_worker_kdd(params, verbose=False):
    user_id, seeds, budget, num_segments, sim_opts, queue = params

    T = sim_opts.end_time
    seg_len = T / num_segments
    num_followers = len(sim_opts.sink_ids)

    wall_mgr = sim_opts.create_manager_for_wall()
    wall_mgr.run_dynamic()
    wall_df = wall_mgr.state.get_dataframe()

    # f_i_c = _follower_intensity_factory(T, num_segments)
    # wall_flat = (wall_df.groupby('sink_id')
    #                 .apply(f_i_c)
    #                 .reset_index())

    # print('columns = {}'.format(wall_flat.columns))

    # wall_intensities = (wall_flat.pivot_table(values=0, index='sink_id', columns='segment'))

    # followers_wall_intensities = wall_intensities.fillna(0).values

    followers_wall_intensities = np.zeros((num_followers, num_segments), dtype=float)

    for idx, sink_id in enumerate(sim_opts.sink_ids):
        df = wall_df[wall_df.sink_id == sink_id]
        for seg_idx in range(num_segments):
            followers_wall_intensities[idx, seg_idx] = np.sum((df.t >= seg_idx * seg_len) & (df.t <= (seg_idx + 1) * seg_len)) / seg_len

    followers_conn_prob = np.ones((num_followers, num_segments), dtype=float)
    followers_weights = np.ones(num_followers) / num_followers

    upper_bounds = np.array([1e11] * num_segments)
    threshold = 0.02

    kdd_opts = []
    iters_ = []
    ops = []

    for k in Ks:
        if k != 1:
            def _util(x):
                return Bopt.utils.weighted_top_k(x,
                                                 followers_wall_intensities,
                                                 followers_conn_prob,
                                                 followers_weights,
                                                 k)
            def _util_grad(x):
                return Bopt.utils.weighted_top_k_grad(x,
                                                      followers_wall_intensities,
                                                      followers_conn_prob,
                                                      followers_weights,
                                                      k)
        else:
            # For k = 1, special case of gradient calculation
            def _util(x):
                return Bopt.utils.weighted_top_one(x,
                                                 followers_wall_intensities,
                                                 followers_conn_prob,
                                                 followers_weights)
            def _util_grad(x):
                return Bopt.utils.weighted_top_one_grad(x,
                                                      followers_wall_intensities,
                                                      followers_conn_prob,
                                                      followers_weights)


        x0 = np.ones(num_segments) * budget / num_segments

        kdd_opt, iters = Bopt.optimize(
            util         = _util,
            util_grad    = _util_grad,
            budget       = budget,
            upper_bounds = upper_bounds,
            threshold    = threshold,
            x0           = x0,
            verbose      = verbose,
            with_iter    = True
        )

        if verbose:
            logTime('Done for user_id = {}, k = {} in {} iterations'.format(user_id, k, iters))

        kdd_opts.append(kdd_opt)
        iters_.append(iters)

    for seed in seeds:
        best_avg_rank, best_avg_k = np.inf, -1
        best_r_2, best_r_2_k = np.inf, -1

        op = {
            'user_id'  : user_id,
            'type'     : 'kdd',
            'seed'     : seed,
            'sim_opts' : sim_opts,
        }

        for k_idx, k in enumerate(Ks):
            kdd_opt = kdd_opts[k_idx]
            op['kdd_opt_' + str(k)] = kdd_opt
            op['kdd_opt_iters_' + str(k)] = iters_[k_idx]

            piecewise_const_mgr = sim_opts.create_manager_with_piecewise_const(
                seed=seed,
                change_times=window_start + np.arange(num_segments) * seg_len,
                rates=kdd_opt / seg_len
            )
            piecewise_const_mgr.state.time = window_start
            piecewise_const_mgr.run_dynamic()
            df = piecewise_const_mgr.state.get_dataframe()
            perf = time_in_top_k(df=df, K=k, sim_opts=sim_opts)

            op['top_' + str(k)] = perf
            op['top_' + str(k) + '_num_events'] = len(df.event_id[df.src_id == sim_opts.src_id].unique())

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

        # These just record the last value.
        op['world_events'] = len(df.event_id[df.src_id != sim_opts.src_id].unique())
        op['num_events'] = len(df.event_id[df.src_id == sim_opts.src_id].unique())

        if queue is not None:
            queue.put(op)

        ops.append(op)

    return ops


@optioned(option_arg='opts')
def run_real_queue(N, num_segments, files_user_ids, min_user_capacity=2, num_procs=None, raw_results=None):
    """Run inference for the given sim_opts_gen by sweeping over 'q' and
    running the simulation for different seeds."""

    if num_procs is None:
        num_procs = mp.cpu_count() - 1

    if raw_results is None:
        raw_results = []
    elif len(raw_results) != 0:
        logging.info('raw_results passed are not empty, want to continue [y/n]?')
        ans = input('Continue [Y/n]?')
        if ans[0] != 'Y':
            return None

    def worker(input_queue, output_queue):
        while True:
            broadcaster_type, broadcaster_args = input_queue.get()

            if broadcaster_type == 'Stop':
                break

            try:
                all_args = broadcaster_args + (output_queue,)
                if broadcaster_type == 'RealData':
                    real_worker_base(all_args)
                elif broadcaster_type == 'Poisson':
                    real_worker_poisson(all_args)
                elif broadcaster_type == 'Opt':
                    real_worker_opt(all_args)
                elif broadcaster_type == 'kdd':
                    real_worker_kdd(all_args)
                else:
                    raise RuntimeError('Unknown broadcaster type: {}'.format(broadcaster_type))
            except Exception as e:
                output_queue.put({
                    'type'             : 'Exception',
                    'error'            : e,
                    'broadcaster_type' : broadcaster_type,
                    'broadcaster_args' : broadcaster_args
                })
                raise

    in_queue = mp.Queue()
    out_queue = mp.Queue()
    results = []
    capacities = {}

    # Start consumers
    processes = [mp.Process(target=worker, args=(in_queue, out_queue))
                 for _ in range(num_procs)]

    for p in processes:
        # p.daemon = True # Terminate if the parent dies.
        # The process may itself start a pool of processes.
        p.start()

    active_procs = 0
    type_procs = defaultdict(lambda: 0)

    def add_task(task_type, args):
        in_queue.put((task_type, args))

    try:
        for user_file, user_id in files_user_ids:
            with open(user_file, 'rb') as pickle_file:
                d = pickle.load(pickle_file)
                if d['num_user_events'] <= min_user_capacity:
                    logTime('User {} had only {} tweets, ignoring.'.format(user_id, d['num_user_events']))
                else:
                    sim_opts = SimOpts(**d['sim_opts_dict'])
                    if len(sim_opts.sink_ids) > 1:
                        in_queue.put(('RealData', (user_id, d['user_event_times'], sim_opts)))
                        active_procs += 1
                    else:
                        logTime('User {} has no followers.'.format(user_id))

        type_procs['RealData'] = active_procs
        sub_task_size = N
        while active_procs > 0:
            r = out_queue.get()
            active_procs -= 1
            type_procs[r['type']] -= 1

            if active_procs % 10 == 0:
                logTime('active_procs = {}, procs = {}'
                        .format(active_procs, list(type_procs.items())))

            if r['type'] == 'Exception':
                logging.error('Exception while handling: ', r)
            else:
                raw_results.append(r)
                results.append(extract_real_perf_fields(r))

                if r['type'] == 'RealData':
                    seeds = range(N)
                    capacity = r['capacity']
                    sim_opts = r['sim_opts']
                    user_id = r['user_id']
                    world_events = r['world_events']

                    add_task('Poisson', (user_id, seeds, capacity, sim_opts))
                    active_procs += sub_task_size
                    type_procs['Poisson'] += sub_task_size

                    add_task('Opt', (user_id, seeds, capacity, sim_opts))
                    active_procs += sub_task_size
                    type_procs['Opt'] += sub_task_size

                    add_task('kdd', (user_id, seeds, capacity, num_segments, sim_opts))
                    active_procs += sub_task_size
                    type_procs['kdd'] += sub_task_size

                    # add_task('Oracle', (seed, capacity, world_events, sim_opts))
                    # active_procs += 1

                    # add_task('kdd', (seed, capacity, num_segments, sim_opts, None))
                    # active_procs += 1

        for p in range(num_procs):
            in_queue.put(('Stop', None))

    except:
        # In case of exceptions, do not block the parent thread and just
        # discard all data on the queues.
        in_queue.cancel_join_thread()
        out_queue.cancel_join_thread()
        raise
    finally:
        logging.info('Cleaning up {} processes'.format(len(processes)))
        for p in processes:
            p.terminate()
            p.join()

    return Options(df=pd.DataFrame.from_records(results),
                   raw_results=raw_results,
                   capacities=capacities)



