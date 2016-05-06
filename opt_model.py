#!/usr/bin/env python

from __future__ import print_function
import scipy.stats.distributions as Distr
import numpy as np
import pandas as pd
import sys
import abc
import bisect

from utils import mb, is_sorted

class Event:
    def __init__(self, event_id, time_delta, cur_time,
                 src_id, sink_ids, metadata=None):
        self.event_id   = event_id
        self.time_delta = time_delta
        self.cur_time   = cur_time
        self.src_id     = src_id
        self.sink_ids   = sink_ids
        self.metadata   = metadata

    def __repr__(self):
        return ('[ Event_id: {}, time_delta: {}, src_id: {} ]'
                .format(self.event_id, self.time_delta, self.src_id))


class State:
    def __init__(self, cur_time, sink_ids):
        self.num_sinks = len(sink_ids)
        self.time      = cur_time
        self.sinks     = dict((x,[]) for x in sink_ids)
        self.events    = []

    def apply_event(self, event):
        """Apply the given event to the state."""
        if event is None:
            # This was the first event, ignore
            return

        self.events.append(event)
        self.time += event.time_delta

        # Add the event (tweet) to the corresponding lists
        for sink_id in event.sink_ids:
            self.sinks[sink_id].append(event)

    def get_dataframe(self):
        """Return the list of events."""
        df = pd.DataFrame.from_records(
            [{'event_id'   : x.event_id,
              'time_delta' : x.time_delta,
              'src_id'     : x.src_id,
              't'          : x.cur_time,
              'sink_id'    : y}
             for x in self.events
             for y in x.sink_ids]
        )
        return df

    def get_wall_rank(self, src_id, follower_ids, dict_form=True):
        """Return a dictionary or vectors of rank of the src_id on the wall of
        the followers. If the follower does not have any events from the given
        source, the corresponding rank is `None`. The resulting vector will
        be formed after sorting the sink_ids.
        """
        rank = dict((sink_id, None) for sink_id in follower_ids)
        for sink_id in follower_ids:
            for ii in range(len(self.sinks[sink_id]) - 1, -1, -1):
                if self.sinks[sink_id][ii].src_id == src_id:
                    rank[sink_id] = len(self.sinks[sink_id]) - ii - 1
                    break # breaks the inner for loop

        if dict_form:
            return rank
        else:
            return np.asarray([rank[sink_id] for sink_id in sorted(rank.keys())])


class Manager:
    def __init__(self, sources, sink_ids=None, end_time=None,
                 edge_list=None, sim_opts=None):
        """The edge_list defines the network.
        Default is that all sources are connected to all sinks."""

        if sim_opts is not None:
            # sources   = mb(sources, sim_opts.other_sources))
            edge_list = mb(edge_list, sim_opts.edge_list)
            sink_ids  = mb(sink_ids, sim_opts.sink_ids)
            end_time  = mb(end_time, sim_opts.end_time)

        assert len(sources) > 0, "No sources."
        assert len(sink_ids) > 0, "No sinks."
        assert len(set(sink_ids)) == len(sink_ids), "Duplicates in sink_ids."

        if edge_list is None:
            edge_list = []
            for src in sources:
                src_id = src.src_id
                for dst_id in sink_ids:
                    edge_list.append((src_id, dst_id))
        else:
            known_src_ids = set(src.src_id for src in sources)
            edge_src_ids = set(x[0] for x in edge_list)
            assert edge_src_ids.issubset(known_src_ids), "Unknown sources in edge_list."

            edge_sink_ids = set(x[1] for x in edge_list)
            assert edge_sink_ids.issubset(set(sink_ids)), "Unknown sinks in edge_list."

        self.end_time  = end_time
        self.edge_list = edge_list
        self.sink_ids  = sink_ids
        self.state     = State(0, sink_ids)
        self.sources   = sources

    def get_state(self):
        """Returns the current state of the simulation."""
        return self.state

    def run(self):
        return self.run_till()

    def run_till(self, end_time=None):
        if end_time is not None:
            print('Warning: deprecation warning: end_time should not be set.')
        else:
            end_time = self.end_time

        # Step 1: Inform the sources of the sinks associated with them.
        # Step 2: Give them the initial state
        for src in self.sources:
            if not src.is_fresh():
                raise ValueError('Source with id: {} is not fresh.'
                                 .format(src.src_id))

            follower_ids = [x[1] for x in self.edge_list if x[0] == src.src_id]
            src.init_state(self.state.time, self.sink_ids, follower_ids, end_time)

        last_event = None
        event_id = 100

        while True:
            # Step 3: Generate one t_delta from each source and form the
            # event which is going to happen next.

            # This step can be made parallel.
            # If multiple events happen at the same time, then the ordering
            # will still be deterministic: by the ID of the source.
            t_delta, next_src_id = sorted((src.get_next_event_time(last_event),
                                           src.src_id)
                                          for src in self.sources)[0]

            assert t_delta >= 0, "Next event must be now or in the future."

            # Step 5: If cur_time + t_delta < end_time, go to step 4, else Step 7
            cur_time = self.state.time
            if cur_time + t_delta > end_time:
                break

            # Step 4: Execute the first event
            last_event = Event(event_id, t_delta,
                               cur_time + t_delta, next_src_id,
                               [x[1] for x in self.edge_list
                                     if x[0] == next_src_id])
            event_id += 1
            self.state.apply_event(last_event)

            # Step 6: Go to step 3

        # Step 7: Stop
        return self


# Broadcasters
##############

# TODO: Make Broadcasters serializable

class Broadcaster:
    __metaclass__ = abc.ABCMeta

    def __init__(self, src_id, seed):
        self.src_id               = src_id
        self.seed                 = seed
        self.random_state         = np.random.RandomState(seed)
        self.t_delta              = None
        self.end_time             = None
        self.last_self_event_time = None
        self.used                 = False

    def init_state(self, start_time, all_sink_ids, follower_sink_ids, end_time):
        self.sink_ids   = sorted(follower_sink_ids)
        self.state      = State(start_time, all_sink_ids)
        self.start_time = start_time
        self.end_time   = end_time

    def is_fresh(self):
        """Returns true if the source is fresh and ready to be used, False
        if the next event times have already been requested once."""
        return not self.used

    def get_next_event_time(self, event):
        cur_time = self.get_current_time(event)
        self.used = True

        if event is None or event.src_id == self.src_id:
            self.last_self_event_time = cur_time

        t_delta = self.get_next_interval(event)
        if t_delta is not None:
            self.t_delta = t_delta

        ret_t_delta = self.last_self_event_time + self.t_delta - cur_time

        if ret_t_delta < 0:
            print('returned t_delta = {} < 0, set to 0 instead.'.format(ret_t_delta), file=sys.stderr)
            ret_t_delta = 0.0

        return ret_t_delta

    def get_current_time(self, event):
        return event.cur_time if event is not None else self.start_time

    @abc.abstractmethod
    def get_next_interval(self):
        """Should return a number to replace the current time to next event or
        None if no change should be made."""
        raise NotImplemented()


class Poisson2(Broadcaster):
    def __init__(self, src_id, seed, rate=1.0):
        super(Poisson2, self).__init__(src_id, seed)
        self.rate      = rate

        self.init      = False
        self.times     = None
        self.t_diff    = None
        self.start_idx = None

    def get_next_interval(self, event):
        if not self.init:
            self.init      = True
            duration       = self.end_time - self.start_time
            num_events     = self.random_state.poisson(self.rate * duration)
            event_times    = self.random_state.uniform(low=self.start_time,
                                                       high=self.end_time,
                                                       size=num_events)
            self.times     = sorted(np.concatenate([[self.start_time], event_times]))
            self.t_diff    = np.diff(self.times)
            self.start_idx = 0

        if event is None:
            return self.t_diff[0]
        elif event.src_id == self.src_id:
            # Re-use times drawn before
            self.start_idx += 1
            if self.start_idx < len(self.t_diff):
                assert self.times[self.start_idx] <= event.cur_time
                assert self.times[self.start_idx + 1] > event.cur_time
                return self.t_diff[self.start_idx]
            else:
                return np.inf


class Poisson(Broadcaster):
    def __init__(self, src_id, seed, rate=1.0):
        super(Poisson, self).__init__(src_id, seed)
        self.rate = rate

    def get_next_interval(self, event):
        if event is None or event.src_id == self.src_id:
            # Draw a new time, one event at a time
            return Distr.expon.rvs(scale=1.0 / self.rate,
                                   random_state=self.random_state)


class Hawkes(Broadcaster):
    def __init__(self, src_id, seed, l_0=1.0, alpha=1.0, beta=10.0):
        super(Hawkes, self).__init__(src_id, seed)
        self.l_0   = l_0
        self.alpha = alpha
        self.beta  = beta
        self.prev_excitations = []

    def get_rate(self, t):
        """Returns the rate of current Hawkes at time `t`."""
        return self.l_0 + \
            self.alpha * sum(np.exp([self.beta * -1.0 * (t - s)
                                     for s in self.prev_excitations
                                     if s < t]))

    def get_next_interval(self, event):
        t = self.get_current_time(event)
        if event is None or event.src_id == self.src_id:
            rate_bound = self.get_rate(t)
            t_delta = 0

            # Ogata sampling for one t-delta
            while True:
                t_delta += Distr.expon.rvs(scale=1.0 / rate_bound,
                                           random_state=self.random_state)
                # Rejection sampling
                if self.random_state.rand() < self.get_rate(t + t_delta) / rate_bound:
                    break

            self.prev_excitations.append(t + t_delta)
            return t_delta


class Opt(Broadcaster):
    def __init__(self, src_id, seed, q_vec=1.0, s=1.0):
        super(Opt, self).__init__(src_id, seed)
        self.q = q_vec
        self.s = s
        self.old_rate = 0
        self.init = False

    def get_next_interval(self, event):
        self.state.apply_event(event)

        if not self.init:
            self.init = True
            if isinstance(self.q, dict):
                self.q_vec = np.asarray(self.q[x]
                                        for x in sorted(self.sink_ids))
            else:
                # Assuming that the self.q is otherwise a scalar number.
                self.q_vec = np.ones(len(self.sink_ids), dtype=float) * self.q

        if event is None:
            # Tweet immediately if this is the first event.
            self.old_rate = 0
            return 0
        elif event.src_id == self.src_id:
            # No need to tweet if we are on top of all walls
            self.old_rate = 0
            return np.inf
        else:
            # check status of all walls and find position in it.
            r_t = self.state.get_wall_rank(self.src_id, self.sink_ids,
                                           dict_form=False)

            # TODO: If multiple walls are updated at the same time, should the
            # drawing happen only once after all the updates have been applied
            # or one at a time? Does that make a difference? Probably not. A
            # lot more work if the events are sent one by one per wall, though.
            new_rate = np.sqrt(self.q_vec / self.s).dot(r_t)
            diff_rate = new_rate - self.old_rate
            self.old_rate = new_rate

            t_delta_new = Distr.expon.rvs(scale=1.0 / diff_rate,
                                          random_state=self.random_state)
            cur_time = event.cur_time

            if self.last_self_event_time + self.t_delta > cur_time + t_delta_new:
                return cur_time + t_delta_new - self.last_self_event_time


class PiecewiseConst(Broadcaster):
    def __init__(self, src_id, seed, change_times, rates):
        """Creates a broadcaster which tweets with the given rates."""
        super(PiecewiseConst, self).__init__(src_id, seed)

        assert is_sorted(change_times)

        self.change_times = change_times
        self.rates        = rates

        self.init         = False
        self.times        = None
        self.t_diff       = None
        self.start_idx    = None

    def get_rate_at(self, t):
        """Finds what the instantaneous rate at time 't' is."""
        return self.rates[bisect.bisect(self.change_times, t) - 1]

    def get_next_interval(self, event):
        if not self.init:
            self.init = True

            assert self.start_time <= self.change_times[0]
            assert self.end_time   >= self.change_times[-1]

            duration = self.end_time - self.start_time
            max_rate = np.max(self.rates)

            # Using Ogata to determine the event times.
            num_all_events = self.random_state.poisson(max_rate * duration)
            all_event_times = self.random_state.uniform(low=self.start_time,
                                                        high=self.end_time,
                                                        size=num_all_events)
            thinned_event_times = []
            for t in sorted(all_event_times):
                # Rejection sampling
                if self.random_state.rand() < self.get_rate_at(t) / max_rate:
                    thinned_event_times.append(t)

            self.times = np.concatenate([[self.start_time], thinned_event_times])
            self.t_diff = np.diff(self.times)
            self.start_idx = 0


        if event is None:
            return self.t_diff[0]
        elif event.src_id == self.src_id:
            self.start_idx += 1

            if self.start_idx < len(self.t_diff):
                assert self.times[self.start_idx] <= event.cur_time
                assert self.times[self.start_idx + 1] > event.cur_time
                return self.t_diff[self.start_idx]
            else:
                return np.inf



class RealData(Broadcaster):
    def __init__(self, src_id, times):
        super(RealData, self).__init__(src_id, 0)
        self.times = np.asarray(times)
        self.t_diff = np.diff(np.concatenate([[0], self.times]))
        self.start_idx = None

    def get_num_events(self):
        return len(self.times)

    def init_state(self, start_time, all_sink_ids, follower_sink_ids, end_time):
        super(RealData, self).init_state(start_time,
                                         all_sink_ids,
                                         follower_sink_ids,
                                         end_time)
        self.start_idx = 0
        while self.start_idx < len(self.times) and self.times[self.start_idx] < start_time:
            self.start_idx += 1

    def get_next_interval(self, event):
        if event is None:
            if len(self.t_diff) > self.start_idx:
                return self.t_diff[self.start_idx]
            else:
                return np.inf
        elif event.src_id == self.src_id:
            if self.start_idx < len(self.t_diff) - 1:
                self.start_idx += 1
                assert self.times[self.start_idx] > event.cur_time, "Skipped a real event."
                return self.t_diff[self.start_idx]
            else:
                return np.inf


# This should only contain immutable objects and create mutable objects on
# demand.
class SimOpts:
    """This class holds the options with methods which can return a manager for running the simulation."""
    def __init__(self, **kwargs):
        self.src_id        = kwargs['src_id']
        self.q_vec         = kwargs['q_vec']
        self.s             = kwargs['s']
        self.other_sources = kwargs['other_sources']
        self.sink_ids      = kwargs['sink_ids']
        self.edge_list     = kwargs['edge_list']
        self.end_time      = kwargs['end_time']

    def create_other_sources(self):
        """Instantiates the other_sources."""
        return [x[0](**x[1]) for x in self.other_sources]

    def create_manager_with_opt(self, seed):
        """Create a manager to run the simulation with Optimal broadcaster as
        one of the sources with the given seed."""
        opt = Opt(src_id=self.src_id, seed=seed, q_vec=self.q_vec, s=self.s)
        return Manager(sim_opts=self,
                       sources=[opt] + self.create_other_sources())

    def create_manager_with_poisson(self, seed, rate=None, capacity=None):
        """Create a manager to run the simulation with the given seed and the
        one source as Poisson with the provided capacity or rate.
        Only one of the two should be specified."""

        if rate is None and capacity is None:
            raise ValueError('One of rate or capacity must be specified.')
        elif rate is None:
            rate = capacity / self.end_time
        elif capacity is None:
            pass
        else:
            raise ValueError('Only one of rate or capacity must be specified.')

        poisson = Poisson2(src_id=self.src_id, seed=seed, rate=rate)
        return Manager(sim_opts=self,
                       sources=[poisson] + self.create_other_sources())

    def create_manager_with_piecewise_const(self, seed, change_times, rates):
        """This returns a manager which runs the simulation with a piece-wise
        constant broadcaster and the other sources."""
        assert len(change_times) == len(rates)
        piecewise = PiecewiseConst(src_id=self.src_id,
                                   seed=seed,
                                   change_times=change_times,
                                   rates=rates)
        return Manager(sim_opts=self,
                       sources=[piecewise] + self.create_other_sources())

    def create_manager_for_wall(self):
        """This generates the tweets of the rest of the other_sources only.
        Useful for heuristics or oracle."""
        edge_list = [x for x in self.edge_list if x[0] != self.src_id]
        return Manager(sim_opts=self.update({ 'edge_list': edge_list }),
                       sources=self.create_other_sources())

    def create_manager_with_times(self, event_times):
        """Returns a manager which runs the wall as dictated by the options
        and tweets at the specified times."""
        deterministic = RealData(self.src_id, event_times)
        return Manager(sim_opts=self,
                       sources=[deterministic] + self.create_other_sources())

    def get_dict(self):
        """Returns dictionary form of the options."""
        return {
            'src_id'        : self.src_id,
            'q_vec'         : self.q_vec,
            's'             : self.s,
            'other_sources' : self.other_sources,
            'sink_ids'      : self.sink_ids,
            'edge_list'     : self.edge_list,
            'end_time'      : self.end_time
        }

    def update(self, changes):
        new_opts = self.get_dict()
        new_opts.update(changes)
        return SimOpts(**new_opts)

    @staticmethod
    def std_poisson(world_seed, world_rate):
        """Returns a new SimOpts with fresh sources and default initialization."""
        return SimOpts(src_id=1,
                       other_sources=[(Poisson2,
                                       {'src_id': 2,
                                        'seed': world_seed,
                                        'rate': world_rate})],
                       end_time=1.0,
                       sink_ids=[1001],
                       q_vec=np.asarray([1.0]),
                       s=1.0,
                       edge_list=[(1, 1001), (2, 1001)])

    @staticmethod
    def std_hawkes(world_seed, world_lambda_0, world_alpha, world_beta):
        """Returns a new SimOpts with a Hawkes wall model."""
        assert world_alpha / world_beta <= 1.0, "The Hawkes wall will explode."

        return SimOpts(src_id=1,
                       other_sources=[(Hawkes,
                                       {'src_id': 2,
                                        'seed': world_seed,
                                        'l_0': world_lambda_0,
                                        'alpha': world_alpha,
                                        'beta': world_beta})],
                       end_time=1.0,
                       sink_ids=[1001],
                       q_vec=np.asarray([1.0]),
                       s=1.0,
                       edge_list=[(1, 1001), (2, 1001)])

    @staticmethod
    def std_piecewise_const(world_seed, world_change_times, world_rates):
        """Returns a new SimOpts with a Piecewise constant wall model."""
        return SimOpts(src_id=1,
                       other_sources=[(PiecewiseConst,
                                       {'src_id': 2,
                                        'seed': world_seed,
                                        'change_times': world_change_times,
                                        'rates': world_rates})],
                       end_time=1.0,
                       sink_ids=[1001],
                       q_vec=np.asarray([1.0]),
                       s=1.0,
                       edge_list=[(1, 1001), (2, 1001)])



def test_simOpts():
    init_opts = {
            'src_id'        : 1,
            'end_time'      : 100.0,
            'q_vec'         : np.array([1,2]),
            's'             : 1.0,
            'other_sources' : [(Poisson, {'src_id': 2, 'seed': 1}),
                               (Poisson, {'src_id': 3, 'seed': 1})],
            'sink_ids'      : [1001, 1000],
            'edge_list'     : [(1, 1001), (1, 1000), (2, 1000), (3, 1001)]
        }

    s = SimOpts(**init_opts)
    assert s.get_dict() == init_opts

    s2 = s.update({ 'src_id': 2 })
    assert s2.src_id == 2

    assert s.create_other_sources()[0].src_id == 2

test_simOpts()
