#!/usr/bin/env python

from __future__ import print_function
import scipy.stats.distributions as Distr
import numpy as np
import pandas as pd
import sys
import abc

class Event:
    def __init__(self, event_id, time_delta, src_id, sink_ids, metadata=None):
        self.event_id   = event_id
        self.time_delta = time_delta
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
              'sink_id'    : y}
             for x in self.events
             for y in x.sink_ids]
        )
        df['t'] = np.cumsum(df['time_delta'])
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
    def __init__(self, sink_ids, sources, edge_list=None):
        """The edge_list defines the network.
        Default is that all sources are connected to all sinks."""

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

        self.edge_list = edge_list
        self.sink_ids = sink_ids
        self.state = State(0, sink_ids)
        self.sources = sources

    def run_till(self, end_time, seed=42):
        # TODO: This only makes runs correct on a single seed system.
        # Will need to extend this to allow running this in a distributed
        # manner.
        np.random.seed(seed)

        # Step 1: Inform the sources of the sinks associated with them.
        # Step 2: Give them the initial state
        for src in self.sources:
            follower_ids = [x[1] for x in self.edge_list if x[0] == src.src_id]
            src.init_state(self.state.time, self.sink_ids, follower_ids)

        last_event = None
        event_id = 100

        while True:
            # Step 3: Generate one t_delta from each source and form the
            # event which is going to happen next.

            # This step can be made parallel
            t_delta, next_src_id = sorted((src.get_next_event_time(last_event),
                                           src.src_id)
                                          for src in self.sources)[0]

            assert t_delta >= 0, "Next event must be now or in the future."

            # Step 5: If cur_time + t_delta < end_time, go to step 4, else Step 7
            cur_time = self.state.time
            if cur_time + t_delta > end_time:
                break

            # Step 4: Execute the first event
            last_event = Event(event_id, t_delta, next_src_id,
                               [x[1] for x in self.edge_list
                                     if x[0] == next_src_id])
            event_id += 1
            self.state.apply_event(last_event)

            # Step 6: Go to step 3

        # Step 7: Stop


# Broadcasters
##############

class Broadcaster:
    __metaclass__ = abc.ABCMeta

    def __init__(self, src_id):
        self.src_id = src_id
        self.t_delta = None
        self.last_self_event_time = None

    def init_state(self, start_time, all_sink_ids, follower_sink_ids):
        self.sink_ids = sorted(follower_sink_ids)
        self.state = State(start_time, all_sink_ids)

    def get_next_event_time(self, event):
        self.state.apply_event(event)
        cur_time = self.state.time

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


    @abc.abstractmethod
    def get_next_interval(self):
        """Should return a number to replace the current time to next event or
           None if no change should be made."""
        raise NotImplemented()


class Poisson(Broadcaster):
    def __init__(self, src_id, rate=1.0):
        super(Poisson, self).__init__(src_id)
        self.rate = rate

    def get_next_interval(self, event):
        if event is None or event.src_id == self.src_id:
            # Draw a new time
            return Distr.expon.rvs(scale=1.0 / self.rate)


class Opt(Broadcaster):
    def __init__(self, src_id, q=1.0, s=1.0):
        super(Opt, self).__init__(src_id)
        self.q = q
        self.s = s
        self.old_rate = 0

    def init_state(self, start_time, all_sink_ids, follower_sink_ids):
        super(Opt, self).init_state(start_time, all_sink_ids, follower_sink_ids)
        if isinstance(self.q, dict):
            self.q_vec = np.asarray(self.q[x]
                                    for x in sorted(follower_sink_ids))
        else:
            # Assuming that the self.q is otherwise a scalar number.
            self.q_vec = np.ones(len(follower_sink_ids), dtype=float) * self.q

    def get_next_interval(self, event):
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
            # assert wall_ranks.shape[0] == 1, "Not implemented for multi follower case."

            new_rate = np.sqrt(self.q_vec / self.s).dot(r_t)
            rate = new_rate - self.old_rate
            self.old_rate = new_rate

            t_delta_new = Distr.expon.rvs(scale = 1.0 / rate)
            cur_time = self.state.time

            if self.last_self_event_time + self.t_delta > cur_time + t_delta_new:
                return cur_time + t_delta_new - self.last_self_event_time


class Hawkes(Broadcaster):
    def __init__(self, src_id, l_0=1.0, alpha=1.0, beta=1.0):
        super(Hawkes, self).__init__(src_id)
        self.l_0   = 1.0
        self.alpha = 1.0
        self.beta  = 1.0
        self.prev_excitations = []

    def get_next_interval(self, event):
        t = self.state.time
        if event is None or event.src_id == self.src_id:
            rate = self.l_0 + \
                   self.alpha * sum(np.exp([self.beta * -1.0 * (t - s)
                                            for s in self.prev_excitations]))

            t_delta = Distr.expon.rvs(scale=1.0 / rate)
            self.prev_excitations.append(t_delta)
            return t_delta


# TODO: Write a real-data reader/generator.



m = Manager([1000], [Poisson(1, 1.0), Poisson(2, 1.0), Opt(3)])


