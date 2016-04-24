#!/usr/bin/env python

from __future__ import print_function
import scipy.stats.distributions as Distr
import numpy as np
import sys

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

    def clone(state):
        return State(state.time, state.sinks.keys())

    def apply_event(self, event):
        """Apply the given event to the state."""
        if event is None:
            # This was the first event, ignore
            return

        self.time += event.time_delta

        # Add the event (tweet) to the corresponding lists
        for sink_id in event.sink_ids:
            self.sinks[sink_id].append(event)

    def time_on_top(self):
        """Returns how much time each source spent on the top."""
        pass

    def get_wall_rank(self, src_id, follower_ids):
        """Return a dictionary of rank of the src_id on the wall of the
        followers. If the follower does not have any events from the given
        source, the corresponding rank is `None`.
        """
        rank = dict((sink_id, None) for sink_id in follower_ids)
        for sink_id in follower_ids:
            for ii in range(len(self.sinks[sink_id]) - 1, -1, -1):
                if self.sinks[sink_id][ii].src_id == src_id:
                    rank[sink_id] = len(self.sinks[sink_id]) - ii - 1
                    break # breaks the inner for loop
        return rank


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
            src.init_state(self.state, follower_ids)

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


class Poisson:
    def __init__(self, src_id, rate=1.0):
        self.rate = rate
        self.src_id = src_id
        self.last_event_time = None
        self.t_delta = None

    def init_state(self, state, follower_sink_ids):
        self.sink_ids = follower_sink_ids
        self.state = State.clone(state)

    def get_next_event_time(self, event):
        self.state.apply_event(event)
        cur_time = self.state.time

        # Assuming that all events will be executed at some point.
        if event is None or event.src_id == self.src_id:
            self.t_delta = Distr.expon.rvs(scale=1.0 / self.rate)
            self.last_event_time = cur_time

        return self.last_event_time + self.t_delta - cur_time


class Opt:
    def __init__(self, src_id, q=1.0, s=1.0):
        self.q = q
        self.s = s
        self.src_id = src_id
        self.last_event_time = None
        self.u_delta = None
        self.t_delta = None

    def init_state(self, state, follower_sink_ids):
        self.sink_ids = follower_sink_ids
        self.state = State.clone(state)

    def get_next_event_time(self, event):
        self.state.apply_event(event)
        cur_time = self.state.time

        if event is None:
            # Tweet immediately if this is the first event.
            self.u_delta = None
            self.t_delta = 0
            self.last_event_time = cur_time
        elif event.src_id == self.src_id:
            # No need to tweet if we are on top of all walls
            self.u_delta = None
            self.t_delta = np.inf
            self.last_event_time = cur_time
        else:
            # check status of all walls and find position in it.
            wall_ranks = self.state.get_wall_rank(self.src_id, self.sink_ids)
            assert len(wall_ranks) == 1, "Not implemented for multi follower case."

            rate = np.sqrt(self.q / self.s) * wall_ranks[self.sink_ids[0]]

            if self.u_delta is None:
                # Have to draw the first sample
                self.u_delta = np.random.rand()

            # Now re-evaluate the t_delta since last event
            self.t_delta = Distr.expon.ppf(self.u_delta, scale=1.0 / rate)

        ret_t_delta = self.last_event_time + self.t_delta - cur_time

        if ret_t_delta < 0:
            print('returned t_delta = {} < 0, set to 0 instead.'.format(ret_t_delta), file=sys.stderr)
            ret_t_delta = 0.0

        return ret_t_delta


m = Manager([1000], [Poisson(1, 1.0), Poisson(2, 1.0), Opt(3)])


