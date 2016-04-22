#!/usr/bin/env python

import scipy.stats.distributions as Distr
import numpy as np

class Event:
    def __init__(self, event_id, time_delta, src_id, sink_ids, metadata=None):
        self.event_id   = event_id
        self.time_delta = time_delta
        self.src_id     = src_id
        self.sink_ids   = sink_ids
        self.metadata   = metadata

    def __repr__(self):
        return ('Event_id: {}, time_delta: {}, src_id: {}'
                .format(self.event_id, self.time_delta, self.src_id))


class State:
    def __init__(self, cur_time, sink_ids):
        self.num_sinks = len(sink_ids)
        self.time      = cur_time
        self.sinks     = dict((x,[]) for x in sink_ids)

    def apply_event(self, event):
        """Apply the given event to the state."""
        if event is None:
            # This was the first event, ignore
            return

        self.time += event.time_delta

        # Add the event (tweet) to the corresponding lists
        for sink_id in event.sink_ids:
            self.sinks[sink_id].append(event)


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
        np.random(seed)

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

            # Step 5: If cur_time + Δt < end_time, go to step 4, else Step 7
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
        self.last_time = None
        self.t_delta = None

    def init_state(self, state, follower_sink_ids):
        self.sink_ids = follower_sink_ids
        self.state = state
        self.last_time = state.time
        self.t_delta = Distr.expon.rvs(scale=1.0 / self.rate)

    def get_next_event_time(self, event):
        self.state.apply_event(event)

        if self.state.time >= self.last_time + self.t_delta:
            self.t_delta = Distr.expon.rvs(scale=1.0 / self.rate)

        return self.t_delta


class Opt:
    def __init__(self, src_id, q=1.0, s=1.0):
        self.q = q
        self.s = s

    def init_state(self, state, follower_sink_ids):
        self.sink_ids = follower_sink_ids
        self.state = state

    def get_next_event_time(self, event):
        self.state.apply_event(event)
        # TODO: Do magic


m = Manager([1000], [Poisson(1, 1.0), Poisson(2, 1.0)])


