#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import logging
import pandas as pd
import itertools
import warnings
import abc
import bisect

try:
    from .utils import mb, is_sorted
except ModuleNotFoundError:
    # May have been imported via explicit %run -i
    warnings.warn('Unable to import is_sorted from utils.')
    pass


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
        return ('[ Event_id: {}, time_delta: {}, cur_time: {}, src_id: {} ]'
                .format(self.event_id, self.time_delta, self.cur_time, self.src_id))


class State:
    def __init__(self, cur_time, sink_ids):
        self.num_sinks         = len(sink_ids)
        self.time              = cur_time
        self.sinks             = dict((x, []) for x in sink_ids)
        self.walls_updated     = False
        self.events            = []
        self.track_src_id      = None
        self._tracked_ranks    = None
        self._tracked_sink_ids = None
        self._sorted_sink_ids  = sorted(self.sinks.keys())

    def set_track_src_id(self, src_id, follower_sink_ids):
        self.track_src_id = src_id
        # Assume that the rank person tweeted first.
        self._tracked_ranks = dict((sink_id, 0) for sink_id in follower_sink_ids)
        self._tracked_sink_ids = follower_sink_ids

    def update_walls(self):
        """Adds the events to the walls. Needed for calculating the ranks."""
        assert not self.walls_updated
        self.walls_updated = True
        for ev in self.events:
            for sink_id in ev.sink_ids:
                self.sinks[sink_id].append(ev)

    def apply_event(self, event, force_wall_update=False):
        """Apply the given event to the state."""
        if event is None:
            # This was the first event, ignore
            return

        self.events.append(event)
        self.time += event.time_delta

        if self.track_src_id is not None:
            if event.src_id == self.track_src_id:
                self._tracked_ranks = dict((sink_id, 0) for sink_id in self._tracked_sink_ids)
            else:
                for sink_id in event.sink_ids:
                    if sink_id in self._tracked_sink_ids:
                        self._tracked_ranks[sink_id] += 1

        if force_wall_update:
            self.walls_updated = True
            # Add the event (tweet) to the corresponding lists
            # But do this only when requested.
            for sink_id in event.sink_ids:
                self.sinks[sink_id].append(event)

    def get_dataframe(self):
        """Return the list of events."""
        # Using a list here appears faster than using a generator expression
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

    def get_num_events(self):
        """Returns number of events which have happened."""
        return len(self.events)

    def get_wall_rank(self, src_id, follower_ids, dict_form=True, force_recalc=False, assume_first=False):
        """Return a dictionary or vectors of rank of the src_id on the wall of
        the followers. If the follower does not have any events from the given
        source, the corresponding rank is `None`. The resulting vector will
        be formed after sorting the sink_ids.

        'assume_first' indicates that we assume that the src_id posted a tweet at time 0-, i.e.
        just before the start of the experiment. This means that none of the ranks will be NaN.
        """
        if self.track_src_id != src_id or force_recalc:

            if not self.walls_updated:
                self.update_walls()

            rank = dict((sink_id, None) for sink_id in follower_ids)
            for sink_id in follower_ids:

                if assume_first:
                    rank[sink_id] = len(self.sinks[sink_id])

                for ii in range(len(self.sinks[sink_id]) - 1, -1, -1):
                    if self.sinks[sink_id][ii].src_id == src_id:
                        rank[sink_id] = len(self.sinks[sink_id]) - ii - 1
                        break  # breaks the inner for loop
        else:
            rank = self._tracked_ranks

            if assume_first:
                rank = rank.copy()
                for sink_id in rank.keys():
                    if rank[sink_id] is None:
                        rank[sink_id] = len(self.sinks[sink_id])

        if dict_form:
            return rank
        else:
            return np.asarray([rank[sink_id]
                               for sink_id in self._sorted_sink_ids
                               if sink_id in follower_ids])


class Manager:
    def __init__(self, sources, sink_ids=None, end_time=None,
                 edge_list=None, sim_opts=None, start_time=0):
        """The edge_list defines the network.
        Default is that all sources are connected to all sinks."""

        if sim_opts is not None:
            # sources   = mb(sources, sim_opts.other_sources))
            edge_list = mb(edge_list, sim_opts.edge_list)
            sink_ids  = mb(sink_ids, sim_opts.sink_ids)
            end_time  = mb(end_time, sim_opts.end_time)

        assert len(sources) > 0, "No sources."
        assert len(sink_ids) > 0, "No sinks."
        assert len(set(x.src_id for x in sources)) == len(sources), "Duplicates in sources."
        assert len(set(sink_ids)) == len(sink_ids), "Duplicates in sink_ids."

        if edge_list is None:
            # If the edge_list is None, then assume that all sources are
            # connected to all sinks.
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
        self.state     = State(start_time, sink_ids)
        self.sources   = sources

    def get_state(self):
        """Returns the current state of the simulation."""
        return self.state

    def run(self):
        warnings.warn('Consider using `run_dynamic` instead of `run`.')
        return self.run_till()

    def run_till(self, end_time=None):
        if end_time is not None:
            logging.warn('Warning: deprecation warning: end_time should not be set.')
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

            # assert t_delta >= 0, "Next event must be now or in the future."

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

    def run_dynamic(self, max_events=float('inf')):
        if max_events is None:
            max_events = float('inf')

        end_time = self.end_time

        # Step 1: Inform the sources of the sinks associated with them.
        # Step 2: Give them the initial state
        dynamic_sources = []
        static_source_times = []
        for src in self.sources:
            if not src.is_fresh():
                raise ValueError('Source with id: {} is not fresh.'
                                 .format(src.src_id))

            follower_ids = [x[1] for x in self.edge_list if x[0] == src.src_id]
            src.init_state(self.state.time, self.sink_ids, follower_ids, end_time)

            if src.is_dynamic:
                dynamic_sources.append(src)
            else:
                src.initialize()
                static_source_times.extend(zip(src.get_all_times(),
                                               itertools.repeat(src.src_id)))

        last_event = None
        event_id = 100
        static_source_times = sorted(static_source_times)
        static_idx = 0

        while self.state.get_num_events() < max_events:
            # Step 3: Generate one t_delta from each source and form the
            # event which is going to happen next.

            # This step can be made parallel.
            # If multiple events happen at the same time, then the ordering
            # will still be deterministic: by the ID of the source.
            if len(dynamic_sources) > 0:
                t_delta, next_src_id = sorted((src.get_next_event_time(last_event),
                                               src.src_id)
                                              for src in dynamic_sources)[0]
            else:
                t_delta, next_src_id = np.inf, None

            # assert t_delta >= 0, "Next event must be now or in the future."

            cur_time = self.state.time

            if static_idx < len(static_source_times) and \
                    cur_time + t_delta > static_source_times[static_idx][0]:
                # Play the event of the static source
                event_time, event_src = static_source_times[static_idx]
                static_idx += 1
            else:
                # Play the event of the dynamic source
                event_time = cur_time + t_delta
                event_src = next_src_id

            # Step 5: If cur_time + t_delta < end_time, go to step 4, else Step 7
            if event_time > end_time:
                break

            # Step 4: Execute the first event
            last_event = Event(event_id, event_time - cur_time,
                               event_time, event_src,
                               [x[1] for x in self.edge_list
                                if x[0] == event_src])
            event_id += 1
            self.state.apply_event(last_event)

            # Step 6: Go to step 3

        # Step 7: Stop
        return self


# Broadcasters
##############

# Make Broadcasters serializable
# -- The broadcasters do not need to be serializable, the sim_opts does.

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
        self.is_dynamic           = True

    def get_all_times(self):
        assert not self.is_dynamic
        raise NotImplementedError()

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
            logging.warning('src_id: {}, event_id: {}, returned t_delta = {} < 0, set to 0 instead.'
                            .format(self.src_id, event.event_id, ret_t_delta))
            ret_t_delta = 0.0

        return ret_t_delta

    def get_current_time(self, event):
        return event.cur_time if event is not None else self.start_time

    @abc.abstractmethod
    def get_next_interval(self, event):
        """Should return a number to replace the current time to next event or
        None if no change should be made."""
        raise NotImplemented()


class Poisson2(Broadcaster):
    def __init__(self, src_id, seed, rate=1.0):
        super(Poisson2, self).__init__(src_id, seed)
        self.rate       = rate
        self.is_dynamic = False
        self.init       = False
        self.times      = None
        self.t_diff     = None
        self.start_idx  = None

    def get_all_times(self):
        assert self.init
        # Drop the start_time which was spuriously entered.
        return self.times[1:]

    def initialize(self):
        self.init      = True
        duration       = self.end_time - self.start_time
        num_events     = self.random_state.poisson(self.rate * duration)
        event_times    = self.random_state.uniform(low=self.start_time,
                                                   high=self.end_time,
                                                   size=num_events)
        self.times     = sorted(np.concatenate([[self.start_time], event_times]))
        self.t_diff    = np.diff(self.times)
        self.start_idx = 0

    def get_next_interval(self, event):
        if not self.init:
            self.initialize()

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
        self.is_dynamic = True
        self.rate = rate

    def get_next_interval(self, event):
        if event is None or event.src_id == self.src_id:
            # Draw a new time, one event at a time
            return self.random_state.exponential(scale=1.0 / self.rate)

# class Hawkes2(Broadcaster):
#     def __init__(self, src_id, seed, l_0, alpha, beta):
#         super(Hawkes2, self).__init__(src_id, seed)
#         self.l_0 = l_0
#         self.alpha = alpha
#         self.beta = beta
#         self.prev_interactions = []
#         self.is_dynamic = False
#         self.init = False
#
#     def get_rate(self, t):
#         """Returns the rate of current Hawkes at time `t`."""
#         return self.l_0 + \
#             self.alpha * sum(np.exp([self.beta * -1.0 * (t - s)
#                                      for s in self.prev_excitations
#                                      if s < t]))


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
                t_delta = self.random_state.exponential(scale=1.0 / rate_bound)

                # Rejection sampling
                if self.random_state.rand() < self.get_rate(t + t_delta) / rate_bound:
                    break

            self.prev_excitations.append(t + t_delta)
            return t_delta


class Opt(Broadcaster):
    def __init__(self, src_id, seed, q=1.0, s=1.0):
        super(Opt, self).__init__(src_id, seed)
        self.q = q
        self.s = s
        self.sqrt_s_by_q = None
        self.old_rate = 0
        self.init = False

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            self.state.set_track_src_id(self.src_id, self.sink_ids)

            if isinstance(self.s, dict):
                self.s_vec = np.asarray([self.s[x]
                                         for x in sorted(self.sink_ids)])
            else:
                # Assuming that the self.q is otherwise a scalar number.
                # Or a vector with the same number of elements as sink_ids
                self.s_vec = np.ones(len(self.sink_ids), dtype=float) * self.s

            self.sqrt_s_by_q = np.sqrt(self.s_vec / self.q)

        self.state.apply_event(event)

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
            new_rate = self.sqrt_s_by_q.dot(r_t)
            diff_rate = new_rate - self.old_rate
            self.old_rate = new_rate

            t_delta_new = self.random_state.exponential(scale=1.0 / diff_rate)
            cur_time = event.cur_time

            if self.last_self_event_time + self.t_delta > cur_time + t_delta_new:
                return cur_time + t_delta_new - self.last_self_event_time


class OptPWSignificance(Broadcaster):
    def __init__(self, src_id, seed, s_vec, time_period, q=1.0):
        super(OptPWSignificance, self).__init__(src_id, seed)
        # The assumption is that all changes happen at the same time across users.
        self.s_pw = np.asarray(s_vec)  # size = |sink_ids| * |segments|
        self.q = q
        self.old_ranks = 0
        self.time_period = time_period
        self.init = False

    def take_one_sample(self, event, pw_intensity):
        """Takes one sample from the given pw_intensity, using the correct phase."""
        phase = self.get_current_time(event) % self.time_period
        # Rejection sampling with self.random_state
        new_sample = 0
        num_segments = pw_intensity.shape[0]
        s_max = np.max(pw_intensity)
        while True:
            new_sample += self.random_state.exponential(scale=1.0 / s_max)
            current_piece_index = int(num_segments * ((new_sample + phase) % self.time_period) / self.time_period)
            if self.random_state.rand() < pw_intensity[current_piece_index] / s_max:
                # print('Sample chosen: ', new_sample)
                return new_sample

    def get_next_interval(self, event):
        if not self.init:
            self.init = True
            self.state.set_track_src_id(self.src_id, self.sink_ids)
            self.old_ranks = np.asarray([0] * len(self.sink_ids))

            if len(self.s_pw.shape) == 1:
                num_followers = len(self.sink_ids)
                # Spread the same s_pw to all the followers
                #
                # Note that here the vector s is interpreted differently
                # from in Opt where the shape of s has to be either 1 or
                # equal to the number of followers. Here, the size is treated
                # as the number of segments in the piecewise continuous
                # significance of each follower.
                #
                # This will come back to bite us at some point.
                self.s_pw = (self.s_pw
                             .repeat(num_followers)
                             .reshape((num_followers, -1), order='F'))
            self.s_max = np.max(self.s_pw.sum(0))

        self.state.apply_event(event)

        if event is None:
            # Tweet immediately if this is the first event.
            self.old_ranks = np.asarray([0] * len(self.sink_ids))
            return 0
        elif event.src_id == self.src_id:
            # No need to tweet if we are on top of all walls
            self.old_rate = 0
            self.old_ranks = np.asarray([0] * len(self.sink_ids))
            return np.inf
        else:
            # check status of all walls and find position in it.
            new_ranks = self.state.get_wall_rank(self.src_id, self.sink_ids,
                                                 dict_form=False)

            # If multiple walls are updated at the same time, should the
            # drawing happen only once after all the updates have been applied
            # or one at a time? Does that make a difference? Probably not. A
            # lot more work if the events are sent one by one per wall, though.
            rank_diff = new_ranks - self.old_ranks
            pw_intensity = (np.sqrt(self.s_pw / self.q) * rank_diff[:, None]).sum(0)

            # Now to actually take a sample
            t_delta_new = self.take_one_sample(event, pw_intensity)
            cur_time = event.cur_time

            self.old_ranks = new_ranks

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
        self.is_dynamic   = False

    def initialize(self):
        self.init = True
        assert self.start_time <= self.change_times[0]
        assert self.end_time   >= self.change_times[-1]

        duration = self.end_time - self.start_time
        max_rate = np.max(self.rates)

        # Using thinning to determine the event times.
        num_all_events = self.random_state.poisson(max_rate * duration)
        all_event_times = self.random_state.uniform(low=self.start_time,
                                                    high=self.end_time,
                                                    size=num_all_events)
        thinned_event_times = []
        for t in sorted(all_event_times):
            # Rejection sampling
            if self.random_state.rand() < self.get_rate(t) / max_rate:
                thinned_event_times.append(t)

        self.times = np.concatenate([[self.start_time], thinned_event_times])
        self.t_diff = np.diff(self.times)
        self.start_idx = 0

    def get_all_times(self):
        assert self.init
        return self.times[1:]

    def get_rate(self, t):
        """Finds what the instantaneous rate at time 't' is."""
        return self.rates[bisect.bisect(self.change_times, t) - 1]

    def get_next_interval(self, event):
        if not self.init:
            self.initialize()

        if event is None:
            return self.t_diff[0]
        elif event.src_id == self.src_id:
            self.start_idx += 1

            if self.start_idx < len(self.t_diff):
                # These assertions may not hold in case of rounding errors.
                # TODO: How to handle those cases? Return t_delta as well as tweet time?
                # assert self.times[self.start_idx] <= event.cur_time
                # assert self.times[self.start_idx + 1] > event.cur_time
                return self.t_diff[self.start_idx]
            else:
                return np.inf


class RealData2(Broadcaster):
    def __init__(self, src_id, times):
        super(RealData2, self).__init__(src_id, 0)
        self.times = times
        self.is_dynamic = False

    def get_num_events(self):
        return len(self.times)

    def init_state(self, start_time, all_sink_ids, follower_sink_ids, end_time):
        super(RealData2, self).init_state(start_time,
                                          all_sink_ids,
                                          follower_sink_ids,
                                          end_time)
        self.start_idx = 0
        self.relevant_times = self.times[self.times >= start_time]
        self.t_diff = np.diff(np.concatenate([[start_time], self.relevant_times]))


class RealData(Broadcaster):
    def __init__(self, src_id, times):
        super(RealData, self).__init__(src_id, 0)
        self.times = np.asarray(times)
        self.t_diff = None
        self.is_dynamic = False
        self.start_idx = None

    def get_num_events(self):
        return len(self.times)

    def init_state(self, start_time, all_sink_ids, follower_sink_ids, end_time):
        super(RealData, self).init_state(start_time,
                                         all_sink_ids,
                                         follower_sink_ids,
                                         end_time)
        self.start_idx = 0
        self.relevant_times = self.times[self.times >= start_time]
        self.t_diff = np.diff(np.concatenate([[start_time], self.relevant_times]))

    def get_all_times(self):
        return self.relevant_times

    def initialize(self):
        pass

    def get_next_interval(self, event):
        if event is None:
            if len(self.t_diff) > self.start_idx:
                assert self.relevant_times[self.start_idx] >= self.get_current_time(event), "Skipped an initial real event."
                return self.t_diff[self.start_idx]
            else:
                return np.inf
        elif event.src_id == self.src_id:
            if self.start_idx < len(self.t_diff) - 1:
                self.start_idx += 1
                assert self.relevant_times[self.start_idx] >= event.cur_time, "Skipped a real event."
                return self.t_diff[self.start_idx]
            else:
                return np.inf


# This should only contain immutable objects and create mutable objects on
# demand.
class SimOpts:
    """This class holds the options with methods which can return a manager for running the simulation."""

    broadcasters = {
        'Hawkes': Hawkes,
        'RealData': RealData,
        'Opt': Opt,
        'PiecewiseConst': PiecewiseConst,
        'Poisson': Poisson,
        'Poisson2': Poisson2,
        'OptPWSignificance': OptPWSignificance
    }

    @classmethod
    def registerSource(cls, sourceName, sourceConstructor):
        """Register another kind of broadcaster."""
        cls.broadcasters[sourceName] = sourceConstructor

    def __init__(self, **kwargs):
        self.src_id        = kwargs['src_id']
        self.s             = kwargs['s']
        self.q             = kwargs['q']
        self.other_sources = kwargs['other_sources']
        self.sink_ids      = kwargs['sink_ids']
        self.edge_list     = kwargs['edge_list']
        self.end_time      = kwargs['end_time']

    def create_other_sources(self):
        """Instantiates the other_sources."""
        others = []
        for x in self.other_sources:
            if callable(x[0]):
                others.append(x[0](**x[1]))
            elif x[0] in self.broadcasters:
                others.append(self.broadcasters[x[0]](**x[1]))
            else:
                raise ValueError('Unknown type of broadcaster: {}'.format(x[0]))

        return others

    def randomize_other_sources(self, using_seed):
        """Returns a new sim_opts after randomizing the seeds of the other sources."""
        other_sources = []
        for idx, (x, y) in enumerate(self.other_sources):
            assert 'seed' in y, 'Do not know how to randomize {}.'.format(x)
            y_new = y.copy()
            y_new['seed'] = using_seed + 99 * idx
            other_sources.append((x, y_new))

        return self.update({'other_sources': other_sources})

    def create_manager_with_opt(self, seed):
        """Create a manager to run the simulation with Optimal broadcaster as
        one of the sources with the given seed."""
        opt = Opt(src_id=self.src_id, seed=seed, s=self.s, q=self.q)
        return Manager(sim_opts=self,
                       sources=[opt] + self.create_other_sources())

    def create_manager_with_broadcaster(self, broadcaster):
        """Create a manager to run the simulation with the provided broadcaster
        as one of the sources with the given seed."""
        assert broadcaster.src_id == self.src_id, "Broadcaster has src_id = {}; expected = {}".format(broadcaster.src_id, self.src_id)

        return Manager(sim_opts=self,
                       sources=[broadcaster] + self.create_other_sources())

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

    def create_manager_with_significance(self, seed, time_period, significance=None, num_segments=None):
        """Creates a manager to run the simulation with the given seed and with
        the passed significance. If not passed, it will attempt to use the
        s as the significance vector. Failing that, if num_segments is
        provided, it will extend s to the required size and return the
        manager."""

        num_followers = len(self.sink_ids)

        if significance is not None:
            significance = np.asarray(significance).astype(float)
        else:
            # We are extending s_vec to significance across time-periods for
            # corresponding followers here (in order the sinks appear.
            s_vec = np.asarray(self.s)
            if s_vec.shape[0] == 1 and num_segments is not None:
                s_vec = np.ones((num_followers, num_segments), dtype=float) * s_vec[:, None]

            if num_segments is not None:
                s_vec = np.ones((num_followers, num_segments)) * s_vec[:, None]

            significance = s_vec

        assert len(significance.shape) == 2, "Significance must be 2 dimensional."
        assert significance.shape[1] == num_segments or num_segments is None, "Number of segments in significance do not match"
        assert significance.shape[0] == len(self.sink_ids), "Number of sink_ids is not the same as size of significance."

        opt_pw = OptPWSignificance(src_id=self.src_id,
                                   seed=seed,
                                   s_vec=significance,
                                   time_period=time_period,
                                   q=self.q)

        return Manager(sim_opts=self,
                       sources=[opt_pw] + self.create_other_sources())

    def create_manager_for_wall(self):
        """This generates the tweets of the rest of the other_sources only.
        Useful for heuristics or oracle."""
        edge_list = [x for x in self.edge_list if x[0] != self.src_id]
        return Manager(sim_opts=self.update({'edge_list': edge_list}),
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
            'q'             : self.q,
            's'             : self.s,
            'other_sources' : self.other_sources,
            'sink_ids'      : self.sink_ids,
            'edge_list'     : self.edge_list,
            'end_time'      : self.end_time
        }

    def copy(self):
        """Create a copy of this SimOpts."""
        return self.update({})

    def update(self, changes):
        """Make the supplied changes and return a new SimOpts."""
        new_opts = self.get_dict()
        new_opts.update(changes)
        return SimOpts(**new_opts)

    @staticmethod
    def std_poisson(world_seed, world_rate):
        """Returns a new SimOpts with fresh sources and default initialization."""
        return SimOpts(src_id=1,
                       other_sources=[('Poisson2',
                                       {'src_id': 2,
                                        'seed': world_seed,
                                        'rate': world_rate})],
                       end_time=1.0,
                       sink_ids=[1001],
                       s=np.asarray([1.0]),
                       q=1.0,
                       edge_list=[(1, 1001), (2, 1001)])

    @staticmethod
    def std_hawkes(world_seed, world_lambda_0, world_alpha, world_beta):
        """Returns a new SimOpts with a Hawkes wall model."""
        assert world_alpha / world_beta <= 1.0, "The Hawkes wall will explode."

        return SimOpts(src_id=1,
                       other_sources=[('Hawkes',
                                       {'src_id': 2,
                                        'seed': world_seed,
                                        'l_0': world_lambda_0,
                                        'alpha': world_alpha,
                                        'beta': world_beta})],
                       end_time=1.0,
                       sink_ids=[1001],
                       s=np.asarray([1.0]),
                       q=1.0,
                       edge_list=[(1, 1001), (2, 1001)])

    @staticmethod
    def std_piecewise_const(world_seed, world_change_times, world_rates):
        """Returns a new SimOpts with a Piecewise constant wall model."""
        return SimOpts(src_id=1,
                       other_sources=[('PiecewiseConst',
                                       {'src_id': 2,
                                        'seed': world_seed,
                                        'change_times': world_change_times,
                                        'rates': world_rates})],
                       end_time=1.0,
                       sink_ids=[1001],
                       s=np.asarray([1.0]),
                       q=1.0,
                       edge_list=[(1, 1001), (2, 1001)])


def test_simOpts():
    init_opts = {
        'src_id'        : 1,
        'end_time'      : 100.0,
        's'             : np.array([1, 2]),
        'q'             : 1.0,
        'other_sources' : [(Poisson, {'src_id': 2, 'seed': 1}),
                           (Poisson, {'src_id': 3, 'seed': 1})],
        'sink_ids'      : [1001, 1000],
        'edge_list'     : [(1, 1001), (1, 1000), (2, 1000), (3, 1001)]
    }

    s = SimOpts(**init_opts)
    assert s.get_dict() == init_opts

    s2 = s.update({'src_id': 2})
    assert s2.src_id == 2

    assert s.create_other_sources()[0].src_id == 2

    init_opts_2 = {
        'src_id'        : 1,
        'end_time'      : 100.0,
        's'             : np.array([1, 2]),
        'q'             : 1.0,
        'other_sources' : [('Poisson', {'src_id': 2, 'seed': 1, 'rate': 1000.0}),
                           ('Poisson2', {'src_id': 3, 'seed': 1, 'rate': 1000.0}),
                           ('Hawkes', {'src_id': 4, 'seed': 1, 'l_0': 1.0, 'alpha': 1.0, 'beta': 10.0}),
                           ('PiecewiseConst', {'src_id': 5, 'seed': 1, 'rates': [0.0, 0.5, 1.0], 'change_times': [0, 50, 75]}),
                           ('Opt', {'src_id': 6, 'seed': 1, 's': np.array([1.0]), 'q': 1.0}),
                           ('RealData', {'src_id': 7, 'times': [0, 50, 75]})],
        'sink_ids'      : [1001, 1000],
        'edge_list'     : [(1, 1001), (1, 1000), (2, 1000), (3, 1001), (4, 1000), (5, 1000), (6, 1000), (7, 1000)]
    }

    s = SimOpts(**init_opts_2)

    assert isinstance(s.create_other_sources()[0], Poisson)
    assert isinstance(s.create_other_sources()[1], Poisson2)
    assert isinstance(s.create_other_sources()[2], Hawkes)
    assert isinstance(s.create_other_sources()[3], PiecewiseConst)
    assert isinstance(s.create_other_sources()[4], Opt)
    assert isinstance(s.create_other_sources()[5], RealData)


test_simOpts()
