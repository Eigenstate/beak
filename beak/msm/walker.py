#!/usr/bin/env python
"""
Methods necessary for my hacky fake graph MSMs
"""

import numpy as np
from msmbuilder.msm import MarkovStateModel
from msmbuilder.lumping import PCCAPlus
from msmbuilder.tpt import hub_scores
from sklearn.utils import check_random_state

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class FakeMSM(object):
    """
    A fake MSM that can easily be walked on and transition matrix
    modified
    """
    def __init__(self, n_states):
        self.n_states_ = n_states
        self.transmat_ = np.zeros((n_states, n_states))
        self.populations_ = np.zeros(n_states)
        self.times_visited = np.zeros(n_states)

    #==========================================================================

    def sample_steps(self, state=None, n_steps=100, random_state=None):
        random = check_random_state(random_state)
        r = random.rand(1 + n_steps)

        if state is None:
            initial = np.sum(np.cumsum(self.populations_) < r[0])
        elif hasattr(state, '__len__') and len(state) == self.n_states_:
            initial = np.sum(np.cumsum(state) < r[0])
        else:
            initial = state

        cstr = np.cumsum(self.transmat_, axis=1)

        chain = [initial]
        for i in range(1, n_steps):
            visit = np.sum(cstr[chain[i - 1], :] < r[i])
            self.times_visited[visit] += 1
            chain.append(visit)

        return chain

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class NaiveWalker(object):
    """
    Walks around naively on the graph, with no
    adaptive sampling
    """
    def __init__(self, msm, nsteps, nwalkers):
        self.graph = FakeMSM(msm.n_states_)
        self.graph.transmat_ = np.copy(msm.transmat_)
        self.nsteps = nsteps
        self.walkers = nwalkers
        # Start all walkers at most populous state
        minpop = msm.inverse_transform(np.argsort(msm.populations_))[0][-1]
        self.start = [minpop] * self.walkers
        self.total = 0

    #==========================================================================

    def walk_once(self):
        for w in range(self.walkers):
            news = self.graph.sample_steps(state=self.start[w],
                                           n_steps=self.nsteps)
            self.start[w] = news[-1]
            self.total += self.nsteps

    #==========================================================================

    def walk_until(self, findme, sub_generation=False):
        """
        Check walkers individually, return minimum needed
        That is, don't count all walker steps in generation that
        found the bound state. This makes nsteps less discretized

        Args:
            findme (int): Cluster label to find
            sub_generation (bool): If mid-generation results should
                be counted nsteps individually
        """
        found = False
        while not found:
            for w in range(self.walkers):
                news = self.graph.sample_steps(state=self.start[w],
                                               n_steps=self.nsteps)
                found = found or findme in news
                if sub_generation and found:
                    self.total += news.index(findme)+1
                    break
                else:
                    self.total += self.nsteps

        return self.total

    #==========================================================================

    def walk(self):
        while np.where(self.graph.times_visited != 0)[0]:
            self.walk_once()

        return self.total

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class AdaptiveWalker(object):
    """
    Walks around adaptively on the graph, building a MSM as it goes.
    Adaptive samples according to either populations or hub_scores or counts
    """
    def __init__(self, msm, nsteps, criteria, nwalkers, lag=1):
        self.graph = FakeMSM(msm.n_states_)
        self.graph.transmat_ = np.copy(msm.transmat_)
        self.nsteps = nsteps
        self.criteria = criteria
        self.walkers = nwalkers
        self.lag = lag

        # Start all walkers at most populous state
        minpop = msm.inverse_transform(np.argsort(msm.populations_))[0][-1]
        self.start = [minpop] * self.walkers
        self.sampled = []
        self.total = 0

    #==========================================================================

    def rebuild_starts(self):
        """
        Constructs the microstate MSM, lumps it, and selects
        starting points (self.start) for next sampling run
        """
        micromsm = MarkovStateModel(lag_time=self.lag,
                                    prior_counts=1e-6,
                                    reversible_type="transpose",
                                    ergodic_cutoff="off")
        micromsm.fit(self.sampled)

        # Now make a macrostate MSM by lumping these into max 50 states
        # that way hub_scores never is overwhelmed, just like in my
        # actual simulation runs
        if len(micromsm.populations_) > 50:
            pcca = PCCAPlus.from_msm(micromsm, n_macrostates=50)
            mclustered = pcca.transform(self.sampled, mode="fill")

            estmsm = MarkovStateModel(lag_time=self.lag,
                                      prior_counts=1e-6,
                                      reversible_type="transpose",
                                      ergodic_cutoff="off")
            estmsm.fit(mclustered)
        else:
            estmsm = micromsm
            mclustered = self.sampled

        if self.criteria == "hub_scores":
            # Handle too few nodes found
            if len(estmsm.mapping_) <= 2:
                self.start = estmsm.inverse_transform(range(len(estmsm.populations_)))[0][:]
            else:
                scores = hub_scores(estmsm)
                self.start = estmsm.inverse_transform(np.argsort(scores))[0][:self.walkers]

        elif self.criteria == "populations":
            self.start = estmsm.inverse_transform(np.argsort(estmsm.populations_))[0][:self.walkers]

        elif self.criteria == "counts":
            self.start = np.argsort(self.graph.times_visited)[:self.walkers]

        else:
            raise ValueError("invalid criteria %s" % self.criteria)

        self.pad_starts()

    #==========================================================================

    def pad_starts(self):
        """
        Handles insufficient states being discovered to populate a starting
        state for all walkers from available data.
        Just pads with randomly selected already found clusters.
        """
        if len(self.start) < self.walkers:
            missing = self.walkers - len(self.start)
            additionals = np.random.choice(self.start, size=missing)
            self.start = np.append(self.start, additionals)

    #==========================================================================

    def walk_once(self):
        for w in range(self.walkers):
            news = self.graph.sample_steps(state=self.start[w],
                                           n_steps=self.nsteps)
            self.sampled.append(news)
            self.total += self.nsteps
        self.rebuild_starts()

    #==========================================================================

    def walk_until(self, findme, sub_generation=False):
        """
        Check walkers individually, return minimum needed
        That is, don't count all walker steps in generation that
        found the bound state. This makes nsteps less discretized

        Args:
            findme (int): Cluster to find
            sub_generation (bool): If mid-generation results should be
                individually counted
        """
        found = False
        while not found:
            for w in range(self.walkers):
                news = self.graph.sample_steps(state=self.start[w],
                                               n_steps=self.nsteps)
                self.sampled.append(news)

                found = found or findme in news
                if sub_generation and found:
                    self.total += news.index(findme)+1
                    break
                else:
                    self.total += self.nsteps

            if not found:
                self.rebuild_starts()

        return self.total

    #==========================================================================

    def walk(self):
        while np.where(self.graph.times_visited != 0)[0]:
            self.walk_once()

        return self.total

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
