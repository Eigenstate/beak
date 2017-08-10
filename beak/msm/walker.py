#!/usr/bin/env python

import numpy as np

from msmbuilder.msm import MarkovStateModel
from msmbuilder.tpt import hub_scores
from sklearn.utils import check_random_state

# Methods necessary for my hacky fake graph MSMs
class FakeMSM(object):
    def __init__(self, n_states):
        self.n_states_ = n_states
        self.transmat_ = np.zeros((n_states, n_states))
        self.populations_ = np.zeros(n_states)
        self.times_visited = np.zeros(n_states)

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

class NaiveWalker(object):
    def __init__(self, msm, nsteps, nwalkers):
        self.graph = FakeMSM(msm.n_states_)
        self.graph.transmat_ = np.copy(msm.transmat_)
        self.nsteps = nsteps
        self.walkers = nwalkers

        # Start all walkers at most populous state
        self.start = msm.inverse_transform(np.argsort(msm.populations_))[0][0]
        self.start = [self.start] * self.walkers
        self.sampled = []

        self.total = 0
        self.found = set(self.start)

    def walk_once(self):
        for w in range(self.walkers):
            news = self.graph.sample_steps(state=self.start[w],
                                           n_steps=self.nsteps)
            self.found.update(news)
            self.start[w] = news[-1]
            self.sampled.append(news)

            self.total += self.nsteps

    def walk_until(self, findme):
        while findme not in self.found:
            self.walk_once()
        return self.total

    def walk(self):
        while len(self.found) < self.graph.n_states_:
            self.walk_once()
            print("Now found %d" % len(self.found))

        return self.total

class AdaptiveWalker(object):
    def __init__(self, msm, nsteps, criteria, nwalkers):
        self.graph = FakeMSM(msm.n_states_)
        self.graph.transmat_ = np.copy(msm.transmat_)
        self.nsteps = nsteps
        self.criteria = criteria
        self.walkers = nwalkers

        # Start all walkers at most populous state
        self.start = msm.inverse_transform(np.argsort(msm.populations_))[0][0]
        self.start = [self.start] * self.walkers
        self.sampled = []
        self.total = 0
        self.found = set(self.start)

    def walk_once(self):
        for w in range(self.walkers):
            news = self.graph.sample_steps(state=self.start[w],
                                           n_steps=self.nsteps)
            self.found.update(news)
            self.sampled.append(news)
            self.total += self.nsteps

        estmsm = MarkovStateModel(lag_time=1,
                                  prior_counts=1e-6,
                                  reversible_type="transpose",
                                  ergodic_cutoff="off")
        estmsm.fit_transform(self.sampled)

        if self.criteria == "hub_scores":
            #print("Scoring")
            scores = hub_scores(estmsm)
            self.start = estmsm.inverse_transform(np.argsort(scores))[0][:self.walkers]
            #print("Start: %s" % self.start)
        elif self.criteria == "populations":
            self.start = estmsm.inverse_transform(np.argsort(estmsm.populations_))[0][:self.walkers]

    def walk_until(self, findme):
        while findme not in self.found:
            self.walk_once()
        return self.total

    def walk(self):
        while len(self.found) < self.graph.n_states_:
            self.walk_once()
            print("Now found %d" % len(self.found))

        return self.total

