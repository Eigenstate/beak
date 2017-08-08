#!/usr/bin/env python

import random
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
    def __init__(self, msm, nsteps):
        self.graph = FakeMSM(msm.n_states_)
        self.graph.transmat_ = np.copy(msm)
        self.nsteps = nsteps

        self.start = msm.inverse_transform(np.argsort(msm.populations_))[0][:3]
        self.sampled = [random.choice(self.start)]
        self.total = 0

    def walk_once(self):
        news = self.graph.sample_steps(state=self.sampled[-1],
                                       n_steps=self.nsteps)
        self.sampled.extend(news)
        self.total += self.nsteps

    def walk(self):
        found = set(self.sampled)
        while len(found) < self.graph.n_states_:
            self.walk_once()
            found.update(self.sampled)

        return self.total

class AdaptiveWalker(object):
    def __init__(self, msm, nsteps, criteria):
        self.graph = FakeMSM(msm.n_states_)
        self.graph.transmat_ = np.copy(msm)
        self.nsteps = nsteps
        self.criteria = criteria

        self.start = msm.inverse_transform(np.argsort(msm.populations_))[0][:3]
        self.start = random.choice(self.start)
        self.sampled = []
        self.total = 0

    def walk_once(self):
        news = self.graph.sample_steps(state=self.start,
                                       n_steps=self.nsteps)
        self.sampled.append(news)

        estmsm = MarkovStateModel(lag_time=1,
                                  prior_counts=1e-6,
                                  reversible_type="transpose",
                                  ergodic_cutoff="off")
        estmsm.fit_transform(self.sampled)

        if self.criteria == "hub_scores":
            scores = hub_scores(estmsm)
            self.start = estmsm.inverse_transform([np.argmin(scores)])[0][0]
        elif self.criteria == "populations":
            self.start = estmsm.inverse_transform([np.argmin(estmsm.populations_)])[0][0]

        self.total += self.nsteps
        return news

    def walk(self):
        found = set()
        while len(found) < self.graph.n_states_:
            news = self.walk_once()
            found.update(news)

        return self.total
