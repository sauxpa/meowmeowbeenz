import abc
from collections import deque
import numpy as np
import scipy
import networkx as nx
from functools import lru_cache
from copy import copy
from tqdm import tqdm
from typing import List


RATINGS = [1, 2, 3, 4, 5]
MAX_RATING = 5
MIN_RATING = 1


class MeowRandomRatings(abc.ABC):
    """Generic class to simulate random MeowMeowBeenz rating dynamics.
    """
    def __init__(self, G: nx.Graph):
        self._G = G

        self.X = np.empty(0)
        self.ratings = np.empty(0)
        self.giver_history = np.empty(0)
        self.receiver_history = np.empty(0)
        self.T = 0

    @property
    def G(self) -> nx.Graph:
        return self._G

    @G.setter
    def G(self, new_G: nx.Graph) -> None:
        self.flush_graph()
        self._G = new_G

    @property
    def N(self) -> int:
        return self.G.number_of_nodes()

    @property
    def N_edges(self) -> int:
        """Count edge (i, j) AND (j, i).
        """
        return 2 * self.G.number_of_edges()

    @property
    @lru_cache(maxsize=None)
    def nodes_list(self) -> list:
        return list(self.G.nodes)

    @property
    @lru_cache(maxsize=None)
    def edges_list(self) -> list:
        """Count edge (i, j) AND (j, i).
        """
        return list(self.G.edges) + [(j, i) for i, j in self.G.edges]

    def random_seed_nodes(self, k):
        """Select k nodes uniformly at random to be infected.
        Returns a vector x0 such that x0[i] = 1 if i is in the seed group,
        0 otherwise.
        """
        x0 = np.zeros(self.N)
        seed_patients = np.random.choice(self.G.nodes, size=k, replace=False)
        x0[seed_patients] = self.infected
        return x0

    @property
    @lru_cache(maxsize=None)
    def A(self) -> scipy.sparse.csr.csr_matrix:
        """Adjacency matrix of G.
        """
        return nx.adjacency_matrix(self.G)

    @property
    @lru_cache(maxsize=None)
    def spectrum(self) -> List:
        """Calculate and cache adjacency spectrum
        (sorted in decreasing order).
        """
        _spectrum = nx.adjacency_spectrum(self.G)
        idx = _spectrum.argsort()[::-1]
        return np.real(_spectrum[idx])

    @property
    @lru_cache(maxsize=None)
    def spectral_radius(self) -> float:
        return np.max(np.abs(self.spectrum))

    @property
    @lru_cache(maxsize=None)
    def spectral_gap(self) -> float:
        return self.spectrum[0] - self.spectrum[1]

    @property
    @lru_cache(maxsize=None)
    def cheeger_lower_bound(self) -> float:
        """Lower bound for the isoperimetric constant
        of the graph G given by its adjacency spectral gap.
        """
        return self.spectral_gap / 2

    @property
    @lru_cache(maxsize=None)
    def cheeger_upper_bound(self) -> float:
        """Upper bound for the isoperimetric constant
        of the graph G given by its adjacency spectral gap.
        """
        # Maximum degree
        dmax = np.max(self.A.dot(np.ones(self.N)))
        return np.sqrt(2 * dmax * self.spectral_gap)

    @property
    def cheeger_halfway_approx(self) -> float:
        """Approximate the isoperimetric constant of G
        by the average of its spectral upper and lower bounds.
        """
        return 0.5 * (self.cheeger_lower_bound + self.cheeger_upper_bound)

    def flush_graph(self) -> None:
        """Clear LRU cache of graph related properties.
        """
        type(self).nodes_list.fget.cache_clear()
        type(self).edges_list.fget.cache_clear()
        type(self).A.fget.cache_clear()
        type(self).spectrum.fget.cache_clear()
        type(self).spectral_radius.fget.cache_clear()
        type(self).spectral_gap.fget.cache_clear()
        type(self).cheeger_lower_bound.fget.cache_clear()
        type(self).cheeger_upper_bound.fget.cache_clear()

    def weight(self,
               new_rating: float,
               giver_rating: float,
               receiver_rating: float,
               mode: str,
               params: dict = {},
               ):
        if mode == 'giver_influence':
            return giver_rating / MAX_RATING
        elif mode == 'receiver_attractive':
            alpha = params.get('alpha')
            exponent = params.get('exponent', 1)
            return (
                alpha / (1 + np.abs(new_rating - receiver_rating)) ** exponent
                + (1 - alpha) * giver_rating / MAX_RATING
                )
        elif mode == 'receiver_repulsive':
            alpha = params.get('alpha')
            exponent = params.get('exponent', 1)
            return (
                alpha * np.abs(new_rating - receiver_rating) ** exponent
                / (MAX_RATING - MIN_RATING)
                + (1 - alpha) * giver_rating / MAX_RATING
                )
        else:
            raise ValueError('Unknown mode : {:s}'.format(mode))

    def generate_new_rating(self,
                            mode,
                            receiver,
                            giver,
                            params,
                            ):
        if mode == 'random':
            rating_distr = params.get('rating_distr')
            return np.random.choice(RATINGS, p=rating_distr)
        elif mode == 'retaliate':
            rating_distr_base = np.array(params.get('rating_distr'))
            alpha = params.get('alpha')
            rating_distr = (
                alpha * self.grudges[receiver][giver]
                + (1 - alpha) * rating_distr_base
                )
            rating_distr /= np.sum(rating_distr)
            return np.random.choice(RATINGS, p=rating_distr)
        else:
            raise ValueError('Unknown mode : {:s}'.format(mode))

    def simulate(self,
                 T: int,
                 x0: List = np.empty(0),
                 buffer_size: int = -1,
                 weight_mode: str = '',
                 weight_params: dict = {},
                 rating_mode: str = '',
                 rating_params: dict = {},
                 verbose: bool = False,
                 ) -> None:
        """Simulate rating dynamic for T steps.
        # By default, start everyone with average ratings.
        """
        if len(x0) == 0:
            x0 = np.ones(self.N) * np.mean(RATINGS)
        Xt = x0
        ratings_t = np.round(Xt)

        # Rating vectors
        self.X = np.empty((T + 1, self.N))
        self.ratings = np.empty((T + 1, self.N))
        self.X[0] = Xt
        self.ratings[0] = ratings_t

        # Initialise rating history
        rating_history = dict(zip(self.nodes_list, [deque([r]) for r in x0]))
        weights_history = dict(zip(self.nodes_list, [deque([r]) for r in x0]))

        self.giver_history = np.empty(T)
        self.receiver_history = np.empty(T)

        grudge_management = rating_mode == 'retaliate'
        if grudge_management:
            self.grudges = dict()
            for receiver in self.nodes_list:
                grudges_node = dict()
                for giver in self.G.neighbors(receiver):
                    grudges_node[giver] = np.zeros(len(RATINGS))
                self.grudges[receiver] = grudges_node
        else:
            self.grudges = None

        for t in tqdm(range(1, T + 1), disable=not verbose):
            # Pick a random edge (receiver, giver)
            idx_edge = np.random.randint(self.N_edges)
            receiver, giver = self.edges_list[idx_edge]

            self.giver_history[t - 1] = giver
            self.receiver_history[t - 1] = receiver

            # Generate new rating from giver to receiver
            new_rating = self.generate_new_rating(
                rating_mode,
                receiver,
                giver,
                rating_params,
                )

            # Generate weight for the newly generated rating
            weights_history[receiver].append(
                self.weight(
                    new_rating,
                    ratings_t[receiver],
                    ratings_t[giver],
                    weight_mode,
                    weight_params,
                    )
                )
            rating_history[receiver].append(new_rating)
            if grudge_management:
                new_rating_idx = int(new_rating - 1)
                self.grudges[receiver][giver][new_rating_idx] += 1

            # Trim buffer of past ratings if need be
            if buffer_size > 0 and len(rating_history[receiver]) > buffer_size:
                rating_history[receiver].popleft()
                weights_history[receiver].popleft()

            # Change rating of transitioned node
            Xnew = copy(Xt)
            Xnew[receiver] = np.average(
                rating_history[receiver],
                weights=weights_history[receiver])
            Xt = Xnew
            ratings_t = np.round(Xt)

            self.X[t] = Xt
            self.ratings[t] = ratings_t

        self.T = T

        # Compute best and worst on the raw ratings X rather than the
        # truncated ones, otherwise there are potentially many ties,
        # which numpy breaks by taking the smallest index.
        self.best = np.argmax(self.X, axis=1)
        self.worst = np.argmin(self.X, axis=1)

        self.ratings_best = np.array(
            [self.ratings[t, self.best[t]] for t in range(T + 1)]
            )
        self.ratings_worst = np.array(
            [self.ratings[t, self.worst[t]] for t in range(T + 1)]
            )
