import abc
from collections import deque
import numpy as np
import scipy
import networkx as nx
from functools import lru_cache
from copy import copy
from tqdm import tqdm


RATINGS = [1, 2, 3, 4, 5]


class MeowRandomRatings(abc.ABC):
    """Generic class to simulate random MeowMeowBeenz rating dynamics.
    """
    def __init__(self,
                 G: nx.Graph,
                 rating_distr: list,
                 ) -> None:
        self._G = G
        self._rating_distr = rating_distr

        self.X = np.empty(0)

    @property
    def G(self) -> nx.Graph:
        return self._G

    @G.setter
    def G(self, new_G: nx.Graph) -> None:
        self.flush_graph()
        self._G = new_G

    @property
    def rating_distr(self) -> list:
        return self._rating_distr

    @rating_distr.setter
    def rating_distr(self, new_rating_distr: list):
        self._rating_distr = new_rating_distr

    @property
    def N(self) -> int:
        return self.G.number_of_nodes()

    @property
    def N_edges(self) -> int:
        return self.G.number_of_edges()

    @property
    @lru_cache(maxsize=None)
    def nodes_list(self) -> list:
        return list(self.G.nodes)

    @property
    @lru_cache(maxsize=None)
    def edges_list(self) -> list:
        return list(self.G.edges)

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
    def spectrum(self) -> np.ndarray:
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

    def simulate(self,
                 T: int,
                 x0: np.ndarray = np.empty(0),
                 buffer_size: int = -1,
                 verbose: bool = False,
                 ) -> None:
        """Simulate rating dynamic for T steps.
        # By default, start everyone with average ratings.
        """
        if len(x0) == 0:
            x0 = np.ones(self.N) * np.mean(RATINGS)
        Xt = x0

        # Rating vectors
        X = np.empty((T + 1, self.N))
        X[0] = Xt

        # Initialise rating history
        rating_history = dict(zip(self.nodes_list, [deque([r]) for r in x0]))
        weights_history = dict(zip(self.nodes_list, [deque([r]) for r in x0]))

        for t in tqdm(range(1, T + 1), disable=not verbose):
            idx_edge = np.random.randint(self.N_edges)
            giver, receiver = self.edges_list[idx_edge]

            new_rating = np.random.choice(RATINGS, p=self.rating_distr)
            rating_history[receiver].append(new_rating)
            weights_history[receiver].append(rating_history[giver][-1])

            if buffer_size > 0 and len(rating_history[receiver]) > buffer_size:
                rating_history[receiver].popleft()
                weights_history[receiver].popleft()

            # Change rating of transitioned node
            Xnew = copy(Xt)
            Xnew[receiver] = np.average(
                rating_history[receiver],
                weights=weights_history[receiver])
            Xt = Xnew

            X[t] = Xt

        self.X = np.array(X)
        self.T = T