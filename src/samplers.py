import numpy as np
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors

class Triplets(object):
    def __init__(self, n_neighbors=5, random=True, len_lim=True, **kwargs):
        self.n_neighbors = n_neighbors
        self.random = random
        self.len_lim = len_lim

    def fit_resample(self, x, y):
        strategy = self._sample_strategy(y)
        self.n_neighbors = max(self.n_neighbors, self.counts.max() // self.counts.min())

        gen_x = []
        gen_y = []
        for c, size in enumerate(strategy):
            if size == 0: continue
            weight = self._weights(x, y, c)
            gen_x_c, gen_y_c = self._sample_one(x, y, c, size, weight)
            gen_x += gen_x_c
            gen_y += gen_y_c
        gen_x = np.vstack(gen_x)
        gen_y = np.array(gen_y)
        return np.concatenate((x, gen_x), axis=0), np.concatenate((y, gen_y), axis=0)
    
    def _sample_strategy(self, y):
        _, self.counts = np.unique(y, return_counts=True)
        return max(self.counts) - self.counts
    
    def _weights(self, x, y, c):
        return np.ones(self.counts[c])
    
    def _sample_one(self, x, y, c, size, weight):
        gen_x = []
        gen_y = []
        if size == 0: return gen_x, gen_y

        # get the indices of minority and majority instances
        min_idxs = np.where(y == c)[0]
        maj_idxs = np.where(y != c)[0]

        # find nearest majority neighbors for each minority instance
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(x[maj_idxs])
        _, indices = nbrs.kneighbors(x[min_idxs])

        # generate synthetic data
        for j in np.random.choice(len(min_idxs), size, p=weight / weight.sum()):
            tp1 = x[min_idxs[j]]
            tp2 = x[maj_idxs[indices[j][:5]]].mean(axis=0)
            # tp3_ord = np.random.randint(1, self.n_neighbors)
            tp3_ord = np.random.randint(self.n_neighbors)
            tp3 = x[maj_idxs[indices[j][tp3_ord]]]
            if (tp2 == tp1).all():
                gen_x.append(tp1)
                gen_y.append(c)
                continue

            offset = tp3 - tp2
            if self.len_lim: offset = offset * min(1, norm(tp1 - tp2) / norm(offset))
            coef = np.random.rand() if self.random is True else 1.0
            new_x = tp1 + coef * offset
            gen_x.append(new_x)
            gen_y.append(c)
        return gen_x, gen_y