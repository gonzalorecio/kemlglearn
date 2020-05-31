import numpy as np
from scipy.stats import mode
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt


class PowerIterationClustering:
    labels_ = None
    cluster_centroids_ = None
    embedding_ = None

    def __init__(self, n_clusters, max_iter=500):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X, affinity='rbf'):
        """Compute Power Iteration Clustering (PIC).
        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
        affinity: similarity metric for the affinity matrix
        """
        if affinity == 'rbf':
            A = np.exp(-distance.cdist(X, X, 'seuclidean')
                       ** 2/(2*np.var(X))).astype(np.float64)
        elif affinity == 'n_neighbors':
            A = kneighbors_graph(X, 10, mode='distance',
                                 include_self=True).toarray()
        else:
            A = 1.0 - distance.cdist(X, X, metric=affinity).astype(np.float64)
        D = np.zeros(A.shape)
        np.fill_diagonal(D, np.sum(A, axis=1))
        W = np.linalg.inv(D) @ A
        v0 = np.sum(A, axis=1) / np.sum(A)

        # Run Power Iteration method
        embedding = self.PI(v0, W)
        self.embedding_ = embedding

        # Copute K-Means on the embedding computed from PI
        kmeans = KMeans(n_clusters=self.n_clusters).fit(
            embedding.reshape(-1, 1))
        self.labels_ = kmeans.labels_
        self.cluster_centroids_ = kmeans.cluster_centers_
        return self

    def PI(self, v, W):
        """ Power Iteration method

        v - Initial iteration vector
        W - Normalized affinity matrix
        """
        tol = 1e-5
        epsilon = 1e-5/len(W)
        delta = float('inf')
        d_prev = float('inf')
        v_prev = v.copy()
        for _ in range(self.max_iter):
            aux = W@v
            v = aux / np.linalg.norm(aux, ord=2)
            delta = np.linalg.norm(v-v_prev, ord=1)
            if abs(delta-d_prev) < epsilon:
                break
            v_prev = v.copy()
            d_prev = delta.copy()
        return v

    def plot_embedding(self, y_true, ax=None):
        assert self.embedding_ is not None
        v = self.embedding_
        if ax:
            ax.scatter(range(len(v)), v, c=y_true)
            ax.set_ylim(top=sorted(v)[-1], bottom=sorted(v)[0])
            plt.tight_layout()
        else:
            plt.scatter(range(len(v)), v, c=y_true)
            plt.ylim(top=sorted(v)[-1], bottom=sorted(v)[0])
            plt.tight_layout()

    def fit_predict(self, X, similarity='euclidean'):
        return self.fit(X, similarity).labels_
