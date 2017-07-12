import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import itertools as it

from sklearn.base import BaseEstimator

from sklearn.metrics.pairwise import manhattan_distances

from scipy.special import xlogy
from scipy.stats import entropy

from tqdm import *

from .abdg import ABDGraph


class ABDGImput(ABDGraph):
    """
    Imputation via Attribute-based Decision Graph

    Parameters
    ----------
    discretization: 'caim' or 'mdlp'
        Method for discretization continuous variables

    categorical_features : 'auto' or 'all' or list/array of indices or list of labels
        Specify what features are treated as categorical (not using discretization).
        - 'auto' (default): Only those features whose number of unique values exceeds
                            the number of classes
                            of the target variable by 2 times or more
        - array of indices: array of categorical feature indices
        - list of labels: column labels of a pandas dataframe


    continous_distribution: 'normal' or 'laplace'
        The distribution using for sampling (reconstruct) continous variables
        after oversampling

    alpha: float
        A threshold for continuous data reconstuction that allows to change
        between considering all instances within the interval or just those
        belonging to the same class [1]

    L: float
        The factor that defines a width within the distribution using to reconstruct
        continuous data [1]

    verbose: int
        If greather than 0, enable verbose output

    random_state : int, or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    References
    ----------
    [1] João Roberto Bertini Junior, Maria do Carmo Nicoletti, Liang Zhao,
        "An embedded imputation method via Attribute-based Decision Graphs",
        Expert Systems with Applications, Volume 57, 2016, Pages 159-177,
        ISSN 0957-4174, http://dx.doi.org/10.1016/j.eswa.2016.03.027.
        http://www.sciencedirect.com/science/article/pii/S0957417416301208

    """

    def __init__(self, discretization='caim', categorical_features='auto', n_iter=4,
                 alpha=0.6, L=0.5, sampling='normal',
                 update_step=10, random_state=None):
        super().__init__(discretization=discretization,
                         categorical_features=categorical_features)
        self.n_iter = n_iter
        self.alpha = alpha
        self.L = L
        self.random_state = random_state
        self.update_step = update_step
        if sampling == 'normal':
            self.f_sample = np.random.normal
        elif sampling == 'laplace':
            self.f_sample = np.random.laplace
        else:
            raise SamplingParameterException('Invalid sampling parameter! Only "laplace" or "normal" are allowed.')

    def fit(self, X, y):
        super().fit(X, y)
        return self

    def predict(self, X, y, refit=False, n_iter=None, alpha=None, L=None, sampling=None):
        if isinstance(X, pd.DataFrame):
            self.pdflag = True
            self.pd_index = X.index.values
            self.pd_columns = X.columns
            self.pd_yname = y.name
            X = X.values
            y = y.values
        if alpha is None:
            alpha = self.alpha
        if L is None:
            L = self.L
        if n_iter is None:
            n_iter = self.L

        X_imput = X.copy()
        init_idx = 0
        if not np.allclose(self.X, X, equal_nan=True):
            init_idx, _ = self.update_x_y(X, y, refit=refit)

        self.nan_matrix = np.zeros(self.X.shape, dtype='bool')
        for i in range(self.nan_matrix.shape[0]):
            for j in range(self.nan_matrix.shape[1]):
                if np.isnan(self.X[i, j]):
                    self.nan_matrix[i, j] = True
        X_di = self.X_di
        for n in range(self.n_iter):
            print('ITERATION # ', n)
            update_count = 0
            for i in tqdm(range(init_idx, X_di.shape[0])):
                ncolumns = np.random.permutation(X_di.shape[1])  # To enchance imputation variance
                for col in ncolumns.tolist():
                    if self.nan_matrix[i, col] == True:
                        idx_y = self.target_dict[self.y[i]]
                        xi = X_di[i]
                        sgraph = self.make_subgraph(xi, col)
                        n_interval = self.estimate_interval(col, sgraph, idx_y)
                        imput_val = self.value_by_node((col, n_interval))
                        self.X_di[i, col] = imput_val
                        if update_count > self.update_step:
                            self.update_weights_all()
                            update_count = 0
                update_count = update_count + 1
        # categorical = self.disc.categorical
        for i in tqdm(range(init_idx, X_di.shape[0])):
            for col in range(X_di.shape[1]):
                if self.nan_matrix[i, col] == True:
                    yyi = self.y[i]
                    val = self.X_di[i, col]
                    node = self.find_node_by_value(col, val)
                    xj, yi = self.real_interval(node)

                    notnan = np.invert(np.isnan(xj))
                    mu0 = xj[notnan].mean()
                    std0 = xj[notnan].std()

                    imput = None
                    c_idx = np.where(yi == yyi)[0]
                    xjc = xj[c_idx]
                    notnan2 = np.invert(np.isnan(xjc))
                    mu1 = xjc[notnan2].mean()
                    std1 = xjc[notnan2].std()

                    if mu1 / mu0 > alpha:
                        if self.random_state is not None:
                            np.random.seed(self.random_state + i + col)
                        imput = self.f_sample(mu1, L * std1)

                    else:
                        if self.random_state is not None:
                            np.random.seed(self.random_state + i + col)
                        imput = self.f_sample(mu0, L * std0)

                    X_imput[i - init_idx, col] = imput
        if self.pdflag:
            X_imput = pd.DataFrame(X_imput, columns=self.pd_columns, index=self.pd_index)

        return X_imput, y

    def make_subgraph(self, xi, j):
        notnan = np.where(np.isnan(xi) == False)[0]
        subgraph = nx.Graph()
        for i in notnan.tolist():
            if i == j:
                continue  # or break
            nod = self.find_node_by_value(i, xi[i])
            gamma = self.G.node[nod]['weight']
            deltas = {}
            edges = self.G.edge[nod]
            for k in edges:
                if k[0] == j:
                    deltas[k[1]] = edges[k]['weight']

            subgraph.add_node(nod, gamma=gamma, deltas=deltas)

        return subgraph

    def estimate_interval(self, col, subgraph, nclass, beta=None):
        b = self.attribute_weights
        G = self.G
        s = 0
        inter_count = 0
        best_interval = 0
        for k in self.xval[col]:
            nod = self.find_node_by_value(col, k)
            gamma_k = G.node[nod]['weight'][nclass]
            psum = 0
            for n in subgraph.nodes():
                i = n[0]
                psum = psum + b[i] * subgraph.node[n]['gamma'][nclass] * \
                    subgraph.node[n]['deltas'][inter_count][nclass]
            temp = gamma_k * psum
            if temp > s:
                best_interval = inter_count
                s = temp
            inter_count = inter_count + 1
        return best_interval

    def value_from_interval(self):
        pass

    # def information_gain(self, F):
    #     P = F / F.sum(axis=0, keepdims=True)
    #     k = -(1 / np.log(n))
    #     H = k * np.sum(P * np.log(P), axis=0)  # entropy


class SamplingParameterException(Exception):
    # Raise if the beta level not found
    pass
