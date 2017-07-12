"""
ABDG
=====

Attribute-based Decision Graph

.. note::
    João Roberto Bertini Junior, Maria do Carmo Nicoletti, Liang Zhao,
    "An embedded imputation method via Attribute-based Decision Graphs",
    Expert Systems with Applications, Volume 57, 2016, Pages 159-177,
    doi: http://dx.doi.org/10.1016/j.eswa.2016.03.027.
    .. _a link: http://www.sciencedirect.com/science/article/pii/S0957417416301208

.. module:: abdg
   :platform: Windows, Unix
   :synopsis: data imputation

"""

import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
import itertools as it

from sklearn.base import BaseEstimator

from sklearn.metrics.pairwise import manhattan_distances

from scipy.special import xlogy
from scipy.stats import entropy

from caimcaim import CAIMD
# from mdlp import MDLP

from tqdm import *


class ABDGraph(BaseEstimator):

    """
    Attribute-based Decision Graph class

    Parameters
    ----------

    discretization: 'caim'
        Method for discretization continuous variables

    categorical_features : 'auto' or 'all' or list/array of indices or list of labels
        Specify what features are treated as categorical (not using discretization).
        - 'auto' (default): Only those features whose number of unique values exceeds
                            the number of classes
                            of the target variable by 2 times or more
        - array of indices: array of categorical feature indices
        - list of labels: column labels of a pandas dataframe

    attribute_weights: 'entropy_enchanced'[2] or 'information_gain'[1]
        Method which will be used for calculating weights of attributes


    References
    ----------
    [1] João Roberto Bertini Junior, Maria do Carmo Nicoletti, Liang Zhao,
        "An embedded imputation method via Attribute-based Decision Graphs",
        Expert Systems with Applications, Volume 57, 2016, Pages 159-177,
        ISSN 0957-4174, http://dx.doi.org/10.1016/j.eswa.2016.03.027.
        http://www.sciencedirect.com/science/article/pii/S0957417416301208

    [2] S. Ouyang, Z. W. Liu, Q. Li, Y. L. Shi,
        "A New Improved Entropy Method and its Application in Power Quality Evaluation",
         Advanced Materials Research, Vols. 706-708, pp. 1726-1733, 2013
         http://dx.doi.org/10.4028/www.scientific.net/AMR.706-708.1726
         https://www.scientific.net/AMR.706-708.1726

    Example
    ---------

    """

    def __init__(self, discretization='caim', categorical_features='auto',
                 attribute_weights='entropy_enchanced'):
        self.type_disc = discretization
        self.categorical = categorical_features
        self.i_categorical = []

        if attribute_weights == 'entropy_enchanced':
            self.f_weights = self.entropy_enchanced
        elif attribute_weights == 'information_gain':
            self.f_weights = self.information_gain
        else:
            raise IvalidWeightsFunction("'attribute_weights' parameter must be 'entropy_enchanced' \
                                        or'information_gain'!")

    def fit(self, X, y):
        self.i_categorical = []
        self.pdflag = False
        if isinstance(X, pd.DataFrame):
            if isinstance(self.categorical, list):
                self.i_categorical = [X.columns.get_loc(label) for label in self.categorical]
            X = X.values
            y = y.values
            self.pdflag = True

        if self.categorical != 'all':
            if self.categorical == 'auto':
                self.i_categorical = self.check_categorical(X, y)
            elif (isinstance(self.categorical, list)) or (isinstance(self.categorical, np.ndarray)):
                if not self.pdflag:
                    self.i_categorical = self.categorical[:]
        else:
            self.i_categorical = np.arange(X.shape[1]).tolist()

        if len(self.i_categorical) < X.shape[1]:
            if self.type_disc == 'caim':
                self.disc = CAIMD(categorical_features=self.i_categorical)

        return self._fit(X, y)

    def _fit(self, X, y):
        self.G = nx.Graph()

        self.X = X
        self.y = y

        X_di = X.copy()
        if len(self.i_categorical) < X.shape[1]:
            X_di = self.disc.fit_transform(X, y)

        self.attribute_weights = self.f_weights(X_di + 1)
        self.X_di = X_di

        prior_target = self.priors_label(y)
        alltarget = np.unique(y)
        self.target_dict = dict(zip(alltarget, np.arange(alltarget.shape[0])))
        self.xval = np.zeros(X.shape[1], dtype='object')
        self.inv_xval = np.zeros(X.shape[1], dtype='object')
        for j in range(X_di.shape[1]):
            xi = X_di[:, j]
            xvalues = np.unique(xi[np.invert(np.isnan(xi))])
            self.xval[j] = dict(zip(xvalues, np.arange(xvalues.shape[0])))
            self.inv_xval[j] = dict(zip(np.arange(xvalues.shape[0]), xvalues))
            for i in range(xvalues.shape[0]):
                xval = xvalues[i]
                gamma = np.zeros(alltarget.shape)
                sump = 0
                for k in range(alltarget.shape[0]):
                    yval = alltarget[k]
                    Pwj = prior_target[yval]
                    Pinter = self.prob_interval_node(xi, xval, y, yval)
                    PP = Pinter * Pwj
                    gamma[k] = PP
                    sump = sump + PP
                gamma = gamma / sump
                self.G.add_node((j, i), weight=gamma)
        for a in range(X_di.shape[1]):
            for b in range(X_di.shape[1]):
                if a == b:
                    continue
                xa = X_di[:, a]
                xb = X_di[:, b]
                aval = list(self.xval[a].keys())
                bval = list(self.xval[b].keys())
                for k in range(len(aval)):
                    for q in range(len(bval)):
                        delta = np.zeros(alltarget.shape[0])
                        sump = 0
                        for z in range(alltarget.shape[0]):
                            yval = alltarget[z]
                            Pwj = prior_target[yval]
                            Pab = self.prob_interval_edge(xa, aval[k], xb, bval[q], y, yval)
                            PP = Pwj * Pab
                            delta[z] = PP
                            sump = sump + PP
                        if sump == 0:
                            sump = 1
                        delta = delta / sump
                        self.G.add_edge((a, k), (b, q), weight=delta)

    def init_graph(self):
        pass

    def update_x_y(self, X, y, refit=False):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        prev_shape = self.X.shape[0]
        self.X = np.vstack((self.X, X))
        self.y = np.hstack((self.y, y))
        Xdi = self.disc.transform(X)  # FIXME
        self.X_di = np.vstack((self.X_di, Xdi))
        new_shape = self.X.shape[0]
        if refit:
            self.fit(self.X, self.y)
        return prev_shape, new_shape

    def update_weights_all(self):
        X_di = self.X_di

        y = self.y
        yvalues = list(self.target_dict.keys())
        prior_target = self.priors_label(y)
        for i in range(len(self.xval)):
            xi = X_di[:, i]
            for k in self.xval[i]:
                gamma = np.zeros(len(yvalues))
                sump = 0
                nod = self.find_node_by_value(i, k)
                for z in range(len(yvalues)):
                    yval = yvalues[z]
                    Pwj = prior_target[yval]
                    Pinter = self.prob_interval_node(xi, k, y, yval)
                    PP = Pinter * Pwj
                    gamma[z] = PP
                    sump = sump + PP
                gamma = gamma / sump
                self.G.node[nod]['weight'] = gamma
        self.attribute_weights = self.entropy_enchanced(X_di + 1, 'all')

        for a in range(X_di.shape[1]):
            for b in range(self.X_di.shape[1]):
                if a == b:
                    continue
                xa = X_di[:, a]
                xb = X_di[:, b]
                aval = list(self.xval[a].keys())
                bval = list(self.xval[b].keys())
                for k in range(len(aval)):
                    for q in range(len(bval)):
                        delta = np.zeros(len(yvalues))
                        sump = 0
                        for z in range(len(yvalues)):
                            yval = yvalues[z]
                            Pwj = prior_target[yval]
                            Pab = self.prob_interval_edge(xa, aval[k], xb, bval[q], y, yval)
                            PP = Pwj * Pab
                            delta[z] = PP
                            sump = sump + PP
                        if sump == 0:
                            sump = 1
                        delta = delta / sump
                        self.G.edge[(a, k)][(b, q)]['weight'] = delta

    def update_weights(self, node):
        X_di = self.X_di
        col, interval = node
        xi = self.X_di[:, col]
        y = self.y
        yvalues = list(self.target_dict.keys())
        prior_target = self.priors_label(y)
        for k in self.xval[col]:
            gamma = np.zeros(len(yvalues))
            sump = 0
            for z in range(len(yvalues)):
                yval = yvalues[z]
                Pwj = prior_target[yval]
                Pinter = self.prob_interval_node(xi, k, y, yval)
                PP = Pinter * Pwj
                gamma[z] = PP
                sump = sump + PP
            gamma = gamma / sump
            self.G.node[node]['weight'] = gamma

        for b in range(self.X_di.shape[1]):
            if b == col:
                continue
            xa = xi
            xb = X_di[:, b]
            aval = list(self.xval[col].keys())
            bval = list(self.xval[b].keys())
            for k in range(len(aval)):
                for q in range(len(bval)):
                    delta = np.zeros(len(yvalues))
                    sump = 0
                    for z in range(len(yvalues)):
                        yval = yvalues[z]
                        Pwj = prior_target[yval]
                        Pab = self.prob_interval_edge(xa, aval[k], xb, bval[q], y, yval)
                        PP = Pwj * Pab
                        delta[z] = PP
                        sump = sump + PP
                    if sump == 0:
                        sump = 1
                    delta = delta / sump
                    self.G.edge[(col, k)][(b, q)]['weight'] = delta

    def update_delta(self):
        pass

    def complete_structure(self):
        pass

    def priors_label(self, y):
        keys, counts = np.unique(y, return_counts=True)
        P = dict(zip(keys, counts / y.shape[0]))
        return P

    def prob_interval_node(self, xi, xval, y, yval):
        indx_val = np.where(xi == xval)[0]
        nom = np.where(y[indx_val] == yval)[0]
        if nom.size:
            nom = nom.shape[0]
            den = np.where(y == yval)[0].shape[0]
            return nom / den
        return 0

    def prob_interval_edge(self, xa, aval, xb, bval, y, yval):
        index_aval = np.where(xa == aval)[0].tolist()
        index_bval = np.where(xb == bval)[0].tolist()
        intersc = list(set(index_aval).intersection(index_bval))
        if intersc:
            nom = np.where(y[intersc] == yval)[0]
            if nom.size:
                nom = nom.shape[0]
                den = np.where(y == yval)[0].shape[0]
                return nom / den
        return 0

    def find_node_by_value(self, j, val):
        return (j, self.xval[j][val])

    def value_by_node(self, node):
        a, k = node
        return self.inv_xval[a][k]

    def real_interval(self, node):
        val = self.value_by_node(node)
        col = node[0]
        idx = np.where(self.X_di[:, col] == val)[0]
        xj = self.X[:, col]
        return xj[idx], self.y[idx]

    # def entropy_enchanced2(self, X):
    #     m = X.shape[1]
    #     w = np.zeros(m)
    #     H = np.zeros(m)
    #     for j in range(X.shape[1]):
    #         xj = X[:, j]
    #         xj = xj[~np.isnan(xj)]
    #         n = xj.shape[0]
    #         P = xj / np.sum(xj)
    #         k = -(1 / np.log(n))
    #         H[j] = k * np.sum(P * np.log(P))  # entropy
    #     H1 = H[H < 1]
    #     H1_mean = H1.mean()
    #     w = np.zeros(np.shape(H))
    #     for j in range(H.shape[0]):
    #         h = H[j]
    #         if h < 1:
    #             w21 = (1 - h) / np.sum(1 - H1)
    #             w22 = (1 + H1_mean - h) / np.sum(1 + H1_mean - H1)
    #             w[j] = (1 - H1_mean) * w21 + H1_mean * w22
    #     return w

    def entropy_enchanced(self, X, categorical='all'):
        m = X.shape[1]
        w = np.zeros(m)
        H = np.zeros(m)
        if categorical == 'all':
            categorical = np.arange(X.shape[1]).tolist()
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[~np.isnan(xj)]
            n = xj.shape[0]
            P = None
            if j in categorical:
                keys, counts = np.unique(xj, return_counts=True)
                P = counts / n
            else:
                P = xj / np.sum(xj)
            k = -(1 / np.log(n))
            H[j] = k * np.sum(sp.special.xlogy(P, P))  # entropy
        H1 = H[H < 1]
        H1_mean = H1.mean()
        w = np.zeros(np.shape(H))
        for j in range(H.shape[0]):
            h = H[j]
            if h < 1:
                w21 = (1 - h) / np.sum(1 - H1)
                w22 = (1 + H1_mean - h) / np.sum(1 + H1_mean - H1)
                w[j] = (1 - H1_mean) * w21 + H1_mean * w22
        return w

    def information_gain(self, X):
        w = np.zeros(X.shape[1])
        P = X[~np.isnan(X)] / X[~np.isnan(X)].sum()
        E = -np.sum(sp.special.xlogy(P, P))
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[~np.isnan(xj)]
            n = xj.shape[0]
            uniq, counts = np.unique(xj, return_counts=True)
            es = 0
            for u in uniq:
                ii = np.where(xj == u)[0]
                p = xj[ii] / xj[ii].sum()
                es = es + (ii.shape[0] / n) * (-np.sum(sp.special.xlogy(p, p)))
            w[j] = np.sum(E - es)
        return w / w.sum()

    def check_categorical(self, X, y):
        categorical = []
        ny2 = 3 * np.unique(y).shape[0]
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            if np.unique(xj).shape[0] < ny2:
                categorical.append(j)
        return categorical


class IvalidWeightsFunction(Exception):
    # Raise if
    pass
