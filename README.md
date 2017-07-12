ABDG
=====

A Python implementation of Attribute-based Decision Graph and a method for missing-data imputation using it


Reference
----------
[1] João Roberto Bertini Junior, Maria do Carmo Nicoletti, Liang Zhao,
    "An embedded imputation method via Attribute-based Decision Graphs",
    Expert Systems with Applications, Volume 57, 2016, Pages 159-177,
    doi: http://dx.doi.org/10.1016/j.eswa.2016.03.027.
    [http://www.sciencedirect.com/science/article/pii/S0957417416301208](http://www.sciencedirect.com/science/article/pii/S0957417416301208)

[2] João Roberto Bertini, Maria do Carmo Nicoletti, Liang Zhao,
    "Attribute-based Decision Graphs: A framework for multiclass data classification",
    Neural Networks, Volume 85, 2017, Pages 69-84,
    http://dx.doi.org/10.1016/j.neunet.2016.09.008.
    [http://www.sciencedirect.com/science/article/pii/S0893608016301381](http://www.sciencedirect.com/science/article/pii/S0893608016301381)

[3] Mikhail Belkin, Partha Niyogi, and Vikas Sindhwani,
    "Manifold Regularization: A Geometric Framework for Learning from Labeled and Unlabeled Examples",
    J. Mach. Learn. Res. 7 (December 2006), 2399-2434.
    [http://www.jmlr.org/papers/v7/belkin06a.html](http://www.jmlr.org/papers/v7/belkin06a.html)

Installation
-------------

Requirements:
 * [numpy](www.numpy.org)
 * [scipy](https://www.scipy.org/)
 * [networkx](https://networkx.github.io)
 * [pandas](http://pandas.pydata.org/)
 * [sklearn](scikit-learn.org)
 * [caimcaim](https://github.com/airysen/caimcaim)
 * [tqdm](https://pypi.python.org/pypi/tqdm)


 Example of usage
------------------

```python
>>> import numpy as np
>>> from abdg import ABDGImput
>>> from sklearn.datasets import make_classification
>>>
>>> X, y = make_classification(n_samples=2000, n_features=7, n_redundant=2,
>>>                            n_informative=4, n_classes=3)
>>>
>>> idx = np.random.choice(np.arange(X.shape[0]), X.shape[0]//2)
>>> X[idx, 0] = np.nan
>>>
>>> abdg = ABDGImput(categorical_features='auto', n_iter=4, alpha=0.6, L=0.5,
>>>                  sampling='normal', update_step=100, random_state=None)
>>> abdg.fit(X, y)
>>>
>>> X_imp, y_imp = abdg.predict(X, y)

```
