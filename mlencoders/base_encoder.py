# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division


class BaseEncoder(object):

    def __init__(self):
        pass

    def fit_transform(self, X, y):
        """ncode given columns of X according to y, and transform X based on the learnt mapping.

        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).

        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
            replaced with encoded columns. DataFrame passed in argument is unchanged.
        :rtype: pandas.DataFrame
        """
        self.fit(X, y)
        return self.transform(X)
