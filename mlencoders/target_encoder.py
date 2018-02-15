# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division

from mlencoders.base_encoder import BaseEncoder

import numpy as np
import pandas as pd


class TargetEncoder(BaseEncoder):
    """
    Target Encoder for categorical features.
    """

    def __init__(self, cols=None, handle_unseen='impute', min_samples=1, smoothing=1):
        """Instantiation

        :param [str] cols: list of columns to encode, or None (then all dataset columns will be encoded at fitting time)
        :param str handle_unseen:
            'impute' - default value, impute a -1 category
            'error'  - raise an error if a category unseen at fitting time is found
            'ignore' - skip unseen categories
        :param int min_samples: minimum samples to take category average into account
        :param int smoothing: coefficient used to balance categorical average (posterior) vs prior,
            the higher this number, the higher the prior is taken into account in the average

        :return: None
        """
        super(TargetEncoder, self).__init__()
        self.cols = cols
        self.handle_unseen = handle_unseen
        self.min_samples = min_samples
        self.smoothing = smoothing
        self._mapping = {}
        self._mean = None

    def fit(self, X, y):
        """Encode given columns of X according to y.

        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).

        :return: None
        """
        if self.cols is None:
            self.cols = X.columns
        else:
            assert all(c in X.columns for c in self.cols)
        assert X.shape[0] == y.shape[0]

        self._mean = y.mean()
        for col in self.cols:
            if self.handle_unseen == 'error':
                if np.isnan(X[col]).sum() > 0:
                    raise ValueError(
                        'NaN values found in `{}` column.'
                        ' Switch to handle_unseen=`impute` to encode them with the target mean, or'
                        ' handle_unseen=`error` to skip them.'.format(col)
                    )

            mapping = y.groupby(X[col]).agg(['mean', 'count'])
            corr_count = mapping['count'] - self.min_samples
            coef = (corr_count > 0) / (1 + np.exp(-corr_count / self.smoothing))
            mapping['smooth'] = self._mean * (1 - coef) + mapping['mean'] * coef
            self._mapping[col] = mapping

    def transform(self, X):
        """Transform categorical data based on mapping learnt at fitting time.

        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.

        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
            replaced with encoded columns. DataFrame passed in argument is unchanged.
        :rtype: pandas.DataFrame
        """
        if not self._mapping:
            raise ValueError('`fit` method must be called before `transform`.')
        assert all(c in X.columns for c in self.cols)

        X_encoded = X.copy(deep=True)
        for col, mapping in self._mapping.items():
            X_encoded[col] = mapping['smooth'].loc[X_encoded[col]].values

            if self.handle_unseen == 'impute':
                X_encoded[col].fillna(self._mean, inplace=True)
            elif self.handle_unseen == 'error':
                if np.unique(X_encoded[col]).shape > mapping.shape[0]:
                    raise ValueError('Unseen categories found in `{}` column.'.format(col))

        return X_encoded
