# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import numpy as np

from mlencoders.base_encoder import BaseEncoder
from mlencoders.base_encoder import NAN_CATEGORY


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
        :param int min_samples: minimum samples to take category average into account, must be >= 1
        :param int smoothing: coefficient used to balance categorical average (posterior) vs prior,
            the higher this number, the higher the prior is taken into account in the average

        :return: None
        """
        super(TargetEncoder, self).__init__(cols, handle_unseen, min_samples, None)
        self.smoothing = smoothing

    def fit(self, X, y):
        """Encode given columns of X according to y.

        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).

        :return: None
        """
        self._before_fit_check(X, y)
        self._imputed = y.mean()
        for col in self.cols:
            mapping = y.groupby(X[col].fillna(NAN_CATEGORY)).agg(['mean', 'count'])
            corr_count = mapping['count'] - self.min_samples
            coef = (corr_count > 0) / (1 + np.exp(-corr_count / self.smoothing))
            mapping['value'] = self._imputed * (1 - coef) + mapping['mean'] * coef
            self._mapping[col] = mapping
