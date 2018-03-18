# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from mlencoders.base_encoder import BaseEncoder
from mlencoders.base_encoder import NAN_CATEGORY


class WeightOfEvidenceEncoder(BaseEncoder):
    """
    Weight of Evidence Encoder for categorical features and binary outcome.

    For a categorical variable X, with possible values X_i, i=1..n, and target binary variable Y,
    it is defined as

        WOE_i = ln[ P(X=X_i | Y=1) / P(X=X_i | Y=0) ]

    """

    def __init__(self, cols=None, handle_unseen='impute', min_samples=1):
        """Instantiation

        :param [str] cols: list of columns to encode, or None (then all dataset columns will be encoded at fitting time)
        :param str handle_unseen:
            'impute' - default value, impute a -1 category
            'error'  - raise an error if a category unseen at fitting time is found
            'ignore' - skip unseen categories
        :param int min_samples: minimum samples to compute WOE of category, must be >= 1.

        :return: None
        """
        self._input_check('handle_unseen', handle_unseen, ['impute', 'error', 'ignore'])
        super(WeightOfEvidenceEncoder, self).__init__(cols, handle_unseen, min_samples, 0)

    def fit(self, X, y):
        """Encode given columns of X according to y.

        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).

        :return: None
        """
        self._before_fit_check(X, y)
        for col in self.cols:
            # Share of positive (resp. negative) labels for each category P(X=X_i | Y=1) (resp. P(X=X_i | Y=0))
            mapping = y.groupby(X[col].fillna(NAN_CATEGORY)).agg(['sum', 'count']).rename({'sum': 'pos'}, axis=1)
            mapping['neg'] = mapping['count'] - mapping['pos']
            mapping[['pos', 'neg']] /= mapping[['pos', 'neg']].sum()
            # For corner cases, defaulting to WOE = 0 (meaning no info). To avoid division by 0 we use default values.
            undef = (mapping['count'] < self.min_samples) | (mapping['pos'] == 0) | (mapping['neg'] == 0)
            mapping.loc[undef, ['pos', 'neg']] = -1
            # Final step, log of ratio of probabily estimates
            mapping['value'] = np.log(mapping['pos'] / mapping['neg'])
            self._mapping[col] = mapping
