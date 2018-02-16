# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import numpy as np

from mlencoders.base_encoder import BaseEncoder


class WeightOfEvidenceEncoder(BaseEncoder):
    """
    Weight of Evidence Encoder for categorical features and binary outcome.

    For a categorical variable X, with possible values X_i, i=1..n, and target binary variable Y,
    it is defined as

        WOE_i = ln[ P(X=X_i | Y=1) / P(X=X_i | Y=0)]

    """

    def __init__(self, cols=None, handle_unseen='impute', min_samples=1):
        """Instantiation

        :param [str] cols: list of columns to encode, or None (then all dataset columns will be encoded at fitting time)
        :param str handle_unseen:
            'impute' - default value, impute a -1 category
            'error'  - raise an error if a category unseen at fitting time is found
            'ignore' - skip unseen categories
        :param int min_samples: minimum samples to take category average into account, must be >= 1

        :return: None
        """
        super(WeightOfEvidenceEncoder, self).__init__()
        self.cols = cols
        self.handle_unseen = handle_unseen
        self.min_samples = max(1, min_samples)
        self._mapping = {}
        self._default = 0

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

        for col in self.cols:
            if self.handle_unseen == 'error':
                if np.isnan(X[col]).sum() > 0:
                    raise ValueError(
                        'NaN values found in `{}` column.'
                        ' Switch to handle_unseen=`impute` to encode them with default value, or'
                        ' handle_unseen=`error` to skip them.'.format(col)
                    )

            # Share of positive (resp. negative) labels for each category P(X=X_i | Y=1) (resp. P(X=X_i | Y=0))
            mapping = y.groupby(X[col]).agg(['sum', 'count']).rename({'sum': 'pos'}, axis=1)
            mapping['neg'] = mapping['count'] - mapping['pos']
            mapping[['pos', 'neg']] /= mapping[['pos', 'neg']].sum()
            # For corner cases, defaulting to WOE = 0 (meaning no info). To avoid division by 0 we use default values.
            undef = (mapping['count'] < self.min_samples) | (mapping['pos'] == 0) | (mapping['neg'] == 0)
            mapping.loc[undef, ['pos', 'neg']] = -1
            # Final step, log of ratio of probabily estimates
            mapping['woe'] = np.log(mapping['pos'] / mapping['neg'])
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
            X_encoded[col] = mapping['woe'].loc[X_encoded[col]].values

            if self.handle_unseen == 'impute':
                X_encoded[col].fillna(self._default, inplace=True)
            elif self.handle_unseen == 'error':
                if np.unique(X_encoded[col]).shape > mapping.shape[0]:
                    raise ValueError('Unseen categories found in `{}` column.'.format(col))

        return X_encoded
