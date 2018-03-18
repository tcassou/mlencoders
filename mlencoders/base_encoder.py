# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

NAN_CATEGORY = -99999


class BaseEncoder(object):

    def __init__(self, cols, handle_unseen, min_samples, imputed):
        self.cols = cols
        self.handle_unseen = handle_unseen
        self.min_samples = max(1, min_samples)
        # In case of unseen value or not enough data to learn the mapping, we use this value for imputation
        self._imputed = imputed
        # dict {str: pandas.DataFrame} column name --> mapping from category (index of df) to value (column of df)
        self._mapping = {}

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
            X_encoded[col] = mapping['value'].loc[X_encoded[col].fillna(NAN_CATEGORY)].values

            if self.handle_unseen == 'impute':
                X_encoded[col].fillna(self._imputed, inplace=True)
            elif self.handle_unseen == 'error':
                if np.unique(X_encoded[col]).shape[0] > mapping.shape[0]:
                    raise ValueError('Unseen categories found in `{}` column.'.format(col))

        return X_encoded

    def fit_transform(self, X, y=None):
        """Encode given columns of X according to y, and transform X based on the learnt mapping.

        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).
            Required only for encoders that need it: TargetEncoder, WeightOfEvidenceEncoder

        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
            replaced with encoded columns. DataFrame passed in argument is unchanged.
        :rtype: pandas.DataFrame
        """
        self.fit(X, y)
        return self.transform(X)

    def _input_check(self, name, value, options):
        if value not in options:
            raise ValueError('Wrong input: {} parameter must be in {}'.format(name, options))

    def _before_fit_check(self, X, y):
        # Checking columns to encode
        if self.cols is None:
            self.cols = X.columns
        else:
            assert all(c in X.columns for c in self.cols)
        # Checking input, depending on encoder type
        assert self.__class__.__name__ == 'LabelEncoder' or y is not None
        if y is not None:
            assert X.shape[0] == y.shape[0]

    def save_as_object_file(self, path):
        if not self._mapping:
            raise ValueError('`fit` method must be called before `save_as_object_file`.')
        pickle.dump(self.__dict__, open(path, 'wb'))

    def load_from_object_file(self, path):
        for k, v in pickle.load(open(path, 'rb')).items():
            setattr(self, k, v)
