# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pandas as pd

from mlencoders.base_encoder import BaseEncoder
from mlencoders.base_encoder import NAN_CATEGORY


class LabelEncoder(BaseEncoder):
    """
    Target Encoder for categorical features.
    """

    def __init__(self, cols=None, handle_unseen='ignore'):
        """Instantiation

        :param [str] cols: list of columns to encode, or None (then all dataset columns will be encoded at fitting time)
        :param str handle_unseen:
            'error'  - raise an error if a category unseen at fitting time is found
            'ignore' - skip unseen categories
        :param int min_samples: minimum samples to take category average into account, must be >= 1
        :param int smoothing: coefficient used to balance categorical average (posterior) vs prior,
            the higher this number, the higher the prior is taken into account in the average

        :return: None
        """
        self._input_check('handle_unseen', handle_unseen, ['error', 'ignore'])
        super(LabelEncoder, self).__init__(cols, handle_unseen, 1, NAN_CATEGORY)

    def fit(self, X, y=None):
        """Label Encode given columns of X.

        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.

        :return: None
        """
        self._before_fit_check(X, y)
        for col in self.cols:
            self._mapping[col] = pd.Series(pd.unique(X[col].fillna(NAN_CATEGORY)), name=col)\
                .reset_index()\
                .rename(columns={'index': 'value'})\
                .set_index(col)
