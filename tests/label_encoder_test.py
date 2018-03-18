# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import unittest

import numpy as np
import pandas as pd
from genty import genty
from genty import genty_dataset
from nose.tools import eq_
from nose.tools import ok_
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from mlencoders.label_encoder import LabelEncoder


@genty
class LabelEncoderTest(unittest.TestCase):

    @genty_dataset(
        default_params=({}, None, 'ignore'),
        some_params=({'cols': ['a'], 'handle_unseen': 'error'}, ['a'], 'error'),
    )
    def test_init(self, kwargs, cols, handle_unseen):
        enc = LabelEncoder(**kwargs)
        eq_(enc.cols, cols)
        eq_(enc.handle_unseen, handle_unseen)
        eq_(enc._imputed, -99999)
        eq_(enc._mapping, {})

    @genty_dataset(
        impute=('impute',),
        typo=('ignores',),
    )
    def test_init_wrong_input(self, handle_unseen):
        assert_raises(ValueError, LabelEncoder, None, handle_unseen)

    def test_transform_before_fit(self):
        enc = LabelEncoder()
        assert_raises(ValueError, enc.transform, 1)

    @genty_dataset(
        categories_2=(['a', 'a', 'b', 'b'], [0, 0, 1, 1], ['a', 'b']),
        categories_3=(['a', 'a', 'b', 'c'], [0, 0, 1, 2], ['a', 'b', 'c']),
    )
    def test_encode_col(self, X, expected, columns):
        enc = LabelEncoder(cols=['cat'])
        result = enc.fit_transform(pd.DataFrame(X, columns=['cat']))
        assert_array_equal(result, pd.DataFrame(expected))
        eq_(enc._imputed, -99999)
        ok_('cat' in enc._mapping)
        ok_(isinstance(enc._mapping['cat'], pd.DataFrame))
        assert_array_equal(enc._mapping['cat'].index, columns)
        assert_array_equal(enc._mapping['cat'].columns, ['value'])

    @genty_dataset(
        some_input=(['a', 'a', np.nan, 'b'], [0, 0, 1, 2], ['a', -99999, 'b']),
    )
    def test_encode_nans(self, X, expected, columns):
        enc = LabelEncoder(cols=['cat'])
        result = enc.fit_transform(pd.DataFrame(X, columns=['cat']))
        assert_array_equal(result, pd.DataFrame(expected))
        ok_('cat' in enc._mapping)
        ok_(isinstance(enc._mapping['cat'], pd.DataFrame))
        eq_(enc._mapping['cat'].index[1], -99999)
        assert_array_equal(enc._mapping['cat'].index, pd.Series(columns))
        assert_array_equal(enc._mapping['cat'].columns, ['value'])

    @genty_dataset(
        ignore=(['a', 'a', 'b', 'b'], 'ignore', [np.nan, 0, 1, 1]),
    )
    def test_transform_unseen(self, X, handle_unseen, expected):
        enc = LabelEncoder(cols=['cat'], handle_unseen=handle_unseen)
        X = pd.DataFrame(X, columns=['cat'])
        enc.fit(X)
        X.iloc[0, 0] = 'foo'
        result = enc.transform(X)
        assert_array_equal(result, pd.DataFrame(expected))

    @genty_dataset(
        impute=(['a', 'a', 'b', 'b'], [0, 0, 1, 1]),
    )
    def test_transform_error(self, X, expected):
        enc = LabelEncoder(cols=['cat'], handle_unseen='error')
        X = pd.DataFrame(X, columns=['cat'])
        enc.fit(X)
        X.iloc[0, 0] = 'foo'
        assert_raises(ValueError, enc.transform, X)

    @genty_dataset(
        some_input=([['a', 'foo'], ['a', 'bar'], ['b', 'foo']], [[0, 0], [0, 1], [1, 0]]),
    )
    def test_encode_multiple_cols(self, X, expected):
        enc = LabelEncoder(cols=['cat1', 'cat2'])
        result = enc.fit_transform(pd.DataFrame(X, columns=['cat1', 'cat2']))
        assert_array_equal(result, pd.DataFrame(expected))
        ok_('cat1' in enc._mapping)
        ok_('cat2' in enc._mapping)
        ok_(isinstance(enc._mapping['cat1'], pd.DataFrame))
        ok_(isinstance(enc._mapping['cat2'], pd.DataFrame))
        assert_array_equal(enc._mapping['cat1'].index, ['a', 'b'])
        assert_array_equal(enc._mapping['cat2'].index, ['foo', 'bar'])
        assert_array_equal(enc._mapping['cat1'].columns, ['value'])
        assert_array_equal(enc._mapping['cat2'].columns, ['value'])

    @genty_dataset(
        some_input=([['a', 'foo'], ['a', 'bar'], ['b', 'foo']], [[0, 0], [0, 1], [1, 0]]),
    )
    def test_encode_all(self, X, expected):
        # all columns are encoded if no cols arg passed
        enc = LabelEncoder()
        result = enc.fit_transform(pd.DataFrame(X, columns=['cat1', 'cat2']))
        assert_array_equal(result, pd.DataFrame(expected))
        assert_array_equal(enc.cols, ['cat1', 'cat2'])
        ok_('cat1' in enc._mapping)
        ok_('cat2' in enc._mapping)
        ok_(isinstance(enc._mapping['cat1'], pd.DataFrame))
        ok_(isinstance(enc._mapping['cat2'], pd.DataFrame))
        assert_array_equal(enc._mapping['cat1'].index, ['a', 'b'])
        assert_array_equal(enc._mapping['cat2'].index, ['foo', 'bar'])
        assert_array_equal(enc._mapping['cat1'].columns, ['value'])
        assert_array_equal(enc._mapping['cat2'].columns, ['value'])
