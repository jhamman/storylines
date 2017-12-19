from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np

import pytest

from storylines.tools.dates import decode_freq, units_from_freq, date_range


def test_decode_freq():

    freq = '4D'
    step, units = decode_freq(freq)
    assert step == 4
    assert 'days since' in units

    freq = 'H'
    step, units = decode_freq(freq)
    assert step == 1
    assert 'hours since' in units


def test_units_from_freq():

    freq = '4D'
    units = units_from_freq(freq)
    assert 'days since' in units

    freq = 'foo'
    with pytest.raises(NotImplementedError):
        units = units_from_freq(freq)


@pytest.mark.parametrize('cal', ['standard', 'gregorian',
                                 'propoleptic_gregorian', 'noleap'])
def test_date_range_matches_pandas(cal):
    expected = pd.date_range(start='2016-01-01', end='2016-01-05')
    actual = date_range(start='2016-01-01', end='2016-01-05', calendar=cal)

    np.testing.assert_equal(actual.values, expected.values)


@pytest.mark.parametrize('cal', ['standard', 'gregorian',
                                 'propoleptic_gregorian', 'noleap'])
def test_date_range_matches_pandas_end_is_none(cal):
    expected = pd.date_range(start='2016-01-01', periods=30)
    actual = date_range(start='2016-01-01', periods=30, calendar=cal)

    np.testing.assert_equal(actual.values, expected.values)


def test_date_range_no_leap(cal='noleap'):
    pd_actual = date_range(start='2016-02-25', end='2016-03-05')
    assert '2016-02-29' in pd_actual

    actual = date_range(start='2016-02-25', end='2016-03-05', calendar=cal)
    assert '2016-02-29' not in actual

    assert len(pd_actual) == (len(actual) + 1)
