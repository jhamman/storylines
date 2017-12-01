from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd

import pytest

from storylines.tools.qc import check_times


def test_check_times_backwards():

    times = pd.date_range(start='2016-02-20', end='2016-03-05').append(
        pd.date_range(start='2016-02-15', end='2016-02-19'))

    with pytest.warns(None) as record:

        check_times(times)
    assert len(record) == 1
    assert 'not monotonically increasing' in str(record[0].message)


def test_check_times_big_jump():

    times = pd.date_range(start='2016-02-20', end='2016-03-05').append(
        pd.date_range(start='2016-03-15', end='2016-03-19'))

    with pytest.warns(None) as record:

        check_times(times)
    assert len(record) == 1
    assert 'delta is too large' in str(record[0].message)
