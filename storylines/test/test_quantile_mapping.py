from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr

import pytest

from storylines.tools.quantile_mapping import (calc_endpoints,
                                               plotting_positions,
                                               quantile_mapping,
                                               make_x_and_y, remove_trend)

from storylines.tools.post_process_gard_output import make_gard_like_obs

from .test_data import get_test_data, requires_test_data


def test_calc_endpoints():

    x = np.array([0, 1, 2, 3, 4, 5])

    # Test None
    y = np.array([np.nan, 1, 2, 3, 4, np.nan])
    expected = np.array([np.nan, 1, 2, 3, 4, np.nan])
    y = calc_endpoints(x, y, None, 2)
    np.testing.assert_array_equal(y, expected)

    # Test minimum
    y = np.array([np.nan, 1, 2, 3, 4, np.nan])
    expected = np.array([0, 1, 2, 3, 4, np.nan])
    y = calc_endpoints(x, y, 'min', 2)
    np.testing.assert_array_equal(y, expected)

    # Test maximum
    y = np.array([np.nan, 1, 2, 3, 4, np.nan])
    expected = np.array([np.nan, 1, 2, 3, 4, 5])
    y = calc_endpoints(x, y, 'max', 2)
    np.testing.assert_array_equal(y, expected)

    # raise error for invalid n_endpoints
    with pytest.raises(ValueError):
        y = calc_endpoints(x, y, 'max', 1)


def test_plotting_positions():
    pp = plotting_positions(1000, 0.4, 0.4)
    assert pp.max() < 1.0  # all are less than 1
    assert pp.min() > 0.0  # all are greater than zero
    assert np.all(np.diff(pp) > 0)  # monotonically increasing


@pytest.mark.parametrize('ex', [None, 'min', 'max', 'both'])
@pytest.mark.parametrize('detrend', [False, True])
def test_quantile_mapping_numpy(ex, detrend):
    # input datasets -- dims (time, y, x)
    input_data = xr.DataArray(
        np.random.random((400, 2, 3)),
        dims=('time', 'y', 'x'),
        coords={'time': pd.date_range('2010-01-01', periods=400)})
    ref_data = xr.DataArray(
        np.random.random((300, 2, 3)),
        dims=('time', 'y', 'x'),
        coords={'time': pd.date_range('2010-01-01', periods=300)})
    data_to_match = xr.DataArray(
        np.random.random((200, 2, 3)),
        dims=('time', 'y', 'x'),
        coords={'time': pd.date_range('2010-01-01', periods=200)})

    new = quantile_mapping(input_data, ref_data, data_to_match,
                           extrapolate=ex, detrend=detrend)

    assert new.shape == input_data.shape
    assert new.dims == input_data.dims


@pytest.mark.parametrize('ex', [None, 'min', 'max', 'both'])
@pytest.mark.parametrize('detrend', [False, True])
def test_quantile_mapping_dask(ex, detrend):
    # input datasets -- dims (time, y, x)
    input_data = xr.DataArray(
        da.random.random((400, 16, 21), chunks=(400, 4, 7)),
        dims=('time', 'y', 'x'), name='input_data',
        coords={'time': pd.date_range('2010-01-01', periods=400)})
    ref_data = xr.DataArray(
        da.random.random((300, 16, 21), chunks=(300, 4, 7)),
        dims=('time', 'y', 'x'), name='input_data',
        coords={'time': pd.date_range('2010-01-01', periods=300)})
    data_to_match = xr.DataArray(
        da.random.random((200, 16, 21), chunks=(200, 4, 7)),
        dims=('time', 'y', 'x'), name='data_to_match',
        coords={'time': pd.date_range('2010-01-01', periods=200)})

    new = quantile_mapping(input_data, ref_data, data_to_match,
                           detrend=detrend, extrapolate=ex).load()

    assert new.shape == input_data.shape
    assert new.dims == input_data.dims

    # input datasets -- dims (time, y, x)
    input_data = xr.DataArray(
        da.random.random((400, 16, 21), chunks=(400, 16, 21)),
        dims=('time', 'y', 'x'), name='input_data',
        coords={'time': pd.date_range('2010-01-01', periods=400)})
    ref_data = xr.DataArray(
        da.random.random((300, 16, 21), chunks=(300, 16, 21)),
        dims=('time', 'y', 'x'), name='input_data',
        coords={'time': pd.date_range('2010-01-01', periods=300)})
    data_to_match = xr.DataArray(
        da.random.random((200, 16, 21), chunks=(200, 16, 21)),
        dims=('time', 'y', 'x'), name='data_to_match',
        coords={'time': pd.date_range('2010-01-01', periods=200)})

    new = quantile_mapping(input_data, ref_data, data_to_match,
                           detrend=detrend, extrapolate=ex).load()

    assert new.shape == input_data.shape
    assert new.dims == input_data.dims


def test_remove_trend():
    input_data = xr.DataArray(
        da.random.random((400, 16, 21), chunks=(400, 4, 3)),
        dims=('time', 'y', 'x'), name='input_data',
        coords={'time': pd.date_range('2016-01-01', periods=400)})

    detrended, trend_ts = remove_trend(input_data)

    assert detrended.shape == input_data.shape
    assert detrended.shape == trend_ts.shape


def test_make_x_and_y():
    n = 30
    y = np.arange(n)

    # More obs than input data, all optiopns should be equivalent to None
    for ex in [None, 'min', 'max', 'both']:
        y = np.arange(n)
        x, y, s = make_x_and_y(y, 0.4, 0.4, ex)
        assert len(x) == len(y) == n + 2
        assert np.all(np.diff(x) >= 0)  # monotonically increasing

    # More input data than obs
    for ex, add in [(None, 2), ('min', 2), ('max', 2), ('both', 2)]:
        y = np.arange(n)
        x, y, s = make_x_and_y(y, 0.4, 0.4, ex)
        assert len(x) == n + add
        assert np.all(np.diff(x) >= 0)  # monotonically increasing


@requires_test_data
def test_with_real_data():
    gard_ds = get_test_data('gard')
    domain = get_test_data('domain')
    obs = get_test_data('obs_stacked')
    print('obs', obs)

    gard_ds['time'] = pd.date_range('1950-01-01', freq='D',
                                    periods=len(gard_ds['time']))
    gard_ds = make_gard_like_obs(gard_ds, domain)

    quantile_mapping(gard_ds['pcp'], None, obs['pcp'], use_ref_data=False)
