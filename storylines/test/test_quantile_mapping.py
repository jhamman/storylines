import numpy as np
import dask
import dask.array as da
import xarray as xr

import pytest

from storylines.tools.quantile_mapping import (calc_endpoints,
                                               plotting_positions,
                                               quantile_mapping,
                                               _qmap_wrapper)


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


def test_quantile_mapping_numpy():
    # input datasets -- dims (time, y, x)
    input_data = xr.DataArray(np.random.random((400, 2, 3)),
                              dims=('time', 'y', 'x'))
    data_to_match = xr.DataArray(np.random.random((200, 2, 3)),
                                 dims=('time', 'y', 'x'))

    new = quantile_mapping(input_data, data_to_match)

    assert new.shape == input_data.shape
    assert new.dims == input_data.dims

    xr.testing.assert_allclose(new.median(dim='time'),
                               data_to_match.median(dim='time'))


def test_quantile_mapping_dask():

    # input datasets -- dims (time, y, x)
    input_data = xr.DataArray(da.random.random((400, 16, 21),
                                               chunks=(400, 4, 7)),
                              dims=('time', 'y', 'x'), name='input_data')
    data_to_match = xr.DataArray(da.random.random((200, 16, 21),
                                                  chunks=(200, 4, 7)),
                                 dims=('time', 'y', 'x'), name='data_to_match')

    new = quantile_mapping(input_data, data_to_match).load()

    assert new.shape == input_data.shape
    assert new.dims == input_data.dims

    xr.testing.assert_allclose(new.median(dim='time'),
                               data_to_match.load().median(dim='time'))

    # input datasets -- dims (time, y, x)
    input_data = xr.DataArray(da.random.random((400, 16, 21),
                                               chunks=(400, 16, 21)),
                              dims=('time', 'y', 'x'), name='input_data')
    data_to_match = xr.DataArray(da.random.random((200, 16, 21),
                                                  chunks=(200, 16, 21)),
                                 dims=('time', 'y', 'x'), name='data_to_match')

    new = quantile_mapping(input_data, data_to_match).load()

    assert new.shape == input_data.shape
    assert new.dims == input_data.dims

    xr.testing.assert_allclose(new.median(dim='time'),
                               data_to_match.load().median(dim='time'))


def test_qmap_wrapper():

    # input arrays
    a = da.random.random((400, 16, 21), chunks=(400, 4, 7))
    b = da.random.random((200, 16, 21), chunks=(200, 4, 7))
    c = da.random.random((16, 21), chunks=(4, 7)) > 0.5  # boolean mask

    da.map_blocks(_qmap_wrapper, a, b, c, dtype=a.dtype, chunks=a.chunks,
                  extrapolate='min').compute()