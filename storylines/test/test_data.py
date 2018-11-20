from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import xarray as xr
import pytest

requires_test_data = pytest.mark.skipif(
    os.environ.get('STORYLINES_TEST_DATA', None) is None,
    reason='requires test data')


def get_test_data(kind=None,
                  data_dir=os.environ.get('STORYLINES_TEST_DATA', None)):

    if kind == 'domain':
        return xr.open_dataset(os.path.join(
            data_dir, 'domain.vic.test0.0125degnewman.20171027.nc'))

    if kind == 'obs':
        return xr.open_mfdataset(os.path.join(data_dir, 'conus_ens_*.nc'),
                                 concat_dim='ensemble')

    if kind == 'obs_stacked':
        return xr.open_mfdataset(os.path.join(data_dir, 'conus_ens_*.nc'),
                                 concat_dim='time').drop('time')

    if kind == 'scrf':
        return xr.open_dataset(os.path.join(
            data_dir, 'test_scrfs_newmann_150years.nc'))

    if kind == 'gard':
        return xr.open_mfdataset(os.path.join(data_dir, 'gard_output*.nc'))

    raise ValueError('Unrecognized kind: %s' % kind)
