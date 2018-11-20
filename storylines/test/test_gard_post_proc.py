from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dask.array as da
import pandas as pd
import xarray as xr
from xarray.core.pycompat import dask_array_type

from storylines.tools.post_process_gard_output import (add_random_effect,
                                                       make_gard_like_obs)

from .test_data import get_test_data, requires_test_data


def test_add_random_effect_without_logistic():
    ds = xr.Dataset()
    ds['t_mean'] = xr.DataArray(
        da.random.random((400, 16, 21), chunks=(400, 16, 21)) + 20,
        dims=('time', 'y', 'x'), name='t_mean',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    ds['t_mean_error'] = xr.DataArray(
        da.random.normal(size=(400, 16, 21), chunks=(400, 16, 21)),
        dims=('time', 'y', 'x'), name='t_mean',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    da_normal = xr.DataArray(
        da.random.normal(size=(400, 16, 21), chunks=(400, 16, 21)),
        dims=('time', 'y', 'x'), name='t_mean',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    actual = add_random_effect(ds, da_normal, var='t_mean', root=1.)
    expected = add_random_effect(ds.load(), da_normal.load(), var='t_mean',
                                 root=1.)
    assert isinstance(actual.data, dask_array_type)
    xr.testing.assert_equal(actual.compute(), expected)


def test_add_random_effect_with_logistic():
    ds = xr.Dataset()
    ds['pcp'] = xr.DataArray(
        da.random.random((400, 16, 21), chunks=(400, 16, 21)) + 20,
        dims=('time', 'y', 'x'), name='pcp',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    ds['pcp_error'] = xr.DataArray(
        da.random.normal(size=(400, 16, 21), chunks=(400, 16, 21)),
        dims=('time', 'y', 'x'), name='pcp_error',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    ds['pcp_exceedence_probability'] = xr.DataArray(
        da.random.uniform(0, 1, size=(400, 16, 21), chunks=(400, 16, 21)),
        dims=('time', 'y', 'x'), name='pcp_exceedence_probability',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    da_normal = xr.DataArray(
        da.random.normal(size=(400, 16, 21), chunks=(400, 16, 21)),
        dims=('time', 'y', 'x'), name='random_normal',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    da_uniform = xr.DataArray(
        da.random.uniform(0, 1, size=(400, 16, 21), chunks=(400, 16, 21)),
        dims=('time', 'y', 'x'), name='random_uniform',
        coords={'time': pd.date_range('2010-01-01', periods=400)})

    actual = add_random_effect(ds, da_normal, da_uniform=da_uniform,
                               var='pcp', root=3.)
    expected = add_random_effect(ds.load(), da_normal.load(),
                                 da_uniform=da_uniform.load(),
                                 var='pcp', root=3.)
    assert isinstance(actual.data, dask_array_type)
    xr.testing.assert_equal(actual.compute(), expected)


@requires_test_data
def test_with_real_data():
    scrf = get_test_data('scrf')
    gard_ds = get_test_data('gard')
    domain = get_test_data('domain')

    scrf['time'] = pd.date_range('1950-01-01', freq='D',
                                 periods=len(scrf['time']))
    gard_ds['time'] = pd.date_range('1950-01-01', freq='D',
                                    periods=len(gard_ds['time']))

    gard_ds = make_gard_like_obs(gard_ds, domain)

    for var, rand_key, root in [('pcp', 'p_rand', 3.),
                                ('t_mean', 't_rand', 1)]:
        expected = add_random_effect(gard_ds, scrf[rand_key],
                                     da_uniform=scrf['p_rand_uniform'],
                                     var=var, root=root)

        print(expected.compute())
