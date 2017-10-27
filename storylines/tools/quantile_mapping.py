#!/usr/bin/env python
import sys
import numpy as np
from scipy import stats

import os
import argparse

import dask.array as da
import xarray as xr

from storylines.tools.encoding import attrs, encoding, make_gloabl_attrs

SYNTHETIC_MIN = -1e20
SYNTHETIC_MAX = 1e20

variables = ['pcp', 't_mean', 't_range']
detrend = {'pcp': False, 't_mean': True, 't_range': True}
extrapolate = {'pcp': 'max', 't_mean': 'both', 't_range': 'max'}
zeros = {'pcp': True, 't_mean': False, 't_range': False}


def quantile_mapping(input_data, ref_data, data_to_match, mask=None,
                     alpha=0.4, beta=0.4, detrend=False,
                     extrapolate=None, n_endpoints=10,
                     use_ref_data=True):
    '''quantile mapping between `input_data` and `data_to_match`

    Parameters
    ----------
    input_data : xr.DataArray
        Input data to be quantile mapped to match the distribution of
        `data_to_match`
    ref_data : xr.DataArray
        Reference data to be used to adjust `input_data`
    data_to_match : xr.DataArray
        Target data for quantile mapping
    mask : xr.DataArray (optional, boolean)
        2-dimensional mask where quantile mapping should be performed
    alpha, beta : float
        Plotting positions parameter. Default is 0.4.
    detrend : bool
        If True, detrend `input_data` prior to performing quantile mapping.
        Default is False.
    extrapolate : str
        Option specifying how to handle endpoints/extreme values. Valid options
        are {'max', 'min', 'both', None}. If `extrapolate` is not `None`, the
        end point(s) of the CDF (0, 1) will be linearly extrapolated using the
        last `n_endpoints` from the tail of the distribution. Default is None.
    n_endpoints : int
        Number of data points to use when the `extrapolate` option is set.

    Returns
    -------
    new : xr.DataArray
        Quantile mapped data with shape from `input_data` and probability
            distribution from `data_to_match`.

    See Also
    --------
    scipy.stats.mstats.plotting_positions

    Note
    ----
    This function will use `dask.array.map_blocks` if the input arguments are
    of type `dask.array.Array`.
    '''
    print('quantile mapping now')

    assert input_data.get_axis_num('time') == 0
    assert data_to_match.get_axis_num('time') == 0
    shape = input_data.shape[1:]
    assert shape == data_to_match.shape[1:]

    if ref_data is not False:
        assert ref_data.get_axis_num('time') == 0
        assert shape == ref_data.shape[1:]
    if use_ref_data and ref_data is False:
        raise ValueError('cannot use ref_data without ref_data')

    # Make mask if mask is one was not provided
    if mask is None:
        d0 = input_data.isel(time=0, drop=False)
        mask = d0.notnull().astype(int)
    else:
        d0 = mask.astype(int)

    chunks = d0.chunks

    # keyword args to qmap
    kwargs = dict(alpha=alpha, beta=beta, extrapolate=extrapolate,
                  n_endpoints=n_endpoints, detrend=detrend)

    if isinstance(input_data.data, da.Array):
        print('inputs are dask arrays')
        # dask arrays
        mask = mask.chunk(chunks)

        assert chunks == input_data.data.chunks[1:]
        assert chunks == data_to_match.data.chunks[1:]

        if use_ref_data:
            assert chunks == ref_data.data.chunks[1:]

            new = da.map_blocks(_qmap_wrapper, input_data.data, ref_data.data,
                                data_to_match.data, mask.data,
                                dtype=input_data.data.dtype,
                                chunks=input_data.data.chunks,
                                name='qmap', use_ref_data=use_ref_data,
                                **kwargs)
        else:
            new = da.map_blocks(_qmap_wrapper, input_data.data, False,
                                data_to_match.data, mask.data,
                                dtype=input_data.data.dtype,
                                chunks=input_data.data.chunks,
                                name='qmap', use_ref_data=use_ref_data,
                                **kwargs)

    else:
        # numpy arrays
        if use_ref_data:
            new = _qmap_wrapper(input_data.data, ref_data.data,
                                data_to_match.data,
                                mask.data, use_ref_data=use_ref_data, **kwargs)
        else:
            new = _qmap_wrapper(input_data.data, False, data_to_match.data,
                                mask.data, use_ref_data=use_ref_data, **kwargs)

    return xr.DataArray(new, dims=input_data.dims, coords=input_data.coords,
                        attrs=input_data.attrs,
                        name=input_data.name).where(mask)


def quantile_mapping_by_group(input_data, ref_data, data_to_match,
                              grouper='time.month', **kwargs):
    '''quantile mapping between `input_data` and `data_to_match by group`

    Parameters
    ----------
    input_data : xr.DataArray
        Input data to be quantile mapped to match the distribution of
        `data_to_match`
    ref_data : xr.DataArray
        Reference data to be used to adjust `input_data`
    data_to_match : xr.DataArray
        Target data for quantile mapping
    grouper : str, array, Grouper
        Object to pass to `DataArray.groupby`, default ``'time.month'``
    kwargs : any
        Additional named arguments to `quantile_mapping`

    Returns
    -------
    new : xr.DataArray
        Quantile mapped data with shape from `input_data` and probability
            distribution from `data_to_match`.

    See Also
    --------
    quantile_mapping
    scipy.stats.mstats.plotting_positions

    Note
    ----
    This function will use `dask.array.map_blocks` if the input arguments are
    of type `dask.array.Array`.
    '''

    # Allow grouper to be None
    if grouper is None:
        return quantile_mapping(input_data, ref_data, data_to_match, **kwargs)

    # Create the groupby objects
    obs_groups = data_to_match.groupby(grouper)
    input_groups = input_data.groupby(grouper)

    # Iterate over the groups, calling the quantile method function on each
    results = []
    if ref_data is not False:
        ref_groups = ref_data.groupby(grouper)
        for (key_obs, group_obs), (key_ref, group_ref), (key_input, group_input) \
                in zip(obs_groups, ref_groups, input_groups):
            results.append(quantile_mapping(group_input, group_ref, group_obs,
                                            **kwargs))
    else:
        for (key_obs, group_obs), (key_input, group_input) \
                in zip(obs_groups, input_groups):
            results.append(quantile_mapping(group_input, False, group_obs,
                                            **kwargs))

    # put the groups back together
    new_concat = xr.concat(results, dim='time')
    # Now sort the time dimension again
    sort_inds = np.argsort(new_concat.time.values)
    new_concat = new_concat.isel(time=sort_inds)

    return new_concat


def plotting_positions(n, alpha=0.4, beta=0.4):
    '''Returns a monotonic array of plotting positions.

    Parameters
    ----------
    n : int
        Length of plotting positions to return.
    alpha, beta : float
        Plotting positions parameter. Default is 0.4.

    Returns
    -------
    positions : ndarray
        Quantile mapped data with shape from `input_data` and probability
            distribution from `data_to_match`.

    See Also
    --------
    scipy.stats.mstats.plotting_positions

    '''
    return (np.arange(1, n + 1) - alpha) / (n + 1. - alpha - beta)


def make_x_and_y(y, alpha, beta, extrapolate,
                 x_min=SYNTHETIC_MIN, x_max=SYNTHETIC_MAX):
    '''helper function to calculate x0, conditionally adding endpoints'''
    n = len(y)

    temp = plotting_positions(n, alpha, beta)

    x = np.empty(n + 2)
    y_new = np.full(n + 2, np.nan)
    rs = slice(1, -1)
    x[rs] = temp

    # move the values from y to the new y array
    # repeat the first/last values to make everything consistant
    y_new[rs] = y
    y_new[0] = y[0]
    y_new[-1] = y[-1]

    # Add endpoints to x0
    if (extrapolate is None) or (extrapolate == '1to1'):
        x[0] = temp[0]
        x[-1] = temp[-1]
    elif extrapolate == 'both':
        x[0] = x_min
        x[-1] = x_max
    elif extrapolate == 'max':
        x[0] = temp[0]
        x[-1] = x_max
    elif extrapolate == 'min':
        x[0] = x_min
        x[-1] = temp[-1]
    else:
        raise ValueError('unknown value for extrapolate: %s' % extrapolate)

    return x, y_new, rs


def _extrapolate(y, alpha, beta, n_endpoints, how='both', ret_slice=False,
                 x_min=SYNTHETIC_MIN, x_max=SYNTHETIC_MAX):

    x_new, y_new, rs = make_x_and_y(y, alpha, beta,
                                    extrapolate=how, x_min=x_min, x_max=x_max)
    y_new = calc_endpoints(x_new, y_new, how, n_endpoints)

    if ret_slice:
        return x_new, y_new, rs
    else:
        return x_new, y_new


def _custom_extrapolate_x_data(x, y, n_endpoints):
    lower_inds = np.nonzero(-np.inf == x)[0]
    upper_inds = np.nonzero(np.inf == x)[0]
    if len(lower_inds):
        s = slice(lower_inds[-1] + 1, lower_inds[-1] + 1 + n_endpoints)
        print(s, lower_inds[-1], n_endpoints, x[s])
        assert not np.isinf(x[s]).any()
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        x[lower_inds] = (y[lower_inds] - intercept) / slope
    if len(upper_inds):
        s = slice(upper_inds[0] - n_endpoints, upper_inds[0])
        assert not np.isinf(x[s]).any()
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        x[upper_inds] = (y[upper_inds] - intercept) / slope
    return x


def calc_endpoints(x, y, extrapolate, n_endpoints):
    '''extrapolate the tails of the CDF using linear interpolation on the last
    n_endpoints

    This function modifies `y` in place'''

    if n_endpoints < 2:
        raise ValueError('Invalid number of n_endpoints, must be >= 2')

    if extrapolate in ['min', 'both']:
        s = slice(1, n_endpoints + 1)
        # fit linear model to slice(1, n_endpoints + 1)
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        # calculate the value of y at x[0]
        y[0] = intercept + slope * x[0]
    if extrapolate in ['max', 'both']:
        s = slice(-n_endpoints - 1, -1)
        # fit linear model to slice(-n_endpoints - 1, -1)
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        # calculate the value of y at x[-1]
        y[-1] = intercept + slope * x[-1]

    return y


def remove_trend(y, inplace=False):
    if inplace:
        detrended = y
    else:
        detrended = y.copy()

    t = np.arange(len(y))
    slope = stats.linregress(t, y)[0]  # extract slope only
    trend = t * slope
    detrended -= trend

    return detrended, trend


def qmap(data, ref, like, alpha=0.4, beta=0.4, extrapolate=None,
         n_endpoints=10, detrend=None, use_ref_data=True):
    '''quantile mapping for a single point'''

    inplace = False

    if detrend:
        # remove linear trend, saving the slope/intercepts for use later
        data, data_trend = remove_trend(data, inplace=inplace)
        like, _ = remove_trend(like, inplace=inplace)

    # x is the percentiles
    # y is the sorted data
    sort_inds = np.argsort(data)
    x_data, y_data, rs = _extrapolate(data[sort_inds], alpha, beta,
                                      n_endpoints,
                                      how=extrapolate, ret_slice=True,
                                      x_min=0, x_max=1)

    x_like, y_like = _extrapolate(np.sort(like), alpha, beta,
                                  n_endpoints, how=extrapolate,
                                  x_min=-1e15, x_max=1e15)

    # map the quantiles from ref-->data
    # TODO: move to its own function
    if use_ref_data and ref is not False:
        if detrend:
            ref, _ = remove_trend(ref, inplace=inplace)

        x_ref, y_ref = _extrapolate(np.sort(ref), alpha, beta, n_endpoints,
                                    how=extrapolate, x_min=-1e10, x_max=1e10)

        left = -np.inf if extrapolate in ['min', 'both'] else None
        right = np.inf if extrapolate in ['max', 'both'] else None
        x_data = np.interp(y_data, y_ref, x_ref, left=left, right=right)

    if np.isinf(x_data).any():
        # Extrapolate the tails beyond 1.0 to handle "new extremes"
        x_data = _custom_extrapolate_x_data(x_data, y_data, n_endpoints)

    # empty array, prefilled with nans
    new = np.full_like(data, np.nan)

    # Do the final mapping
    new[sort_inds] = np.interp(x_data, x_like, y_like)[rs]

    # If extrapolate is 1to1, apply the offset between ref and like to the
    # tails of new
    if use_ref_data and (ref is not False) and (extrapolate == '1to1'):
        ref_max = ref.max()
        ref_min = ref.min()
        inds = (data > ref_max)
        if inds.any():
            if len(ref) == len(like):
                new[inds] = like.max() + (data[inds] - ref_max)
            elif len(ref) > len(like):
                ref_at_like_max = np.interp(x_like[-1], x_ref, y_ref)
                new[inds] = like.max() + (data[inds] - ref_at_like_max)
            elif len(ref) < len(like):
                like_at_ref_max = np.interp(x_ref[-1], x_like, y_like)
                new[inds] = like_at_ref_max + (data[inds] - ref_max)
        inds = (data < ref_min)
        if inds.any():
            if len(ref) == len(like):
                new[inds] = like.min() + (data[inds] - ref_min)
            elif len(ref) > len(like):
                ref_at_like_min = np.interp(x_like[0], x_ref, y_ref)
                new[inds] = like.min() + (data[inds] - ref_at_like_min)
            elif len(ref) < len(like):
                like_at_ref_min = np.interp(x_ref[0], x_like, y_like)
                new[inds] = like_at_ref_min + (data[inds] - ref_min)

    # put the trend back
    if detrend:
        new += data_trend

    return new


def _qmap_wrapper(data, ref, like, mask, **kwargs):

    new = np.full_like(data, np.nan)
    ii, jj = np.nonzero(mask)
    if kwargs.get('use_ref_data', True) and ref is not False:
        for i, j in zip(ii, jj):
            new[:, i, j] = qmap(data[:, i, j], ref[:, i, j], like[:, i, j],
                                **kwargs)
    else:
        for i, j in zip(ii, jj):
            new[:, i, j] = qmap(data[:, i, j], False, like[:, i, j],
                                **kwargs)

    return new


def main():
    """
    Generate high-resolution meteorologic forcings by downscaling the GCM
    and/or RCM using the Generalized Analog Regression Downscaling (GARD) tool.

    Inputs:
    Configuration file formatted with the following options:
    TODO: Add sample config
    """
    # Define usage and set command line arguments
    parser = argparse.ArgumentParser(
        description='Downscale ensemble forcings')
    parser.add_argument('--data', help='data file')
    parser.add_argument('--ref', help='reference data file', default=False)
    parser.add_argument('--obs_files', nargs='+', help='obs data file(s)',
                        default=['/glade/u/home/jhamman/workdir/GARD_inputs/newman_ensemble/conus_ens_00%d.nc' % i for i in range(1, 6)])
    parser.add_argument('--kind', help='input data type', default='gard')
    parser.add_argument('--variables', nargs='+', default=variables,
                        help='list of variables to quantile map')
    parser.add_argument('--skip_existing', action='store_true')
    args = parser.parse_args()

    print('opening obs files %s' % args.obs_files)
    obs = xr.open_mfdataset(args.obs_files,
                            decode_times=False, concat_dim='time')
    obs['time'].values = np.arange(obs.dims['time'])
    obs['time'].encoding = {}
    obs['time'].attrs = {}

    if args.kind == 'gard':
        data, ref, new_fname = _gard_func(args, obs)
    elif args.kind == 'icar':
        data, ref, new_fname = _icar_func(args, obs)

    if 'mask' in data:
        mask = data['mask']
    elif 'mask' in obs:
        mask = obs['mask']
    else:
        mask = None
    if mask is not None and 'time' in mask.coords:
        mask = mask.isel(time=0, drop=True).astype(np.int)
    if mask is not None:
        print('number of points in mask:', mask.values.sum())
        mask = mask.load()

    if args.skip_existing and os.path.isfile(new_fname):
        print('skipping: output file exists')
        with open("QM_EXISTING.txt", "a") as f:
            f.write(new_fname + '\n')
        return

    qm_ds = xr.Dataset()
    for var in args.variables:
        print(var, flush=True)
        if ref is not False:
            ref_da = ref[var].load()
            use_ref_data = True
        else:
            use_ref_data = False
            ref_da = False
        qm_ds[var] = quantile_mapping(
            data[var].load(),
            ref_da,
            obs[var].load(),
            mask=mask,
            detrend=detrend[var],
            extrapolate='1to1',
            use_ref_data=use_ref_data)  # extrapolate[var])
        if np.isnan(data[var]).all():
            print('data[%s] is all nans' % var)
        if np.isnan(obs[var]).all():
            print('obs[%s] is all nans' % var)
        if np.isnan(qm_ds[var]).all():
            print('qm_ds[%s] is all nans' % var)

        if zeros[var]:
            # make sure a zero in the input data comes out as a zero
            qm_ds[var].values = np.where(data[var].values <= 0,
                                         0, qm_ds[var].values)
    qm_ds['time'] = data['time']

    for var in ['elevation', 'mask']:
        if var in obs:
            qm_ds[var] = obs[var]
        elif var in data:
            qm_ds[var] = data[var]
        if 'time' in qm_ds[var].coords:
            qm_ds[var] = qm_ds[var].isel(time=0, drop=True)
    qm_ds['mask'] = qm_ds['mask'].astype(np.int)

    if 't_mean' in args.variables and 't_range' in args.variables:
        qm_ds['t_max'] = qm_ds['t_mean'] + 0.5 * qm_ds['t_range']
        qm_ds['t_min'] = qm_ds['t_mean'] - 0.5 * qm_ds['t_range']

    use_encoding = {}
    for key in qm_ds.data_vars:
        if key in encoding:
            use_encoding[key] = encoding[key]

    for var in qm_ds.data_vars:
        try:
            qm_ds[var].attrs = attrs.get(var, obs[var].attrs)
            qm_ds[var].encoding = use_encoding.get(var, obs[var].encoding)
        except KeyError:
            print('unable to find attributes for %s' % var)
            pass

    qm_ds.attrs = make_gloabl_attrs(title='Quantile mapped downscaled dataset')

    print('writing output file %s' % new_fname)
    print(qm_ds)
    qm_ds.info()
    qm_ds.to_netcdf(new_fname, unlimited_dims=['time'],
                    format='NETCDF4', encoding=use_encoding)


def _gard_func(args, obs):
    print('opening data file %s' % args.data)
    data = xr.open_dataset(args.data)

    if 't_mean' not in data and 't_min' in data and 't_max' in data:
        data['t_mean'] = (data['t_min'] + data['t_max']) / 2

    if args.ref == 'auto':
        template = 'gard_output.{gset}.{dset}.{gcm}.{scen}.{date_range}.dm.nc'
        _, gset, dset, gcm, scen, drange, step, _ = args.data.split('.')

        ref_time = {'NCAR_WRF_50km': '19510101-20051231',
                    'NCAR_WRF_50km_reanalysis': '19790101-20151231'}

        ref = template.format(gset=gset, dset=dset, gcm=gcm,
                              scen='hist', date_range=ref_time[dset])

    if ref is not False and ref != args.data:
        print('opening ref file %s' % ref)
        ref = xr.open_dataset(ref)
        if 't_mean' not in ref and 't_min' in ref and 't_max' in ref:
            ref['t_mean'] = (ref['t_min'] + ref['t_max']) / 2
    else:
        print('skipping reference data')
        ref = False

    new_fname = args.data[:-3] + '.qm.nc'

    return data, ref, new_fname


def _icar_func(args, obs):

    if os.path.isfile(args.data):
        data = xr.open_dataset(args.data).rename({'icar_pcp': 'pcp',
                                                  'avg_ta2m': 't_mean'})
        case_name = os.path.basename(args.data)
        pattern = args.data
        for scen in ['hist', 'rcp45', 'rcp85']:
            if scen in case_name:
                break
    else:
        dirname = args.data
        case_name = os.path.split(dirname)[-1]
        pattern = os.path.join(dirname, 'merged', 'merged_%s_*.nc' % case_name)
        print(case_name, pattern, flush=True)

        data = xr.open_mfdataset(pattern).rename({'icar_pcp': 'pcp',
                                                  'avg_ta2m': 't_mean'})
        scen = case_name.split('_')[1]

    data['lon'].data[data['lon'].data > 180] -= 360.

    data['t_range'] = data['max_ta2m'] - data['min_ta2m']

    if scen == 'hist':
        ref = False
    else:
        if args.ref == 'auto':
            ref_pattern = pattern.replace(scen, 'hist')
        else:
            ref_pattern = args.ref
        print('  ref->', ref_pattern, flush=True)
        ref = xr.open_mfdataset(ref_pattern).rename(
            {'icar_pcp': 'pcp', 'avg_ta2m': 't_mean'})
        ref['lon'].data[ref['lon'].data > 180] -= 360.
        ref['t_range'] = ref['max_ta2m'] - ref['min_ta2m']

    new_fname = '/glade/u/home/jhamman/workdir/icar_qm/%s' % case_name

    return data, ref, new_fname


if __name__ == "__main__":
    try:
        main()
    except:
        # write out failed arguments
        with open("FAILED.txt", "a") as f:
            line = ' '.join(sys.argv) + '\n'
            f.write(line)
        raise
