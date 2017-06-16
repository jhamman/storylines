#!/usr/bin/env python
import numpy as np
from scipy import stats

import argparse

import dask.array as da
import xarray as xr

SYNTHETIC_MIN = -1e20
SYNTHETIC_MAX = 1e20


def quantile_mapping(input_data, ref_data, data_to_match, mask=None,
                     alpha=0.4, beta=0.4, detrend=False,
                     extrapolate=None, n_endpoints=10):
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

    assert input_data.get_axis_num('time') == 0
    assert ref_data.get_axis_num('time') == 0
    assert data_to_match.get_axis_num('time') == 0
    shape = input_data.shape[1:]
    assert shape == ref_data.shape[1:]
    assert shape == data_to_match.shape[1:]

    # Make mask if mask is one was not provided
    if mask is None:
        d0 = input_data.isel(time=0, drop=False)
        mask = xr.DataArray(~da.isnull(d0), dims=d0.dims,
                            coords=d0.coords)
    else:
        d0 = mask

    chunks = d0.chunks

    # keyword args to qmap
    kwargs = dict(alpha=alpha, beta=beta, extrapolate=extrapolate,
                  n_endpoints=n_endpoints, detrend=detrend)

    if ref_data is input_data:
        use_ref_data = False
    else:
        use_ref_data = True

    if isinstance(input_data.data, da.Array):
        # dask arrays
        mask = mask.chunk(chunks)

        assert chunks == input_data.data.chunks[1:]
        assert chunks == ref_data.data.chunks[1:]
        assert chunks == data_to_match.data.chunks[1:]

        new = da.map_blocks(_qmap_wrapper, input_data.data, ref_data.data,
                            data_to_match.data, mask.data,
                            dtype=input_data.data.dtype,
                            chunks=input_data.data.chunks,
                            name='qmap', use_ref_data=use_ref_data, **kwargs)
    else:
        # numpy arrays
        new = _qmap_wrapper(input_data.data, ref_data.data, data_to_match.data,
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
    ref_groups = ref_data.groupby(grouper)
    input_groups = input_data.groupby(grouper)

    # Iterate over the groups, calling the quantile method function on each
    results = []
    for (key_obs, group_obs), (key_ref, group_ref), (key_input, group_input) \
            in zip(obs_groups, ref_groups, input_groups):
        results.append(quantile_mapping(group_input, group_obs, **kwargs))

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
    if extrapolate is None:
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
        assert len(x[s]) == n_endpoints
        assert not np.isinf(x[s]).any()
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        x[lower_inds] = (y[lower_inds] - intercept) / slope
    if len(upper_inds):
        s = slice(upper_inds[0] - n_endpoints, upper_inds[0])
        slope, intercept, _, _, _ = stats.linregress(x[s], y[s])
        assert len(x[s]) == n_endpoints
        assert not np.isinf(x[s]).any()
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
    slope, intercept, _, _, _ = stats.linregress(t, y)
    trend = intercept + t * slope
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
    new[sort_inds] = np.interp(x_data, x_like, y_like)

    # put the trend back
    if detrend:
        new += data_trend

    return new


def _qmap_wrapper(data, ref, like, mask, **kwargs):
    new = np.full_like(data, np.nan)
    ii, jj = np.nonzero(mask)
    if kwargs.get('use_ref_data', True):
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
    parser.add_argument('data', help='data file')
    parser.add_argument('--ref', help='reference data file', default=False)
    args = parser.parse_args()

    n_mems = 5

    # chunks = {'lat': 56, 'lon': 58, 'time': 1e20}
    chunks = {}
    variables = ['pcp', 't_mean', 't_range']
    detrend = {'pcp': False, 't_mean': True, 't_range': True}
    extrapolate = {'pcp': 'max', 't_mean': 'both', 't_range': 'max'}
    zeros = {'pcp': True, 't_mean': False, 't_range': False}

    obs_files = ['/glade/u/home/jhamman/workdir/GARD_inputs/newman_ensemble/conus_ens_00%d.nc' % i for i in range(1, n_mems+1)]
    print('opening obs files %s' % obs_files)
    obs = xr.open_mfdataset(obs_files, chunks=chunks,
                            decode_times=False, concat_dim='time')

    print('opening data file %s' % args.data)
    data = xr.open_dataset(args.data, chunks=chunks)
    if 't_mean' not in data:
        data['t_mean'] = (data['t_min'] + data['t_max']) / 2

    if args.ref == 'auto':
        template = 'gard_output.{gset}.{dset}.{gcm}.{scen}.{date_range}.dm.nc'
        _, gset, dset, gcm, scen, drange, step, _ = args.data.split('.')

        ref_time = {'NCAR_WRF_50km': '19510101-20051231',
                    'NCAR_WRF_50km_reanalysis': '19790101-20151231'}

        ref = template.format(gset=gset, dset=dset, gcm=gcm,
                              scen='hist', date_range=ref_time[dset])

    if ref and ref != args.data:
        print('opening ref file %s' % args.ref)
        ref = xr.open_dataset(args.ref, chunks=chunks)
        if 't_mean' not in ref:
            ref['t_mean'] = (ref['t_min'] + ref['t_max']) / 2
    else:
        print('skipping reference data')
        ref = data

    qm_ds = xr.Dataset()
    for var in variables:
        print(var, flush=True)
        qm_ds[var] = quantile_mapping_by_group(
            data[var].load(), ref[var].load(), obs[var].load(),
            grouper=None,
            detrend=detrend[var],
            extrapolate=extrapolate[var])

        if zeros[var]:
            # make sure a zero in the input data comes out as a zero
            qm_ds[var].values = np.where(data[var].values == 0,
                                         0, qm_ds[var].values)

    qm_ds['tmax'] = qm_ds['t_mean'] + 0.5 * qm_ds['t_range']
    qm_ds['tmin'] = qm_ds['t_mean'] - 0.5 * qm_ds['t_range']

    new_fname = args.data[:-3] + '.qm.nc'

    print('writing output file %s' % new_fname)
    qm_ds.to_netcdf(new_fname)


if __name__ == "__main__":
    main()
