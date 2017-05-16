import numpy as np
from scipy import stats

import dask.array as da
import xarray as xr


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
        ref_data = None

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
                            name='qmap', **kwargs)
    else:
        # numpy arrays
        new = _qmap_wrapper(input_data.data, ref_data.data, data_to_match.data,
                            mask.data, **kwargs)

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


def make_x_and_y(y, alpha, beta, extrapolate):
    '''helper function to calculate x0, conditionally adding endpoints'''
    n = len(y)

    temp = plotting_positions(n, alpha, beta)

    if extrapolate is not None:
        # Add endpoints to x0
        if extrapolate == 'both':
            x = np.empty(n + 2)
            y_new = np.full(n + 2, np.nan)
            rs = slice(1, -1)
            x[rs] = temp
            x[0] = 0.
            x[-1] = 1.
        elif extrapolate == 'min':
            x = np.empty(n + 1)
            y_new = np.full(n + 1, np.nan)
            rs = slice(1, None)
            x[rs] = temp
            x[0] = 0.
        elif extrapolate == 'max':
            x = np.empty(n + 1)
            y_new = np.full(n + 1, np.nan)
            rs = slice(None, -1)
            x[rs] = temp
            x[-1] = 1.
        else:
            raise ValueError('unknown value for extrapolate: %s' % extrapolate)
        # move the values from y to the new y array
        y_new[rs] = y
    else:
        rs = slice(None)
        extrapolate = False
        x = temp
        y_new = y

    return x, y_new, rs


def _extrapolate(y, alpha, beta, n_endpoints, how='both', ret_slice=False):

    x_new, y_new, rs = make_x_and_y(y, alpha, beta, extrapolate=how)
    y_new = calc_endpoints(x_new, y_new, how, n_endpoints)

    if ret_slice:
        return x_new, y_new, rs
    else:
        return x_new, y_new


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


def qmap(data, ref, like, alpha=0.4, beta=0.4,
         extrapolate=None, n_endpoints=10, detrend=None):
    '''quantile mapping for a single point'''

    if detrend:
        # remove linear trend, saving the slope/intercepts for use later
        data, data_trend = remove_trend(data, inplace=True)
        ref, _ = remove_trend(ref, inplace=True)
        like, _ = remove_trend(like, inplace=True)

    # x is the percentiles
    # y is the sorted data
    sort_inds = np.argsort(data)
    x_data, y_data, rs = _extrapolate(data[sort_inds], alpha, beta,
                                      n_endpoints,
                                      how=extrapolate, ret_slice=True)
    x_ref, y_ref = _extrapolate(np.sort(ref), alpha, beta, n_endpoints,
                                how=extrapolate)
    x_like, y_like = _extrapolate(np.sort(like), alpha, beta,
                                  n_endpoints, how=extrapolate)

    # map the quantiles from ref-->data
    x_data = np.interp(y_data, y_ref, x_ref)

    # empty array, prefilled with nans
    new = np.full_like(data, np.nan)

    # Indicies that would sort the input data
    new[sort_inds] = np.interp(x_data, x_like, y_like)

    # put the trend back
    if detrend:
        new += data_trend

    return new


def _qmap_wrapper(data, ref, like, mask, **kwargs):
    new = np.full_like(data, np.nan)
    ii, jj = np.nonzero(mask)
    for i, j in zip(ii, jj):
        new[:, i, j] = qmap(data[:, i, j], ref[:, i, j], like[:, i, j],
                            **kwargs)
    return new
