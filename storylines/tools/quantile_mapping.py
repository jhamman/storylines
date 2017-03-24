import numpy as np
import dask.array as da
import xarray as xr


def quantile_mapping(input_data, data_to_match, mask=None,
                     alpha=0.4, beta=0.4):
    '''quantile mapping between `input_data` and `data_to_match`

    Parameters
    ----------
    input_data : xr.DataArray
        Input data to be quantile mapped to match the distribution of
        `data_to_match`
    data_to_match : xr.DataArray
        Target data for quantile mapping
    mask : xr.DataArray (optional, boolean)
        2-dimensional mask where quantile mapping should be performed
    alpha, beta : float
        Plotting positions parameter. Default is 0.4.

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
    assert data_to_match.get_axis_num('time') == 0
    assert input_data.shape[1:] == data_to_match.shape[1:]

    # Make mask if mask is one was not provided
    if mask is None:
        d0 = input_data.isel(time=0, drop=True)
        mask = xr.Variable(d0.dims, ~da.isnull(d0))

    # quantiles for the input data
    x1 = plotting_positions(len(input_data['time']), alpha, beta)

    # quantiles for the obs
    x0 = plotting_positions(len(data_to_match['time']), alpha, beta)

    def qmap(data, like, mask):
        '''helper function to apply the meat of the quantile mapping method

        x0 and x1 are inherited from the scope above.
        '''
        # Use numpy to sort these arrays before we loop through each variable
        sort_inds_all = np.argsort(data, axis=0)
        sorted_all = np.sort(like, axis=0)

        # empty array, prefilled with nans
        new = np.full_like(data, np.nan)

        # i,j indicies for the valid grid cells
        ii, jj = mask.nonzero()

        # loop over valid indicies
        for i, j in zip(ii, jj):
            # Sorted Observations
            y0 = sorted_all[:, i, j]
            # Indicies that would sort the input data
            sort_inds = sort_inds_all[:, i, j]
            new[sort_inds, i, j] = np.interp(x1, x0, y0)  # TODO: handle edges

        return new

    if isinstance(input_data.data, da.Array):
        # dask arrays
        new = da.map_blocks(qmap, input_data.data, data_to_match.data,
                            mask.data, chunks=input_data.data.chunks,
                            name='qmap')
    else:
        # numpy arrays
        new = qmap(input_data.data, data_to_match.data, mask.data)

    return xr.DataArray(new, dims=input_data.dims, coords=input_data.coords,
                        attrs=input_data.attrs, name=input_data.name)


def apply_quantile_mapping_by_group(input_data, data_to_match,
                                    grouper='time.month', **kwargs):
    '''quantile mapping between `input_data` and `data_to_match by group`

    Parameters
    ----------
    input_data : xr.DataArray
        Input data to be quantile mapped to match the distribution of
        `data_to_match`
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
        return quantile_mapping(input_data, data_to_match, **kwargs)

    # Create the groupby objects
    obs_groups = data_to_match.groupby(grouper)
    input_groups = input_data.groupby(grouper)

    # Iterate over the groups, calling the quantile method function on each
    results = []
    for (key_obs, group_obs), (key_input, group_input) in zip(obs_groups,
                                                              input_groups):
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
