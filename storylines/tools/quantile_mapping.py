import numpy as np
import dask.array as da
import xarray as xr


def quantile_mapping(input_data, data_to_match, mask=None,
                     alpha=0.4, beta=0.4):
    '''quantile mapping'''

    assert input_data.get_axis_num('time') == 0
    assert data_to_match.get_axis_num('time') == 0
    assert input_data.shape[1:] == data_to_match.shape[1:]

    # Make mask if mask is one was not provided
    if mask is None:
        d0 = input_data.isel(time=0, drop=True)
        mask = xr.Variable(d0.dims, ~da.isnull(d0))

    # quantiles for the input data
    n = len(input_data['time'])
    x1 = (np.arange(1, n + 1) - alpha) / (n + 1. - alpha - beta)

    # quantiles for the obs
    n = len(data_to_match['time'])
    x0 = (np.arange(1, n + 1) - alpha) / (n + 1. - alpha - beta)

    def qmap(data, like, mask):
        # Use numpy to sort these arrays before we loop through each variable
        sort_inds_all = np.argsort(data, axis=0)
        sorted_all = np.sort(like, axis=0)

        ii, jj = mask.nonzero()

        new = np.full_like(data, np.nan)

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


def apply_quantile_mapping_by_month(input_data, data_to_match, **kwargs):
    '''apply quantile mapping by month
    '''
    obs_groups = data_to_match.groupby('time.month')
    input_groups = input_data.groupby('time.month')

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
