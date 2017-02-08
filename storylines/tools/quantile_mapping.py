import numpy as np
import pandas as pd
import xarray as xr


def quantile_mapping(input_data, data_to_match, mask=None,
                     alpha=0.4, beta=0.4):
    '''quantile mapping'''

    # Allocate memory for new array
    new = xr.full_like(input_data, np.nan)

    # Make mask if mask is one was not provided
    if mask is None:
        d0 = input_data.isel(time=0, drop=True)
        mask = xr.Variable(d0.dims, ~pd.isnull(d0))

    # quantiles for the input data
    n = len(input_data['time'])
    x1 = (np.arange(1, n + 1) - alpha) / (n + 1. - alpha - beta)

    # quantiles for the obs
    n = len(data_to_match['time'])
    x0 = (np.arange(1, n + 1) - alpha) / (n + 1. - alpha - beta)

    for (i, j), m in np.ndenumerate(mask):
        if m:
            # Sorted Observations
            y0 = np.sort(data_to_match[:, i, j])
            # Indicies that would sort the input data
            sort_inds = np.argsort(input_data[:, i, j])
            new[sort_inds, i, j] = np.interp(x1, x0, y0)  # TODO: handle edges

    return new


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
