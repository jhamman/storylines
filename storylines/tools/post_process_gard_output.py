#!/usr/bin/env python
import argparse

import numpy as np
import dask
import xarray as xr
from scipy.special import cbrt
from scipy.stats import norm

from netCDF4 import num2date
from xarray.conventions import nctime_to_nptime

from .gard_utils import read_config
from .quantile_mapping import quantile_mapping_by_group

attrs = {'pcp': {'units': 'mm', 'long_name': 'precipitation',
                 'comment': 'random effects applied'},
         't_mean': {'units': 'C', 'long_name': 'air temperature',
                    'comment': 'random effects applied'},
         't_range': {'units': 'C', 'long_name': 'daily air temperature range',
                     'comment': 'random effects applied'},
         't_min': {'units': 'C', 'long_name': 'minimum daily air temperature',
                   'comment': 'random effects applied'},
         't_max': {'units': 'C', 'long_name': 'maximum daily air temperature',
                   'comment': 'random effects applied'}}

encoding = {'pcp': {'_FillValue': -9999},
            't_mean': {'_FillValue': -9999},
            't_range': {'_FillValue': -9999},
            't_min': {'_FillValue': -9999},
            't_max': {'_FillValue': -9999}}


def _get_units_from_drange(d):
    return 'days since %s-%s-%s' % (d[:4], d[4:6], d[6:8])


def make_gard_like_obs(ds, obs, mask=None):

    ds = ds.rename({'x': 'lon', 'y': 'lat'})
    ds.coords['lon'] = obs['lon']
    ds.coords['lat'] = obs['lat']

    if mask is not None:
        ds = ds.where(mask)

    return ds


def add_random_effect(ds, da_normal, da_uniform=None,
                      var=None, root=1., logistic_thresh=None):
    '''Add random effects to dataset `ds`

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with variables variable (e.g. `tas`) and error terms.
        If `{var}_exceedence_probability` is a variable in ``ds``, it will also
        be applied to the error term.
    da_normal : xarray.DataArray
        Input spatially correlated random field (normal distribution)
    da_uniform : xarray.DataArray
        Input spatially correlated random field (uniform distribution)
    var : str
        Variable name in ``ds`` that random effects will be added to
    root : float
        Root used to transform

    Returns
    -------
    da_errors : xarray.DataArray
        Like `ds[var]` except da_errors includes random effects.
    '''

    t0, t1 = str(ds.time.data[0]), str(ds.time.data[-1])

    da = ds[var]
    da_errors = ds['{}_error'.format(var)]

    exceedence_var = '{}_exceedence_probability'.format(var)
    if exceedence_var in ds:

        # Get the array of uniform errors
        r_uniform = da_uniform.sel(time=slice(t0, t1))
        # Get the exceedence variable (e.g. POP)
        da_ex = ds[exceedence_var]

        # Allocate a few arrays
        new_uniform = xr.zeros_like(r_uniform)
        r_normal = xr.zeros_like(r_uniform)

        # Mask where precip occurs
        mask = r_uniform > (1 - da_ex)

        # Rescale the uniform distribution
        new_uniform = ((r_uniform - 1) / da_ex) + 1

        # Where precip occurs, get the normal distribution equivalent of
        # new_uniform
        # r_normal.data[mask] = norm.ppf(new_uniform.data[mask])
        r_normal = new_uniform.pipe(norm.ppf)
    else:
        mask = None
        r_normal = da_normal.sel(time=slice(t0, t1))

    if root == 1.:
        da_errors = da + (da_errors * r_normal)
    elif root == 3:
        if isinstance(da.data, dask.array.Array):
            da_errors.data = (dask.array.map_blocks(cbrt, da.data) +
                              (da_errors.data * r_normal.data)) ** root
        else:
            da_errors = (cbrt(da) + (da_errors * r_normal)) ** root
    else:
        da_errors = ((da ** (1./root)) + (da_errors * r_normal)) ** root

    if mask is not None:
        da_errors.data = dask.array.where(mask, da_errors.data, 0)
        da_errors.data = dask.array.maximum(da_errors.data, 0.)

    return da_errors


def process_gard_output(se, scen, periods, obs_ds, rand_ds,
                        template=None, obs_mask=None,
                        calendar='standard',
                        variables=['pcp', 't_mean', 't_range'],
                        rename_vars=None,
                        roots={'pcp': 3., 't_mean': 1, 't_range': 1},
                        rand_vars={'pcp': 'p_rand', 't_mean': 't_rand',
                                   't_range': 't_rand'},
                        quantile_mapping=False, chunks=None):

    '''Top level function for processing raw gard output

    Parameters
    ----------
    se : str
        Set name
    scen : str
        Scenario name
    periods : list of str
        List of periods (e.g. ['19200101-19291231', '19300101-19391231'])
    obs_ds : xr.Dataset
        Dataset with observations
    rand_ds : xr.Dataset
        Dataset with random fields
    template : str
        Filepath template string
    obs_mask : xr.DataArray
        Mask to apply to GARD datasets (optional)
    calendar : str
        Calendar to use for time coordinates
    variables : list of str
        List of variable names to process
    rename_vars : dict
        Dictionary mapping from ``variables`` to obs variable names
    roots : dict
        Dictionary of transform roots
    rand_vars : dict
        Dictionary mapping from ``variables`` to ``rand_ds`` variable names
    quantile_mapping : boolean or str or array
        If True, perform quantile mapping over full record. If str or array,
        perform quantile mapping by group. If False, do not perform any
        quantile mapping.

    Returns
    -------
    ds : xr.Dataset
        Post processed GARD dataset

    See Also
    --------
    add_random_effect
    make_gard_like_obs
    quantile_mapping_by_group
    '''

    if not isinstance(obs_ds, xr.Dataset):
        # we'll assume that obs_ds is a string/path/or something that
        # xr.open_dataset can open, if not, we'll raise an error right away
        obs_ds = xr.open_dataset(obs_ds, chunks=chunks)
    if rename_vars is not None:
        obs_ds = obs_ds.rename(rename_vars)
    if not isinstance(rand_ds, xr.Dataset):
        # we'll assume that obs_ds is a string/path/or something that
        # xr.open_dataset can open, if not, we'll raise an error right away
        rand_ds = xr.open_dataset(rand_ds, chunks=chunks)

    ds_out = xr.Dataset()
    for var in variables:

        ds_list = []
        for drange in periods:

            pre = template.format(se=se, drange=drange, scen=scen)

            ds = xr.open_dataset(pre + '{}.nc'.format(var))
            ds.merge(xr.open_dataset(pre + '{}_errors.nc'.format(var)),
                     inplace=True)
            try:
                ds.merge(xr.open_dataset(pre + '{}_logistic.nc'.format(var)),
                         inplace=True)
            except (FileNotFoundError, OSError):
                pass

            ds = make_gard_like_obs(ds, obs_ds)
            if chunks is not None:
                ds = ds.chunk(chunks=chunks)

            units = _get_units_from_drange(drange)
            ds['time'] = nctime_to_nptime(
                num2date(np.arange(0, ds.dims['time']),
                         units, calendar=calendar))

            ds_list.append(ds)
        ds = xr.concat(ds_list, dim='time')

        if rename_vars is not None and var in rename_vars:
            rename_dict = rename_vars.copy()
            ervar = '{}_error'.format(rename_vars[var])
            rename_dict['{}_error'.format(var)] = ervar
            exvar = '{}_exceedence_probability'.format(rename_vars[var])
            rename_dict['{}_exceedence_probability'.format(var)] = exvar
            ds = ds.rename(rename_dict)
            var = rename_vars[var]

        assert var in attrs
        assert var in encoding

        if se is not 'pass_through':
            root = roots[var]
            rand_var = rand_vars[var]
            da_errors = add_random_effect(ds, rand_ds[rand_var],
                                          rand_ds['p_rand_uniform'],
                                          var=var, root=root)
        else:
            da_errors = ds[var]

        if quantile_mapping is not False:
            t0 = str(da_errors.coords['time'].data[0])
            t1 = str(da_errors.coords['time'].data[-1])
            # offset zeros in GARD data for precip
            # TODO: update this with new dask/xarray where
            if '{}_exceedence_probability'.format(var) in ds:
                da_errors.data = dask.array.where(
                    da_errors.data > 0,  # where precip is non-zero
                    da_errors.data + 1,  # add 1
                    rand_ds['p_rand_uniform'].sel(time=slice(t0, t1)).data)

            # Unpack the grouper for quantile mapping
            if quantile_mapping is True:
                grouper = None
            else:
                grouper = quantile_mapping
            # Quantile mapping
            da_errors = quantile_mapping_by_group(da_errors, obs_ds[var],
                                                  mask=obs_mask,
                                                  grouper=grouper)
        ds_out[var] = da_errors
        ds_out[var].attrs = attrs[var]
        ds_out[var].encoding = encoding[var]

    if 't_range' in variables and 't_mean' in variables:
        ds_out['t_min'] = ds_out['t_mean'] - 0.5 * ds_out['t_range']
        ds_out['t_max'] = ds_out['t_mean'] + 0.5 * ds_out['t_range']
        for var in ['t_max', 't_min']:
            ds_out[var].attrs = attrs[var]
        ds_out = ds_out.drop('t_mean')

    # mask sure the dataset is properly masked
    if obs_mask is not None:
        ds_out = ds_out.where(obs_mask)

    # Add metadata
    ds_out['time'].encoding['calendar'] = calendar
    ds_out['time'].encoding['units'] = units

    return ds_out


def command_line_tool():
    parser = argparse.ArgumentParser(
        description='Post Process GARD output')
    parser.add_argument('config_file', metavar='config_file',
                        help='configuration file for downscaling matrix')
    parser.add_argument('--outdir', metavar='outdir', default=None,
                        help='output directory for post processed files')
    args = parser.parse_args()

    config = read_config(args.config_file)

    # move logic from notebook here

# drange = '{}-{}'.format(periods[0].split('-')[0],
#                         periods[-1].split('-')[1])
# pre = out_template.format(se=se, drange=drange, scen=scen)
# mm_fname_out = pre + 'mm.nc'
# dm_fname_out = pre + 'dm.nc'
#
# print(dm_fname_out)
#
# ds_out.resample('MS', dim='time', how='mean',
#                 keep_attrs=True).to_netcdf(mm_fname_out, encoding=encoding,
#                                            unlimited_dims=['time'])
#
# ds_out.to_netcdf(dm_fname_out, encoding=encoding, unlimited_dims=['time'])

    print(config)


if __name__ == '__main__':
    print('post_process_gard_output')
    command_line_tool()
