
import argparse

# import numpy as np
import dask
import dask.array as np
import xarray as xr
from scipy.special import cbrt

from netCDF4 import num2date
from xarray.conventions import nctime_to_nptime

from .gard_utils import read_config

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


def add_random_effect(ds, da_rand, var=None, root=1.,
                      is_precip=False, pop_thresh=None):
    '''Add random effects to dataset `ds`

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with variables variable (e.g. `tas`) and error terms.
        if `is_precip=True`, must also include `pcp_exceedence_probability`
        variable.
    da_rand : xarray.DataArray
        Input spatially correlated random field
    root : float
        Root used to transform
    is_precip : bool
        True if the quantity being calculated is precipitation
    pop_thresh : float or xr.DataArray
        Threshold value for POP (required if `is_precip==True`).

    Returns
    -------
    da_errors : xarray.DataArray
        Like `ds[var]` except da_errors includes random effects.
    '''

    t0, t1 = str(ds.time.data[0]), str(ds.time.data[-1])

    e_var = '{}_error'.format(var)
    rand = da_rand.sel(time=slice(t0, t1))
    if root == 1.:
        da_errors = ds[var] + (ds[e_var] * rand)
    if root == 3:
        if isinstance(ds[var].data, dask.array.Array):
            da_errors = (np.map_blocks(cbrt, ds[var]) +
                         (ds[e_var] * rand)) ** root
        else:
            da_errors = (cbrt(ds[var]) + (ds[e_var] * rand)) ** root
    else:
        da_errors = ((ds[var] ** (1./root)) + (ds[e_var] * rand)) ** root

    if is_precip:
        da_pop = ds['{}_exceedence_probability'.format(var)]

        if isinstance(pop_thresh, float):
            # mask where POP is less than the threshold
            da_errors = np.where(da_pop > pop_thresh, da_errors, 0)
        elif isinstance(pop_thresh, (xr.DataArray)):
            t0, t1 = str(ds.time.data[0]), str(ds.time.data[-1])
            # mask where POP is less than a uniform transform of rand
            p_rand_uniform = pop_thresh.sel(time=slice(t0, t1))
            da_errors = np.where(np.logical_and(da_pop > (1 - p_rand_uniform),
                                                (da_pop > pop_thresh)),
                                 da_errors, 0)
        else:
            raise TypeError('cannot apply POP mask with %s' % type(pop_thresh))

    return da_errors


def process_gard_output(se, scen, periods, obs_ds, rand_ds,
                        template=None, obs_mask=None,
                        calendar='standard',
                        variables=['pcp', 't_mean', 't_range'],
                        roots={'pcp': 3., 't_mean': 1, 't_range': 1},
                        rand_vars={'pcp': 'p_rand', 't_mean': 't_rand',
                                   't_range': 't_rand'},
                        quantile_mapping=False):

    if not isinstance(obs_ds, xr.Dataset):
        # we'll assume that obs_ds is a string/path/or something that
        # xr.open_dataset can open, if not, we'll raise an error right away
        obs_ds = xr.open_dataset(obs_ds)
    if not isinstance(rand_ds, xr.Dataset):
        # we'll assume that obs_ds is a string/path/or something that
        # xr.open_dataset can open, if not, we'll raise an error right away
        rand_ds = xr.open_dataset(rand_ds)

    ds_out = xr.Dataset()
    for var in variables:
        root = roots[var]
        rand_var = rand_vars[var]
        ds_list = []
        for drange in periods:

            pre = template.format(se=se, drange=drange, scen=scen)

            ds = xr.open_dataset(pre + '{}.nc'.format(var))
            ds.merge(xr.open_dataset(pre + '{}_errors.nc'.format(var)),
                     inplace=True)
            try:
                ds.merge(xr.open_dataset(pre + '{}_logistic.nc'.format(var)),
                         inplace=True)
                is_precip = True
            except FileNotFoundError:
                is_precip = False

            ds = make_gard_like_obs(ds, obs_ds)

            units = _get_units_from_drange(drange)
            ds['time'] = nctime_to_nptime(
                num2date(np.arange(0, ds.dims['time']),
                         units, calendar=calendar))

            ds_list.append(ds)
        ds = xr.concat(ds_list, dim='time')

        if se is not 'pass_through':
            # TODO: add additional args
            if is_precip:
                pop_thresh = rand_ds['p_rand_uniform']
            else:
                pop_thresh = None
            da_errors = add_random_effect(ds, rand_ds[rand_var],
                                          var=var, root=root,
                                          is_precip=is_precip,
                                          pop_thresh=pop_thresh)
        else:
            da_errors = ds[var]

        t0, t1 = str(da_errors.data[0]), str(da_errors.data[-1])

        if quantile_mapping:
            # offset zeros in GARD data for precip
            # TODO: update this with new dask/xarray where
            if is_precip:
                da_errors.data = np.where(
                    da_errors.data > 0,  # where precip is non-zero
                    da_errors.data + 1,  # add 1
                    rand_ds['p_rand_uniform'].sel(time=slice(t0, t1)).data)

            # Quantile mapping
            if isinstance(da_errors.data, dask.array.Array):
                da_errors = np.map_blocks(quantile_mapping, da_errors.data,
                                          obs_ds[var].data)
            else:
                da_errors = quantile_mapping(da_errors, obs_ds[var],
                                             mask=obs_mask)

        ds_out[var] = da_errors
        ds_out[var].attrs = attrs[var]
        ds_out[var].encoding = encoding[var]

    if 't_range' in variables and 't_mean' in variables:
        ds_out['t_min'] = ds['t_mean'] - 0.5 * ds['t_range']
        ds_out['t_max'] = ds['t_mean'] + 0.5 * ds['t_range']

    if obs_mask:
        ds_out = ds_out.where(obs_mask)

    # Add metadata
    ds_out['time'].encoding['calendar'] = calendar
    ds_out['time'].encoding['units'] = units

    return ds_out

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

    print(config)


if __name__ == 'main':
    command_line_tool()
