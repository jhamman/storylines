from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import cbrt as _cbrt
from scipy.stats import norm as norm

from netCDF4 import num2date
from xarray.conventions import nctime_to_nptime

from storylines.tools.gard_utils import (read_config, get_drange_chunks,
                                         list_like, _tslice_to_str)
from storylines.tools.encoding import attrs, encoding, make_gloabl_attrs

PASS_THROUGH_PCP_MULT = 24  # units mm/hr to mm/day
KELVIN = 273.15

reindex_dates = {
    'hist': xr.DataArray(pd.date_range('1950-01-01', '2005-12-31', freq='D'),
                         dims='time', name='time'),
    'rcp45': xr.DataArray(pd.date_range('2006-01-01', '2099-12-31', freq='D'),
                          dims='time', name='time'),
    'rcp85': xr.DataArray(pd.date_range('2006-01-01', '2099-12-31', freq='D'),
                          dims='time', name='time')}

raw_chunks = {'x': 58, 'y': 56}


def cbrt(data):
    '''parallized version of scipy cube root'''
    return xr.apply_ufunc(_cbrt, data, dask='parallelized',
                          output_dtypes=[data.dtype])


def ppf(data):
    '''parallized version of scipy cube root'''
    return xr.apply_ufunc(norm.ppf, data, dask='parallelized',
                          output_dtypes=[data.dtype])


def _get_units_from_drange(d):

    dsplit = d.split('-')

    if len(dsplit) <= 2:
        # e.g. `d = 19200101-19291231` or just `d = 19200101`
        return 'days since {0}-{1}-{2}'.format(d[:4], d[4:6], d[6:8])
    else:
        # e.g. `d = 1920-01-01`
        return 'days since {:04}-{:02}-{:02}'.format(*map(int, dsplit))


def make_gard_like_obs(ds, domain):

    ds = ds.rename({'x': 'lon', 'y': 'lat'})
    ds.coords['lon'] = domain['lon']
    ds.coords['lat'] = domain['lat']

    return ds


def get_rand_ds(rand_file, chunks=None, calendar='standard',
                units='days since 1950-01-01'):
    rand_ds = xr.open_dataset(rand_file, chunks=chunks)
    rand_ds['time'] = nctime_to_nptime(
        num2date(np.arange(0, rand_ds.dims['time']), units,
                 calendar=calendar))

    return rand_ds


def add_random_effect(ds, da_normal, da_uniform=None, var=None, root=1.):
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

    t0, t1 = str(ds.indexes['time'][0]), str(ds.indexes['time'][-1])

    da = ds[var]
    da_errors = ds['{}_error'.format(var)]

    exceedence_var = '{}_exceedence_probability'.format(var)
    if exceedence_var in ds:

        # Get the array of uniform errors
        r_uniform = da_uniform.sel(time=slice(t0, t1))

        # Get the exceedence variable (e.g. POP)
        da_ex = ds[exceedence_var]

        # Mask where precip occurs
        mask = r_uniform > (1 - da_ex)

        # Rescale the uniform distribution
        new_uniform = (r_uniform - (1 - da_ex)) / da_ex

        # Get the normal distribution equivalent of new_uniform
        r_normal = ppf(new_uniform)
    else:
        mask = None
        r_normal = da_normal.sel(time=slice(t0, t1))

    # apply the errors in transform space
    if root == 1.:
        da_errors = da + (da_errors * r_normal)
    elif root == 3:
        da_errors = (cbrt(da) + (da_errors * r_normal)) ** root
    else:
        da_errors = ((da ** (1./root)) + (da_errors * r_normal)) ** root

    # if this var used logistic regression, apply that mask now
    if mask is not None:
        valids = xr.ufuncs.logical_or(mask, da_errors >= 0)
        da_errors = da_errors.where(valids, 0)
    return da_errors


def process_gard_output(se, scen, periods, obs_ds, rand_ds,
                        template=None, obs_mask=None,
                        calendar='standard',
                        variables=['pcp', 't_mean', 't_range'],
                        rename_vars=None,
                        roots={'pcp': 3., 't_mean': 1, 't_range': 1},
                        rand_vars={'pcp': 'p_rand', 't_mean': 't_rand',
                                   't_range': 't_rand'},
                        chunks=None):

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

    Returns
    -------
    ds : xr.Dataset
        Post processed GARD dataset

    See Also
    --------
    add_random_effect
    make_gard_like_obs
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

    if rename_vars is not None:
        rename_dict = rename_vars.copy()

    for var in variables:
        exceedence_var = False
        ds_list = []
        for drange in periods:

            drange = _tslice_to_str(drange)
            pre = template.format(se=se, drange=drange, scen=scen)

            fname = pre + '{}.nc'.format(var)
            ds = xr.open_dataset(fname, chunks=raw_chunks)
            fname = pre + '{}_errors.nc'.format(var)
            ds = ds.merge(xr.open_dataset(fname, chunks=raw_chunks))
            try:
                fname = pre + '{}_logistic.nc'.format(var)
                ds.merge(xr.open_dataset(fname, chunks=raw_chunks))
                exceedence_var = True
            except (FileNotFoundError, OSError):
                pass

            ds = make_gard_like_obs(ds, obs_ds)
            units = _get_units_from_drange(drange)
            ds['time'] = pd.DatetimeIndex(
                nctime_to_nptime(num2date(np.arange(0, ds.dims['time']),
                                          units, calendar=calendar)))

            ds_list.append(ds)

        ds = xr.concat(ds_list, dim='time')

        for v in ['mask', 'elevation']:
            if v in obs_ds:
                ds[v] = obs_ds[v]

        # ds = ds.reindex_like(reindex_dates[scen])
        ds = ds.chunk(chunks=dict(time=ds.dims['time'], **chunks))
        # ds[var] = ds[var].interpolate_na(
        #     dim='time', kind='index', fill_value='extrapolate')
        # calendar = 'standard'

        if rename_vars is not None and var in rename_vars:
            rename_dict = {}
            rename_dict[var] = rename_vars[var]
            ervar = '{}_error'.format(rename_vars[var])
            rename_dict['{}_error'.format(var)] = ervar
            if exceedence_var:
                exvar = '{}_exceedence_probability'.format(rename_vars[var])
                rename_dict['{}_exceedence_probability'.format(var)] = exvar
            ds = ds.rename(rename_dict)
            var = rename_dict[var]

        t0 = str(ds.coords['time'].data[0])
        t1 = str(ds.coords['time'].data[-1])
        rand_ds = rand_ds.sel(time=slice(t0, t1))

        if se != 'pass_through':
            root = roots[var]
            rand_var = rand_vars[var]
            da_errors = add_random_effect(ds, rand_ds[rand_var],
                                          rand_ds['p_rand_uniform'],
                                          var=var, root=root)

        else:
            if var == 'pcp':
                da_errors = ds[var] * PASS_THROUGH_PCP_MULT
            elif var == 't_mean':
                da_errors = ds[var] - KELVIN
            else:
                da_errors = ds[var]

        # QC data after adding random effects
        if var in ['pcp', 't_range']:
            da_errors = xr.where(da_errors > 0., da_errors, 0)

        ds_out[var] = da_errors
        ds_out[var].attrs.update(attrs[var])
        ds_out[var].encoding.update(encoding[var])

    # mask sure the dataset is properly masked
    if obs_mask is not None:
        ds_out = ds_out.where(obs_mask)
        ds_out['mask'] = obs_mask
        ds_out['mask'].attrs = attrs['mask']
        ds_out['mask'].encoding.update(encoding['mask'])

    # Add metadata
    ds_out['time'].encoding['calendar'] = calendar
    ds_out['time'].encoding['units'] = units

    return ds_out


def run(config_file, gcms=None, sets=None, variables=None,
        return_processed=False):

    config = read_config(config_file)

    # Define variables from configuration file
    if not variables:
        variables = list_like(config['PostProc']['variables'])
    roots = config['PostProc']['roots']
    rand_vars = config['PostProc']['rand_vars']
    rename_vars = config['PostProc']['rename_vars']
    rand_file = config['PostProc']['rand_file']

    chunks = config['PostProc'].get('chunks', None)

    # create directories if they don't exist yet
    data_dir = config['Options']['DataDir']
    processed_dir = os.path.join(data_dir, 'post_processed')
    for d in [data_dir, processed_dir]:
        os.makedirs(d, exist_ok=True)

    out = {}  # for when return_processed==True

    chunk_years = relativedelta(years=int(config['Options']['ChunkYears']))

    # Get obs dataset
    obs_ds = xr.open_dataset(config['ObsDataset']['ObsInputPattern'],
                             chunks=chunks)
    obs_mask = obs_ds['mask']

    for dataset, dset_config in config['Datasets'].items():

        if not sets:
            run_sets = list_like(dset_config.get('RunSets',
                                 config['Sets'].keys()))
        else:
            run_sets = sets
        if not gcms:
            run_gcms = list_like(dset_config['GCMs'])
        else:
            # get the intersection of gcms with this dataset's config
            run_gcms = list_like(set(gcms).intersection(
                set(list_like(dset_config['GCMs']))))
            run_gcms = list(run_gcms)
            if not gcms:
                print('skipping because this gcm isnt in the dataset config')
                continue
        if run_gcms and isinstance(run_gcms[0], int):
            # work around for these being cast as ints
            run_gcms = ['{0:03}'.format(i) for i in run_gcms]

        for gcm, setname in itertools.product(run_gcms, run_sets):
            print(gcm, setname)
            calendar = config['Calendars'].get(
                gcm, config['Calendars'].get('all', 'standard'))

            rand_calendar = 'standard'  # TODO: get this from PostProc dict

            # Get random dataset
            rand_ds = get_rand_ds(rand_file, chunks=chunks,
                                  calendar=rand_calendar)

            for scen, drange in dset_config['scenario'].items():

                template = os.path.join(
                    data_dir, dataset,
                    '{se}/{drange}/gard_output.{se}.%s.%s.{scen}.{drange}.' %
                    (dataset, gcm))

                out_template = os.path.join(
                    processed_dir,
                    'gard_output.{se}.%s.%s.{scen}.{drange}.' % (dataset, gcm))

                periods = get_drange_chunks(tuple(drange),
                                            max_chunk_size=chunk_years)

                drange = '{}-{}'.format(periods[0][0].strftime('%Y%m%d'),
                                        periods[-1][1].strftime('%Y%m%d'))
                pre = out_template.format(se=setname, drange=drange,
                                          scen=scen)
                # mm_fname_out = pre + 'mm.nc'
                dm_fname_out = pre + 'dm.nc'

                if not return_processed and os.path.isfile(dm_fname_out):
                    continue

                ds_out = process_gard_output(
                    setname, scen, periods, obs_ds, rand_ds,
                    template=template,
                    obs_mask=obs_mask,
                    calendar=calendar,
                    variables=variables,
                    rename_vars=rename_vars,
                    roots=roots,
                    rand_vars=rand_vars,
                    chunks=chunks)

                out_encoding = {}
                for key in ds_out.variables:
                    out_encoding[key] = ds_out[key].encoding
                    out_encoding[key].update(encoding.get(key, {}))

                ds_out.attrs = make_gloabl_attrs(
                    title='Post-processed GARD output downscaled dataset')

                if return_processed:
                    out[dm_fname_out] = ds_out
                else:
                    print('writing %s' % dm_fname_out)
                    ds_out.load().to_netcdf(dm_fname_out,
                                            unlimited_dims=['time'],
                                            format='NETCDF4',
                                            encoding=out_encoding,
                                            engine='h5netcdf')

    if return_processed:
        return out
