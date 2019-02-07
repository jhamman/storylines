from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import itertools
from dateutil.relativedelta import relativedelta

# import pandas as pd
import xarray as xr
from scipy.special import cbrt as _cbrt
from scipy.stats import norm as norm

from storylines.tools.gard_utils import (read_config, get_drange_chunks,
                                         list_like, _tslice_to_str)
from storylines.tools.encoding import attrs, encoding, make_gloabl_attrs

PASS_THROUGH_PCP_MULT = 24  # units mm/hr to mm/day
KELVIN = 273.15

raw_chunks = {'x': 58, 'y': 56}
chunks = {'lat': raw_chunks['y'], 'lon': raw_chunks['x']}


def cbrt(data):
    '''parallized version of scipy cube root'''
    return xr.apply_ufunc(_cbrt, data, dask='parallelized',
                          output_dtypes=[data.dtype])


def ppf(data):
    '''parallized version of scipy cube root'''
    return xr.apply_ufunc(norm.ppf, data, dask='parallelized',
                          output_dtypes=[data.dtype])


def tidy_gard(ds):
    lat = ds['lat'].isel(x=0, drop=True).load()
    if 'lon' in lat.coords:
        lat = lat.drop('lon')
    lon = ds['lon'].isel(y=0, drop=True).load()
    if 'lat' in lon.coords:
        lon = lon.drop('lat')
    lon.values[lon.values > 180] -= 360

    ds.coords['lat'] = lat
    ds.coords['lon'] = lon

    ds = ds.rename({'y': 'lat', 'x': 'lon'})

    ds = ds.sortby('time')

    return ds


def get_rand_ds(rand_file, chunks=None, calendar='standard',
                start='1950-01-01', freq='D'):
    rand_ds = xr.open_dataset(rand_file, chunks=chunks)
    rand_ds['time'] = xr.cftime_range(start, periods=rand_ds.dims['time'],
                                      calendar=calendar, freq=freq)

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
    t0 = str(ds.indexes['time'][0].strftime('%Y-%m-%d'))
    t1 = str(ds.indexes['time'][-1].strftime('%Y-%m-%d'))

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
    if root == 1:
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


def process_gard_output(setname, scen, periods, obs_ds, rand_ds,
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
    setname : str
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
    tidy_gard
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
            pre = template.format(se=setname, drange=drange, scen=scen)

            files = [pre + '{}.nc'.format(var),
                     pre + '{}_errors.nc'.format(var)]
            exfile = pre + '{}_logistic.nc'.format(var)
            if os.path.exists(exfile):
                files.append(exfile)
                exceedence_var = True

            ds = xr.open_mfdataset(files, preprocess=tidy_gard).chunk(chunks)

            ds_list.append(ds)

        ds = xr.concat(ds_list, dim='time')

        for v in ['mask', 'elevation']:
            if v in obs_ds:
                ds[v] = obs_ds[v]

        date0 = ds.indexes['time'][0].strftime('%Y-%m-%d')
        ds['time'] = xr.cftime_range(date0, periods=ds.dims['time'],
                                     calendar=calendar, freq='1D')
        ds = ds.chunk(chunks=dict(time=ds.dims['time'], **chunks))

        print('renaming vars', flush=True)
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

        print('clipping random data', flush=True)
        # t0 = str(ds.indexes['time'][0].isoformat())
        # t1 = str(ds.indexes['time'][-1].isoformat())
        t0 = str(ds.indexes['time'][0].strftime('%Y-%m-%d'))
        t1 = str(ds.indexes['time'][-1].strftime('%Y-%m-%d'))
        rand_ds = rand_ds.sel(time=slice(t0, t1))
        # print('rand_ds', t0, t1, rand_ds)

        print('adding random effects', flush=True)
        if setname != 'pass_through':
            root = roots[var]
            rand_var = rand_vars[var]
            da_errors = add_random_effect(ds, rand_ds[rand_var],
                                          rand_ds['p_rand_uniform'],
                                          var=var, root=root)
        else:
            print('converting units', flush=True)
            if var == 'pcp':
                da_errors = ds[var] * PASS_THROUGH_PCP_MULT
            elif var == 't_mean':
                da_errors = ds[var] - KELVIN
            else:
                da_errors = ds[var]

        # QC data after adding random effects
        print('masking after the fact', flush=True)
        if var in ['pcp', 't_range']:
            da_errors = xr.where(da_errors > 0., da_errors, 0)

        ds_out[var] = da_errors
        ds_out[var].attrs.update(attrs[var])
        ds_out[var].encoding.update(encoding[var])

    # mask sure the dataset is properly masked
    print('masking again, also metadata', flush=True)
    if obs_mask is not None:
        ds_out = ds_out.where(obs_mask > 0)
        ds_out['mask'] = obs_mask
        ds_out['mask'].attrs = attrs['mask']
        ds_out['mask'].encoding.update(encoding['mask'])

    # Add metadata
    ds_out['time'].encoding['calendar'] = calendar
    # ds_out['time'].encoding['units'] = units
    print('returning ds: \n%s' % ds_out, flush=True)

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

    # chunks = config['PostProc'].get('chunks', None)

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

            # Get random dataset
            rand_ds = get_rand_ds(rand_file, chunks=chunks,
                                  calendar=calendar)

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

                print('done constructing output dataset', flush=True)

                out_encoding = {}
                for key in variables:
                    out_encoding[key] = ds_out[key].encoding
                    out_encoding[key].update(encoding.get(key, {}))
                print(out_encoding)

                ds_out.attrs = make_gloabl_attrs(
                    title='Post-processed GARD output downscaled dataset')

                if return_processed:
                    out[dm_fname_out] = ds_out
                else:
                    print('dataset:\n', ds_out, flush=True)
                    print('size: %s' % (ds_out.nbytes / 1e9))
                    print('loading data', flush=True)
                    ds_out = ds_out.load(timeout='10s')
                    print('writing %s' % dm_fname_out, flush=True)
                    ds_out.to_netcdf(dm_fname_out,
                                     unlimited_dims=['time'],
                                     encoding=out_encoding)
                    del ds_out

    if return_processed:
        return out
