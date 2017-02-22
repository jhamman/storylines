#!/usr/bin/env python
import argparse
import glob
import os.path
import itertools
from dateutil.relativedelta import relativedelta
import pprint
from collections import namedtuple
import warnings
import shutil

import numpy as np
import pandas as pd
import xarray as xr

from tonic.io import read_configobj as read_config

import logging
import logging.config


def set_logger(name='logname', loglvl='DEBUG'):
    """Set up logger"""

    loglvl_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO,
                   'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
                   'CRITICAL': logging.CRITICAL}
    logger = logging.getLogger(name)
    logger.setLevel(loglvl_dict[loglvl])
    ch = logging.StreamHandler()
    ch.setLevel(loglvl_dict[loglvl])
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s ' +
                                  '- %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def replace_var_pythonic_config(src, dst, header=None, **kwargs):
    ''' Python style ASCII configuration file from src to dst. Dost noe remove
    comments or empty lines. Replace keywords in brackets with variable values
    in **kwargs dict. '''
    with open(src, 'r') as fsrc:
        with open(dst, 'w') as fdst:
            lines = fsrc.readlines()
            if header is not None:
                fdst.write(header)
            for line in lines:
                line = line.format(**kwargs)
                fdst.write(line)
    file_chmod(dst)


def file_chmod(infile, mode='664'):
    '''Changes file privileges with default of -rw-rw-r--. Convert mode from
    string to base-8  to be compatible with python 2 and python 3. Checks
    for file ownership before attempting to run chmod. '''
    if os.stat(infile).st_uid == os.getuid():
        os.chmod(infile, int(mode, 8))

pp = pprint.PrettyPrinter(indent=4)

FILELIST_TEMPLATE = \
    'gard_filelist.{dset}.{id}.{scenario}.{drange}.{var}.txt'
NAMELIST_TEMPLATE = \
    'gard_namelist.{setname}.{dset}.{id}.{scenario}.{drange}.{var}.nml'
OUTPUT_TEMPLATE = \
    'gard_output.{setname}.{dset}.{id}.{scenario}.{drange}.'

CHECK_TIMEVARS = True

CUBE_ROOT_TRANSFORM = 3
FIFTH_ROOT_TRANSFORM = 4
NO_TRANSFORM = 0
QUANTILE_TRANSFORM = 1

NO_NORM = 0
SELF_NORM = 1
TRAIN_NORM = 2

# TODO: Move to config file
p_transform = CUBE_ROOT_TRANSFORM
TRANSFORM = {'TMP_2maboveground': NO_TRANSFORM,
             'PRES_meansealevel': NO_TRANSFORM,
             'APCP_surface': p_transform,
             'pcp': p_transform,
             'pr': p_transform,
             't_mean': NO_TRANSFORM,
             'tmp_2m': NO_TRANSFORM,
             'apcp_sfc': p_transform,
             'T2': NO_TRANSFORM,
             'tas': NO_TRANSFORM,
             'tasmax': NO_TRANSFORM,
             'tasmin': NO_TRANSFORM,
             'T2max': NO_TRANSFORM,
             'T2min': NO_TRANSFORM,
             'PREC_ACC_NC': p_transform,
             'PREC_ACC_C': p_transform,
             'U': NO_TRANSFORM,
             'V': NO_TRANSFORM,
             'PSFC': NO_TRANSFORM,
             'TREFHT': NO_TRANSFORM,
             'PRECT': p_transform,
             'UBOT': NO_TRANSFORM,
             'VBOT': NO_TRANSFORM,
             'PSL': NO_TRANSFORM,
             'PRECC': p_transform,
             'PRECL': p_transform,
             '2T_GDS4_SFC': NO_TRANSFORM,
             'TP_GDS4_SFC': p_transform,
             '10U_GDS4_SFC': NO_TRANSFORM,
             '10V_GDS4_SFC': NO_TRANSFORM,
             'MSL_GDS4_SFC': NO_TRANSFORM,
             'CP_GDS4_SFC': p_transform,
             'LSP_GDS4_SFC': p_transform,
             'T_MEAN': NO_TRANSFORM,
             'T_RANGE': NO_TRANSFORM,
             't_range': NO_TRANSFORM,
             'PREC_TOT': p_transform,
             }

TRAINVARMAP = {'TREFHT': '2T_GDS4_SFC',
               'PRECT': 'TP_GDS4_SFC',
               'UBOT': '10U_GDS4_SFC',  # NOTE: Check that these are equivalent
               'VBOT': '10V_GDS4_SFC',  # NOTE: Check that these are equivalent
               'PSL': 'MSL_GDS4_SFC',
               'PRECC': 'CP_GDS4_SFC',
               'PRECL': 'LSP_GDS4_SFC'}

defaults = dict(n_analogs=200,
                pure_regression=False,
                pure_analog=False,
                analog_regression=True,
                sample_analog=False,
                pass_through=False,
                timezone_offset=0,
                weight_analogs=True,
                logistic_from_analog_exceedance=False,
                normalization_method=SELF_NORM)

kFILL_VALUE = -9999
LOGISTIC_THRESH = {'pcp': 0,
                   'pr': 0,
                   'tas': kFILL_VALUE,
                   'tasmin': kFILL_VALUE,
                   'tasmax': kFILL_VALUE,
                   't_mean': kFILL_VALUE,
                   't_range': kFILL_VALUE}

# TODO: add mechanisim for timezone offset
GARD_TIMEFORMAT = '%Y-%m-%d %H:%M:%S'

filelistkey = namedtuple('filelistkey',
                         ('dset', 'drange', 'id', 'scenario', 'var'))


# -------------------------------------------------------------------------#
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
    parser.add_argument('config_file', metavar='config_file',
                        help='configuration file for downscaling matrix')
    parser.add_argument('--outfile', metavar='outfile', default='namelist.txt',
                        help='output file downscaling namelists')
    args = parser.parse_args()

    # Read configuration file into a dictionary
    config = read_config(args.config_file)
    outfile = args.outfile

    log_level = config['Options']['LogLevel']
    chunk_years = relativedelta(years=int(config['Options']['ChunkYears']))

    # Set up logging for messaging
    logger = set_logger(os.path.splitext(os.path.split(__file__)[-1])[0],
                        log_level)

    logger.info('Downscaling Configuration Options:')
    pp.pprint(config)

    # Define variables from configuration file
    # cores = config['Options']['Cores']

    # create directories if they don't exist yet
    data_dir = config['Options']['DataDir']
    filelist_dir = os.path.join(data_dir, 'gard_filelists')
    namelist_dir = os.path.join(data_dir, 'gard_namelists')
    [os.makedirs(d, exist_ok=True) for d in
        [data_dir, filelist_dir, namelist_dir]]
    # if outfile is default, put in data_dir
    if outfile == 'namelist.txt':
        outfile = os.path.join(data_dir, outfile)

    # GARD namelist template
    namelist_template = config['Options']['NamelistTemplate']

    prediction_sets = config['Sets']

    # Make training file lists
    file_lists = {}
    file_lists_len = {}
    set_dirs = {}
    namelists = []

    for dataset, dset_config in config['Datasets'].items():
        predict_ranges = {}
        gcms = list_like(dset_config['GCMs'])

        if isinstance(gcms[0], int):
            # work around for these being cast as ints
            gcms = ['{0:03}'.format(i) for i in gcms]
            print(gcms)

        train_calendar = dset_config.get('TrainCalendar', None)
        obs_calendar = config['Obs_Dataset'].get('ObsCalendar', None)

        for setname, set_config in prediction_sets.items():

            logger.info('Creating configuration files for set:  %s', setname)

            # Make set directory
            set_dirs[setname] = os.path.join(data_dir, dataset, setname)
            os.makedirs(set_dirs[setname], exist_ok=True)

            # Training/prediction/obs variables
            obs_vars = list_like(set_config['ObsVars'])
            logger.info('Obs Vars:  %s', obs_vars)

            # For now, we can assume the training / prediction variable names
            # are he same
            vars_list = []
            for var in obs_vars:
                vars_list.append(set_config[var])
            vars_list = list(set(flatten(vars_list)))
            logger.info('Variables: %s', vars_list)

            # Get scenario to process
            logger.debug(dset_config['PredictPattern'])
            scenarios = dset_config['scenario']
            train_range = tuple(dset_config['TrainPeriod'])
            transform_range = tuple(dset_config['TransformPeriod'])
            transform_scen = dset_config['TransformScenario']
            logger.info('training range: %s', train_range)

            for scen, drange in scenarios.items():
                predict_ranges[scen] = tuple(drange)

            # Make the filelists
            for var in obs_vars:
                # Obs filelist
                key = filelistkey(dset=dataset, var=var,
                                  id='obs', drange=train_range,
                                  scenario='training')

                file_lists[key], file_lists_len[key] = make_filelist(
                    key, config['Obs_Dataset']['ObsInputPattern'],
                    prefix=filelist_dir, calendar=obs_calendar)

            for var in vars_list:
                # training filelist
                key = filelistkey(dset=dataset, var=TRAINVARMAP.get(var, var),
                                  id='training', drange=train_range,
                                  scenario='training')
                file_lists[key], file_lists_len[key] = make_filelist(
                    key, dset_config['TrainPattern'], prefix=filelist_dir,
                    calendar=train_calendar)

                # prediction filelists
                for gcm, scen in itertools.product(gcms, scenarios):
                    for drange in get_drange_chunks(
                            predict_ranges[scen], max_chunk_size=chunk_years):
                        key = filelistkey(dset=dataset, var=var, id=gcm,
                                          drange=drange, scenario=scen)
                        file_lists[key], file_lists_len[key] = make_filelist(
                            key, dset_config['PredictPattern'],
                            prefix=filelist_dir,
                            transform_range=transform_range,
                            transform_scen=transform_scen,
                            calendar=config['Calendars'].get(
                                gcm, config['Calendars'].get('all', None)))

        for gcm, var, setname in itertools.product(
                gcms, obs_vars, prediction_sets):

            for scen in dset_config['scenario']:
                # drange_str = _tslice_to_str(predict_ranges[scen])
                for drange in get_drange_chunks(predict_ranges[scen],
                                                max_chunk_size=chunk_years):
                    drange_str = _tslice_to_str(drange)
                    # Make GARD namelist
                    namelist = os.path.join(
                        namelist_dir, NAMELIST_TEMPLATE.format(
                            setname=setname, var=var, id=gcm, dset=dataset,
                            scenario=scen, drange=drange_str))

                    kwargs = defaults.copy()
                    kwargs.update(prediction_sets[setname])

                    # Set the downscaling mode
                    mode = prediction_sets[setname]['Mode']
                    for m in ['pure_regression', 'analog_regression',
                              'pure_analog', 'pass_through']:
                        kwargs[m] = (m == mode)

                    # Now, add a bunch of computed variables
                    # parameters section
                    out_dir = os.path.join(set_dirs[setname], drange_str)
                    os.makedirs(out_dir, exist_ok=True)

                    kwargs['output_file_prefix'] = os.path.join(
                        out_dir, OUTPUT_TEMPLATE.format(
                            setname=setname, drange=drange_str, dset=dataset,
                            id=gcm, scenario=scen))
                    kwargs['start_date'] = pd.to_datetime(
                        drange[0]).strftime(GARD_TIMEFORMAT)
                    kwargs['end_date'] = pd.to_datetime(
                        drange[1]).strftime(GARD_TIMEFORMAT)
                    kwargs['start_train'] = pd.to_datetime(
                        train_range[0]).strftime(GARD_TIMEFORMAT)
                    kwargs['end_train'] = pd.to_datetime(
                        train_range[1]).strftime(GARD_TIMEFORMAT)
                    kwargs['start_transform'] = pd.to_datetime(
                        transform_range[0]).strftime(GARD_TIMEFORMAT)
                    kwargs['end_transform'] = pd.to_datetime(
                        transform_range[1]).strftime(GARD_TIMEFORMAT)

                    kwargs['logistic_threshold'] = LOGISTIC_THRESH[var]

                    # training_parameters section
                    var_list = list_like(prediction_sets[setname][var])
                    train_var_list = [TRAINVARMAP.get(v, v)
                                      for v in var_list]
                    train_kwargs = dict(dset=dataset, drange=train_range,
                                        id='training', scenario='training', )
                    kwargs['train_nfiles'] = file_lists_len[filelistkey(
                        var=train_var_list[0], **train_kwargs)]
                    kwargs['train_nvars'] = len(train_var_list)
                    kwargs['train_vars'] = ','.join(train_var_list)
                    kwargs['train_transform'] = get_transform_str(
                        train_var_list, mode)
                    kwargs['train_filelists'] = _make_filelist_str(
                        file_lists, train_var_list, **train_kwargs)
                    kwargs['train_calendar'] = train_calendar

                    # prediction_parameters section
                    predict_kwargs = dict(dset=dataset,
                                          drange=drange,
                                          id=gcm, scenario=scen)
                    kwargs['predict_nfiles'] = file_lists_len[filelistkey(
                        var=var_list[0], **predict_kwargs)]
                    nvars = len(var_list)
                    kwargs['predict_nvars'] = nvars
                    kwargs['predict_vars'] = ','.join(var_list)
                    transformations = ','.join(
                        [str(QUANTILE_TRANSFORM)] * nvars)
                    kwargs['transformations'] = transformations
                    kwargs['predict_transform'] = get_transform_str(var_list,
                                                                    mode)
                    kwargs['predict_filelists'] = _make_filelist_str(
                        file_lists, var_list, **predict_kwargs)
                    kwargs['predict_calendar'] = config['Calendars'].get(
                        gcm, config['Calendars'].get('all', None))

                    # obs_parameters section
                    obs_kwargs = dict(dset=dataset,
                                      drange=train_range,
                                      id='obs', scenario='training')
                    kwargs['obs_nvars'] = 1
                    key = filelistkey(
                        var=var, **obs_kwargs)
                    kwargs['obs_nfiles'] = file_lists_len[filelistkey(
                        var=var, **obs_kwargs)]
                    kwargs['obs_vars'] = var
                    kwargs['obs_transform'] = TRANSFORM[var]
                    kwargs['obs_filelists'] = _make_filelist_str(
                        file_lists, [var], **obs_kwargs)
                    kwargs['obs_calendar'] = obs_calendar

                    # Special cases for "pass_through" option:
                    if mode == 'pass_through':
                        kwargs['train_transform'] = NO_TRANSFORM
                        kwargs['predict_transform'] = NO_TRANSFORM
                        kwargs['obs_transform'] = NO_TRANSFORM
                        kwargs['normalization_method'] = NO_NORM
                        kwargs['transformations'] = NO_TRANSFORM

                    # Store the namelist filepath for execuation
                    namelists.append(namelist)
                    # Create the namelist by filling in the template
                    replace_var_pythonic_config(namelist_template, namelist,
                                                **kwargs)

    # Write file of namelists
    logger.info('writing outfile %s', outfile)
    with open(outfile, 'w') as f:
        f.writelines('\n'.join(namelists))

    config = os.path.join(data_dir, os.path.basename(args.config_file))
    logger.info('writing config file %s', config)
    shutil.copyfile(args.config_file, config)

    return


def get_filelist(pattern, date_range=None, timevar='time', calendar=None):
    '''given a glob pattern, return a list of files between daterange'''
    files = glob.glob(pattern)

    if date_range is not None:
        date_range = pd.to_datetime(list(date_range)).values
        sublist = []
        for f in files:
            try:
                kwargs = dict(mask_and_scale=False, concat_characters=False,
                              decode_coords=False)
                if calendar:
                    ds = xr.open_dataset(f, decode_cf=False,
                                         decode_times=False, **kwargs)

                    if (('XTIME' in ds) and not
                            ('calendar' not in ds['XTIME'].attrs)):
                        ds['XTIME'].attrs['calendar'] = calendar

                    elif 'calendar' not in ds[timevar].attrs:
                        ds[timevar].attrs['calendar'] = calendar
                    # else decode using callendar attribute in file

                    ds = xr.decode_cf(ds, decode_times=True, **kwargs)
                else:
                    ds = xr.open_dataset(f, decode_cf=True, decode_times=True,
                                         **kwargs)
            except Exception as e:
                warnings.warn('failed to open {}: {}'.format(f, e))

            try:
                ds[timevar] = ds['XTIME']
            except KeyError:
                pass

            if CHECK_TIMEVARS:
                try:
                    check_times(ds[timevar].values, f=f)
                except ValueError as e:
                    warnings.warn(
                        'time check raised an error for file %s: %s' % (f, e))

            start = ds[timevar].values[0]
            end = ds[timevar].values[-1]
            ds.close()
            if (((start >= date_range[0]) and (start <= date_range[1])) or
                    ((end >= date_range[0]) and (end <= date_range[1])) or
                    (start <= date_range[0]) and (end >= date_range[1])):
                sublist.append(f)
        files = sublist
    files.sort()
    return files


def check_times(times, min_delta=np.timedelta64(1, 's'),
                max_delta=np.timedelta64(49, 'h'), f=None):
    '''QC time variable from a netcdf file.

    Raise a ValueError if a check is violated.

    Current checks:
    1) Timestamps must be monotonic (increasing)
    2) Maximum timestep size must less than a certain threshold (max_delta)
    '''
    diffs = np.diff(times)
    negs = np.nonzero(diffs < min_delta)[0]
    too_big = np.nonzero(diffs > max_delta)[0]

    if len(negs) > 0:
        datestamps = pd.to_datetime(times[negs[0]-1: negs[0]+2])
        warnings.warn('%s: times are not monotonically increasing. '
                      'Found timestamp < %s at %s, first example: '
                      '%s' % (f, min_delta, negs, datestamps))
    if len(too_big) > 0:
        datestamps = pd.to_datetime(times[too_big[0]-1: too_big[0]+2])
        warnings.warn('%s: found a timestep where its delta is too '
                      'large (greater than %s) at %s, first example: '
                      '%s' % (f, max_delta, too_big, datestamps))


def write_filelist(fname, files, check=False):
    '''write GARD style file list'''

    if check:
        for f in files:
            if not os.path.isfile(f):
                raise ValueError('{} is not a valid file name'.format(f))
    with open(fname, 'w') as f:
        for fil in files:
            f.write('"{}"\n'.format(fil))


def _tslice_to_str(s, timeformat='%Y%m%d'):
    '''
    return a string that represents the time period in s, which should be a
    length 2 iterable and have datetime objects in both positions.  If both
    elements of s are the same date, only one will be converted to a string.
    '''
    assert len(s) == 2
    ts = pd.to_datetime(list(s))
    if ts[0] == ts[1]:
        return ts[0].strftime(timeformat)
    return '{}-{}'.format(ts[0].strftime(timeformat),
                          ts[1].strftime(timeformat))


def _lines_in_file(fname):
    '''this is basically equivalent to `wc -l` in unix'''
    i = 0
    try:
        with open(fname) as f:
            for i, _ in enumerate(f):
                pass
        i += 1
    except UnboundLocalError as e:
        UnboundLocalError('failed to get number of lines in %s' % fname)
    return i


def _make_filelist_str(flists, variables, **kwargs):
    '''
    helper function to create a list of file paths with quotes around each
    path name
    '''
    l = []
    for var in variables:
        l.append('"{path}"'.format(
            path=flists[filelistkey(var=var, **kwargs)]))
    s = ','.join(l)
    return s


def make_glob_pattern(template, drange, **kwargs):
    '''helper function to infer a glob pattern for the data store'''

    start, stop = pd.to_datetime(list(drange))

    # replace date place holders
    kwargs.update({'yyyy': '????',
                   'mm': '??',
                   'dd': '??',
                   'yyyymmdd': '????????'})

    same = False
    if start.year == stop.year:
        # start and stop year is the same
        kwargs['yyyy'] = '{0:04d}'.format(start.year)
        same = True
    if same and (start.month == stop.month):
        # start and stop month is the same
        kwargs['mm'] = '{0:02d}'.format(start.month)
    else:
        same = False
    if same and (start.day == stop.day):
        # start and stop day is the same
        kwargs['dd'] = '{0:02d}'.format(start.day)

    kwargs['yyyymmdd'] = '{}{}{}'.format(kwargs['yyyy'], kwargs['mm'],
                                         kwargs['dd'])

    return template.format(**kwargs)


def flatten(container):
    '''helper function to flatten nested lists'''
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def list_like(i):
    '''helper function to ensure that a variable is a list-like object'''
    if isinstance(i, str) or not hasattr(i, '__iter__'):
        return [i]
    return i


def get_transform_str(var_list, mode):
    '''helper function to determine transform types'''
    if mode == 'pass_through':
        return ','.join(['0'] * len(var_list))
    else:
        return ','.join(
            map(str, [TRANSFORM[v] for v in var_list]))


def make_filelist(key, template, prefix='', transform_range=None,
                  transform_scen='hist', calendar=None):
    # create the namelist filename
    drange_str = _tslice_to_str(key.drange)
    filename = os.path.join(
        prefix, FILELIST_TEMPLATE.format(
            id=key.id, dset=key.dset, scenario=key.scenario,
            var=key.var, drange=drange_str))
    # If the filelist doesn't already exist, "intellegently"
    # create a new one
    list_len = -1
    pattern = 'not set'
    if os.path.isfile(filename):
        list_len = _lines_in_file(filename)

    if list_len < 1:
        trans_flist = []

        drange = list(key.drange)

        if transform_range is not None:
            # prepend training data that preceeds the prediction period
            tran_dates = pd.to_datetime(list(transform_range))
            predict_dates = pd.to_datetime(list(drange))

            d0 = min(tran_dates[0], predict_dates[0])
            d1 = min(tran_dates[1], predict_dates[0])

            if tran_dates[1] > drange[1]:
                drange[1] = tran_dates[1]

            if d0 < d1:
                trans_pattern = make_glob_pattern(template,
                                                  (d0, d1),
                                                  scenario=transform_scen,
                                                  id=key.id, var=key.var)
                trans_flist = get_filelist(trans_pattern,
                                           date_range=(d0, d1),
                                           calendar=calendar)

        pattern = make_glob_pattern(template,
                                    drange, scenario=key.scenario,
                                    id=key.id, var=key.var)

        flist = get_filelist(pattern, date_range=drange, calendar=calendar)
        flist = list(unique(trans_flist + flist))  # combine and drop dups

        if len(flist) > 0:
            write_filelist(filename, flist, check=False)

        list_len = len(flist)

    if list_len < 1:
        warnings.warn('filelist length is {} for {}. Pattern was: {}'.format(
            list_len, key, pattern))

    return filename, list_len


def unique(ll):
    '''equivalent to list(set(ll)) but keeps order'''
    new = []
    for l in ll:
        if l not in new:
            new.append(l)
    return new


def get_drange_chunks(dates, max_chunk_size=relativedelta(years=5)):
    '''helper function that returns a series of dates
    that span dates[0]-dates[1]'''

    # Coerce dates to datetime objects
    dates = pd.to_datetime(dates)

    # start with the begining through (begining + max_chunk_size - 1day)
    beg = dates[0]
    end = dates[0] + max_chunk_size - relativedelta(days=1)

    chunked_dates = []
    while True:
        if end >= dates[1]:
            # we've reached the end
            end = dates[1]
            chunked_dates.append((beg, end))
            break
        else:
            chunked_dates.append((beg, end))
        # move everything forward one chunk
        beg += max_chunk_size
        end += max_chunk_size

    return chunked_dates


# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    main()
