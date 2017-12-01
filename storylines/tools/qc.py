from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
import pandas as pd


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
