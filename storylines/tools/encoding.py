from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import socket
import time as time_mod
from getpass import getuser

attrs = {'pcp': {'units': 'mm', 'long_name': 'precipitation',
                 'comment': 'random effects applied'},
         't_mean': {'units': 'C', 'long_name': 'air temperature',
                    'comment': 'random effects applied'},
         't_range': {'units': 'C', 'long_name': 'daily air temperature range',
                     'comment': 'random effects applied'},
         't_min': {'units': 'C', 'long_name': 'minimum daily air temperature',
                   'comment': 'random effects applied'},
         't_max': {'units': 'C', 'long_name': 'maximum daily air temperature',
                   'comment': 'random effects applied'},
         'mask': {'long_name': 'domain mask', 'note': 'unitless',
                  'comment': '0 value indicates cell is not active'}}

encoding = {'pcp': {'dtype': 'f4', '_FillValue': -9999},  # 'zlib': True, # 'complevel': 1},
            't_mean': {'dtype': 'f4', '_FillValue': -9999},  # 'zlib': True, # 'complevel': 1},
            't_range': {'dtype': 'f4', '_FillValue': -9999},  # 'zlib': True, # 'complevel': 1},
            't_min': {'dtype': 'f4', '_FillValue': -9999},  # 'zlib': True, # 'complevel': 1},
            't_max': {'dtype': 'f4', '_FillValue': -9999},  # 'zlib': True, # 'complevel': 1},
            'mask': {}}  # 'zlib': True, 'complevel': 3}}


def make_gloabl_attrs(**kwargs):
    attrs = dict(
        history='Created: {}'.format(time_mod.ctime(time_mod.time())),
        institution='National Center for Atmospheric Research',
        source=sys.argv[0],
        Conventions='CF-1.6',
        hostname=socket.gethostname(),
        username=getuser())

    attrs.update(kwargs)
    return attrs
