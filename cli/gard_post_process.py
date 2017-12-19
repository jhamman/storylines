#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket
import click
from dask.distributed import LocalCluster, Client

from storylines.tools.post_process_gard_output import run


@click.command()
@click.option('--sets', multiple=True, default=None,
              help='list of GARD setnames to process')
@click.option('--gcms', multiple=True, default=None,
              help='list of GARD gcms to process')
@click.option('--variables', multiple=True, default=None,
              help='list of GARD variables to process')
@click.option('--scheduler', default='LocalCluster',
              help='path to dask scheduler file')
@click.argument('config')
def cli(config, sets, gcms, variables, scheduler):
    '''Command line interface for running the GARD post-processing tools'''

    if scheduler == 'LocalCluster':
        client = Client(LocalCluster())
    elif scheduler is not None:
        client = Client(scheduler_file=scheduler)
    else:
        client = None
    print('Dask.distributed client information:')
    print('\tHostname: %s' % socket.gethostname())
    print(client, flush=True)

    run(config, gcms=gcms, sets=sets, variables=variables)


if __name__ == '__main__':
    cli()
