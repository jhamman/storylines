#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click

from stoylines.tools.post_process_gard_output import run


@click.command()
@click.option('--sets', multiple=True, default='',
              help='list of GARD setnames to process')
@click.option('--gcms', multiple=True, default='',
              help='list of GARD gcms to process')
@click.option('--variables', multiple=True, default='',
              help='list of GARD variables to process')
@click.argument('config')
def cli(config, sets, gcms, variables):
    '''Command line interface for running the GARD post-processing tools'''

    run(config, gcms=gcms, sets=sets, variables=variables,)


if __name__ == '__main__':
    cli()
