#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click

from storylines.tools.gard_utils import run


@click.command()
@click.option('--outfile', default='namelist.txt',
              help='output file downscaling namelists')
@click.argument('config')
def cli(config, outfile):
    '''Command line interface for running the GARD pre-processing tools'''

    run(config, outfile)


if __name__ == '__main__':
    cli()
