#!/usr/bin/env python
import click

from storylines.tools.quantile_mapping import run


@click.command()
@click.option('--data_file', help='dataset to be quantile mapped')
@click.option('--ref_file', help='reference datasets for quantile adjustment')
@click.option('--obs_files', help='observation dataset')
@click.option('--kind', type=click.Choice(['gard', 'icar']))
@click.option('-v', '--variables', multiple=True)
def cli(data_file, ref_file, obs_files, kind, variables):
    '''Command line interface for running the GARD quantile mapping routines'''

    import dask
    dask.config.set(scheduler='single-threaded')

    run(data_file, ref_file, obs_files, kind, variables)


if __name__ == '__main__':
    cli()
