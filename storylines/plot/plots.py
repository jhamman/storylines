import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

projection = ccrs.LambertConformal()


states_shp = shpreader.natural_earth(resolution='50m',
                                     category='cultural',
                                     name='admin_1_states_provinces')
state_geoms = list(shpreader.Reader(states_shp).geometries())


def set_ax_props(ax):
    ax.coastlines('50m', lw=0.25)
    ax.add_geometries(state_geoms, ccrs.PlateCarree(),
                      facecolor='', edgecolor='black', zorder=10000, lw=0.15)


def plot_thumbnails(obs_da, gard_da, isdiff=False,
                    months=list(range(1, 13, 2)),
                    kwargs={}, dkwargs={}, cbar_labels={}, extends={}):

    sets = gard_da.sets.values

    fig, axes = plt.subplots(nrows=len(sets) + 1, ncols=len(months),
                             figsize=(12, 2 + len(sets)),
                             subplot_kw=dict(projection=projection))

    default_kwargs = dict(transform=ccrs.PlateCarree(), add_colorbar=False)
    default_kwargs.update(kwargs)
    kwargs = default_kwargs

    if isdiff:
        default_dkwargs = dict(transform=ccrs.PlateCarree(),
                               add_colorbar=False)
        default_dkwargs.update(dkwargs)
        gkws = default_dkwargs
    else:
        gkws = kwargs

    for j, month in enumerate(range(1, 13, 2)):
        # add obs
        ax = axes[0, j]
        m0 = obs_da.sel(month=month).plot.pcolormesh('longitude', 'latitude',
                                                     ax=ax, **kwargs)

        if j == 0:
            ax.text(-0.07, 0.55, 'Observations', va='center', ha='right',
                    rotation='horizontal', rotation_mode='anchor',
                    transform=ax.transAxes)

        # add GARD
        for i, s in enumerate(sets):
            ax = axes[i + 1, j]
            m1 = gard_da.sel(sets=s, month=month).plot.pcolormesh(
                'longitude', 'latitude',  ax=ax, **gkws)
            ax.set_title('')

            if j == 0:
                ax.text(-0.07, 0.55, s, va='center', ha='right',
                        rotation='horizontal', rotation_mode='anchor',
                        transform=ax.transAxes)

    for ax in axes.flat:
        set_ax_props(ax)

    if not isdiff:
        cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.725])
    else:
        cbar_ax = fig.add_axes([0.95, 0.6, 0.025, 0.3])
        dcbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.3])

        dcbar = plt.colorbar(m1, orientation='vertical', cax=dcbar_ax,
                             extend=extends.get('dcbar', 'both'))
        dcbar.set_label(cbar_labels.get('dcbar', 'unset'))

    cbar = plt.colorbar(m0, orientation='vertical', cax=cbar_ax,
                        extend=extends.get('cbar', 'neither'))
    cbar.set_label(cbar_labels.get('cbar', 'unset'))

    return fig, axes
