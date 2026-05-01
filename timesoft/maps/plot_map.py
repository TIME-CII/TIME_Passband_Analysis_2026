from __future__ import division, print_function, unicode_literals, absolute_import

import logging
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.signal
from scipy.stats.mstats import gmean

import timefpu.coordinates as coords
import timefpu.params as params
import timefpu.mce_data as mce_data

def plot_maps(outfile, data=None, markers=None, title='',
    clabel='', cmin=None, cmax=None, scale=1.0, missing_color='grey',
    cmap='rainbow', mux_space=False, marker_labels=None,
    cticks=None, cticklabels=None, fontsize=14, titlesize=16, legendsize=11,
    logscale=False):

    val_a = []
    val_b = []
    val_c = []

    # print('plotting')
    marker_legend = (marker_labels is not None)
    colorbar = (data is not None)

    if mux_space:
        print('mux')
        mshape = 's'
        msize = 70

        # a, b = mux_c, mux_r
        a_range = list(range(params.N_MUX_COLS))
        b_range = list(range(params.N_MUX_ROWS))
        transpose = False

        if colorbar and marker_legend:
            figsize = (10,7.5)
            legend_bbox = (1.18, 0.5)
            subplot_right = 0.7
        elif colorbar:
            figsize = (7,7.5)
        elif marker_legend:
            figsize = (9,7.5)
            legend_bbox = (1.02, 0.5)
            subplot_right = 0.75
        else:
            figsize = (8,7.5)

    else:
        # Slightly rectangular for reasonable aspect ratio
        mshape = [(-0.55,-1),(-0.55,1),(0.55,1),(0.55,-1)]
        msize = 120

        # a, b = det_x, det_f
        a_range = list(range(params.N_CHAN_SPATIAL))
        b_range = list(range(params.N_CHAN_SPECTRAL))
        transpose = True

        # print(a_range)
        # print(b_range)
        if colorbar and marker_legend:
            figsize = (12.1,4.95)
            legend_bbox = (1.12, 0.5)
            subplot_right = 0.785
        elif colorbar:
            figsize = (10.42,4.95)
        elif marker_legend:
            figsize = (11.7,4.95)
            legend_bbox = (1.01, 0.5)
            subplot_right = 0.84
        else:
            figsize = (9.6,4.95)

        #~ if colorbar and marker_legend:
            #~ figsize = (15,5)
            #~ legend_bbox = (1.10, 0.5)
            #~ subplot_right = 0.83
        #~ elif colorbar:
            #~ figsize = (13.5,5)
        #~ elif marker_legend:
            #~ figsize = (15,5)
            #~ legend_bbox = (1.01, 0.5)
            #~ subplot_right = 0.85
        #~ else:
            #~ figsize = (13,5)

    fig = plt.figure(figsize=figsize)

    markers_a = {}
    markers_b = {}

    for a in a_range:
        for b in b_range:
            if markers is not None and (a, b) in markers:
                print('..')
                mc = markers[(a, b)]
            elif data is not None: #and (a, b) in data:
                # print('storign data')
                # print(a, b, data[a,b])
                val_a.append(a)
                val_b.append(b)
                val_c.append(scale * data[a, b])
                continue
            else:
                mc = missing_color

            # This is a marker, not a value
            if mc not in markers_a.keys():
                markers_a[mc] = []
                markers_b[mc] = []
            markers_a[mc].append(a)
            markers_b[mc].append(b)

    if transpose:
        val_a, val_b = val_b, val_a
        markers_a, markers_b = markers_b, markers_a

    # Show markers
    for mc in markers_a.keys():
        plt.scatter(markers_a[mc], markers_b[mc], marker=mshape, s=msize, c=mc)

    cnorm = None
    if logscale:
        cnorm = matplotlib.colors.LogNorm()

    # Show values
    plt.scatter(val_a, val_b, marker=mshape, s=msize, c=val_c, cmap=cmap, norm=cnorm)
    plt.title(title, fontsize=titlesize)

    if mux_space:

        plt.xlim(-1, params.N_MUX_COLS)
        plt.ylim(-1, params.N_MUX_ROWS)
        plt.xlabel("Multiplexing Column", fontsize=fontsize)
        plt.ylabel("Multiplexing Row", fontsize=fontsize)

    else:

        plt.xlim(-1, params.N_CHAN_SPECTRAL)
        plt.ylim(-1, params.N_CHAN_SPATIAL)
        plt.xlabel("Frequency Index ($f$ Coordinate, Lowest Frequency at $f=0$)", fontsize=fontsize)
        plt.ylabel("Spatial Index ($x$ Coordinate)", fontsize=fontsize)

        for f in [0,8,16,24,36,48,60]:
            plt.axvline(f - 0.5, color = 'k', lw=0.6)
        for m in range(0,17,4):
            plt.axhline(m - 0.5, color = 'k', lw=0.6)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if marker_legend:
        # Plot dummy scatter plots for the labels
        for key, val in marker_labels.items():
            plt.scatter([100], [100], marker=mshape, s=msize, c=key, label=val)
        plt.legend(loc='center left', bbox_to_anchor=legend_bbox, fontsize=legendsize)

    if colorbar:
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes("right", "2%", pad="2%")
        cb = plt.colorbar(cax=cax, ticks=cticks)
        if cticklabels is not None:
            cb.set_ticklabels(cticklabels)
            cb.ax.tick_params(labelsize=fontsize)
        cb.set_label(clabel, fontsize=fontsize)
        plt.clim(cmin, cmax)

    plt.tight_layout()

    if marker_legend:
        plt.subplots_adjust(right=subplot_right)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
    plt.close(fig)
