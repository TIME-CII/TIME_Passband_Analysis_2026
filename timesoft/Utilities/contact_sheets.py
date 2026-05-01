import numpy as np
import matplotlib.pyplot as plt
from timesoft.helpers.nominal_frequencies import f_ind_to_val
from matplotlib import colors

def plot_perdet_maps(per_det_data, xfs, title, savepath, sublabel_xf=True, colormax=100, colormin=0, clabel='', colorspine=False, colorbar=False, imshow=True, suptitles=False, paper_plot=False, log_scale=False, planet=False):
    """
    per_det_data: numpy.ndarray with shape (#detectors , map_height , map_width)
        The per-detector maps to plot.

    xfs: numpy.ndarray with shape (#detectors, 2)
        Each entry is (x, f) for the corresponding per-det_data.

    title: str
        Suptitle of the plot.

    savepath: str
        Directory path (without trailing /) to save the output PNG.

    sublabel_xf: bool
        If True, label each map with its (x, f).

    colormax: float
        Color scale limit (symmetric about 0). if 0, then colormax is calculated to be 3*std of all maps.

    colorspine: bool
        If True, color the spines of the plots by frequency.
    """
    center_frequencies = [f_ind_to_val(c) for c in np.arange(60)]


    if paper_plot:


        per_det_data = per_det_data[11:53]
        center_frequencies = center_frequencies[11:53]

        ncols = 6
    else: 
        ncols = 8

    print('hello', paper_plot)
    nrows = int(np.ceil(per_det_data.shape[0] / ncols))
    # f_of_perdet = xfs[:, 1]
    # f_of_perdet = [xf[1] for xf in xfs]

    # fig = plt.figure(figsize=(20, nrows))
    plt.rcParams.update({'font.size':10})
    if paper_plot:
        fig, axs = plt.subplots(nrows, ncols, figsize=(8,8))
    else:
        fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 6*nrows),gridspec_kw={'wspace': 0.0, 'hspace': 0.00, 'left':0, 'right':1, 'top':1, 'bottom':0})
    plt.rcParams.update({'font.size':10})

    axs = axs.ravel()  # flatten in case of multiple rows/cols

    # fig.suptitle(title, fontsize=16)
    cmap = 'RdBu'

    im = None  # keep reference for colorbar
    # per_det_data = per_det_data[11:53]
    

    if colorspine:
        spine_colors = plt.get_cmap('Set3').colors * 5  # extended to cover 60 freqs


    for i in range(per_det_data.shape[0]):
        ax = axs[i]
        this_data = per_det_data[i]
        if imshow:
            if paper_plot:
                ax.set_title('%.0F GHz' % center_frequencies[i], pad=0.01, fontsize=8)
            if log_scale:
                im = ax.imshow(this_data, origin='lower', cmap='RdBu', aspect='equal', clim=(colormin,colormax),norm=colors.LogNorm(vmin=this_data.min(), vmax=this_data.max()))
            else:
                im = ax.imshow(this_data, origin='lower', cmap='RdBu', aspect='equal', clim=(colormin,colormax))  # add vmin/vmax if needed
            ax.set_xlim(0, this_data.shape[1])
            ax.set_ylim(0, this_data.shape[0])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.invert_xaxis()
            # ax.set_title('x%s' % xfs[i][0])
            # ax.text(0.5,0.98, 'x%s' % i, transform=ax.transAxes, fontsize=24,color='white',bbox=dict(boxstyle='round,pad=0.3',facecolor='black',edgecolor='black',alpha=0.8))
            if planet:
                ax.text(0.5,0.98, 'x%sf%s' % (xfs[i][0],xfs[i][1]), transform=ax.transAxes, fontsize=24,color='white',bbox=dict(boxstyle='round,pad=0.3',facecolor='black',edgecolor='black',alpha=0.8))
            else:
                ax.text(0.5,0.98, '%s' % xfs[i], transform=ax.transAxes, fontsize=24,color='white',bbox=dict(boxstyle='round,pad=0.3',facecolor='black',edgecolor='black',alpha=0.8))
            # if suptitles:
                # ax.set_title('%)
            ax.set_frame_on(False)
            ax.set_aspect('equal', adjustable='box')
        else:
            ax.plot(this_data)
            ax.set_title('x%s f%s' % (xfs[i][0], xfs[i][1]))
    


    # for i in range(per_det_data.shape[0]):
    #     row, col = divmod(i, ncols)
    #     ax = fig.add_subplot(nrows, ncols, i + 1)

    #     this_data = per_det_data[i]
    #     im = ax.imshow(this_data, origin='lower', cmap=cmap)
    #                 #    vmin=-colormax, vmax=colormax, aspect=1.5) #this aspect here was computed from pixel size (i had pixels 0.004 degrees in ra and 0.006 degrees in dec)

    #     x = xfs[i][0]; f = xfs[i][1]
    #     if sublabel_xf:
    #         ax.set_title('x%s f%s' % (x,f), fontsize=8)

    #     if colorspine:
    #         for spine in ax.spines.values():
    #             spine.set_edgecolor(spine_colors[f_of_perdet[i]])
    #             spine.set_linewidth(2)

    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.invert_xaxis()  # RA convention

    # add colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=axs, orientation="vertical", fraction=0.05, pad=0.02)
        cbar.set_label('Jy/Beam', fontsize=10)
        cbar.set_ticks(np.linspace(colormin, colormax, 10))

    # plt.subplots_adjust(wspace=0, hspace=0)
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.savefig(f"{savepath}/{title.replace(' ', '_')}.png",dpi=300, bbox_inches='tight')
    plt.close(fig)
