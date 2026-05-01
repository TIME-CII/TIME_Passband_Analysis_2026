import matplotlib.pyplot as plt 
import numpy as np 


def plot_by_subarrays(save_path, ident, cmap, array):
    
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.colors import LogNorm
    from matplotlib.colors import ListedColormap
    

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('lightgray')
    
    reddest_color = 'red'
    # err_map_cmap = ListedColormap(vik_cmap(np.linspace(0.5, 1.0, 256)))

    cmap = ListedColormap(["#ffffff", "#1f77b4"])

    # vmin = np.nanmin(timesoft_gains[timesoft_gains > 0])  # avoid zero in log
    # vmax = np.nanmax(timesoft_gains)
    im = ax.imshow(array, origin='lower', cmap=cmap)
    
    #make color bar the same height 
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05) 
    
    
    # cbar = fig.colorbar(im, cax=cax)
    # cbar.set_label('Gains [Jy / Beam × Count] ', fontsize=12)
    
    # Set yticks
    ax.set_yticks(np.arange(0, 16, 2))
    for verticle_grid in [7.5,15.5,23.5,35.5,47.5]: 
        ax.axvline(verticle_grid,color=reddest_color,linewidth=3)
    for horizontal_grid in [3.5,7.5,11.5]: 
        ax.axhline(horizontal_grid,color=reddest_color,linewidth=3)
    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('Spatial Index')
    ax.text(47.5+(59-47.5)/2, 17, 'HF 1',
                color='black', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor=reddest_color, edgecolor='none', alpha=0.3, boxstyle='round,pad=0.2'))
    ax.text(35.5+(47-35.5)/2, 17, 'HF 2',
                color='black', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor=reddest_color, edgecolor='none', alpha=0.3, boxstyle='round,pad=0.2'))
    ax.text(23.5+(35-23.5)/2, 17, 'HF 3',
                color='black', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor=reddest_color, edgecolor='none', alpha=0.3, boxstyle='round,pad=0.2'))
    ax.text(15.5+(23-15.5)/2, 17, 'LF 3',
                color='black', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor=reddest_color, edgecolor='none', alpha=0.3, boxstyle='round,pad=0.2'))
    ax.text(7.5+(15-7.5)/2, 17, 'LF 2',
                color='black', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor=reddest_color, edgecolor='none', alpha=0.3, boxstyle='round,pad=0.2'))
    ax.text(0+(7-0)/2, 17, 'LF 1',
                color='black', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor=reddest_color, edgecolor='none', alpha=0.3, boxstyle='round,pad=0.2'))
    #fig.suptitle('Gridded by Sub Arrays', y=0.7)
    plt.savefig(save_path + ident,dpi=300, bbox_inches='tight')
    # pl.show()


def plot_by_mux(save_path, ident, cbar_label,array):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # import coordinates
    import sys 
    sys.path.append('/home/vaughan/TIME-rx/timefpu/')
    import coordinates


    from matplotlib.colors import LogNorm
    from matplotlib.colors import ListedColormap

    
    # timesoft_gains = np.loadtxt('./1644342374/gain_map_for_plot.txt')
    the_x_s = np.arange(0,16,1)
    the_f_s = np.arange(0,60,1)
    the_xfs = np.array(np.meshgrid(the_x_s,the_f_s)).T
    mux_plotting = np.empty((16,60,2))
    for f in range(60):
        for x in range(16):
            this_muxcr = coordinates.xf_to_muxcr(the_xfs[x,f][0],the_xfs[x,f][1])
            mux_plotting[x,f] = this_muxcr
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('lightgray')
    # err_map_cmap = ListedColormap(vik_cmap(np.linspace(0.5, 1.0, 256)))
    # cmap = ListedColormap(["#ffffff", "#1f77b4"])
    # cmap = 'plasma'

    colors31 = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363",
        "#9ecae1", "#fdae6b", "#a1d99b", "#bcbddc", "#d9d9d9",
        "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#969696",
        "#c6dbef",
    ]
    cmap31 = ListedColormap(colors31, name="cmap31")

    vmin = np.nanmin(array[array > 0]) 
    vmax = np.nanmax(array)
    im1 = ax.imshow(array, origin='lower', cmap=cmap31)#, #norm=LogNorm(vmin=vmin,vmax=vmax))
    #im1 = ax.imshow(timesoft_gains, origin='lower', cmap=err_map_cmap)
    
    mux_map = mux_plotting[:, :, 0]
    mux_ids = np.unique(mux_map)
    print(mux_ids)



    #   [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29.]
    # mux_ids_b =  [9,   8,  7,  6,  5,  4,  ]
    # mux_labels = ['a','b','c','d','e','f','d','c','b','a','a','b','c','d','e','f','d','c','b','a','a','b','c','d','e','f','d','c','b','a']
    spec_B_labels = [9,8,7,6,5,4,3,2,1,0,19,18,17,16,15,14,13,12,11,10,29,28,27,26,25,24,23,22,21,20]
    reddest_color='black'
    
    for mi, mux_id in enumerate(mux_ids):
        ys, xs = np.where(mux_map == mux_id)
        if len(xs) == 0 or len(ys) == 0:
            continue
    
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    
        
    
        rect = Rectangle((xmin - 0.5, ymin - 0.5),
                         xmax - xmin + 1,
                         ymax - ymin + 1,
                         linewidth=3,
                         edgecolor=reddest_color,  # Change color if desired
                         facecolor='none')
        ax.add_patch(rect)
    
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        ax.text(x_center, y_center, '%s' % spec_B_labels[mi],
                color='black', fontsize=8, ha='center', va='center',
                bbox=dict(facecolor=reddest_color, edgecolor='none', alpha=0.3, boxstyle='round,pad=0.2'))
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0)
    cbar = fig.colorbar(im1, cax=cax, ticks=np.arange(np.nanmax(mux_ids)))
    cbar.set_label(cbar_label, fontsize=6)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(np.arange(0, mux_map.shape[1], 10))
    ax.set_yticks(np.arange(0, mux_map.shape[0], 2))
    
    ax.set_xlabel('Frequency Index')
    ax.set_ylabel('Spatial Index')
    
    #fig.suptitle('Gains Gridded By MUX Columns', y=0.65)
    #plt.tight_layout()
    plt.savefig(save_path + ident,dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    plot_by_subarrays('/home/vaughan/TIME-analysis/', 'sub_array_mapping', None, np.zeros((16,60)))
    plot_by_mux('/home/vaughan/TIME-analysis', 'mux_mapping', None, np.zeros((16,60)))