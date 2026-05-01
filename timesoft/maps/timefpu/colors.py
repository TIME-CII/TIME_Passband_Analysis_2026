from __future__ import division, print_function, unicode_literals, absolute_import

import matplotlib.pyplot as plt
import colorsys

# tab10 is the same as the default matplotlib color cycle
default_colors = plt.get_cmap("tab10").colors

# Give names to the default matplotlib colors so they can be reordered
# on plots (they are pretty and keep things looking consistent)
blue = default_colors[0]
orange = default_colors[1]
green = default_colors[2]
red = default_colors[3]
purple = default_colors[4]
brown = default_colors[5]
pink = default_colors[6]
grey = default_colors[7]
gold = default_colors[8]
teal = default_colors[9]

black = 'k'

pure_blue = 'b'
pure_green = 'g'
pure_red = 'r'
pure_cyan = 'c'
pure_magenta = 'm'
pure_yellow = 'y'

# Modify the lightness of an RGB color tuple by delta_l, and return 
# another RGB tuple
def modify_lightness(rgb, delta_l=-0.1):
	assert len(rgb) == 3, "Expecting an RGB tuple"
	h,l,s = colorsys.rgb_to_hls(*rgb)
	l += delta_l
	l = max(0, l)
	l = min(1, l)
	return colorsys.hls_to_rgb(h,l,s)

# Darker and lighter versions of the default colors
default_darks = [modify_lightness(c, delta_l=-0.1) for c in default_colors]
default_light = [modify_lightness(c, delta_l=0.1) for c in default_colors]
