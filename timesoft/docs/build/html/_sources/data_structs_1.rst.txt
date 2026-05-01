Level 1: Timestream Data
========================

Once raw data has been reconstructed into a timestream,
it can be saved in this state as a timestream file. We
refer to files containing timestream data as our **level 1**
data product.

Timestream files are generated using tools in the 
:doc:`timesoft.Timestream <timestream>` class and saved in
zipped numpy file archives (``.npz``). They can be loaded,
analyzed, and saved using methods of the ``timesoft.Timestream``
class.

Files saved in the ``.npz`` format can be loaded 
with ``numpy.load``: ``file_in = np.load([filename],allow_pickle=True)``.
The various contents of file archive can then be
listed with ``file_in.files()``. Files can also be
loaded as ``ts = timesoft.Timestream([filename])`` which
provides many additional options for working with the
data.

Contents of the Timestream Files
--------------------------------

The timestream ``.npz`` files contain a number of elements.

====================    ==========   ======================================================================
Data Element            Data Type    Description
====================    ==========   ======================================================================
header                  dictionary   Summary information about the file/observation
data                    2D array     An n_detectors x n_samples array of detector readouts
t                       1D array     The unix timstamps of every sample
ra,dec                  1D array     The sky coordinates of every sample
az,el                   1D array     The telescope azimuth/elevation of every sample
tel_flags               1D array     The status of the telescope at every sample
scan_flags              1D array     The scan id of every sample in the timestream
scan_direction_flags    1D array     The direction the telescope was moving along the scan for every sample
====================    ==========   ======================================================================

A timestream can be saved with a second copy of each item 
(except the header), which is meant to serve as an uneditable
copy of the timestream, making it possible to revert the 
data to the state it was in immediately after being loaded
from the raw data files.

.. _header_ts:

Headers
-------
Most items in the headers of **level 1**, 
**level 2** and **level 3** data (timestreams,
maps, and power spectra) are the same,
and simply inherited from the lower level data
products. However, any given file may have 
arbitrary items added to the header (which 
will in turn be inherited by derived data
products).

Basic header information added when a timestream
is first created and kept throughout the data
reduction process are:

=============   =======================================================
Header key      Description
=============   =======================================================
observer        Observer initials
object          Name of object observed
datetime        Date and time of observation
med_time        Median unix timestamp of dataset
center_ra       Central ra of the observation
center_dec      Central declination of the observation

mc              Which MCE the data originates from
n_samples       Number of samples in the timestream
n_detectors     Number of detectors in the dataset
xf_coords       The feed-frequency coordinates of each detector
cr_coords       The readout row and column of each detector

epoch           The coordinate epoch for the ra and dec timestreams
detector_pars   Dictionary of detector setup information
scan_pars       Dictionary of scan setup information
flags           Dictionary of flags indicating analysis steps completed
=============   =======================================================

The final three items each contain a number of different parameters.
The detector_pars entry contains the settings for detector readout
provided when the observation was initialized. The scan_pars dictionary contains
information about the scan strategy (1D/2D, direction, slew speed, etc.).
The flags dictionary contains a number of boolean flags indicating 
whether a given analysis step has been completed.

.. note::

    The data structure for raw data is still being finalized,
    therefore the exact header data may change as more information
    becomes available in the raw data headers.

.. note::

    For the engineering run, some of the header data was saved
    incorrectly. Fields in the header affected by this typically
    have nans or 'unknown' as their value. This information can
    be recovered from the log.txt files for the raw data.


Telescope Flag Values
---------------------

The timestreams contain three sets of flags. 

**tel_flags:**
This flag is reported by the telescope and indicates what 
the telescope is doing. It can have values from 0 to 4:

=====   ===========================================================
Value   Meaning
=====   ===========================================================
0       Telescope is not on source
1       Telescope is tracking the field center
2       Telescope is scanning across the field in the +RA direction
3       Telescope is scanning across the field in the -RA direction
4       Telescope is turning around
=====   ===========================================================

.. note::

    We do not have documentation of what flag values are given 
    for scans along the declination (as opposed to RA) axis. 
    Presumably these also return 2 and 3 depending on scan direction.


**scan_flags:**
This flag can be generated during analysis of the timestream
by the ``timesoft.Timestream.flag_scans`` method. It is 
determined by breaking the timestream up into individual scans
across the field (in a single direction), and then numbering
these scans starting at 1. The scan_flags then give the scan
number to which each data point belongs. Samples that do not
belong to any scan have a scan_flag value of 0.

**scan_direction_flags:**
This flag is also generated by the ``timesoft.Timestream.flag_scans`` 
method. It is 0 when the telescope is moving in the direction
of increasing R.A./dec (depending on which coordinate is being
scanned across), and 1 when the telescope is moving in the 
direction of decreaseing R.A./dec.