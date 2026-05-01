Level 1: Working with Timestream Data
=====================================

These pages contains the documentation for the class
``timesoft.Timestream`` which is the primary tool for
three tasks:

1.  Loading raw data and processing it into our 
    standardized format for timestream analyses
    (i.e. converting :doc:`level 0 <data_structs>` data
    into :doc:`level 1 <data_structs>` data).
2.  Manipulating timestream data, including making
    data cuts, performing atmospheric filtering,
    and transforming between coordinate systems.
3.  Generating maps of 2D scan data and spatially 
    binned representations of 1D scan data
    (i.e. converting :doc:`level 1 <data_structs>` data
    into :doc:`level 2 <data_structs>` data).

A tutorial walking new users through how to accomplish
all of these tasks is available :doc:`here <notebooks/timesoft_demo>`.


.. note::

   Due to its general usefulness, the ``timestream``
   class for working with timestream data is known to the top
   level of the ``timesoft`` module allowing the import 
   statement ``from timesoft import timestream``. The full
   path to the class would be ``timesoft.timestream.timestream_tools.Timestream``.

.. automodule:: timestream


The Timestream Class
--------------------

.. autoclass:: timestream.Timestream
   :no-members:
   :no-undoc-members:
   :show-inheritance:

Initialization
--------------

By default, ``Timestream`` inspects the path to the data 
provided at initialization and attempts to determine if 
it is raw data (from ``.nc`` netCDF files) or processed 
data (from ``.npz`` files created when data processed 
with ``timesoft`` functionality is saved). Unless the 
`path` parameter ends with ``.npz`` it assumes raw data
is being loaded and processes the file specified 
accordingly. If the automatic identification of the data file
type is failing. You can force ``Timestream`` to
try to load the file as raw data or processed 
data using the `mode` parameter.

When initializing a ``Timestream`` class from raw
data, the MCE to load should be specified with `mc`.
Optionally, the component files to load from the
data directory can be restricted with the `files_in`
parameter. A list of detectors to load can be 
specified in either feed-frequency or readout
column-row coordinates using the `xf` or `cr`
parameters.

When initializing a ``Timestream`` class from a
processed ``.npz`` file, only the file path needs
to be specified. However you can optionally 
restrict the detectors being loaded using `xf`
or `cr`.

Manipulating Timestream Data
----------------------------

We describe each point in time for which a timestream
contains data a sample. Once a ``Timestream`` class is
initialized, the information about these samples is 
stored as a set of arrays which can be accessed as 
attributes of the ``Timestream`` instance.

The basic information is as follows:

``Timestream.t``
    The time when each sample was generated, given in 
    unix time.
``Timestream.ra`` and ``Timestream.dec``
    The R.A. and declination of the telescope when each
    sample was generated. These are initially in 
    the apparent coordinate frame, but can be 
    converted to other frames using the 
    ``Timestream.convert_coordinates``
    method.
``Timestream.az`` and ``Timestream.el``
    The azimuth and elevation of the telescope when 
    each sample was generated. 
``Timestream.data``
    The detector counts data from all loaded detectors
    for each sample.
    These data will be modified directly by things like
    atmospheric filtering or flux calibration
``Timestream.tel_flags``
    Information returned from the telescope indicating
    how the telescope is moving when each sample was 
    generated.
``Timestream.scan_flags``
    This attribute is initially filled with zeros, 
    however once methods designed to identify individual 
    scans across the mapped region have been run, this
    will be populated with the scan number to which each
    sample belonged.
``Timestream.scan_direction_flags``
    This attribute is initially filled with zeros, 
    however once methods designed to determine the direction
    across the scan pattern which the telescope was moving
    it will be filled with that information for each sample. 

These attributes can be accessed directly. But it is 
often most convenient to work with them through pre-defined
methods of the ``Timestream`` class.

There are a number of methods for working with the timestreams.

Methods for Accessing Specific Detectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ordering of detectors in the ``Timestream.data`` 
attribute can varry depending on which detectors are 
loaded in a ``Timestream`` instance and how. The 
``Timestream`` object internally knows this ordering.
To facilitate easy retreival of data the following 
methods return the index in ``Timestream.data`` 
corresponding to a specified detector:

.. autosummary::
    :nosignatures:

    Timestream.get_xf
    Timestream.get_cr
    Timestream.restrict_detectors

Methods for Tidying Up Scans and Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Timestreams are generally constructed by 
collecting detector data as the telescope
is scanned back and forth over some region
of the sky. It is useful to be able to 
break the timestream up in terms of which
scan in the series a given sample belongs
and which direction the telescope is scanning.
The ``Timestream.flag_scans`` method 
acomplishes this. Once it has been run,
a number of other methods are availabe to 
remove bad scans, bad pieces of scans, 
scans where the telescope is moving in one
direction or the other, etc. 

In addition, the ``Timestream._remove_samples``
method provides a way to remove samples 
meeting arbitrary criteria from the timestream.

.. autosummary::
    :nosignatures:

    Timestream.header_entry
    Timestream.flag_scans
    Timestream.renumber_scans
    Timestream.remove_scan_flag
    Timestream.remove_scan_direction
    Timestream.remove_short_scans
    Timestream.remove_end_scans
    Timestream.remove_scan_edge
    Timestream.remove_tel_flag
    Timestream._remove_samples

Methods for Analyzing Timestreams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Once you have identified scans and cropped unwanted 
chunks from the timestream, the next steps involve
transforming the data in various ways. This can include
filtering atmospheric noise and detector drifts,
converting coordinate systems of the position information,
and so on.

.. autosummary::
    :nosignatures:

    Timestream.convert_coordinates
    Timestream.filter_scan_det
    Timestream.filter_scan_det_mask
    Timestream.filter_scan

Methods for Producing Level 2 Data Products
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following methods are useful for processing
timestream data into maps (for 2D scanning patters)
or binning data along the scanned axis (for 1D
scanning patterns). These processed data constitute
Level 2 of TIME's data products, and producing
one of these objects is often the goal of analyzing
timestream data.

Methods for making and doing some basic visualizations
of these products are provided as part of 
the ``Timestream`` class, while additional 
capabilities for processing the data are available
in the ``timesoft.maps`` module.

.. autosummary::
    :nosignatures:

    Timestream.make_1d
    Timestream.plot_1d
    Timestream.make_map
    Timestream.plot_map
    Timestream.detector_grid
    Timestream.make_map_det

Methods for Saving Data Products
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following methods make it possible to 
save timestreams, maps, and binned 1d data
in the format specified by the
:doc:`TIME data formatting <data_structs>`.

.. autosummary::
    :nosignatures:

    Timestream.write
    Timestream.write_map
    Timestream.write_1d
    Timestream.reset


Tutorial
--------

The following sections provide a demonstration of how
to use ``timesoft.Timestream``.

.. toctree::

    notebooks/timesoft_demo


