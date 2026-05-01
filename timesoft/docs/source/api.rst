Level 0 Data API
================
.. currentmodule:: raw

.. autosummary::
   :toctree: auto_api
   :nosignatures:

   get_data

Level 1 Data API
================

timesoft.Timestream
-------------------
.. currentmodule:: timestream

.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Timestream

Accessing Detectors
^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Timestream.get_xf
    Timestream.get_cr
    Timestream.restrict_detectors

Tidying Scan Information
^^^^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: auto_api
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

Analyzing Timestreams
^^^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Timestream.convert_coordinates
    Timestream.filter_scan_det
    Timestream.filter_scan_det_mask
    Timestream.filter_scan

Producing Maps and 1D Binned Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Timestream.make_1d
    Timestream.plot_1d
    Timestream.make_map
    Timestream.plot_map
    Timestream.detector_grid
    Timestream.make_map_det

Saving Data
^^^^^^^^^^^

.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Timestream.write
    Timestream.write_map
    Timestream.write_1d
    Timestream.reset

Level 2 Data API
================

timesoft.Map and timesoft.LineMap
---------------------------------
.. currentmodule:: maps

.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Map
    LineMap

Accessing Detectors
^^^^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Map.get_xf
    Map.get_cr
    Map.restrict_detectors

.. autosummary::
    :toctree: auto_api
    :nosignatures:

    LineMap.get_xf
    LineMap.get_cr
    LineMap.restrict_detectors

Visualizing Maps
^^^^^^^^^^^^^^^^
.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Map.plot
    Map.detector_grid

.. autosummary::
    :toctree: auto_api
    :nosignatures:

    LineMap.plot


Saving Data
^^^^^^^^^^^
.. autosummary::
    :toctree: auto_api
    :nosignatures:

    Map.write

.. autosummary::
    :toctree: auto_api
    :nosignatures:

    LineMap.write
