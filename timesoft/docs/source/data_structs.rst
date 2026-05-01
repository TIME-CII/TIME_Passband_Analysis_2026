Data Structures Used by the TIME Collaboration
==============================================

Four primary types of files are of broadly relevant for
TIME data reduction and analyses. These are:

0. The :doc:`raw data <data_structs_0>` - unedited 
   data from the telescope and 
   detectors. These are stored in netCDF files.
1. :doc:`Timestream data <data_structs_1>` - this is 
   a collection of detector 
   timestreams and some associated information like 
   timestamps, telescope pointing, and flags.
2. :doc:`Map data <data_structs_2>` - this is timestream 
   data for one or more detectors that have been processed 
   into maps or, in the case of line scan observations,
   "linemaps".
3. :doc:`Powerspectrum data <data_structs_3>`.

.. note::

    We have not yet reached the power spectrum analysis phase
    of pipeline development.
