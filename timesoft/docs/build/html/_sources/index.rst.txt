Documentation for TIME's Analysis Software
==========================================

These pages contain the documentation for data anaysis software
written by/for the TIME Collaboration. This code is designed to
interface with raw data written by the TIME Data Acquisition 
software, convert it to timestreams, maps, and power spectra, 
and perform reduction and analysis of these objects. Code for 
general use is contained within the TIMESOFT package, which can
be cloned from the `collaboration github page <https://github.com/TIME-CII/TIME-analysis>`_.

In addition to documenting the code itself, we will try to 
maintain information about the :doc:`format of the data <data_structs>`,
notes on data reduction practices, and other useful information
on this page.

If you plan to develop code that will become part of TIMESOFT,
please review the :doc:`rules <contrib_rules>` for documenting and formatting your 
code and integrating it with the existing software before pushing
anything to the git repo.

.. toctree::
   :caption: Getting Started
   :maxdepth: 3

   installation
   contrib_rules

.. toctree::
   :caption: Data Structures
   :maxdepth: 3

   data_structs
   data_structs_0
   data_structs_1
   data_structs_2
   data_structs_3

.. toctree::
   :caption: Working with Data
   :maxdepth: 3

   raw
   timestream
   maps
   powspec

.. toctree::
   :caption: API
   :maxdepth: 5

   api