Level 0: Working with the Raw Data
==================================

.. warning::
   For most purposes, working with the raw data 
   products directly is unnecessary. It is recommended
   that new users start by reviewing the documentation
   for :doc:`level 1 data <timestream>`.

Our level zero data products are the raw netCDF
files written by during observations by the data
acquisition software. These data can be interfaced
with directly using code from the 
`netCDF4 <https://unidata.github.io/netcdf4-python/>`_ 
package.

Direct Interface with netCDF Data
---------------------------------

A simple example of loading the data from a netCDF 
file is 
.. code-block:: 

   >>> from netCDF4 import MFDataset
   >>> path = '/location/containing/netCDFfiles/for/your/observation'
   >>> files = [path + i for i in files_in if i.endswith(".nc")]
   >>> data_file = MFDataset(files)

The various variables in the netCDF file can then be
accessed using keys, for example the MCE 0 detector 
readouts can be retreived using
.. code-block:: 

   >>> data_file.variables['mce0_raw_data_all']

Generating A Timestream 
-----------------------

Generally there are some very standard 
tasks that need to be performed prior to any analyses 
on the raw data. For instance, the telescope telemetry
and detector data are sampled at different rates and
an interpolation needs to be performed to determine 
telescope telemetry information for any given detector
datapoint. We have standardized this procedure in the
``raw`` module. ``raw`` has a single
important function: ``get_data``. ``get_data`` loads
a specified set of netCDF files and processes them in
order to produce a set of arrays containing the timestamp,
interpolated telescope telemetry information and flags,
and detector data for each detector sample. We refer
to these as :doc:`level 1 data <timestream>`, as the 
minimum processing required to work with the data as a 
timestream has been performed.

Generating a timestream from a given observation can
be acomplished as follows

.. code-block:: 

   >>> from timesoft import get_data 
   >>> path = '/location/containing/netCDFfiles/for/your/observation'
   >>> t,ra,dec,az,el,data,telflag = get_data(path=path)

There are a number of parameters for ``get_data`` which
allow a user to accomplish specific tasks. For example, it 
is possible to load data for only a specific list of 
detectors (instead of the whole array). The following code
could be used to load only the detectors from pixel 7:

.. code-block:: 

   >>> dets_x7 = [(7,i) for i in range(60)]
   >>> t,ra,dec,az,el,data_x7,telflag = get_data(path=path,xf=dets_x7)

The full functionality of ``get_data`` is described in
the documentation below.

The processing of timestream data is handled by the
``raw`` module. The classes and functions in
``raw`` have a built in wrapper around 
``get_data`` such that most users should not need to 
use code from ``raw`` directly, but can skip
to data processing with the ``timestream`` modue, 
which contains much more functionality.

Documentation for the ``raw`` Module
---------------------------------------------

.. note::

   Due to its general usefulness, the ``get_data()``
   function for loading raw data is known to the top
   level of the ``timesoft`` module allowing the import 
   statement to be simplified from ``from timesoft.raw.loading_utils
   import get_data`` to ``from timesoft import get_data``. 

.. automodule:: raw.loading_utils
   :members: