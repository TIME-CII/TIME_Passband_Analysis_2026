Contributing to the TIMESOFT Code and Documentation
===================================================

Some Guiding Principles
-----------------------

When working on code for ``timesoft`` please keep in mind the 
following guidelines:

1. **Contributions encouraged:**
   the ``timesoft`` package is a collaborative development of the
   TIME Collaboration and all members are welcomed and encouraged
   to contribute in whatever ways they can.
2. **What to include:**
   software included in ``timesoft`` should be for general purpose
   processing of the data. If your code exists only to analyze a 
   very specific piece of data, it should be kept in the ``timesandbox``
   section of the github repo instead of added to the ``timesoft``
   package. For example, code with general utilities for cleaning
   up timestreams (e.g. flagging bad data, correcting atmospheric
   varriations, etc.) belongs in ``timesoft``. On the other hand,
   code to load data for a specific observation of Mars, toss out 
   the first 10 minutes of data because of some defect noticed by
   the observer, and then perform some specific analyses on the 
   timestream belongs in ``timesandbox``. In general, if you don't know
   where your code goes, ask at the next telecon and we can figure
   it out collectively.
3. **When to include it:**
   make sure a piece of software is fully developed and adds the
   desired new functionality before adding it to ``timesoft``. For
   experimental code or work in progress, use your personal 
   ``timesandbox`` directory (or a collective working directory) for
   developing or experimenting with things. You may also want to 
   perform work in a development branch, and make a `pull request 
   <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>`_
   once work is ready to be reviewed. Once something is put in 
   ``timesandbox`` it should be the standard procedure that will be 
   used by the collaboration to accomplish a given analysis task.
4. **Read the documentation before making changes:**
   These documentation pages contain lots of information, including
   some guidance on how to use and edit various pieces of code that
   might not be evident from the code itself. Please review the
   documentation for any code you are thinking of modifying prior
   to making changes.
5. **Document your code:** 
   Please thoroughly document your code, as it will be used by
   others for their analysis. In particular, include properly 
   formatted doc strings for major pieces of code 
   that you add to ``timesoft``. You are also invited to add to our
   documentation web pages by writing more extensive tutorials or
   explanations of major pieces of code you contribute. You can 
   find a primer on doc strings and how to format them 
   :ref:`here <docstrings>`. You can find a primer on how to 
   write pages of the documentation :ref:`here <sphinxintro>`.

.. _docstrings:

Doc String Formatting
---------------------

Sphinx works by parsing the doc strings in Python code and 
formatting them in a uniform way for easily readable documentation.
In order for this to work specific formatting conventions must
be followed. Our Sphinx setup supports a few format options, but 
if you are not already familiar with a particular convention I
recommend using the ``numpydoc`` 
`style <https://numpydoc.readthedocs.io/en/latest/format.html>`_.

A doc string is a markdown string placed at the beginning of a function,
class, or module, demarcated by triple double quotes. At its most basic, it 
should be a short description of the behavior of the piece of code. However,
well written doc strings can completely document a piece of code, and 
documentation that lives alongside the code is more likely to stay up to date.
A basic ``numpydoc`` doc string for a function is structured as follows:

.. code-block:: python

    def f(params):
        """One line description

        A few sentences providing a longer description of the code's 
        functionality.

        Parameters
        ----------
        par1 : type
            description of first parameter
        par2 : type
            description of second parameter
        
        Returns
        -------
        return1 : type
            description of the first item returned
        return2 : type
            description of the second item returned

        Notes
        -----
        (Optional) Notes about the code

        Examples
        --------
        (Optional) Examples of code usage
        """

For a concrete example:

.. code-block:: python

    def add(x,y,sayhi=False):
        """Function to add x and y

        This function takes inputs x and y and returns the sum x+y.
        Optionally it also prints a friendly message.

        Parameters
        ----------
        x : float
            The first number to be summed
        y : float
            The second number to be summed
        sayhi : bool, optional
            If set to True, the function will also print a friendly message.
            The default is False.

        Returns
        -------
        z : float
            The sum of `x` and `y`.

        Examples
        --------
        Standard usage of the function:
        >>> z = add(1.0,2.0)
        >>> print(z)
        3.0

        Usage if you need some encouragement:
        >>> z = add(1.0,2.0,sayhi=True)
        Hi friend. You're doing a great job!
        >>> print(z)
        3.0


        """
        z = x+y
        if sayhi:
            print("Hi friend. You're doing a great job!")
        return z

.. note::

    Stick to section headings supported by ``numpydoc`` in
    your doc strings, as Sphinx may ignore things that 
    are not recognized. Standard sections include 
    Parameters, Returns, Raises, Notes, Examples, See Also,
    Methods (for classes), Attributes (for classes),
    and Routine Listing (for modules). See the ``numpydoc``
    `website <https://numpydoc.readthedocs.io/en/latest/format.html>`_
    for a full listing.


.. _sphinxintro:

Basics of Sphinx Documentation
------------------------------

Sphinx is a tool for documenting Python code that combines 
markdown tools with an interpreter that can parse the doc 
strings of code and automatically format them for easy reading.
It is the tool used to generate the documentation for a wide
range of familiar Python packages, including ``numpy`` and 
``matplotlib``.
The `Sphinx website <https://www.sphinx-doc.org/en/master/index.html>`_
provides a good overview and tutorial for setting up documentation.
The specific markdown language used by Sphinx is `reStructuredText
<https://docutils.sourceforge.io/rst.html>`_. There are plenty of
guides available for achieving specific formatting with reStructuredText.

This section provides a short overview of the minimum steps to
create a new page documenting a hypothetical ``funmath`` module 
you might have just added to ``timesoft``. 

**Step 1: Create a module.rst file:** The markdown files for our
docs are kept in the ``TIME-analysis`` repo in ``timesoft/docs/source``.
``index.rst`` contains defines the main page of our docs, and other
``.rst`` files create subsections of the documentation. 

Create a new file ``timesoft/docs/source/funmath.rst``. Open this 
file and add the following markdown:

.. code-block:: markdown

    FUNMATH: A Happy Module For More Fun Doing Math
    ===============================================

    Funmath performs regular mathematical calculations
    such as addition and subtraction, but prints nice
    comments to the console as it works. This page 
    demonstrates how to use the varous functions included 
    in the ``timesoft.funmath`` module.

Headings are indicated by underlining a section of text with
equal signs. Subheadings can be acomplished using dashes instead.

**Step 2: Add your file to the index:** New markdown files must
be added to the index in order to make them discoverable via the
navigation panes of our docs. Open ``timesoft/docs/source/index.rst``
and find the section at the end with ``.. toctree::`` followed by a list of 
the docs pages, and add ``funmath`` to the list in the location
where it best fits. You'll end up with something like this:

.. code-block:: markdown

    .. toctree::
        contrib_rules
        data_structs
        raw
        funmath
        ...

**Step 3: Generate documentation for the module:** Next,
go to ``timesoft/docs/source/api.rst``. This page 
contains listings of the available classes, methods, 
and functions in each module. Determine where your 
module belongs in the list, and add the
following markdown to the end of the document:

.. code-block:: markdown

    timesoft.funmath
    ----------------
    .. currentmodule:: funmath

    .. autosummary::
        :toctree: api

        funaddition
        funsubtraction
        ...

Where the list under ``.. autosummary::`` contains all 
of the important classes, methods, functions, and other
objects in your module (in this example, funaddition,
funsubtraction, and so on).

The autosummary imports the funmath module then 
collects and formats all the doc strings.

**Step 4: Remake the documentation pages:** finally, navigate
to ``timesoft/docs`` in the command line and run ``make html``. 
This will generate the HTML for all the webpages. The HTML 
files are stored in ``timesoft/docs/build/html``. Scroll up 
through the messages output in the console and make sure no
major errors occurred (they will probably be highlighted in 
red text). Often, when new items are added to the API the 
``make html`` will be unable to find them the first time it 
is run. Try running it a second time and many of the errors
may be resolved. Then you can push your updated documentation 
to the github repository.

.. note::

    Once we have a plan for how/where/if to host these docs, we'll
    want to add instructions for updating the code there. 

Jupyter/IPython Notebooks with Sphinx
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sphinx supports directly importing Jupyter notebooks into
your documentation. If you have a notebook demonstrating
how to use a piece of code and providing some commentary 
via the notebook's markdown cells, you can include a copy
in the ``timesoft`` documentation.

To do this, first make sure that all cells in your notebook
have been run and show the desired output. Then place a 
copy in the ``timesoft/docs/source/notebooks`` directory.
From there all you have to do is add it to the toctree on 
the index.rst page. Open ``timesoft/docs/source/index.rst``
and add ``notebooks/[filename]`` at the location where 
tutorial document belongs.

More details on this feature are available 
`here <https://docs.readthedocs.io/en/stable/guides/jupyter.html>`_.


Troubleshooting
---------------

**Relative Imports:** sphinx seems to have trouble with relative
imports. If you're writing code form moduleX and want to import
functionA from moduleY, the relative import you would use is
``from ..moduleY import functionA``. This may cause sphinx's 
autodoc module to fail when loading the code. Instead try using
``from timesoft.moduleY import functionA``. This seems to be a
workable alternative.

**conda environments:**
running ``make docs``` to run ``sphinx`` is 
`problematic <https://stackoverflow.com/a/4249730>`_ inside a ```conda``` 
environment, due to how the ``sphinx`` ``Makefile`` is generated. It calls 
``sphinx-build``, which may be a system install, rather than the one installed 
in the ``conda`` env. To explicitly specify the right ``sphinx-build`` executable, 
do

.. code-block:: console
    
    make html SPHINXBUILD='python <path_to_sphinx>/sphinx-build'

After the ``sphinx`` dependencies are installed in the environment, you can find 
this executable in a place like ``~/.conda/envs/<env-name>/bin/sphinx-build``.