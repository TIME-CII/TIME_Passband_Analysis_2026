Installing ``timesoft``
=======================

Installation of the ``timesoft`` package is simple.
Begin by cloning the TIME-Analysis 
`github repo <https://github.com/TIME-CII/TIME-analysis>`_.

.. code-block:: console

    cd [location where you want to copy the repo]
    git clone https://github.com/TIME-CII/TIME-analysis.git

Which will create a local copy of the TIME software 
on your computer. Now navigate to the ``TIME-analysis/timesoft``
directory and install ``timesoft`` using pip:

.. code-block:: console

    cd TIME-analysis/timesoft
    pip install -e .

Since ``timesoft`` is under active development the code 
is likely to change. Installing with the ``-e`` flag 
tells your python interpreter to automatically pull in
changes made to the code. This way, all you have to do
to keep your installation of ``timesoft`` synced with 
the latest changes is get the latest changes to the 
git repo using

.. code-block:: console

    git pull


