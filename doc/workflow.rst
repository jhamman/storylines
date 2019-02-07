Chains Workflow Tutorial
========================

The Storylines Project is developing a large ensemble of hydrologic model
projections. It is using a Snakemake workflow currently stored in a separate
repository - Chains_. This tutorial provides the basic steps needed to setup
this workflow.

.. _Chains: https://github.com/jhamman/chains


Setup Python Environment
~~~~~~~~~~~~~~~~~~~~~~~~

These steps will setup the Python environemnt needed to run the Chains workflow.
If you already have `conda` installed, you can skip step (1). You will only run
these steps once.

0. Clone this Repository
------------------------

.. code-block:: bash

  git clone https://github.com/jhamman/chains.git

1. Install Miniconda
--------------------

.. code-block:: bash

  # download Miniconda
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;

  # install Miniconda
  bash miniconda.sh -b -p $HOME/miniconda

Follow the installer instructions and be sure to add the miniconda installation
to your path.

2. Setup the Storylines Environment
-----------------------------------

.. code-block:: bash

  conda env create --name storylines --file environment.yml

Follow the instructions and when the installer is done, activate the environment.

.. code-block:: bash

  source activate storylines

Setup the workflow inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

TODO. For now, I'll provide these.

Run the workflow
~~~~~~~~~~~~~~~~

1. Run the `hydrology_models` rule on the local node. If you are running on
Cheyenne, this should be done on an interactive node.

.. code-block:: bash

  snakemake hydrology_models --configfile ../storylines_test_data/test_config.yml

The contents of the config file is described in the previous section.

2. Run the `hydrology_models` rule on the full cluster.

.. code-block:: bash

  snakemake hydrology_models --jobs 5000 --local-cores 1 \
  --configfile config.yml --cluster-config config.yml \
  --cluster "{cluster.submit_cmd} {cluster.walltime} {cluster.resources} {cluster.account} {cluster.queue}" \
  --jobscript jobscript.sh \
  --jobname {rulename}.{jobid}.sh --latency-wait 30 -p \
  --max-jobs-per-second 2
