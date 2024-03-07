PySilCam
===============================

![Docker build and test](https://github.com/SINTEF/PySilCam/workflows/Docker%20build%20and%20test/badge.svg)

A Python interface to the SilCam.

For user-level descriptions and guidance, please see the [wiki](https://github.com/SINTEF/PySilCam/wiki).

Documentation for PySilCam code can be found here: https://pysilcam.readthedocs.io

Features
--------

* Acqusition of images from SilCam.
* Real-time processing of SilCam images, particle statistics.
* Processing images from disk (batch or real-time).

Requirements
------------

* Python = 3.5

* Pymba (camera acquisition) with Python 3 support. Install using

```bash
pip install git+https://github.com/mabl/pymba@python3
```

Installing
----------

If you are using windows, you need Microsoft Visual C++ Redistributable, which can be downloaded [here](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

Install [Python 3.6 or later](https://github.com/conda-forge/miniforge/#download)

A prompt such as is provided by the [miniforge installation](https://github.com/conda-forge/miniforge/#download) may be used for the following:

Create a virtual environment using the environment.yml (will create an environment called silcam)

```bash
conda env create -f environment.yml
```

to update, we recommend a forced re-install:

```bash
conda env create -f environment.yml --force
```

to activate:

```bash
conda activate silcam
```

Test that it works with

```bash
python setup.py develop
```

```bash
python setup.py test
```

Build the documentation

```bash
python setup.py build_sphinx
```

Auto-building of the documentation is performed by Read The Docs, with this project: https://readthedocs.org/projects/pysilcam/

Jupyter notebook setup
------

After installing, activate the silcam conda environment. Then run:

'''bash
conda install -c anaconda ipykernel jupyter
'''

Then you should be able to see silcam as an available kernel after starting a Jupyter notebook

Docker
------

The Dockerfile will spin up an environment where Anaconda Python 3.5 is installed, and the source code folder (this folder) is mapped to /silcam. The _first_ time you must run this command to build a container (or any time the Dockerfile changes)

```bash
docker-compose build
```

Note: you might need sudo here.

To run the tests inside the Docker environment:

```bash
docker-compose up
```

To get an interactive shell session with the PySilcam Docker environment run:

```bash
docker-compose run --rm --entrypoint /bin/bash silcam
```

Note: make sure to remove any .pyc files generated on the host system, as these can cause problems inside the Docker environment.

If you want to mount a local folder inside the docker environment, use -v [local path:docker mount point]:

```bash
docker-compose run -v /mnt/ARRAY/:/mnt/ARRAY/ --rm --entrypoint /bin/bash silcam
```

Contributions
-------------

We welcome additions and improvements to the code!

However, we also request that you follow a few guidelines. These are in place to make sure the code improves over time.

1. All code changes must be submitted as pull requests, either from a branch or a fork.
2. All pull requests are required to pass all tests. Please do not disable or remove tests just to make your branch pass the pull request.
3. All pull requests must be reviewed by a person. The benefits from code review are plenty, but we like to emphasise that code reviews help spreading the awarenes of code changes. Please note that code reviews should be a pleasant experience, so be plesant, polite and remember that there is a human being with good intentions on the other side of the screen.
4. All contributions are linted with flake8. We recommend that you run flake8 on your code while developing to fix any issues as you go.
5. We recommend using autopep8 to autoformat your Python code (but please check the code behaviour is not affected by autoformatting before pushing). This makes flake8 happy, and makes it easier for us all to maintain a consistent and readable code base.

License
-------

PySilCam is licensed under the BSD3 license. See LICENSE.
