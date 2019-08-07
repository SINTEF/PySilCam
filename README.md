PySilCam
===============================

A Python interface to the SilCam.

Features
--------

* Acqusition of images from SilCam.
* Real-time processing of SilCam images, particle statistics.
* Processing images from disk (batch or real-time).

Pymba Requirements
------------------

* Python = 3.4

* Pymba with Python 3 support. Install using

    pip install git+https://github.com/mabl/pymba@python3


* For Python 2, the master branch Pymba is required. Install it with

    pip install -e git+https://github.com/morefigs/pymba#egg=pymba


Installing
----------
Install Anaconda:  https://www.anaconda.com/download/ Python 3.6 version

Anaconda Prompt may be used for the following

Create a virtual environment (preferably containing a username)

```
    conda create -n <name of the environment> python=3.5 scikit-image pandas seaborn
```

Unix: 

```
    source activate sctest 
```
    
Windows: 

```
    activate sctest
```


Install packages

```
    conda install --yes -c cogsci pygame psutil sphinx sphinxcontrib pyserial seaborn pyserial setuptools
```

```
    yes | pip install tensorflow==1.1.0 tflearn h5py cmocean psutil openpyxl Sphinx sphinxcontrib-napoleon
```


Test that it works with

```
    python setup.py test
```


Build the documentation

```
    python setup.py build_sphinx
```

See the [wiki](https://github.com/emlynjdavies/PySilCam/wiki) for more details on running PySilCam.


Docker
------
The Dockerfile will spin up an environment where Anaconda Python 3.5 is installed, and the source code folder (this folder) is mapped to /silcam. The _first_ time you must run this command to build a container (or any time the Dockerfile changes)

    docker-compose build

To run the tests inside the Docker environment:

    docker-compose up

To get an interactive shell session with the PySilcam Docker environment run:

    docker-compose run --rm --entrypoint /bin/bash silcam

Note: make sure to remove any .pyc files generated on the host system, as these can cause problems inside the Docker environment.


License
-------

BSD3 license. See LICENSE
