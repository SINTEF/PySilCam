===============================
PySilCam
===============================

.. image:: docs/silcam.png


A Python interface to the SilCam.

Features
--------

* Acqusition of images from SilCam
* Real-time processing of SilCam images, particle statistics
* Processing images from disk (batch or real-time)

Quick start
-----------
To process images from disk in real time (watching a folder for new images):

    > PYSILCAM_REALTIME_DATA='<path>' MPLBACKEND='TkAgg' silcam-acquire process pysilcam/config_example.ini


Requirements
------------

* Python = 3.4

* numpy

* Pymba with Python 3 support. Install using

    pip install git+https://github.com/mabl/pymba@python3


* For Python 2, the master branch Pymba is required. Install it with

    pip install -e git+https://github.com/morefigs/pymba#egg=pymba


License
-------

MIT licensed. See the bundled `LICENSE` file for more details.
