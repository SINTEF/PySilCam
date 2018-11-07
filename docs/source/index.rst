PySilCam
=================================================

Overview
-------------

The following main processing steps are applied to each image recorded by the silhouette system:

1: Each image is corrected by a clean background to reduce noise

2: The corrected image is segmented (binarized) to produce a logical image (zeros and ones) of the particles detected

3: Particles in the binary image are then counted and particle properties (geometry and particle type) are calculated for each particle

4: The particle size distribution is calculated by counting Equivalent Circular Diameters (ECD) into their appropriate volume size class (this is done in a similar way to the log-spaced size bins used by Sequoia Scientific's LISST and LISST-HOLO (http://holoproc.marinephysics.org) instruments, but extending up to a maximum diameter of several cm).

The background image used for correction of each raw is calculated from an average of images recorded immediately prior to the raw image being processed. This allows for real-time moving average background correction. The correction of images reduces noise and gradients in background illumination and small fouling artifacts that may appear of the optics.

Top-Level functions and entry points
=================================================

.. autofunction:: pysilcam.__main__.silcam
.. autofunction:: pysilcam.silcreport.silcreport

Main processing functions
=================================================

Processing and analysis is performed using the below functions.

.. autofunction:: pysilcam.__main__.processImage
.. autofunction:: pysilcam.process.statextract
.. autofunction:: pysilcam.process.image2blackwhite_fast
.. autofunction:: pysilcam.process.image2blackwhite_accurate
.. autofunction:: pysilcam.process.clean_bw
.. autofunction:: pysilcam.process.measure_particles

Calssification functions
=================================================

.. automodule:: pysilcam.silcam_classify
   :members:
   :undoc-members:
   :show-inheritance:

PySilCam contents
=================================================

.. toctree::
   :maxdepth: 2

   pysilcam