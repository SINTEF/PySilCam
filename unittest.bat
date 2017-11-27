@echo off
REM Default installs
activate sctest && conda install tensorflow -c conda-forge && pip install tflearn && pip install psutil && python setup.py test
