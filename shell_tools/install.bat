rem install conda packages
call conda install --yes -c cogsci pygame=1.9.2a0 psutil=5.4.7 sphinx=1.7.9 sphinxcontrib=1.0 pyserial=3.4 seaborn=0.9.0 setuptools=40.2.0 pytables=3.4.4

rem install other python packages and run tests
cd ..\
python setup.py develop
python setup.py test


