rem install conda packages
call conda install --yes pytables=3.4.4
call conda install --yes opencv=3.4.2

rem install other python packages and run tests
cd ..\
python setup.py develop
python setup.py test


