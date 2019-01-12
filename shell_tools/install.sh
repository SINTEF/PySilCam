sudo apt-get install -y git wget

## vimba installation
#wget https://www.alliedvision.com/fileadmin/content/software/software/Vimba/Vimba_v2.1.3_Linux.tgz
#tar -xvzf Vimba_v2.1.3_Linux.tgz
#bash /Vimba_2_1/VimbaGigETL/Install.sh

## python installation
#wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
#bash Anaconda3-5.1.0-Linux-x86_64.sh

#conda update --yes -n base conda
#yes | pip install --upgrade pip

## pymba installation
#RUN pip install git+https://github.com/mabl/pymba@python3

# pysilcam requirements
conda install --yes opencv=3.4.2
conda install --yes pytables=3.4.4
cd ../
python setup.py develop
python setup.py test
