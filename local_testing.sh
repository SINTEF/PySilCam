#!/bin/bash

# source ~/anaconda3/etc/profile.d/conda.sh
# # # conda deactivate
# conda env create -f environment.yml --force
# # # conda deactivate

# # conda env update --file environment.yml --prune
# conda activate silcam

# # export SILCAM_MODEL_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/COAPNetTorch/COAPNet_model.pt'
# # export UNITTEST_DATA_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/unittest-data/'
# export UNITTEST_DATA_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/unittest-data/'
# export SILCAM_MODEL_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/tflmodel/particle-classifier.tfl'

# Here are all the rests run in test_noskip
# pysilcam/tests/test_background.py .                                                                                                                                               [ 10%]
# pysilcam/tests/test_config.py ...                                                                                                                                                 [ 40%]
# pysilcam/tests/test_process.py FF                                                                                                                                                 [ 60%]
# pysilcam/tests/test_silcam-aquire.py .                                                                                                                                            [ 70%]
# pysilcam/tests/test_standards.py FF                                                                                                                                               [ 90%]
# pysilcam/tests/test_z_classify.py .    

# cd pysilcam/tests
# # pytest -W ignore -s test_process.py
# pytest -s test_process.py
# # pytest -s test_z_classify.py

# python setup.py test_noskip

# # Testing as in main:
# conda env create -f environment.yml --force
# conda activate silcam

# export UNITTEST_DATA_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/unittest-data/'
# export SILCAM_MODEL_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/tflmodel/particle-classifier.tfl'
# python setup.py test_noskip


# Testing as in main:
# conda env update --file environment.yml --prune
# conda activate silcam

# export UNITTEST_DATA_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/unittest-data/'
# export SILCAM_MODEL_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/tflmodel/particle-classifier.tfl'
# export UNITTEST_DATA_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/unittest-data/'
# export SILCAM_MODEL_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata/COAPNetTorch/COAPNet_model.pt'
export UNITTEST_DATA_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata-newtorch/unittest-data/'
export SILCAM_MODEL_PATH='/Users/odin/Sintef/SilCam/pysilcam-testdata-newtorch/COAPNetTorch/COAPModNet_CPU_model.pt'
python setup.py test_noskip