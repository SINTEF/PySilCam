===============================
PySilCam ML tools
===============================

All the code for training the classification models.

In 2020 we moved from mostly using TF to Pytorch, mostly because
of dependency and compatibility issues.

Network training
--------

To train the network/a new network then first you need to get the
training data. This is currently the data from Azure. William split
off 10% of this data for testing, and this is done using the script:
    `train_test_folders.py`
Once you have that data you need to specify 4 folders, will full paths
in the train_network.py script
 1. silcam_directory
 2. train_dir (data for training, will be split into train/val)
 3. test_dir (not actually used currently, so can be any string)
 4. model_dir (a folder in which the model files will be saved)

Once all the data is in place you can navigate to the 
`PySilCam/torch_tools` folder, with the conda silcam conda env
running, then simply: `python train_network.py`.

Additional help
--------

* `test_network.py` is main purpose of this folder, and should train networks.
* `test_trained_networks.py` is a quickly written file. Should be updated when we get full testing image sets.
* `train_test_folders.py` is simply used to split data between two folder (for test/train splitting).

Problems, possible future work
--------

* There is a header.tfl.txt file, which contains a list of classes. This could be renamed and maybe change to a standard config file or such.