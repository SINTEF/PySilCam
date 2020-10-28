from pysilcam.silcam_classify import check_model, load_model, predict
from skimage.io import imread
import glob
import os
import numpy as np
import unittest

# Get user-defined path to unittest data folder
ROOTPATH = os.environ.get('UNITTEST_DATA_PATH', None)

# Get user-defined tensorflow model path from environment variable
MODEL_PATH = os.environ.get('SILCAM_MODEL_PATH', None)

print('ROOTPATH', ROOTPATH)
print('MODEL_PATH', MODEL_PATH)


@unittest.skipIf((ROOTPATH is None),
                 "test path not accessible")
def test_classify():
    '''
    Basic check of classification prediction against the training database.
    Therefore, if correct positive matches are not high percentages, then something is wrong with the prediction.

    @todo include more advanced testing of the classification feks. assert values in a confusion matrix.
    '''

    # location of the training data
    database_path = os.path.join(ROOTPATH, 'silcam_classification_database')

    check_model(MODEL_PATH)

    # Load the trained tensorflow model and class names
    model, class_labels = load_model(MODEL_PATH)

    # class_labels should match the training data
    classes = glob.glob(os.path.join(database_path, '*'))

    # @todo write a quick check that classes and class_labels agree before doing the proper test.

    def correct_positives(category):
        '''
        calculate the percentage positive matches for a given category
        '''

        # list the files in this category of the training data
        files = glob.glob(os.path.join(database_path, category, '*.tiff'))

        # start a counter of incorrectly classified images
        failed = 0

        # loop through the database images
        for file in files:

            img = imread(file)  # load ROI
            prediction = predict(img, model)  # run prediction from silcam_classify

            ind = np.argmax(prediction)  # find the highest score

            # check if the highest score matches the correct category
            if not class_labels[ind] == category:
                # if not, the add to the failure count
                failed += 1

        # turn failed count into a success percent
        success = 100 - (failed / len(files)) * 100
        return success


    # loop through each category and calculate the success percentage
    for cat in classes:
        name = os.path.split(cat)[-1]
        success = correct_positives(name)
        print(name, success)
        assert success > 96, (name + ' was poorly classified at only ' + str(success) + 'percent.')
