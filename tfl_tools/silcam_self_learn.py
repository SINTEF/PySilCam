from pysilcam.config import load_config, PySilcamSettings
import pysilcam.silcam_classify as sccl
import numpy as np
import pandas as pd
import os
from shutil import copyfile
import logging

logging.basicConfig(filename='self_learn.log',level=logging.DEBUG)

DATABASE_PATH = '/mnt/PDrive/silcam_classification_database'
config_file = '/mnt/ARRAY/ENTICE/Data/configs/config.ini'
stats_csv_file = '/mnt/ARRAY/ENTICE/Data/proc_nn/STN14-STATS.csv'
filepath = '/mnt/ARRAY/ENTICE/Data/export/'
model_path = '/mnt/ARRAY/classifier/model/'

confidence_threshold = [1, 0.05, 0.999, 0.05, 0.999]

DATABASE_selftaught_PATH = os.path.join(DATABASE_PATH,'../','silcam_self_taught_database')

header = pd.read_csv(os.path.join(model_path, 'header.tfl.txt'))
OUTPUTS = len(header.columns)
class_labels = header.columns
logging.info(class_labels)
logging.info(confidence_threshold)

for cl in class_labels:
    os.makedirs(os.path.join(DATABASE_selftaught_PATH,cl),exist_ok=True)

stats = pd.read_csv(stats_csv_file)

choice, confidence = sccl.choise_from_stats(stats)

for i,cl in enumerate(class_labels):
    class_label = 'probability_class' + str(i)
    logging.info(class_label)
    logging.info(class_labels[i])

    sstats = stats[(choice==class_label) & (confidence>confidence_threshold[i])]
    if len(sstats)==0:
        logging.info(0,'images')
        continue

    logging.info(len(sstats),'images')

    for j in np.arange(0,len(sstats)):
        filename = sstats.iloc[j]['export name']
        imfile = os.path.join(filepath,filename)
        copy_to_path = os.path.join(DATABASE_selftaught_PATH,
                class_labels[i],
                filename)

        #print('from:')
        #print(imfile)

        #print('to:')
        #print(copy_to_path)

        copyfile(imfile,copy_to_path)
