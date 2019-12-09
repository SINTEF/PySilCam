import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pysilcam.fakepymba import silcam_load, silcam_name2time
from glob import glob
import time
import pandas as pd

def main(datapath):
    print(datapath)
    while True:
        if True:
            plt.figure()
            plt.ion()
            plt.show()
            while True:
                files = sorted(glob(os.path.join(datapath, '*.silc')))
                if len(files)<5:
                    print('waiting for data')
                    time.sleep(1)
                    continue
                idx = np.random.randint(0, len(files))
                idx = -2
                print(idx)
                img = silcam_load(files[idx])

                timestamp = silcam_name2time(os.path.split(files[idx])[-1])
                print(timestamp)

                start_time = silcam_name2time(os.path.split(files[0])[-1])
                end_time = silcam_name2time(os.path.split(files[-1])[-1])

                dt = pd.Timedelta(end_time - start_time)
                av_freq = 1 / (dt.total_seconds() / len(files))

                start_time = silcam_name2time(os.path.split(files[-5])[-1])
                end_time = silcam_name2time(os.path.split(files[-1])[-1])
                dt = pd.Timedelta(end_time - start_time)
                av_freq_recent = 1 / (dt.total_seconds() / 4)


                plt.cla()
                plt.imshow(img)
                title = ('clock: ' + str(pd.datetime.now()) +
                         '\nLatest data: ' + str(timestamp) + '  ' +
                         str(len(files)) + ' files\n' +
                         str(np.round(av_freq,1)) + ' Hz  dataset average  ' +
                         str(np.round(av_freq_recent, 1)) + ' Hz recent  ')
                plt.title(title)
                plt.gca().axis('off')
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)

                time.sleep(1)
        #except:
        #    pass
        time.sleep(1)

if __name__ == '__main__':
    main(sys.argv[1])