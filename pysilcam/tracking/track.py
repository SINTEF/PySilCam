'''
SilCam Tracking wrapper for particleinstrumenttools/silcamtracker analysis tools

Usage:
  track.py bottle <day> <bottle>
  track.py column <dataset> <offset>
  track.py plastic <dataset> <offset>
  track.py eggs <dataset> <offset>
  track.py image <dataset> <offset>
  track.py plot <dataset>
'''


from docopt import docopt
import cmocean
import sys
from silcamtracker import *
import numpy as np
import sys
import glob
import skimage.io as skio
import pandas as pd
from scipy import interpolate

def make_output_path(datapath):
    dataset_name = os.path.split(datapath)[1]
    outputpath = os.path.join(datapath, ('../output_' + dataset_name))
    os.makedirs(outputpath,exist_ok=True)
    return outputpath


def eggs(datapath, offset=0):

    outputpath = make_output_path(datapath)
    print('* Output path:', outputpath)

    sctr = Tracker()

    sctr.path = datapath
    sctr.DATAFILE = outputpath

    sctr.av_window = 50
    #sctr.files = subsample_files(datapath, offset=offset)
    sctr.initialise()
    sctr.files = sctr.files[offset:]
    sctr.MIN_LENGTH = 500
    sctr.MIN_SPEED = 0.000001 # cm/s
    sctr.GOOD_FIT = 0.1
    sctr.THRESHOLD = 0.95
    sctr.ecd_tollerance = 5 # percent
    sctr.PIX_SIZE = 27.532679738562095
    sctr.process()


def plastic(datapath, offset=0):

    outputpath = make_output_path(datapath)
    print('* Output path:', outputpath)

    sctr = Tracker()

    sctr.path = datapath
    sctr.DATAFILE = outputpath

    sctr.av_window = 50
    #sctr.files = subsample_files(datapath, offset=offset)
    sctr.initialise()
    sctr.files = sctr.files[offset:]
    sctr.MIN_LENGTH = 30
    #sctr.MIN_LENGTH = 200
    sctr.MIN_SPEED = 0.000001 # cm/s
    sctr.GOOD_FIT = 0.1
    sctr.THRESHOLD = 0.95
    sctr.ecd_tollerance = 5 # percent
    sctr.PIX_SIZE = 27.532679738562095 / 2
    sctr.process()

def column_track(datapath, offset):

    outputpath = make_output_path(datapath)

    sctr = Tracker()

    sctr.path = datapath
    sctr.DATAFILE = outputpath

    sctr.av_window = 5
    sctr.files = subsample_files(datapath, offset=offset)
    sctr.initialise()
    sctr.MIN_LENGTH = 500
    sctr.MIN_SPEED = 0.0001 # cm/s
    sctr.GOOD_FIT = 0.2
    sctr.process()


def subsample_files(datapath, approx_files=2000, offset=int(0)):
    print('Subsampling files....')

    files = [os.path.join(datapath, f) for f in sorted(os.listdir(datapath)) if f.endswith('.bmp')]
    files = files[int(offset):]

    times = [f.replace(datapath + '/D','').replace('.bmp','') for f in files]
    times = pd.to_datetime(times)

    t1 = times[0]

    dt = np.abs(times-t1)

    secs = (dt.components.hours*3600) + (dt.components.minutes*60) + dt.components.seconds + (dt.components.milliseconds / 1000)
    image_number = np.arange(0, len(secs))

    base=10
    start=0
    stop=max(secs)
    sample_secs = np.logspace(start, np.log(stop)/np.log(base), num=approx_files, base=base)
    sample_secs = sample_secs[sample_secs<max(secs)]
    f = interpolate.interp1d(secs, image_number, kind='nearest')
    sample_idx = f(sample_secs)
    sample_idx_ = []
    for i in sample_idx:
        sample_idx_.append(int(i))
    sample_idx = np.unique(sample_idx_)

    # plt.plot(image_number, secs,'ko')
    # plt.plot(sample_idx, secs[sample_idx],'r.', markersize=1)

    len(sample_idx)

    files = np.array(np.copy(files))
    resampled_files = sorted(list(files[sample_idx]))

    print('  ', str(len(resampled_files)) ,'files from', str(len(files)))

    return resampled_files


def bottle_track(day, bottle):

    ROOT = '/mnt/nasdrive/Miljoteknologi/302003070_Marine snow flocculation and sedimentation in relation to oil spill responses_OGB/'

    sctr = Tracker()

    sctr.path = os.path.join(ROOT,
	                         'RawData/WP3/d' + day + '/d' + day + '_S' + bottle)
    sctr.DATAFILE = os.path.join(ROOT,
	                             'AutoProcessing/tracking/WP3-d' + day + '-S' + bottle + '/WP3-d' + day + '-S' + bottle)

    sctr.av_window = 15
    sctr.initialise()
    #sctr.files_all = np.copy(sctr.files)
    #sctr.files = sctr.files_all[10:]
    sctr.process()

def plot_single(datapath):
    outputpath = make_output_path(datapath)
    csv_file = (outputpath + '.csv')
    print('CSV file:', csv_file)

    data = pd.read_csv(csv_file)
    print(data.columns)

    f, a = plt.subplots(2,2,figsize=(12,12))

    plt.sca(a[0,0])
    plt.plot(pd.to_datetime(data['Time']), data['ECD [mm]'], '.', color='0.8', alpha=0.1)
    plt.ylabel('ECD [mm]')

    plt.sca(a[1,0])
    plt.plot(pd.to_datetime(data['Time']), data['Speed [cm/s]'], '.', color='0.8', alpha=0.1)
    plt.ylabel('Speed [cm/s]')

    plt.sca(a[0,1])
    plt.plot(data['ECD [mm]'], data['Ws [cm/s]'], '.', color='0.8', alpha=0.1)
    plt.xlabel('ECD [mm]')
    plt.ylabel('Ws [cm/s]')

    plt.sca(a[1,1])
    plt.plot(data['ECD [mm]'], data['Speed [cm/s]'], '.', color='0.8', alpha=0.1)
    plt.xlabel('ECD [mm]')
    plt.ylabel('Speed [cm/s]')

    plt.show()


def show_image(datapath, offset):
    files = sorted(glob.glob(datapath + '/*.bmp'))
    files = files[int(offset):]

    for i, f in enumerate(files):
        print(os.path.split(f)[1])
        im = skio.imread(f)
        plt.figure()
        plt.imshow(im, cmap=cmocean.cm.matter)
        plt.title(i+int(args['<offset>']))
        plt.show()


if __name__ == "__main__":
    args = docopt(__doc__)
    # args['<dataset>'] could be 'settling_d6_19'

    # root directory
    datapath = '/mnt/nasdrive/Miljoteknologi/302003070_Marine snow flocculation and sedimentation in relation to oil spill responses_OGB/RawData/Settling/'

    # functions below here require datapath to point to the actual data
    datapath = os.path.join(datapath, args['<dataset>'])

    if args['bottle']:
        bottle_track(args['day'], args['bottle'])

    if args['column']:
        column_track(datapath, int(args['<offset>']))

    if args['plastic']:
        datapath = args['<dataset>']
        plastic(datapath, int(args['<offset>']))

    if args['eggs']:
        datapath = args['<dataset>']
        eggs(datapath, int(args['<offset>']))

    if args['image']:
        show_image(datapath, args['<offset>'])

    if args['plot']:
        plot_single(datapath)
