from docopt import docopt
import cmocean
import sys
from pysilcam.tracking.silcamtracker import *
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


def subsample_files(datapath, approx_files=2000, offset=int(0)):
    print('Subsampling files....')

    files = [os.path.join(datapath, f) for f in sorted(os.listdir(datapath)) if f.endswith('.bmp') or f.endswith('.silc')
            or f.endswith('.silc_mono')]
    files = files[int(offset):]

    try:
        times = [f.replace(datapath + '/D','').replace('.bmp','') for f in files]
        times = pd.to_datetime(times)
    except:
        try:
            times = [f.replace(datapath + '/D','').replace('.silc','') for f in files]
            times = pd.to_datetime(times)
        except:
            try:
                times = [f.replace(datapath + '/D','').replace('.silc_mono','') for f in files]
                times = pd.to_datetime(times)
            except:
                pass


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


def silctrack():
    """
    does tracking

    Usage:
        silcam-track process <datapath> [--offset=<offset>]
        silcam-track post-process <tracksfile>
        silcam-track plotting <tracksfile> [--gif=<outputdir>] [<rawdatapath>] [--boxplot]
    """

    #@todo intended final usage: silcam-track <configfile> <datapath> [--offset=<offset>]

    PIX_SIZE = 27.532679738562095
    print('!! HARDCODED PIX_SIZE:', PIX_SIZE)

    sctr = Tracker()

    # @todo read these settings from a normal silcam config file, with an extra place for tracking specific settings.

    sctr.av_window = 15
    # sctr.files = subsample_files(datapath, approx_files=200,
    #        offset=offset

    # sctr.files = sctr.files[-200:]
    sctr.MIN_LENGTH = 200
    # sctr.MIN_LENGTH = 200
    sctr.MIN_SPEED = 0.000001  # cm/s
    sctr.GOOD_FIT = 0.1
    sctr.THRESHOLD = 0.95
    sctr.ecd_tollerance = 5  # percent
    sctr.PIX_SIZE = PIX_SIZE

    print('!! HARDCODED SETTINGS')

    args = docopt(silctrack.__doc__)

    if args['process']:
        datapath = args['<datapath>']
        offset = args['--offset']
        if offset is not None:
            try:
                offset = int(offset)
            except ValueError:
                print('Expected type int for --offset.')
                sys.exit(0)
        else:
            offset = 0

        #outputpath = make_output_path(datapath)
        #print('* Output path:', outputpath)

        sctr.path = datapath
        sctr.DATAFILE = datapath
        sctr.initialise()
        sctr.files = sctr.files[offset:]

        print('sctr.DATAFILE',sctr.DATAFILE)
        sctr.process()

    if args['post-process']:
        print('* Load and process')
        data, tracks = load_and_process(args['<tracksfile>'], PIX_SIZE, track_length_limit=5)
        tracks.to_hdf(args['<tracksfile>'], 'Tracking/tracks', mode='r+')
        # @todo add track length limit used in processing to metadata

    if args['plotting']:
        if args['--gif']:


            if not checkgroup(args['<tracksfile>'], 'Tracking/unfiltered_tracks'):
                print('* Extracting unfiltered tracks')
                data = pd.read_hdf(args['<tracksfile>'], 'Tracking/data')
                unfiltered_tracks = extract_continuous_tracks(data, max_starts=10)
                with pd.HDFStore(args['<tracksfile>']) as fh:
                    unfiltered_tracks.to_hdf(fh, 'Tracking/unfiltered_tracks', mode='r+')
                print('  OK.')
            else:
                unfiltered_tracks = pd.read_hdf(args['<tracksfile>'], 'Tracking/unfiltered_tracks')

            outputdir = args['--gif']
            rawdatapath = args['<rawdatapath>']

            make_output_files_for_giffing(unfiltered_tracks, rawdatapath, outputdir, PIX_SIZE,
                                          track_length_limit = 5)
            #make_output_files_for_giffing_old(rawdatapath, outputdir, data, PIX_SIZE)

        if args['--boxplot']:
            print('unfinished code')
            return
            # @todo This should look for a bunch of h5 files in a given directory and use these to make a dictionary of
            #       tracks for the boxplotting
            return
            # do some glob search for h5 files

            # loop trough tracks files and add them to a dict
            data = dict()
            tracks = dict()
            print('* Loading', tracksfile)
            tracks = pd.read_hdf(tracksfile, 'Tracking/tracks')
            print('  OK.')
            dataset_names = list(tracks)
            print('* Creating boxplot from:')
            print(dataset_names)
            print('WAITING FOR INPUT BEFORE PROCEEDING')
            input()
            make_boxplot(dataset_names, tracks, PIX_SIZE,
                         '/mnt/nasdrive/Miljoteknologi/PlasticSettling2020/proc/boxplot')