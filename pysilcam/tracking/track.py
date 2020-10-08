from docopt import docopt
from pysilcam.tracking.silcamtracker import *
import numpy as np
import sys
import glob
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from skimage.transform import rotate
from pysilcam.postprocess import explode_contrast


def make_output_path(datapath):
    dataset_name = os.path.split(datapath)[1]
    outputpath = os.path.join(datapath, ('../output_' + dataset_name))
    os.makedirs(outputpath,exist_ok=True)
    return outputpath


def make_boxplot(dataset_names, tracks, PIX_SIZE, figurename):
    ps = np.arange(0,len(dataset_names))
    ls = dataset_names

    f, a = plt.subplots(3,1,figsize=(6,12))

    plt.sca(a[0])
    box_data = [tracks[dataset_names[i]]['width']/
                tracks[dataset_names[i]]['length']
                for i in range(len(dataset_names))]
    plt.boxplot(box_data, positions=ps, labels=ls)
    plt.ylabel('Minor/Major axis')
    plt.ylim(0, 1)
    plt.gca().xaxis.tick_top()
    plt.xticks(rotation=45, horizontalalignment='left')

    plt.sca(a[1])
    box_data = [tracks[dataset_names[i]]['length']*PIX_SIZE/1000
                for i in range(len(dataset_names))]
    plt.boxplot(box_data, positions=ps, labels=ls)
    plt.ylabel('Maxjor Axis Length [mm]')
    plt.xticks([])
    plt.ylim(0, 12)

    plt.sca(a[2])
    box_data = [tracks[dataset_names[i]]['S_cms']*10
                for i in range(len(dataset_names))]

    plt.boxplot(box_data, positions=ps, labels=ls)
    plt.ylabel('Net speed [mm/s]')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.ylim(0, 2.5)

    figurename = os.path.join(figurename + '.png')
    print('  saving:', figurename)
    plt.savefig(figurename,dpi=600, bbox_inches='tight')
    print('  saved')


def im_from_timestamp(timestamp, rawdatapath):
    searchfilename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')
    hits = glob.glob(os.path.join(rawdatapath, searchfilename + '*'))
    imagename = hits[0]
    im = silcam_load(imagename)
    return im


def make_output_files_for_giffing(data, rawdatapath, outputdir, PIX_SIZE, track_length_limit=15):
    '''
    # use 'convert -delay 12 -loop 0 *.png output.gif' to make a gif
    :param data:
    :param rawdatapath:
    :param outputdir:
    :param PIX_SIZE:
    :param track_length_limit:
    :return:
    '''

    print('* make_output_files_for_giffing')

    os.makedirs(outputdir, exist_ok=True)

    u = np.unique(data['t2'])

    for timestamp in u:
        print('making tracks for', timestamp)
        tmptracks = data[pd.to_datetime(data['t2']) == pd.to_datetime(timestamp)]
        print('len(tmptracks)', len(tmptracks))
        if len(tmptracks) == 0:
            continue

        try:
            im = im_from_timestamp(pd.to_datetime(timestamp), rawdatapath)
            im = np.uint8(np.min(im, axis=2))
        except:
            print('could not load image from:', str(tmptracks.iloc[0]['t2']))
            continue

        plt.figure(figsize=(7, 10))
        r, c = np.shape(im)
        plt.imshow(rotate(explode_contrast(np.uint8(im)),
                          270, resize=True),
                   cmap='gray',
                   extent=[0, r * PIX_SIZE / 1000,
                           0, c * PIX_SIZE / 1000])
        plt.title(str(timestamp))

        subset = data[data['t2'] <= pd.to_datetime(timestamp)]

        for p in subset['ParticleName'].values:
            this_particle = subset[subset['ParticleName'] == p]
            if len(this_particle) == 0:
                continue

            t_arr = min(this_particle['t1'])
            t_dep = max(this_particle['t2'])
            x_arrival = this_particle[this_particle['t1'] == t_arr]['x1'].values[0]
            x_departure = this_particle[this_particle['t2'] == t_dep]['x2'].values[0]
            linex = np.float64([x_arrival, x_departure])
            y_arrival = this_particle[this_particle['t1'] == t_arr]['y1'].values[0]
            y_departure = this_particle[this_particle['t2'] == t_dep]['y2'].values[0]
            liney = np.float64([y_arrival, y_departure])

            speed = calculate_speed(x_arrival, x_departure, y_arrival, y_departure, t_arr, t_dep, PIX_SIZE)

            linex *= PIX_SIZE / 1000
            liney *= PIX_SIZE / 1000
            linex = c * PIX_SIZE / 1000 - linex
            liney = r * PIX_SIZE / 1000 - liney
            n_tracks = len(this_particle)  # should be the same as max(this_particle['n-tracks'])
            plot_color = 'r-'
            if n_tracks > track_length_limit:
                plot_color = 'g-'

                plt.text(liney[0], linex[0], (p + '\n{:0.2f}mm {:0.2f}mm/s'.format(this_particle['length'].values[0] * PIX_SIZE / 1000,
                                                                         speed * 10)),
                        fontsize=4, color='g')

            plt.plot(liney, linex, plot_color, linewidth=1)

        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.gca().invert_yaxis()

        name = pd.to_datetime(timestamp).strftime('D%Y%m%dT%H%M%S.%f')
        print('saving', os.path.join(outputdir, name + '-tr.png'))
        plt.savefig(os.path.join(outputdir, name + '-tr.png'), dpi=300, bbox_inches='tight')


def calculate_speed(x_arr, x_dep, y_arr, y_dep, t_arr, t_dep, PIX_SIZE):
    X = np.array([x_arr, x_dep])
    Y = np.array([y_arr, y_dep])

    dt = pd.Series(np.abs(t_arr - t_dep)).dt.total_seconds()
    dt = dt.values

    Y_mm = Y*PIX_SIZE*1e-3
    X_mm = X*PIX_SIZE*1e-3
    dY_mm = np.diff(Y_mm, axis=0)
    dX_mm = np.diff(X_mm, axis=0)
    dD_mm = np.sqrt(dY_mm**2 + dX_mm**2)
    dD_m = dD_mm / 1000
    S_ms = dD_m/dt # speed in m/s
    S_cms = S_ms * 100 # speed in cm/s

    return S_cms[0]


def checkgroup(h5filename, groupstr):
    '''check if a groupstr exists in the hdf5 file, h5filename
    Args:
        h5filename (str)        : hdf5 file name
        groupstr (str)          : path to search for existence

    Returns:
        bool
    '''
    objs = []
    with h5py.File(h5filename) as f:
        f.visit(objs.append)
        groups = [obj for obj in objs if isinstance(f[obj], h5py.Group)]
    return groupstr in groups


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


def load_and_process(tracksfile, PIX_SIZE,
                     minlength=0, maxlength=1e6, track_length_limit=15):
    data = pd.read_hdf(tracksfile,'Tracking/data')
    tracks = post_process(data, PIX_SIZE,
            track_length_limit=track_length_limit,
                          minlength=minlength, maxlength=maxlength)
    return data, tracks


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
    sctr.MIN_LENGTH = 200
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

        sctr.path = datapath
        sctr.DATAFILE = datapath
        sctr.initialise()
        sctr.files = sctr.files[offset:]

        print('sctr.DATAFILE',sctr.DATAFILE)
        sctr.process()

    if args['post-process']:
        print('* Load and process')
        data, tracks = load_and_process(args['<tracksfile>'], PIX_SIZE, track_length_limit=5)

        with pd.HDFStore(args['<tracksfile>']) as fh:
            tracks.to_hdf(fh, 'Tracking/tracks', mode='r+')
            # @todo add track length limit used in processing to metadata

        unfiltered_tracks = extract_continuous_tracks(data)
        with pd.HDFStore(args['<tracksfile>']) as fh:
            unfiltered_tracks.to_hdf(fh, 'Tracking/unfiltered_tracks', mode='r+')

    if args['plotting']:
        if args['--gif']:

            if not checkgroup(args['<tracksfile>'], 'Tracking/unfiltered_tracks'):
                print('* Extracting unfiltered tracks')
                data = pd.read_hdf(args['<tracksfile>'], 'Tracking/data')
                unfiltered_tracks = extract_continuous_tracks(data)
                with pd.HDFStore(args['<tracksfile>']) as fh:
                    unfiltered_tracks.to_hdf(fh, 'Tracking/unfiltered_tracks', mode='r+')
                print('  OK.')
            else:
                unfiltered_tracks = pd.read_hdf(args['<tracksfile>'], 'Tracking/unfiltered_tracks')

            outputdir = args['--gif']
            rawdatapath = args['<rawdatapath>']

            make_output_files_for_giffing(unfiltered_tracks, rawdatapath, outputdir, PIX_SIZE,
                                          track_length_limit = 5)

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