from docopt import docopt
import pysilcam.tracking.silcamtracker as sctracker
import numpy as np
import sys
import glob
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from skimage.transform import rotate
from pysilcam.postprocess import explode_contrast
from pysilcam.config import PySilcamSettings, settings_from_h5
import os
import h5py


def make_output_path(datapath):
    dataset_name = os.path.split(datapath)[1]
    outputpath = os.path.join(datapath, ('../output_' + dataset_name))
    os.makedirs(outputpath, exist_ok=True)
    return outputpath


def plot_boxplot(dataset_names, tracks, PIX_SIZE, figurename):
    ps = np.arange(0, len(dataset_names))
    ls = dataset_names

    f, a = plt.subplots(3, 1, figsize=(6, 12))

    plt.sca(a[0])
    box_data = [tracks[dataset_names[i]]['width'] /
                tracks[dataset_names[i]]['length']
                for i in range(len(dataset_names))]
    plt.boxplot(box_data, positions=ps, labels=ls)
    plt.ylabel('Minor/Major axis')
    plt.ylim(0, 1)
    plt.gca().xaxis.tick_top()
    plt.xticks(rotation=45, horizontalalignment='left')

    plt.sca(a[1])
    box_data = [tracks[dataset_names[i]]['length'] * PIX_SIZE / 1000
                for i in range(len(dataset_names))]
    plt.boxplot(box_data, positions=ps, labels=ls)
    plt.ylabel('Maxjor Axis Length [mm]')
    plt.xticks([])
    plt.ylim(0, 12)

    plt.sca(a[2])
    box_data = [tracks[dataset_names[i]]['S_cms'] * 10
                for i in range(len(dataset_names))]

    plt.boxplot(box_data, positions=ps, labels=ls)
    plt.ylabel('Net speed [mm/s]')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.ylim(0, 2.5)

    figurename = os.path.join(figurename + '.png')
    print('  saving:', figurename)
    plt.savefig(figurename, dpi=600, bbox_inches='tight')
    print('  saved')


def im_from_timestamp(timestamp, rawdatapath):
    searchfilename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')
    hits = glob.glob(os.path.join(rawdatapath, searchfilename + '*'))
    imagename = hits[0]
    im = sctracker.silcam_load(imagename)
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

                plt.text(liney[0], linex[0],
                         (p + '\n{:0.2f}mm {:0.2f}mm/s'.format(this_particle['length'].values[0] * PIX_SIZE / 1000,
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

    Y_mm = Y * PIX_SIZE * 1e-3
    X_mm = X * PIX_SIZE * 1e-3
    dY_mm = np.diff(Y_mm, axis=0)
    dX_mm = np.diff(X_mm, axis=0)
    dD_mm = np.sqrt(dY_mm ** 2 + dX_mm ** 2)
    dD_m = dD_mm / 1000
    S_ms = dD_m / dt  # speed in m/s
    S_cms = S_ms * 100  # speed in cm/s

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


def plot_single(datapath):
    outputpath = make_output_path(datapath)
    csv_file = (outputpath + '.csv')
    print('CSV file:', csv_file)

    data = pd.read_csv(csv_file)
    print(data.columns)

    f, a = plt.subplots(2, 2, figsize=(12, 12))

    plt.sca(a[0, 0])
    plt.plot(pd.to_datetime(data['Time']), data['ECD [mm]'], '.', color='0.8', alpha=0.1)
    plt.ylabel('ECD [mm]')

    plt.sca(a[1, 0])
    plt.plot(pd.to_datetime(data['Time']), data['Speed [cm/s]'], '.', color='0.8', alpha=0.1)
    plt.ylabel('Speed [cm/s]')

    plt.sca(a[0, 1])
    plt.plot(data['ECD [mm]'], data['Ws [cm/s]'], '.', color='0.8', alpha=0.1)
    plt.xlabel('ECD [mm]')
    plt.ylabel('Ws [cm/s]')

    plt.sca(a[1, 1])
    plt.plot(data['ECD [mm]'], data['Speed [cm/s]'], '.', color='0.8', alpha=0.1)
    plt.xlabel('ECD [mm]')
    plt.ylabel('Speed [cm/s]')

    plt.show()


def load_and_process(tracksfile, PIX_SIZE,
                     minlength=0, maxlength=1e6, track_length_limit=15):
    data = pd.read_hdf(tracksfile, 'Tracking/data')
    tracks = sctracker.post_process(data, PIX_SIZE, track_length_limit=track_length_limit, minlength=minlength,
                                    maxlength=maxlength)
    return data, tracks


def track_process(configfile, datapath, offset=0):
    '''
    This takes a silcam config file (and uses information within the [Tracking] section to track particles in images in
    datapath. This will produce a -TRACKS.h5 file in the [General]datafile location (in a similar manner to normal
    silcam processing).

    Functions like pysilcam.tracking.track.make_boxplot can make plots with these -TRACK.h5 files.

    :param configfile:
    :param datapath:
    :param offset:
    :return:
    '''
    settings = PySilcamSettings(configfile)

    sctr = sctracker.Tracker()
    sctr.av_window = settings.Background.num_images
    sctr.MIN_LENGTH = settings.Tracking.min_length
    sctr.MIN_SPEED = settings.Tracking.min_speed
    sctr.GOOD_FIT = settings.Tracking.good_fit
    sctr.THRESHOLD = settings.Process.threshold
    sctr.ecd_tolerance = settings.Tracking.ecd_tolerance
    sctr.PIX_SIZE = settings.PostProcess.pix_size

    sctr.path = datapath
    dataset_name = os.path.split(datapath)[-1] + '-TRACKS'
    sctr.DATAFILE = os.path.join(settings.General.datafile, dataset_name)
    sctr.track_length_limit = settings.Tracking.track_length_limit
    sctr.initialise()
    sctr.files = sctr.files[offset:]

    os.makedirs(settings.General.datafile, exist_ok=True)

    # setup HDF5 file and metadata
    print('* Setting up', sctr.DATAFILE + '.h5')
    with h5py.File(sctr.DATAFILE + '.h5', "a") as HDF5File:
        meta = HDF5File.require_group('Meta')
        meta.attrs['Modified'] = str(pd.datetime.now())
        settings_dict = {s: dict(settings.config.items(s)) for s in settings.config.sections()}
        meta.attrs['Settings'] = str(settings_dict)

    sctr.process()


def make_boxplot(tracksfile):
    '''
    given the input tracksfile (-TRACKs.h5), this function will look for all the other -TRACKS.h5 files in the same
    folder and use this to create summary boxplots of tracked data, which is saved to the General.datafile location
    specified within the settings contained in the last -TRACKS.h5 file.

    :param tracksfile:
    :return:
    '''
    h5filedir = os.path.split(tracksfile)[0]
    h5file_list = glob.glob(os.path.join(h5filedir, '*-TRACKS.h5'))
    dataset_names = [os.path.split(k)[-1].replace('-TRACKS.h5', '') for k in h5file_list]

    tracks = dict()
    print('* Loading data:')
    for f, k in zip(h5file_list, dataset_names):
        print('  ', f)
        tracks[k] = pd.read_hdf(f, 'Tracking/tracks')

    settings = settings_from_h5(f)

    fig_name = tracksfile.replace('-TRACKS.h5', '-boxplot')
    plot_boxplot(dataset_names, tracks, settings.PostProcess.pix_size, fig_name)
    print('* boxplotting finished.')


def silctrack():
    """
    does tracking

    Usage:
        silcam-track process <configfile> <datapath> [--offset=<offset>]
        silcam-track post-process <tracksfile>
        silcam-track plotting <tracksfile> [--gif=<outputdir>] [<rawdatapath>] [--boxplot]
    """

    args = docopt(silctrack.__doc__)

    if args['process']:
        offset = args['--offset']
        if offset is not None:
            try:
                offset = int(offset)
            except ValueError:
                print('Expected type int for --offset.')
                sys.exit(0)
        else:
            offset = 0

        track_process(args['<configfile>'], args['<datapath>'],
                      offset=offset)

    if args['post-process']:
        print('* Load and process')
        settings = settings_from_h5(args['<tracksfile>'])

        data, tracks = load_and_process(args['<tracksfile>'],
                                        settings.PostProcess.pix_size,
                                        track_length_limit=settings.Tracking.track_length_limit)

        with pd.HDFStore(args['<tracksfile>']) as fh:
            tracks.to_hdf(fh, 'Tracking/tracks', mode='r+')

        unfiltered_tracks = sctracker.extract_continuous_tracks(data)
        with pd.HDFStore(args['<tracksfile>']) as fh:
            unfiltered_tracks.to_hdf(fh, 'Tracking/unfiltered_tracks', mode='r+')

    if args['plotting']:
        settings = settings_from_h5(args['<tracksfile>'])

        if args['--gif']:

            if not checkgroup(args['<tracksfile>'], 'Tracking/unfiltered_tracks'):
                print('* Extracting unfiltered tracks')
                data = pd.read_hdf(args['<tracksfile>'], 'Tracking/data')
                unfiltered_tracks = sctracker.extract_continuous_tracks(data)
                with pd.HDFStore(args['<tracksfile>']) as fh:
                    unfiltered_tracks.to_hdf(fh, 'Tracking/unfiltered_tracks', mode='r+')
                print('  OK.')
            else:
                unfiltered_tracks = pd.read_hdf(args['<tracksfile>'], 'Tracking/unfiltered_tracks')

            outputdir = args['--gif']
            rawdatapath = args['<rawdatapath>']

            make_output_files_for_giffing(unfiltered_tracks, rawdatapath, outputdir,
                                          settings.PostProcess.pix_size,
                                          track_length_limit=5)
            print('* output files finished.')
            print('use ''convert -delay 12 -loop 0 *.png output.gif'' to make a gif')

        if args['--boxplot']:
            tracksfile = args['<tracksfile>']
            make_boxplot(tracksfile)

        print('* plotting finished.')
