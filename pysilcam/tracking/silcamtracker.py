import os
import numpy as np
from skimage.feature import match_template
from skimage.morphology import label, binary_dilation
from skimage.measure import regionprops
import pandas as pd
from tqdm import tqdm
import pysilcam.process as scpr
from pysilcam.background import backgrounder
from pysilcam.fakepymba import silcam_load
import h5py
import names


class Tracker:
    '''
    Class for tracking
    '''

    def __init__(self):
        # define some defaults
        self.av_window = 15
        self.THRESHOLD = 0.98
        self.MIN_LENGTH = 500  # microns
        self.MIN_SPEED = 0.01  # cm/s
        self.cross_correlation_threshold = 0.2
        self.PIX_SIZE = 27.532679738562095
        self.ecd_tolerance = 0
        self.path = ''
        self.DATAFILE = ''
        self.files = None
        self.track_length_limit = 15
        self.search_box_size = 10  # pixels
        self.search_box_steps = 5

    def initialise(self):
        print('* INITIALISE')
        if self.files is None:
            self.files = [os.path.join(self.path, f)
                          for f in sorted(os.listdir(self.path)) if
                          f.endswith('.bmp') or f.endswith('.silc') or f.endswith('.silc_mono')]
        print('  File list obtained:')
        print('    ', len(self.files), 'files found')

        self.aqgen = self.generator_tracker()

        # Get number of images to use for background correction from config
        print('* Initializing background image handler')
        self.bggen = backgrounder(self.av_window, self.aqgen,
                                  bad_lighting_limit=None,
                                  real_time_stats=False)

    def generator_tracker(self, datapath=None):
        for f in tqdm(self.files):

            img = silcam_load(f)

            if np.ndim(img) == 3:
                img = np.min(img, axis=2)
            imx, imy = np.shape(img)
            imc = np.zeros((imx, imy, 3), dtype=np.uint8())
            imc[:, :, 0] = img
            imc[:, :, 1] = img
            imc[:, :, 2] = img
            timestamp = pd.to_datetime(os.path.splitext(os.path.split(f)[-1])[0][1:])
            yield timestamp, imc

    def process(self):
        PIX_SIZE = self.PIX_SIZE
        MIN_LENGTH = self.MIN_LENGTH
        GOOD_FIT = self.cross_correlation_threshold
        DATAFILE = self.DATAFILE
        track_length_limit = self.track_length_limit

        if DATAFILE == '':
            print('DATAFILE not specified')
            print('---- END ----')
            return

        # setup HDF5 file and metadata
        with h5py.File(DATAFILE + '.h5', "a") as HDF5File:
            meta = HDF5File.require_group('Meta')
            meta.attrs['Modified'] = str(pd.datetime.now())
            meta.attrs['DatasetName'] = 'not implemented'
            HDF5File.create_group("Tracking")

        tracks = pd.DataFrame()
        tracks.index.name = 'UPID'
        UPID = -1

        print('* Tracking....')

        img1, t1 = self.load_image()

        i = 0
        while True:

            try:
                img2, t2 = self.load_image()
            except Exception:
                break

            i += 1

            try:
                X, Y, ecd, length, width, im_plot = get_vect(img1, img2,
                                                             PIX_SIZE, MIN_LENGTH, GOOD_FIT,
                                                             thresh=self.THRESHOLD,
                                                             ecd_tolerance=self.ecd_tolerance,
                                                             search_box_size=self.search_box_size,
                                                             search_box_steps=self.search_box_steps)
            except ValueError:
                print('  Error getting vectors')
                continue

            if len(X[0]) == 0:
                continue

            for p in range(len(X[0])):
                UPID += 1
                tracks.loc[UPID, 'particle'] = int(p)
                tracks.loc[UPID, 't1'] = t1
                tracks.loc[UPID, 't2'] = t2
                tracks.loc[UPID, 'x1'] = X[0][p]
                tracks.loc[UPID, 'x2'] = X[1][p]
                tracks.loc[UPID, 'y1'] = Y[0][p]
                tracks.loc[UPID, 'y2'] = Y[1][p]
                tracks.loc[UPID, 'ecd'] = ecd[p]
                tracks.loc[UPID, 'length'] = length[p]
                tracks.loc[UPID, 'width'] = width[p]

            tracks = match_last_pair(tracks)
            with pd.HDFStore(DATAFILE + '.h5', "a") as fh:
                tracks.to_hdf(fh, 'Tracking/data')

            # get ready for next iteration
            img1 = np.copy(img2)
            t1 = pd.to_datetime(str(np.copy(t2)))

        print('* Starting post-process')
        continuous_tracks = post_process(tracks, PIX_SIZE,
                                         track_length_limit=track_length_limit,
                                         max_starts=None)

        with pd.HDFStore(DATAFILE + '.h5', "a") as fh:
            continuous_tracks.to_hdf(fh, 'Tracking/tracks')
        print('Post-processing done.')

    def load_image(self):

        timestamp, imc, imraw = next(self.bggen)
        im = imc
        if len(np.shape(im)) == 3:
            im = np.min(im, axis=2)
        im = np.uint8(im)
        return im, timestamp


def imc2iml(imc, thresh=0.98):
    imbw = scpr.image2blackwhite_fast(imc, thresh)
    imbw = scpr.clean_bw(imbw, 12)
    for i in range(2):
        imbw = binary_dilation(imbw)
    iml = label(imbw)
    return iml


def get_vect(img1, img2, PIX_SIZE, MIN_LENGTH, cross_correlation_threshold,
             thresh=0.98, ecd_tolerance=0, search_box_size=10, search_box_steps=5):
    # label image 2
    iml2 = imc2iml(img2, thresh)
    imbw_out = np.copy(iml2)

    if (np.max(iml2) == 0):
        raise ValueError('NoParticles')

    # calculate geometrical stats from image 2
    props2 = regionprops(iml2, cache=False)

    # label image 1
    iml = imc2iml(img1, thresh)
    if (np.max(iml) == 0):
        raise ValueError('NoParticles')

    # calculate geometrical stats from image 1
    props = regionprops(iml, cache=False)

    # preallocate particle information to extract
    y1 = []
    x1 = []
    x = []
    y = []
    ecd = []
    length = []
    width = []

    # loop through all particles
    for i, el in enumerate(props):

        # skip if the particle is too short
        if (el.major_axis_length * PIX_SIZE) < MIN_LENGTH:
            continue

        bbox = el.bbox
        cr = el.centroid  # cr[0] is y and cr[1] is x because centroid returns row, col

        roi = iml[bbox[0]:bbox[2], bbox[1]:bbox[3]]  # roi of image 1
        roi = roi > 0

        for search_iteration in range(search_box_steps):

            search_box = get_search_box(bbox, img2, search_iteration, search_box_size)

            # extract the roi in which we expect to find a particle
            search_roi = iml2[search_box[0]:search_box[2], search_box[1]:search_box[3]]

            idx, particle_found = find_particle_idx(cross_correlation_threshold, roi, search_roi, ecd_tolerance,
                                                    el.equivalent_diameter)

            # if the particle is found, then stop searching
            if particle_found:
                # if there is a particle here, use its centroid location for the vector
                # calculation
                cr2 = props2[int(idx - 1)].centroid  # subtract 1 from idx because of zero-indexing

                # remove this particle from future calculations
                iml2[iml2 == int(idx)] = 0

                x.append(cr2[1])  # col
                y.append(cr2[0])  # row

                # and append the position of this particle from the image 1
                y1.append(cr[0])  # row
                x1.append(cr[1])  # col

                # if we get here then we also want the particle stats
                ecd.append(el.equivalent_diameter)
                length.append(el.major_axis_length)
                width.append(el.minor_axis_length)
                break

    X = [x1, x]  # horizontal vector
    Y = [y1, y]  # vertical vector
    return X, Y, ecd, length, width, imbw_out


def get_search_box(bbox, img2, search_iteration, search_box_size=10):
    bbexp = search_box_size * (search_iteration + 1)  # expansion by this many pixels in all directions
    # establish a search box within image 2 by expanding the particle
    # bounding box from image 1
    r, c = np.shape(img2)
    search_box = np.zeros_like(bbox)
    search_box[0] = max(0, bbox[0] - bbexp)
    search_box[1] = max(0, bbox[1] - bbexp)
    search_box[2] = min(r, bbox[2] + bbexp)
    search_box[3] = min(c, bbox[3] + bbexp)
    return search_box


def get_idx_from_search_box(ecd_tolerance, ecd_lookup, search_roi):
    '''

    :param ecd_tolerance:
    :param ecd_lookup:
    :param search_roi:
    :return:
    '''
    # we are still looking for a particle
    particle_found = False
    idx = 0

    # and squash to an array of particle indicies from within the box
    unp = np.unique(search_roi)
    # size the particles in the search_roi
    if len(unp) > 1:  # len(unp) == 1 implies no particles
        props = regionprops(search_roi, cache=True)

        # list all the choices of particle size within the search area
        choice_ecd = np.array([p.equivalent_diameter for p in props])

        # find the closest ecd
        closest_ecd = np.min(np.abs(choice_ecd - ecd_lookup))
        closest_ecd_pcent = closest_ecd / ecd_lookup * 100

        if (closest_ecd_pcent < ecd_tolerance):
            idx = (np.abs(choice_ecd - ecd_lookup)).argmin()  # find the ecd that is closest

            # get the labelled particle number at this location
            idx = int(unp[int(idx + 1)])
            particle_found = True
    return idx, particle_found


def find_particle_idx(cross_correlation_threshold, roi, search_roi, ecd_tolerance, ecd_lookup):
    '''

    :param cross_correlation_threshold:
    :param iml2:
    :param roi:
    :param search_roi:
    :param ecd_tolerance:
    :param ecd_lookup:
    :return:
    '''
    # use a Fast Normalized Cross-Correlation
    cross_correlation = match_template(search_roi > 0, roi)

    idx = 0
    particle_found = False
    if np.max(cross_correlation) > cross_correlation_threshold:
        particle_found = True

        # look for a peak in the cross-correlation regardless of how good it is
        # and extract the location of the maximum
        ij = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)
        x_, y_ = ij[::-1]
        idx = int(search_roi[int(y_), int(x_)])

        # if there is no particle here, then we need more analysis
        # this is because cross_correlation was good but there is no particle
        if idx == 0:
            idx, particle_found = get_idx_from_search_box(ecd_tolerance, ecd_lookup, search_roi)

    return idx, particle_found


def match_last_pair(data):
    data_1 = data[data['t2'] == data.iloc[-1].t1]
    x_end = data_1['x2'].values
    y_end = data_1['y2'].values

    data_2 = data[data['t1'] == data.iloc[-1].t1]
    x_start = data_2['x1'].values
    y_start = data_2['y1'].values

    c = len(x_start)
    for i in range(c):
        dxy = abs(x_end - x_start[i]) + abs(y_end - y_start[i])
        ind = np.argwhere(dxy < 1e-4)
        if len(ind) == 1:
            data.loc[data_2.iloc[i].name, 'UPID-backward-match'] = data_1.iloc[int(ind)].name
            data.loc[data_1.iloc[int(ind)].name, 'UPID-forward-match'] = data_2.iloc[i].name
    return data


def calculate_speed_df(data, PIX_SIZE):
    X = np.array([data['x-arrival'], data['x-departure']])
    Y = np.array([data['y-arrival'], data['y-departure']])

    dt = pd.Series(np.abs(pd.to_datetime(data['t-arrival']) -
                          pd.to_datetime(data['t-departure']))).dt.total_seconds()
    dt = dt.values

    Y_mm = Y * PIX_SIZE * 1e-3
    X_mm = X * PIX_SIZE * 1e-3
    dY_mm = np.diff(Y_mm, axis=0)
    dX_mm = np.diff(X_mm, axis=0)
    dD_mm = np.sqrt(dY_mm ** 2 + dX_mm ** 2)
    dD_m = dD_mm / 1000
    S_ms = dD_m / dt  # speed in m/s
    S_cms = S_ms * 100  # speed in cm/s

    Xs_mms = dX_mm / dt
    Xs_cms = Xs_mms / 10

    data['S_cms'] = S_cms[0]
    data['X_cms'] = Xs_cms[0]
    return data


def extract_continuous_tracks(tracks, max_starts=None):
    # find positions in the dataframe where there is a forward match but not a backward match
    # these are the first occurances of a particle  ('arrivals')
    starts = tracks[np.isnan(tracks['UPID-backward-match']) & ~np.isnan(tracks['UPID-forward-match'])].index

    print('starts', len(starts))

    if max_starts is not None:
        starts = starts[0:min([max_starts, len(starts)])]

    for s in tqdm(starts):

        # initial start point
        # we know that this is not the last time due to forward match not being nan
        tracks.loc[s, 'x-arrival'] = tracks.loc[s, 'x1']
        tracks.loc[s, 'y-arrival'] = tracks.loc[s, 'y1']
        tracks.loc[s, 't-arrival'] = tracks.loc[s, 't1']
        c = 1  # counter for number of tracks
        tracks.loc[s, 'n-tracks'] = c
        particle_name = names.get_full_name()
        tracks.loc[s, 'ParticleName'] = particle_name

        # search for future data for this particle
        new_loc = tracks.loc[s, 'UPID-forward-match']
        # we know that the while condition here will be met at least once
        while not np.isnan(new_loc):
            c += 1
            # carry previous information
            tracks.loc[new_loc, 'ParticleName'] = particle_name
            tracks.loc[new_loc, 'x-arrival'] = tracks.loc[s, 'x1']
            tracks.loc[new_loc, 'y-arrival'] = tracks.loc[s, 'y1']
            tracks.loc[new_loc, 't-arrival'] = tracks.loc[s, 't1']

            # obtain new information
            tracks.loc[new_loc, 'x-departure'] = tracks.loc[new_loc, 'x2']
            tracks.loc[new_loc, 'y-departure'] = tracks.loc[new_loc, 'y2']
            tracks.loc[new_loc, 't-departure'] = tracks.loc[new_loc, 't2']
            tracks.loc[new_loc, 'n-tracks'] = c

            new_loc = tracks.loc[new_loc, 'UPID-forward-match']
    return tracks


def post_process(data, PIX_SIZE, track_length_limit=15, max_starts=None, minlength=0, maxlength=1000000000):
    data = data[(data['length'] * PIX_SIZE / 1000 > minlength) &
                ((data['length'] * PIX_SIZE / 1000 < maxlength))]
    data = extract_continuous_tracks(data, max_starts=max_starts)
    # replace tracks so it only contains particle departures with backward matches
    data = data[np.isnan(data['UPID-forward-match']) & ~np.isnan(data['UPID-backward-match'])]
    data = data[data['n-tracks'] > track_length_limit]
    data = calculate_speed_df(data, PIX_SIZE)

    return data
