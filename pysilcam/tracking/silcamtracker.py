import os
import imageio as imo
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template, peak_local_max
from skimage.morphology import label, remove_small_objects, binary_dilation
from skimage.measure import regionprops
import matplotlib.animation as manimation
import pandas as pd
import pysilcam.postprocess as scpp
from tqdm import tqdm
import pysilcam.process as scpr
from pysilcam.acquisition import Acquire
from pysilcam.background import backgrounder
from pysilcam.fakepymba import silcam_load
import pickle
import glob as glob
from skimage.transform import rotate
from pysilcam.postprocess import explode_contrast


class Tracker:
    '''
    Class for tracking
    '''
    def __init__(self):
        # define some defaults
        self.av_window = 15
        self.THRESHOLD = 0.98
        self.MIN_LENGTH = 500 # microns
        self.MIN_SPEED = 0.01 # cm/s
        self.GOOD_FIT = 0.2
        self.PIX_SIZE = 27.532679738562095
        self.ecd_tollerance = 0
        self.FPS = 15
        self.path = ''
        self.DATAFILE = ''
        self.vidname=self.DATAFILE + '.mp4'
        self.files = None


    def initialise(self):
        print('* INITIALISE')
        if self.files == None:
            self.files = [os.path.join(self.path, f)
                    for f in sorted(os.listdir(self.path)) if f.endswith('.bmp') or f.endswith('.silc') or f.endswith('.silc_mono')]
        print('  File list obtained:')
        print(len(self.files), 'files found')

        # imbg = np.float64(imo.imread(self.files[0]))
        # print('Background averaging....')
        # for i in tqdm(range(self.av_window - 1)):
            # self.imbg += np.float64(load_image(self.files[i]))
        # self.imbg /= self.av_window
        # print('  done.')
        # self.imy, self.imx = np.shape(imbg)

        # aq = Acquire(USE_PYMBA=False)
        # aq.get_generator = self.get_generator_tracker
        self.aqgen = self.generator_tracker()

        # print(self.aqgen.pymba)

        #Get number of images to use for background correction from config
        print('* Initializing background image handler')
        self.bggen = backgrounder(self.av_window, self.aqgen,
                bad_lighting_limit = None,
                real_time_stats=False)

    def generator_tracker(self, datapath=None):
        # i = 0
        for f in tqdm(self.files):
            # try:
            # img = imo.imread(f)
            # img = img[250:1750,:]
            img = silcam_load(f)

            if np.ndim(img)==3:
                img = np.min(img, axis=2)
            imx, imy = np.shape(img)
            imc = np.zeros((imx,imy,3), dtype=np.uint8())
            imc[:,:,0] = img
            imc[:,:,1] = img
            imc[:,:,2] = img
            timestamp = pd.to_datetime(os.path.splitext(os.path.split(f)[-1])[0][1:])
            # i += 1
            yield timestamp, imc


    def process_with_video(self):
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Lab Floc Test', artist='Emlyn Davies',
                        comment='Test!')

        writer = FFMpegWriter(fps=self.FPS, metadata=metadata, codec='mpeg4')

        fig = plt.figure()
        with writer.saving(fig, self.vidname, 600):
            self.process(writer=writer)


    def process(self, writer=None):
        files = self.files
        # imbg = self.imbg
        PIX_SIZE = self.PIX_SIZE
        MIN_SPEED = self.MIN_SPEED
        MIN_LENGTH = self.MIN_LENGTH
        GOOD_FIT = self.GOOD_FIT
        DATAFILE = self.DATAFILE
        # imy = self.imy
        # imx = self.imx

        N = len(files) - 1

        # fig, a = plt.subplots(2, 2, figsize=(10, 10))
        fig, a = plt.subplots(2, 2, figsize=(12, 12))

        ecd_mm_ALL = []
        length_mm_ALL = []
        S_cms_ALL = []
        Xs_cms_ALL = []

        tracks = pd.DataFrame()
        tracks.index.name = 'UPID'
        UPID = -1

        cnames = ['Time', 'ECD [mm]', 'Length [mm]',
                  'Width [mm]', 'Speed [cm/s]', 'Ws [cm/s]']

        df = pd.DataFrame(columns=cnames)

        print('Processing....')


        img1, t1 = self.load_image()
        imy, imx = np.shape(img1)

        i=0
        c = 0
        while True:
            # fname = files[i + 1]

            if not (i==0):
                img1 = np.copy(img2)
                t1 = pd.to_datetime(str(np.copy(t2)))

            try:
                img2, t2 = self.load_image()
            except:
                break

            i += 1
            tname = str(t2)

            # img2 = imcor(img2, imbg)

            # img1 = load_image(files[i])
            # img1 = imcor(img1, imbg)

            # timestamp = pd.to_datetime(tname)

            # t1 = timestamp
            # t2 = pd.to_datetime(os.path.splitext(os.path.split(files[i])[-1])[0][1:])
            dt = np.abs(t1 - t2)
            # convert to a dt in decimal second with millisecond precision, assuming less
            # than one minute separation
            dt = dt.components.seconds + (dt.components.milliseconds / 1000)
            dt = np.abs(dt)

            # xlim = int((15 * 1000) / PIX_SIZE)

            # X, Y, ecd, length, width, im_plot = get_vect(img1[:, :xlim], img2[:, :xlim],
            #                                              PIX_SIZE, MIN_LENGTH, GOOD_FIT)

            try:
                X, Y, ecd, length, width, im_plot = get_vect(img1, img2,
                                                            PIX_SIZE, MIN_LENGTH, GOOD_FIT,
                                                            thresh=self.THRESHOLD,
                                                            ecd_tollerance=self.ecd_tollerance)

                # f, a = plt.subplots(2, 1, figsize=(20,20))
                # plt.sca(a[0])
                # plt.imshow(img2, cmap='gray')
                # plt.plot(X, Y)
                # plt.title(tname)
                # plt.sca(a[1])
                # plt.imshow(im_plot>0, cmap='gray')
                # plt.savefig('/mnt/ARRAY/plastic_settling/figs/tmp.png')
                # input()


            except ValueError:
                print('  Error getting vectors')
                continue

            if len(X[0])==0:
                continue

            for p in range(len(X[0])):
                UPID += 1
                tracks.loc[UPID,'particle'] = int(p)
                tracks.loc[UPID,'t1'] = t1
                tracks.loc[UPID,'t2'] = t2
                tracks.loc[UPID,'x1'] = X[0][p]
                tracks.loc[UPID,'x2'] = X[1][p]
                tracks.loc[UPID,'y1'] = Y[0][p]
                tracks.loc[UPID,'y2'] = Y[1][p]
                tracks.loc[UPID,'ecd'] = ecd[p]
                tracks.loc[UPID,'length'] = length[p]
                tracks.loc[UPID,'width'] = width[p]

            tracks = match_last_pair(tracks)

            tracks.to_csv(DATAFILE + '.csv')
            # times_in_data = np.unique(tracks['t1'])
            # if len(times_in_data)>1:
            #     tracks = match_last_pair(tracks)
            #     if c==0:
            #         print('first write')
            #         tracks.to_csv(DATAFILE + '.csv')
            #         c = 125682935
            #     else:
            #         print('subsequent write')
            #         output = tracks[(tracks['t1']==max(tracks['t1']))]
            #         output.to_csv(DATAFILE + '.csv', mode='a', header=False)

            # limit the number of image pairs kept in dataframe
            # if len(times_in_data)>10:
                # tracks = tracks[(tracks['t1']>times_in_data[-10])]

            # print('writing data')
            # output_data = (tracks, PIX_SIZE)
            # pickle.dump(output_data, open(DATAFILE + '.p', "wb") )
            # print('OK')
            # if len(tracks)>100:
                # raise


            continue

            # x_j, y_j, ecd, length, width = join_vectors(X_prev, Y_prev, t_prev, t2, np.array(X), np.array(Y), ecd, length, width)

            # print(x_j, y_j)

            # X_prev = np.copy(x_j)
            # Y_prev = np.copy(y_j)
            # t_prev = np.copy(x_j[3,:])

            ecd_mm, S_cms, length_mm, width_mm, Xs_cms = stuff2speed(x_j, y_j,
                                                                     ecd, length, width,
                                                                     PIX_SIZE)

            # ecd_mm, S_cms, length_mm, width_mm, Xs_cms = stuff2speed(x_j, y_j,
            #                                                          ecd, length, width, dt,
            #                                                          PIX_SIZE)

            Xs_cms[S_cms < MIN_SPEED] = np.nan
            S_cms[S_cms < MIN_SPEED] = np.nan

            for j in range(len(ecd_mm)):
                df = update_dataframe(df, t2, ecd_mm[j],
                                      S_cms[j], Xs_cms[j], length_mm[j], width_mm[j])

            df.to_csv(DATAFILE + '.csv', index=False)

            # if i == 0:
            #     df.to_csv(DATAFILE + '.csv', index=False)
            # else:
                # df.to_csv(DATAFILE + '.csv', mode='a', header=False, index=False)

            dias, bin_limits_um = scpp.get_size_bins()
            necd, edges = np.histogram(np.array(ecd)/1000, bin_limits_um)
            vd = scpp.vd_from_nd(necd, dias)

            # individual_droplet_volume = 4 / 3 * np.pi * ((ecd_mm / 1000) / 2) ** 3  # m3
            # individual_droplet_mass = individual_droplet_volume * 880  # kg
            # horizontal_vectors = -Xs_cms / 100  # m/s
            # flux_volume = (imy * PIX_SIZE) * 0.5 * 0.15  # m3
            # mass_flux = np.sum((individual_droplet_mass / flux_volume) * horizontal_vectors)  # kg/s/m2

            ecd_mm_ALL.append(ecd_mm)
            length_mm_ALL.append(length_mm)
            S_cms_ALL.append(S_cms)
            Xs_cms_ALL.append(Xs_cms)

            # plt.sca(a[0, 0])
            plt.sca(a[0,0])
            plt.cla()
            plot_image(img2, PIX_SIZE, imx, imy)
            # plot_pair_vectors(X, Y, PIX_SIZE)
            plot_pair_vectors(x_j, y_j, PIX_SIZE)
            plt.title(t2)

            plt.sca(a[0,1])
            plt.cla()
            plot_psd(dias, vd)

            plt.sca(a[1,0])
            plt.cla()
            # plot_all_vectors(PIX_SIZE, X, Y, imx, imy)
            plot_all_vectors(PIX_SIZE, x_j, y_j, imx, imy)

            plt.sca(a[1,1])
            plt.cla()
            plt.plot(df['ECD [mm]'], df['Speed [cm/s]'], '.', color='0.8', alpha=0.25)
            plt.plot(ecd_mm, S_cms,'r.')
            plt.xlabel('ECD [mm]')
            plt.ylabel('Speed [cm/s]')

            if not writer == None:
                writer.grab_frame()
            # print(DATAFILE + '_' + tname + '.png')
            plt.savefig(DATAFILE + '_' + tname + '.png', dpi=100,
                        bbox_inches='tight')
            # plt.savefig(DATAFILE + '_' + str(i) + '.png', dpi=100,
            #             bbox_inches='tight')
        print('Processing done.')


    def load_image(self):

        timestamp, imc, imraw  = next(self.bggen)
        # im = imo.imread(filename)1
        im = imc
        if len(np.shape(im))==3:
            im = np.min(im, axis=2)
        # im = np.rot90(im)
        im = np.uint8(im)
        return im, timestamp


def imcor(imraw, imbg):
    imc = np.float64(imraw) - np.float64(imbg)

    m = np.median(imc)
    imc += 1.5 * ((255/2)-m + 20)

    imc[imc>255] = 255
    imc[imc<0] = 0

    imc = np.uint8(imc)

    return imc


def imc2iml(imc, thresh=0.98):
    # imbw = imc < 0.92 * np.median(imc)
    imbw = scpr.image2blackwhite_fast(imc, thresh)
    # imbw = remove_small_objects(imbw > 0, min_size=12)
    imbw = scpr.clean_bw(imbw, 12)
    for i in range(2):
        imbw = binary_dilation(imbw)
    iml = label(imbw)
    return iml


def get_vect(img1, img2, PIX_SIZE, MIN_LENGTH, GOOD_FIT, thresh=0.98,
             ecd_tollerance=0):

    # label image 2
    iml2 = imc2iml(img2, thresh)
    imbw_out = np.copy(iml2)

    if (np.max(iml2)==0):
        raise ValueError('NoParticles')

    # calculate geometrical stats from image 2
    props2 = regionprops(iml2, cache=False, coordinates='xy')

    # label image 1
    iml = imc2iml(img1, thresh)
    if (np.max(iml)==0):
        raise ValueError('NoParticles')

    # calculate geometrical stats from image 1
    props = regionprops(iml, cache=False, coordinates='xy')

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
        cr = el.centroid # cr[0] is y and cr[1] is x because centroid returns row, col

        #roi = img1[bbox[0]:bbox[2], bbox[1]:bbox[3]] # roi of image 1
        roi = iml[bbox[0]:bbox[2], bbox[1]:bbox[3]] # roi of image 1
        roi = roi > 0

        OK = False
        for c in range(5):
            bbexp = 10*(c+1) # expansion by this many pixels in all directions

            # establish a search box within image 2 by expanding the particle
            # bounding box from image 1
            r, c = np.shape(img2)
            search_box = np.zeros_like(bbox)
            search_box[0] = max(0,bbox[0]-bbexp)
            search_box[1] = max(0,bbox[1]-bbexp)
            search_box[2] = min(r,bbox[2]+bbexp)
            search_box[3] = min(c,bbox[3]+bbexp)

            # print('search_box',search_box)

            # extract the roi in which we expect to find a particle
            #search_roi = img2[search_box[0]:search_box[2], search_box[1]:search_box[3]]
            search_roi = iml2[search_box[0]:search_box[2], search_box[1]:search_box[3]]
            search_roi = search_roi > 0

            # use a Fast Normalized Cross-Correlation
            result = match_template(search_roi, roi)
            if np.max(result) < GOOD_FIT:
                # if there is not a good enough match, then continue by
                # expanding the search box until the number of search iterations
                # run out, then forget searching
                continue

            # look for a peak in the cross-correlation regardless of how good it is
            # and extract the location of the maximum
            ij = np.unravel_index(np.argmax(result), result.shape)
            x_, y_ = ij[::-1]

            # convert position from inside the bounding box to a position within
            # the original image
            x_ += ((bbox[3]-bbox[1])/2)
            y_ += ((bbox[2]-bbox[0])/2)

            # get the labelled particle number at this location
            idx = iml2[int(y_+search_box[0]),int(x_+search_box[1])]
            idx = int(idx) # make sure it is int

            # OK = True # I am not sure about this....?!

            # if there is no particle here, then we need more analysis
            if idx == 0:
                #plt.figure()
                #plt.imshow(result)
                #plt.plot(x_, y_,'ro')
                #plt.show()

                # get the labelled particles in the search box
                search_iml2 = iml2[search_box[0]:search_box[2], search_box[1]:search_box[3]]
                # and squash to an array of particle indicies from within the box
                unp = np.unique(search_iml2)

                # size the particles in the search_roi
                if len(unp) > 1: # len(unp) == 1 implies no particles
                    props3 = regionprops(search_iml2,cache=True)
                    ecd_lookup = el.equivalent_diameter

                    # list all the choices of particle size within the search area
                    choice_ecd = []
                    for p3 in props3:
                        choice_ecd.append(p3.equivalent_diameter)
                    choice_ecd = np.array(choice_ecd)

                    closest_ecd = np.min(np.abs(choice_ecd-ecd_lookup))
                    closest_ecd_pcent = closest_ecd/ecd_lookup *100

                    if (closest_ecd_pcent<ecd_tollerance):
                        # print('closest_ecd_pcent', closest_ecd_pcent)
                        idx = (np.abs(choice_ecd-ecd_lookup)).argmin() # find the ecd that is closest

                        # get the labelled particle number at this location
                        idx = int(unp[int(idx+1)])
                        OK = True
                        break
                    else:
                        OK = False
                else: # if there are no particles in the search box then forget it
                    # print('not found')
                    continue
            else:
                OK = True

        if not OK:
            continue

        # if there is a particle here, use its centroid location for the vector
        # calculation
        cr2 = props2[int(idx-1)].centroid # subtract 1 from idx because of zero-indexing

        # remove this particle from future calculations
        iml2[iml2==int(idx)] = 0

        x.append(cr2[1]) # col
        y.append(cr2[0]) # row

        # and append the position of this particle from the image 1
        y1.append(cr[0]) # row
        x1.append(cr[1]) # col

        # if we get here then we also want the particle stats
        ecd.append(el.equivalent_diameter)
        length.append(el.major_axis_length)
        width.append(el.minor_axis_length)

        # plt.figure(figsize=(20,20))
        # plt.imshow(img2, cmap='gray')
        # plt.plot([search_box[1], search_box[1], search_box[3], search_box[3]],
        #          [search_box[0], search_box[2], search_box[0], search_box[2]],
        #          'rx')
        # plt.plot([bbox[1], bbox[1], bbox[3], bbox[3]],
        #          [bbox[0], bbox[2], bbox[0], bbox[2]],
        #          'gx')
        # plt.plot([cr2[1], cr[1]], [cr2[0], cr[0]] ,'r')
        # plt.savefig('/mnt/ARRAY/plastic_settling/figs/tmp.png')
        # input()

    X = [x1,x] # horizontal vector
    Y = [y1,y] # vertical vector
    return X, Y, ecd, length, width, imbw_out


def stuff2speed(X, Y, ecd, length, width, dt, PIX_SIZE):
    Y_mm = np.array(Y)*PIX_SIZE*1e-3
    X_mm = np.array(X)*PIX_SIZE*1e-3
    dY_mm = Y_mm[1,:]-Y_mm[0,:]
    dX_mm = X_mm[1,:]-X_mm[0,:]
    dD_mm = np.sqrt(dY_mm**2 + dX_mm**2)
    dD_m = dD_mm / 1000
    S_ms = dD_m/dt
    S_cms = S_ms * 100

    Xs_mms = dX_mm/dt
    Xs_cms = Xs_mms / 10

    ecd_mm = np.array(ecd)*PIX_SIZE/1000
    length_mm = np.array(length)*PIX_SIZE/1000
    width_mm = np.array(width)*PIX_SIZE/1000

    return ecd_mm, S_cms, length_mm, width_mm, Xs_cms


def v_stokes(rop,rof,d,visc=1.002e-3,C1=18):
    R = (rop-rof)/rof # submerged specific gravity
    w = R*9.81*(d**2)/(C1*visc/rof)

    return w
    # d = np.linspace(100,2500,100) * 1e-6 # m
    # rop = 1030
    # wstokes_m_sec = np.zeros((2,len(d)),dtype=np.float64)
    # wstokes_m_sec[0,:] = v_stokes(rop,rof,d,visc,C1) # m/s
    # rop = 1100
    # wstokes_m_sec[1,:] = v_stokes(rop,rof,d,visc,C1) # m/s
    # wstokes_cm_sec = wstokes_m_sec * 100

    # return w, wstokes_cm_sec


def extract_roi(imc, bbox):
    bbexp = 400

    r, c = np.shape(imc)
    search_box = np.zeros_like(bbox)
    search_box[0] = max(0,bbox[0]-bbexp)
    search_box[1] = max(0,bbox[1]-bbexp)
    search_box[2] = min(r,bbox[2]+bbexp)
    search_box[3] = min(c,bbox[3]+bbexp)

    search_roi = imc[search_box[0]:search_box[2], search_box[1]:search_box[3],:]

    return search_roi


def update_dataframe(df, timestamp, ecd_mm, S_cms, W_cms, length_mm,
        width_mm):

    dat = [[timestamp, ecd_mm, length_mm, width_mm, S_cms, W_cms]]
    dfnew = pd.DataFrame(columns=df.columns, data=dat)

    df = df.append(dfnew)

    return df


def frode_time(tname):
    '''convert frode's filenames into a readable string for pandas'''
    timestring = (tname[-20:-16] + '/' + tname[-15:-13] + '/' + tname[-12:-10] +
                  ' ' + tname[-10:-8] + ':' + tname[-8:-6] + ':' + tname[-6:-4] + '.' + tname[-3:])

    return timestring


def csv_to_xls(DATAFILE):
    print('convert to excel')
    df = pd.read_csv(DATAFILE + '.csv')
    df.to_excel(DATAFILE + '.xlsx')
    print('OK.')


def plot_image(img, PIX_SIZE, imx, imy):
    plt.cla()
    # a[0].imshow(np.uint8(img2),cmap='gray',vmin=0, vmax=255, extent=[0,(2448)*PIX_SIZE/1000,2048*PIX_SIZE/1000,0])
    plt.imshow(np.uint8(img), cmap='gray', extent=[0, imx * PIX_SIZE / 1000, imy * PIX_SIZE / 1000, 0])
    y = (imy * PIX_SIZE / 1000)
    plt.ylim(y, 0)
    plt.xlim((0, imx * PIX_SIZE / 1000))
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')


def plot_pair_vectors(X, Y, PIX_SIZE):
    plt.plot(np.array(X) * PIX_SIZE / 1000, np.array(Y) * PIX_SIZE / 1000, 'r-')


def plot_psd(dias, vd):
    plt.cla()
    plt.plot(dias, vd, 'k')
    plt.xscale('log')
    plt.xlabel('Equivalent circular diameter [um]')
    plt.ylabel('Volume concentration [uL/L')


def plot_mass_flux_timeseries(timestamp, mass_flux):
    plt.plot(pd.to_datetime(timestamp), mass_flux, 'k.')
    plt.ylabel('Mass flux [kg/s/m2]')
    plt.xlabel('Time')


def plot_all_vectors(PIX_SIZE, X, Y, imx, imy):
    plt.plot(np.array(X) * PIX_SIZE / 1000, np.array(Y) * PIX_SIZE / 1000, 'r-', alpha=0.1)
    plt.gca().set_aspect('equal')
    plt.xlim(0, imx * PIX_SIZE / 1000)
    plt.ylim(imy * PIX_SIZE / 1000, 0)
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')


def join_vectors(X_prev, Y_prev, t_prev, t2, X, Y, ecd, length, width):
    match = match_vectors(X_prev, Y_prev, X, Y)

    num_matches = len(match[match>-1])
    x_j = np.zeros((4, num_matches), dtype=float) * np.nan
    y_j = np.zeros((4, num_matches), dtype=float) * np.nan

    j = -1
    for i, m in enumerate(match):
        if m==-1:
            continue
        j += 1

        dt = np.abs(t2-t_prev)
        dt = dt.components.seconds + (dt.components.milliseconds / 1000)
        x_j[:, j] = [X_prev[0,i], X[1,m], dt, t_prev]
        y_j[:, j] = [Y_prev[0,i], Y[1,m], dt, t_prev]

    match = match[match>-1]
    ecd = np.array(ecd)[match]
    length = np.array(length)[match]
    width = np.array(width)[match]

    return x_j, y_j, ecd, length, width


def match_vectors(x_0, y_0, x_1, y_1):
    '''returns the indicies of matching vector'''
    r, c = np.shape(x_0)

    match = np.zeros(c, dtype=int) - 1
    for i in range(c):
        dxy = abs(x_0 - x_1) + abs(y_0 - y_1)
        ind = np.argwhere(dxy < 1e-4)
        if len(ind)==1:
            match[i] = int(ind)

    return match


def match_last_pair(data):

    data_1 = data[data['t2']==data.iloc[-1].t1]
    x_end = data_1['x2'].values
    y_end = data_1['y2'].values

    data_2 = data[data['t1']==data.iloc[-1].t1]
    x_start = data_2['x1'].values
    y_start = data_2['y1'].values

    c = len(x_start)
    match = np.zeros(c, dtype=int) - 1
    for i in range(c):
        dxy = abs(x_end - x_start[i]) + abs(y_end - y_start[i])
        ind = np.argwhere(dxy < 1e-4)
        if len(ind)==1:
            data.loc[data_2.iloc[i].name,'UPID-backward-match'] = data_1.iloc[int(ind)].name
            data.loc[data_1.iloc[int(ind)].name,'UPID-forward-match'] = data_2.iloc[i].name
    return data


def extract_matches(data):
    matched = data[~np.isnan(data['UPID-backward-match']) & ~np.isnan(data['UPID-forward-match'])]
    return matched


def calculate_speed_df(data, PIX_SIZE):
    X = np.array([data['x-arrival'], data['x-departure']])
    Y = np.array([data['y-arrival'], data['y-departure']])
    ecd = np.nan
    length = np.nan
    width = np.nan

    dt = pd.Series(np.abs(pd.to_datetime(data['t-arrival']) -
                      pd.to_datetime(data['t-departure']))).dt.total_seconds()
    dt = dt.values

    Y_mm = Y*PIX_SIZE*1e-3
    X_mm = X*PIX_SIZE*1e-3
    dY_mm = np.diff(Y_mm, axis=0)
    dX_mm = np.diff(X_mm, axis=0)
    dD_mm = np.sqrt(dY_mm**2 + dX_mm**2)
    dD_m = dD_mm / 1000
    S_ms = dD_m/dt
    S_cms = S_ms * 100

    Xs_mms = dX_mm/dt
    Xs_cms = Xs_mms / 10

    data['S_cms'] = S_cms[0]
    data['X_cms'] = Xs_cms[0]
    return data


def calculate_speed(x_arr, x_dep, y_arr, y_dep, t_arr, t_dep, PIX_SIZE):
    X = np.array([x_arr, x_dep])
    Y = np.array([y_arr, y_dep])
    ecd = np.nan
    length = np.nan
    width = np.nan

    dt = pd.Series(np.abs(t_arr - t_dep)).dt.total_seconds()
    dt = dt.values

    Y_mm = Y*PIX_SIZE*1e-3
    X_mm = X*PIX_SIZE*1e-3
    dY_mm = np.diff(Y_mm, axis=0)
    dX_mm = np.diff(X_mm, axis=0)
    dD_mm = np.sqrt(dY_mm**2 + dX_mm**2)
    dD_m = dD_mm / 1000
    S_ms = dD_m/dt
    S_cms = S_ms * 100

    Xs_mms = dX_mm/dt
    Xs_cms = Xs_mms / 10

    return S_cms[0]


def extract_continuous_tracks(tracks, max_starts=None):
    # find positions in the dataframe where there is a forward match but not a backward match
    # these are the first occurances of a particle  ('arrivals')
    starts = tracks[np.isnan(tracks['UPID-backward-match']) & ~np.isnan(tracks['UPID-forward-match'])].index

    print('starts', len(starts))
    # if len(starts)==0:
    #     return tracks[np.isnan(tracks['UPID-backward-match']) & ~np.isnan(tracks['UPID-forward-match'])]

    if not max_starts==None:
        starts = starts[0:min([max_starts, len(starts)])]

    for s in tqdm(starts):

        # max_speed = 0
        # min_speed = np.inf

        # initial start point
        # we know that this is not the last time due to forward match not being nan
        tracks.loc[s, 'x-arrival'] = tracks.loc[s, 'x1']
        tracks.loc[s, 'y-arrival'] = tracks.loc[s, 'y1']
        tracks.loc[s, 't-arrival'] = tracks.loc[s, 't1']
        c = 1 # counter for number of tracks
        tracks.loc[s, 'n-tracks'] = c

        # search for future data for this particle
        new_loc = tracks.loc[s, 'UPID-forward-match']
        # we know that the while condition here will be met at least once
        while not np.isnan(new_loc):
            c += 1
            # carry previous information
            tracks.loc[new_loc, 'x-arrival'] = tracks.loc[s, 'x1']
            tracks.loc[new_loc, 'y-arrival'] = tracks.loc[s, 'y1']
            tracks.loc[new_loc, 't-arrival'] = tracks.loc[s, 't1']

            # obtain new information
            tracks.loc[new_loc, 'x-departure'] = tracks.loc[new_loc, 'x2']
            tracks.loc[new_loc, 'y-departure'] = tracks.loc[new_loc, 'y2']
            tracks.loc[new_loc, 't-departure'] = tracks.loc[new_loc, 't2']
            tracks.loc[new_loc, 'n-tracks'] = c

            # calculate speed in pixels/sec
            # speed_check = calculate_speed_df(tracks.loc[new_loc], 1)
            # max_speed = np.max([max_speed, speed_check['S_cms']])
            # min_speed = np.min([min_speed, speed_check['S_cms']])

            # tracks.loc[new_loc, 'max_speed'] = max_speed
            # tracks.loc[new_loc, 'min_speed'] = min_speed

            new_loc = tracks.loc[new_loc, 'UPID-forward-match']

    # replace tracks so it only contains particle departures with backward matches
    tracks = tracks[np.isnan(tracks['UPID-forward-match']) & ~np.isnan(tracks['UPID-backward-match'])]
    return tracks


def get_stokes(particle_density=1030):
    diams = np.linspace(0.1, 10, 100)/1000 # in m

    water_density = 1025
    particle_density = particle_density+water_density # this means the input is density difference

    w = []
    for d in diams:
        w.append(v_stokes(particle_density, water_density, d)*60*24)

    w = np.array(w)

    diams = diams[w<2000]
    w = w[w<2000]
    return diams*1000, w


def plot_stokes(c='k-', particle_density=1030):
    diams, w = get_stokes(particle_density=particle_density)
    plt.plot(diams, w, c)
    plt.annotate('  ' + str(particle_density),(diams[-2],w[-2]))


def calculate_density(w, r):
    mu = 1.3e-3
    rho_w = 1025
    rho_p = rho_w + (2*9.81*r**2)/(w*9*mu)
    return rho_p


def load_data(datapath, search_sring='output_s*'):
    '''load and merge all data that matches the search string'''
    files = glob.glob(datapath + '/' + search_sring + '.csv')
    for c, f in enumerate(files):
        print(f)

        data = pd.read_csv(f)

        if c==0:
            data_all = data.copy()
        else:
            data_all = data_all.append(pd.read_csv(f))

    return data_all


def post_process(data, PIX_SIZE, track_length_limit=15, max_starts=None):
    data = extract_continuous_tracks(data, max_starts=max_starts)
    data = data[data['n-tracks']>track_length_limit]

    data = calculate_speed_df(data, PIX_SIZE)

    return data


def load_and_process(tracksfile, PIX_SIZE,
                     minlength=0, maxlength=1e6, track_length_limit=15):
    data = pd.read_csv(tracksfile)
    data = data[(data['length']*PIX_SIZE/1000 > minlength) &
                ((data['length']*PIX_SIZE/1000 < maxlength))]
    tracks = post_process(data, PIX_SIZE,
            track_length_limit=track_length_limit, max_starts=50)

    return data, tracks


def make_output_files_for_giffing(datapath, dataset_name, data, PIX_SIZE, track_length_limit=15):

    print('make_output_files_for_giffing')

    outputdir = os.path.join(datapath, 'output_' + dataset_name)
    os.makedirs(outputdir, exist_ok=True)

    sctr = Tracker()

    sctr.path = os.path.join(datapath, dataset_name)

    sctr.av_window = 15
    #sctr.files = subsample_files(datapath, offset=offset)
    sctr.initialise()
    # sctr.files = sctr.files[166:]
    sctr.files = sctr.files[-100:]
    sctr.MIN_LENGTH = 300
    sctr.MIN_SPEED = 0.000001 # cm/s
    sctr.GOOD_FIT = 0.1
    sctr.THRESHOLD = 0.99
    sctr.ecd_tollerance = 5
    sctr.PIX_SIZE = PIX_SIZE

    while True:
        try:
            im, timestamp = sctr.load_image()
        except:
            break
        name = timestamp.strftime('D%Y%m%dT%H%M%S.%f')[:-4] + '.bmp'
        tmptracks = data[pd.to_datetime(data['t2'])==pd.to_datetime(timestamp)]
    #     tmptracks = tracks_[pd.to_datetime(timestamp)==pd.to_datetime(tracks_['t2'])]
        if len(tmptracks)==0:
            print(name, 'no tracks.')
            continue
    #     tmptracks = tracks[pd.to_datetime(timestamp)==pd.to_datetime(tracks['t2'])]

#         plt.close('all')
        plt.figure(figsize=(7,10))
        r, c = np.shape(im)
        plt.imshow(rotate(explode_contrast(np.uint8(im)),
            270, resize=True),
                cmap='gray',
                extent=[0, r * PIX_SIZE / 1000,
                    0, c * PIX_SIZE / 1000])
        #plt.title(str(timestamp))

        tmptracks_ = tmptracks[(tmptracks['n-tracks'] > 1) & (tmptracks['n-tracks'] <= 150)]
        for m in tmptracks_.index:
            match3 = tmptracks_.loc[m]
            match3 = calculate_speed_df(match3, PIX_SIZE)
            linex = np.float64(
                [match3['x-arrival'], match3['x-departure']])
            liney = np.float64(
                [match3['y-arrival'], match3['y-departure']])
            linex *= PIX_SIZE / 1000
            liney *= PIX_SIZE / 1000
            linex = c * PIX_SIZE / 1000 - linex
            liney = r * PIX_SIZE / 1000 - liney
            plt.plot(liney, linex, 'b-', linewidth=1)

            plt.text(liney[0], linex[0], ('{:0.2f}mm {:0.2f}mm/s'.format(match3['length'] * PIX_SIZE / 1000,
                                                                         match3['S_cms'] * 10)),
                     fontsize=12, color='b')

        if False:
            tmptracks_ = tmptracks[(tmptracks['n-tracks']>1) & (tmptracks['n-tracks']<=track_length_limit)]
            for m in tmptracks_.index:
                match3 = tmptracks_.loc[m]
                match3 = calculate_speed_df(match3, PIX_SIZE)
                linex = np.float64(
                        [match3['x-arrival'], match3['x-departure']])
                liney = np.float64(
                        [match3['y-arrival'], match3['y-departure']])
                linex *= PIX_SIZE/1000
                liney *= PIX_SIZE/1000
                linex = c * PIX_SIZE/1000 - linex
                liney = r * PIX_SIZE/1000 - liney
                plt.plot(liney, linex ,'r-', linewidth=1)

                plt.text(liney[0], linex[0], ('{:0.2f}mm {:0.2f}mm/s'.format(match3['length']*PIX_SIZE/1000,
                                                                             match3['S_cms']*10)),
                         fontsize=12, color='r')

            tmptracks = tmptracks[tmptracks['n-tracks']>track_length_limit]
            for m in tmptracks.index:
                match3 = tmptracks.loc[m]
                match3 = calculate_speed_df(match3, PIX_SIZE)
                linex = np.float64(
                        [match3['x-arrival'], match3['x-departure']])
                liney = np.float64(
                        [match3['y-arrival'], match3['y-departure']])
                linex *= PIX_SIZE/1000
                liney *= PIX_SIZE/1000
                linex = c * PIX_SIZE/1000 - linex
                liney = r * PIX_SIZE/1000 - liney
                plt.plot(liney, linex ,'g-', linewidth=2)

                plt.text(liney[0], linex[0], ('{:0.2f}mm {:0.2f}mm/s'.format(match3['length']*PIX_SIZE/1000,
                                                                             match3['S_cms']*10)),
                         fontsize=12, color='g')

    #     except:
    #         pass
        #plt.axis('off')
        plt.xlabel('mm')
        plt.ylabel('mm')
        plt.gca().invert_yaxis()

        plt.savefig(os.path.join(outputdir, name[:-4] + '-tr.png'), dpi=300, bbox_inches='tight')
        #plt.savefig(os.path.join(outputdir, name[:-4] + '-tr.png'), dpi=50)
    #     input()

    # use convert -delay 12 -loop 0 *.png output_ALL.gif to make a gif


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
    # plt.yscale('log')
    plt.xticks(rotation=45, horizontalalignment='right')
    plt.ylim(0, 2.5)

    figurename = os.path.join(figurename + '.png')
    print('  saving:', figurename)
    plt.savefig(figurename,dpi=600, bbox_inches='tight')
    print('  saved')