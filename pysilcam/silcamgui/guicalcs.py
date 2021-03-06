from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
import os
from pysilcam.config import PySilcamSettings
import pysilcam.oilgas as scog
import numpy as np
import pysilcam.postprocess as sc_pp
import pandas as pd
from enum import Enum
import psutil
from tqdm import tqdm
from pysilcam.fakepymba import silcam_load
import pygame


def count_data(datadir):
    silcfiles = [os.path.join(datadir, f) for f in
                 sorted(os.listdir(datadir))
                 if f.endswith('.silc')]
    bmpfiles = [os.path.join(datadir, f) for f in
                sorted(os.listdir(datadir))
                if f.endswith('.bmp')]
    silc = len(silcfiles)
    bmp = len(bmpfiles)
    return silc, bmp


def extract_stats_im(guidata):
    imc = guidata['imc']
    del guidata['imc']
    stats = pd.DataFrame.from_dict(guidata)
    return stats, imc


def export_timeseries(configfile, statsfile):
    settings = PySilcamSettings(configfile)

    print('Loading STATS data: ', statsfile)
    stats = pd.read_hdf(statsfile, 'ParticleStats/stats')

    stats['timestamp'] = pd.to_datetime(stats['timestamp'])

    stats.sort_values(by='timestamp', inplace=True)

    print('Extracting oil and gas')
    stats_oil = scog.extract_oil(stats)
    stats_gas = scog.extract_gas(stats)

    print('Calculating timeseries')
    u = pd.to_datetime(stats['timestamp']).unique()

    sample_volume = sc_pp.get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)

    td = pd.to_timedelta('00:00:' + str(settings.PostProcess.window_size / 2.))

    vdts_all = []
    vdts_oil = []
    vdts_gas = []
    d50_all = []
    d50_oil = []
    d50_gas = []
    timestamp = []
    d50_av_all = []
    d50_av_oil = []
    d50_av_gas = []
    gor = []
    for s in tqdm(u):
        timestamp.append(pd.to_datetime(s))
        dt = pd.to_datetime(s)

        dias, vd_all = sc_pp.vd_from_stats(stats[stats['timestamp'] == s],
                                           settings.PostProcess)
        dias, vd_oil = sc_pp.vd_from_stats(stats_oil[stats_oil['timestamp'] == s],
                                           settings.PostProcess)
        dias, vd_gas = sc_pp.vd_from_stats(stats_gas[stats_gas['timestamp'] == s],
                                           settings.PostProcess)

        nims = sc_pp.count_images_in_stats(stats[stats['timestamp'] == s])
        sv = sample_volume * nims
        vd_all /= sv
        vd_oil /= sv
        vd_gas /= sv
        d50_all.append(sc_pp.d50_from_vd(vd_all, dias))
        d50_oil.append(sc_pp.d50_from_vd(vd_oil, dias))
        d50_gas.append(sc_pp.d50_from_vd(vd_gas, dias))

        vdts_all.append(vd_all)
        vdts_oil.append(vd_oil)
        vdts_gas.append(vd_gas)

        stats_av = stats[(stats['timestamp'] < (dt + td)) & (stats['timestamp'] > (dt - td))]
        stats_av_oil = scog.extract_oil(stats_av)
        stats_av_gas = scog.extract_gas(stats_av)
        d50_av_all.append(sc_pp.d50_from_stats(stats_av, settings.PostProcess))
        d50_av_oil.append(sc_pp.d50_from_stats(stats_av_oil, settings.PostProcess))
        d50_av_gas.append(sc_pp.d50_from_stats(stats_av_gas, settings.PostProcess))

        dias, vdts_av = sc_pp.vd_from_stats(stats_av, settings.PostProcess)
        dias, vdts_av_oil = sc_pp.vd_from_stats(stats_av_oil, settings.PostProcess)
        dias, vdts_av_gas = sc_pp.vd_from_stats(stats_av_gas, settings.PostProcess)
        nims = sc_pp.count_images_in_stats(stats_av)
        sv = sample_volume * nims
        vdts_av /= sv
        vdts_av_oil /= sv
        vdts_av_gas /= sv

        gor.append(np.sum(vdts_av_gas) / np.sum(vdts_av_oil))

    outpath, outfile = os.path.split(statsfile)
    outfile = outfile.replace('-STATS.h5', '')
    outfile = os.path.join(outpath, outfile)

    time_series = pd.DataFrame(data=np.squeeze(vdts_all), columns=dias)
    time_series['D50'] = d50_all
    time_series['Time'] = timestamp
    time_series.to_excel(outfile +
                         '-TIMESERIES' + '' + '.xlsx')

    time_series = pd.DataFrame(data=np.squeeze(vdts_oil), columns=dias)
    time_series['D50'] = d50_oil
    time_series['Time'] = timestamp
    time_series.to_excel(outfile +
                         '-TIMESERIES' + 'oil' + '.xlsx')

    time_series = pd.DataFrame(data=np.squeeze(vdts_gas), columns=dias)
    time_series['D50'] = d50_gas
    time_series['Time'] = timestamp
    time_series.to_excel(outfile +
                         '-TIMESERIES' + 'gas' + '.xlsx')

    print('Exporting averages... ')

    # average all
    dias, vd = sc_pp.vd_from_stats(stats,
                                   settings.PostProcess)
    nims = sc_pp.count_images_in_stats(stats)
    sv = sample_volume * nims
    vd /= sv
    d50 = sc_pp.d50_from_vd(vd, dias)
    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50
    timestamp = np.min(pd.to_datetime(stats['timestamp']))
    dfa['Time'] = timestamp
    dfa.to_excel(statsfile.replace('-STATS.h5', '') +
                 '-AVERAGE' + '' + '.xlsx')

    # average oil
    dias, vd = sc_pp.vd_from_stats(stats_oil,
                                   settings.PostProcess)
    vd /= sv  # sample volume remains the same as 'all'
    d50 = sc_pp.d50_from_vd(vd, dias)
    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50
    timestamp = np.min(pd.to_datetime(stats['timestamp']))  # still use total stats for this time
    dfa['Time'] = timestamp
    dfa.to_excel(statsfile.replace('-STATS.h5', '') +
                 '-AVERAGE' + 'oil' + '.xlsx')

    # average gas
    dias, vd = sc_pp.vd_from_stats(stats_gas,
                                   settings.PostProcess)
    vd /= sv  # sample volume remains the same as 'all'
    d50 = sc_pp.d50_from_vd(vd, dias)
    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50
    timestamp = np.min(pd.to_datetime(stats['timestamp']))  # still use total stats for this time
    dfa['Time'] = timestamp
    dfa.to_excel(statsfile.replace('-STATS.h5', '') +
                 '-AVERAGE' + 'gas' + '.xlsx')

    print('Export done: ', outfile)


def load_image(filename, size):
    im = silcam_load(filename)
    im = pygame.surfarray.make_surface(np.uint8(im))
    im = pygame.transform.flip(im, False, True)
    im = pygame.transform.rotate(im, -90)
    im = pygame.transform.scale(im, size)

    return im


def liveview(datadir, config_file):
    import pysilcam.silcamgui.liveviewer as lv
    lv.liveview(datadir, config_file)


def annotate(datadir, filename):
    a_path = os.path.join(datadir, "annotations.txt")
    if os.path.isfile(a_path):
        f = open(a_path, 'a')
    else:
        f = open(a_path, 'w')
    f.write('{}, \n'.format(os.path.basename(filename)))
    f.close()


def silcview(datadir):
    files = [os.path.join(datadir, f) for f in sorted(os.listdir(datadir)) if f.endswith('.silc') or f.endswith('.bmp')
             or f.endswith('.silc_mono')]
    if len(files) == 0:
        return
    pygame.init()
    info = pygame.display.Info()
    size = (int(info.current_h / (2048 / 2448)) - 100, info.current_h - 100)
    screen = pygame.display.set_mode(size)
    font = pygame.font.SysFont("monospace", 20)
    font_colour = (0, 127, 127)
    c = pygame.time.Clock()
    zoom = False
    counter = -1
    annotate_counter = 0
    direction = 1  # 1=forward 2=backward
    last_direction = direction
    pause = False
    pygame.event.set_blocked(pygame.MOUSEMOTION)

    while True:
        if pause:
            event = pygame.event.wait()
            if event.type in [12, pygame.QUIT]:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    zoom = np.invert(zoom)
                elif event.key == pygame.K_LEFT:
                    counter -= 1
                elif event.key == pygame.K_RIGHT:
                    counter += 1
                elif event.key == pygame.K_HOME:
                    counter = 0
                elif event.key == pygame.K_END:
                    counter = len(files) - 1
                elif event.key == pygame.K_n:
                    annotate_counter += 1
                    if counter > -1:
                        annotate(datadir, files[counter])
                    else:
                        annotate(datadir, files[0])  # This should only happen if a is pressed on the very first frame
                elif event.key == pygame.K_p:
                    pause = np.invert(pause)
                    direction = last_direction
                elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    pygame.quit()
                    return
                else:
                    continue
                pygame.time.wait(10)

        counter += direction
        counter = np.max([counter, 0])
        counter = np.min([len(files) - 1, counter])
        c.tick(15)  # restrict to 15Hz
        f = files[counter]

        if not (counter == 0 | counter == len(files) - 1):
            filename = os.path.join(datadir, f)
            im = load_image(filename, size)

        if zoom:
            label = font.render('ZOOM [F]: ON', 1, font_colour)
            im = pygame.transform.scale2x(im)
            screen.blit(im, (-size[0] / 2, -size[1] / 2))
        else:
            im = pygame.transform.scale(im, size)
            screen.blit(im, (0, 0))
            label = font.render('ZOOM [F]: OFF', 1, font_colour)
        screen.blit(label, (0, size[1] - 20))

        if direction == 1:
            dirtxt = '>>'
        elif direction == -1:
            dirtxt = '<<'
        if pause and not ('AUSED' in dirtxt):
            dirtxt = 'AUSED ' + dirtxt
        label = font.render('DIRECTION [HOME|<-|->|END] [P]' + dirtxt, 1, font_colour)
        screen.blit(label, (0, size[1] - 40))

        if counter == 0:
            label = font.render('FIRST IMAGE', 1, font_colour)
            screen.blit(label, (0, size[1] - 60))
        elif counter == len(files) - 1:
            label = font.render('LAST IMAGE', 1, font_colour)
            screen.blit(label, (0, size[1] - 60))

        timestamp = pd.to_datetime(
            os.path.splitext(os.path.split(f)[-1])[0][1:])

        pygame.display.set_caption('raw image replay:' + os.path.split(f)[0])  # , icontitle=None)
        label = font.render(str(timestamp), 20, font_colour)
        screen.blit(label, (0, 0))
        label = font.render('Frame: {0}/{1}'.format(counter, len(files) - 1), 1, font_colour)
        screen.blit(label, (0, 20))
        label = font.render('Display FPS: {:0.2f}'.format(c.get_fps()),
                            1, font_colour)
        screen.blit(label, (0, 40))

        if annotate_counter > 0:
            label = font.render('Annotations: {}'.format(annotate_counter), 1, font_colour)
            screen.blit(label, (size[0] - 200, size[1] - 20))

        if not pause:
            for event in pygame.event.get():
                if event.type in [12, pygame.QUIT]:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        zoom = np.invert(zoom)
                    elif event.key == pygame.K_LEFT:
                        direction = -1
                    elif event.key == pygame.K_RIGHT:
                        direction = 1
                    elif event.key == pygame.K_HOME:
                        counter = 0
                        direction = 1
                    elif event.key == pygame.K_END:
                        counter = len(files) - 1
                        direction = -1
                    elif event.key == pygame.K_p:
                        pause = np.invert(pause)
                        last_direction = direction
                        direction = 0
                    elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        pygame.quit()
                        return

        pygame.display.flip()

    pygame.quit()


class process_mode(Enum):
    process = 1
    aquire = 2
    real_time = 3


class ProcThread(Process):
    run_type = process_mode.process

    def __init__(self, datadir, configfile, disc_write, run_type, overwriteSTATS, fighandle):
        super(ProcThread, self).__init__()
        self.q = Queue(1)
        self.info = 'ini done'
        self.datadir = datadir
        self.configfile = configfile
        self.settings = ''
        self.rts = ''
        self.disc_write = disc_write
        self.run_type = run_type
        self.overwriteSTATS = overwriteSTATS
        self.fighandle = fighandle

    def run(self):
        import pysilcam.__main__ as psc
        if (self.run_type == process_mode.process):
            psc.silcam_process(self.configfile, self.datadir, multiProcess=True, realtime=False,
                               gui=self.q, overwriteSTATS=self.overwriteSTATS)
        elif (self.run_type == process_mode.aquire):
            psc.silcam_acquire(self.datadir, config_filename=self.configfile, writeToDisk=self.disc_write, gui=self.q)
        elif (self.run_type == process_mode.real_time):
            if 'REALTIME_DISC' in os.environ.keys():
                psc.silcam_process(self.configfile, self.datadir, multiProcess=False, realtime=True,
                                   discWrite=False, gui=self.q, overwriteSTATS=True)
            else:
                psc.silcam_process(self.configfile, self.datadir, multiProcess=True, realtime=True,
                                   discWrite=self.disc_write, gui=self.q, overwriteSTATS=self.overwriteSTATS)

    def go(self):
        self.start()

    def stop_silcam(self):

        if self.is_alive():
            self.terminate()
            # terminate all children processes
            list = psutil.Process(self.pid).children()
            for p in list:
                p.terminate()
            self.join()
            self.info = 'termination sent'
        else:
            self.info = 'nothing to terminate'

    def plot(self):
        infostr = 'waiting to plot'
        if self.rts == '':
            self.rts = scog.rt_stats(self.settings)

        if self.is_alive():

            try:
                guidata = self.q.get(timeout=0.1)
            except:  # noqa: E722
                return

            timestamp = guidata[0]
            imc = guidata[1]
            imraw = guidata[2]
            dias = guidata[3]['dias']
            vd_oil = guidata[3]['vd_oil']
            vd_gas = guidata[3]['vd_gas']
            gor = guidata[3]['gor']
            oil_d50 = guidata[3]['oil_d50']
            gas_d50 = guidata[3]['gas_d50']
            saturation = guidata[3]['saturation']

            infostr = 'got data'

            fig = plt.figure(self.fighandle)
            axispsd = plt.subplot(2, 2, 1)
            axispsd.clear()
            axispsd.plot(dias, vd_oil, 'r')
            axispsd.plot(dias, vd_gas, 'b')
            axispsd.set_xscale('log')
            axispsd.set_xlim((50, 12000))
            axispsd.set_ylabel('Volume Concentration [uL/L]')
            axispsd.set_xlabel('Diameter [um]')

            axis_imraw = plt.subplot(2, 2, 3)
            axis_imraw.clear()
            ttlstr = (
                    'Oil d50: {:0.0f} [um]'.format(oil_d50) + '\n' +
                    'Gas d50: {:0.0f} [um]'.format(gas_d50) + '\n' +
                    'GOR: {:0.2f}'.format(gor) + ' ' + ' Saturation: {:0.0f} [%]'.format(saturation)
            )
            axis_imraw.set_title(ttlstr)
            axis_imraw.imshow(imraw)
            axis_imraw.axis('off')

            axis_imc = plt.subplot(1, 2, 2)
            ttlstr = ('Image time: ' + str(timestamp))
            axis_imc.clear()
            axis_imc.imshow(imc)
            axis_imc.axis('off')
            axis_imc.set_title(ttlstr)

            fig.tight_layout()

            self.info = infostr

    def load_settings(self, configfile):
        self.settings = PySilcamSettings(configfile)
